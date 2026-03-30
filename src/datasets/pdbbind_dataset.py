"""
PDBbindv2020 dataset module for DiGress pretraining.

This module adds a molecular-graph dataset built from the local PDBbindv2020
folder so the diffusion model can be pretrained on a broad ligand chemistry
distribution before being fine-tuned on the MPro dataset.
"""

import os
import os.path as osp
import pathlib
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import subgraph
from tqdm import tqdm

from src.datasets.abstract_dataset import MolecularDataModule, AbstractDatasetInfos


PDBBIND_SPLIT_SEED = 42
PDBBIND_MAX_HEAVY_ATOMS = 60
ATOM_TYPES = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'S': 5, 'Cl': 6, 'Br': 7, 'I': 8}
BOND_TYPES = {
    BT.SINGLE: 0,
    BT.DOUBLE: 1,
    BT.TRIPLE: 2,
    BT.AROMATIC: 3,
}


def files_exist(files) -> bool:
    """Check if all provided paths exist on disk."""
    return len(files) != 0 and all(osp.exists(file_path) for file_path in files)


def to_list(value: Any) -> Sequence:
    """Convert a single value to a one-item list when needed."""
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    return [value]


class RemoveYTransform:
    """Remove regression labels so pretraining stays unconditional."""

    def __call__(self, data):
        data.y = torch.zeros((1, 0), dtype=torch.float)
        return data


class PDBbindDataset(InMemoryDataset):
    """PyG dataset for ligand graphs extracted from a local PDBbindv2020 copy."""

    def __init__(
        self,
        stage: str,
        root: str,
        remove_h: bool = True,
        dataset_dir: str = None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.stage = stage
        self.remove_h = remove_h

        if dataset_dir is None:
            base_path = pathlib.Path(__file__).resolve().parents[3]
            self.dataset_dir = str((base_path / 'PDBbindv2020').resolve())
        else:
            self.dataset_dir = str(pathlib.Path(dataset_dir).expanduser().resolve())

        self.file_idx = {'train': 0, 'val': 1, 'test': 2}[stage]
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[self.file_idx])

    @property
    def raw_file_names(self):
        return ['Total.csv', 'Ligand_SMILES', *self.split_file_name]

    @property
    def split_file_name(self):
        return ['pdbbind_train.csv', 'pdbbind_val.csv', 'pdbbind_test.csv']

    @property
    def split_paths(self):
        files = to_list(self.split_file_name)
        return [osp.join(self.raw_dir, file_name) for file_name in files]

    @property
    def processed_file_names(self):
        if self.remove_h:
            return ['pdbbind_proc_tr_no_h.pt', 'pdbbind_proc_val_no_h.pt', 'pdbbind_proc_test_no_h.pt']
        return ['pdbbind_proc_tr_h.pt', 'pdbbind_proc_val_h.pt', 'pdbbind_proc_test_h.pt']

    @property
    def raw_dir(self) -> str:
        return self.dataset_dir

    def download(self):
        if files_exist(self.split_paths):
            return
        self._create_split_csvs()

    def _record_to_complex_id(self, record: Dict[str, Any]) -> str:
        protein = str(record['[protein]']).strip()
        ligand = str(record['[ligand]']).strip()
        return f'{protein}_{ligand}'

    def _smiles_path(self, complex_id: str) -> str:
        return osp.join(self.raw_dir, 'Ligand_SMILES', f'{complex_id}.smi')

    def _read_smiles_file(self, smiles_path: str) -> str:
        with open(smiles_path, 'r', encoding='utf-8') as handle:
            line = handle.readline().strip()
        if not line:
            raise ValueError(f'Empty SMILES file: {smiles_path}')
        if '\t' in line:
            return line.split('\t', 1)[0].strip()
        return line.split()[0].strip()

    def _heavy_atom_count(self, mol: Chem.Mol) -> int:
        return sum(atom.GetAtomicNum() > 1 for atom in mol.GetAtoms())

    def _has_supported_atoms(self, mol: Chem.Mol) -> bool:
        return all(atom.GetSymbol() in ATOM_TYPES for atom in mol.GetAtoms())

    def _parse_bool(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {'true', '1', 'yes'}

    def _load_valid_records(self) -> pd.DataFrame:
        total_path = osp.join(self.raw_dir, 'Total.csv')
        if not osp.exists(total_path):
            raise FileNotFoundError(f'Total.csv not found at {total_path}')

        total_df = pd.read_csv(total_path)
        valid_records: List[Dict[str, Any]] = []
        skipped = {
            'missing_smiles_file': 0,
            'empty_smiles_file': 0,
            'invalid_smiles': 0,
            'unsupported_atoms': 0,
            'too_many_heavy_atoms': 0,
        }

        for record in tqdm(total_df.to_dict('records'), desc='Filtering PDBbind records'):
            complex_id = self._record_to_complex_id(record)
            smiles_path = self._smiles_path(complex_id)
            if not osp.exists(smiles_path):
                skipped['missing_smiles_file'] += 1
                continue

            try:
                smiles = self._read_smiles_file(smiles_path)
            except (OSError, ValueError):
                skipped['empty_smiles_file'] += 1
                continue

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                skipped['invalid_smiles'] += 1
                continue

            if not self._has_supported_atoms(mol):
                skipped['unsupported_atoms'] += 1
                continue

            if self._heavy_atom_count(mol) > PDBBIND_MAX_HEAVY_ATOMS:
                skipped['too_many_heavy_atoms'] += 1
                continue

            record['complex_id'] = complex_id
            record['smiles'] = smiles
            valid_records.append(record)

        if not valid_records:
            raise ValueError('No valid PDBbind records remained after filtering.')

        valid_df = pd.DataFrame(valid_records)
        print(
            'PDBbind filtering summary:',
            {key: int(value) for key, value in skipped.items()},
            f'usable={len(valid_df)}',
        )
        return valid_df

    def _create_split_csvs(self):
        valid_df = self._load_valid_records()
        rng = np.random.default_rng(PDBBIND_SPLIT_SEED)
        permutation = rng.permutation(len(valid_df))
        shuffled_df = valid_df.iloc[permutation].reset_index(drop=True)

        n_total = len(shuffled_df)
        n_test = max(1, int(0.1 * n_total))
        n_val = max(1, int(0.1 * n_total))
        n_train = n_total - n_val - n_test
        if n_train <= 0:
            raise ValueError('PDBbind split creation failed because the filtered dataset is too small.')

        train_df = shuffled_df.iloc[:n_train].reset_index(drop=True)
        val_df = shuffled_df.iloc[n_train:n_train + n_val].reset_index(drop=True)
        test_df = shuffled_df.iloc[n_train + n_val:].reset_index(drop=True)

        train_df.to_csv(self.split_paths[0], index=False)
        val_df.to_csv(self.split_paths[1], index=False)
        test_df.to_csv(self.split_paths[2], index=False)

        print(f'Created PDBbind splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}')

    def process(self):
        RDLogger.DisableLog('rdApp.*')
        split_df = pd.read_csv(self.split_paths[self.file_idx])

        data_list = []
        invalid_count = 0
        for record in tqdm(split_df.to_dict('records'), desc=f'Processing PDBbind {self.stage} split'):
            complex_id = record.get('complex_id', self._record_to_complex_id(record))
            smiles = record.get('smiles')
            if not isinstance(smiles, str) or not smiles.strip():
                try:
                    smiles = self._read_smiles_file(self._smiles_path(complex_id))
                except (OSError, ValueError):
                    invalid_count += 1
                    continue

            mol = Chem.MolFromSmiles(smiles)
            if mol is None or not self._has_supported_atoms(mol):
                invalid_count += 1
                continue

            if self._heavy_atom_count(mol) > PDBBIND_MAX_HEAVY_ATOMS:
                invalid_count += 1
                continue

            type_idx: List[int] = []
            unsupported_atoms = False
            for atom in mol.GetAtoms():
                atom_type = ATOM_TYPES.get(atom.GetSymbol())
                if atom_type is None:
                    unsupported_atoms = True
                    break
                type_idx.append(atom_type)

            if unsupported_atoms:
                invalid_count += 1
                continue

            row, col, edge_type = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                edge_type += 2 * [BOND_TYPES[bond.GetBondType()] + 1]

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)
            edge_attr = F.one_hot(edge_type, num_classes=len(BOND_TYPES) + 1).to(torch.float)

            if len(row) > 0:
                num_nodes = mol.GetNumAtoms()
                perm = (edge_index[0] * num_nodes + edge_index[1]).argsort()
                edge_index = edge_index[:, perm]
                edge_attr = edge_attr[perm]

            x = F.one_hot(torch.tensor(type_idx), num_classes=len(ATOM_TYPES)).float()
            binding_value = float(record['[-log k]']) if '[-log k]' in record else 0.0
            y = torch.tensor([[binding_value]], dtype=torch.float)

            if self.remove_h:
                atom_type_tensor = torch.tensor(type_idx).long()
                to_keep = atom_type_tensor > 0
                if not torch.any(to_keep):
                    invalid_count += 1
                    continue
                if len(row) > 0:
                    edge_index, edge_attr = subgraph(
                        to_keep,
                        edge_index,
                        edge_attr,
                        relabel_nodes=True,
                        num_nodes=len(to_keep),
                    )
                x = x[to_keep]
                x = x[:, 1:]

            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
                idx=complex_id,
                protein_id=str(record['[protein]']).strip(),
                ligand_id=str(record['[ligand]']).strip(),
                smiles=smiles,
                binding_value=binding_value,
                covalent_bond=self._parse_bool(record.get('[covalent_bond]', False)),
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        if invalid_count > 0:
            print(f'Warning: {invalid_count} invalid molecules in PDBbind {self.stage} split')

        torch.save(self.collate(data_list), self.processed_paths[self.file_idx])


class PDBbindDataModule(MolecularDataModule):
    """DataModule wrapper for the processed PDBbind splits."""

    def __init__(self, cfg):
        self.datadir = cfg.dataset.datadir
        self.remove_h = cfg.dataset.remove_h
        dataset_dir = cfg.dataset.get('raw_data_dir', None)

        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)

        datasets = {
            'train': PDBbindDataset(
                stage='train',
                root=root_path,
                remove_h=cfg.dataset.remove_h,
                dataset_dir=dataset_dir,
                transform=RemoveYTransform(),
            ),
            'val': PDBbindDataset(
                stage='val',
                root=root_path,
                remove_h=cfg.dataset.remove_h,
                dataset_dir=dataset_dir,
                transform=RemoveYTransform(),
            ),
            'test': PDBbindDataset(
                stage='test',
                root=root_path,
                remove_h=cfg.dataset.remove_h,
                dataset_dir=dataset_dir,
                transform=RemoveYTransform(),
            ),
        }

        super().__init__(cfg, datasets)


class PDBbindInfos(AbstractDatasetInfos):
    """Dataset statistics and molecular vocabulary for PDBbind pretraining."""

    def __init__(self, datamodule, cfg, recompute_statistics: bool = True):
        self.remove_h = cfg.dataset.remove_h
        self.need_to_strip = False
        self.name = 'pdbbind'

        if self.remove_h:
            self.atom_encoder = {'C': 0, 'N': 1, 'O': 2, 'F': 3, 'S': 4, 'Cl': 5, 'Br': 6, 'I': 7}
            self.atom_decoder = ['C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'I']
            self.num_atom_types = 8
            self.valencies = [4, 3, 2, 1, 2, 1, 1, 1]
            self.atom_weights = {0: 12, 1: 14, 2: 16, 3: 19, 4: 32, 5: 35, 6: 80, 7: 127}
            self.max_n_nodes = PDBBIND_MAX_HEAVY_ATOMS
            self.max_weight = 800
        else:
            self.atom_encoder = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'S': 5, 'Cl': 6, 'Br': 7, 'I': 8}
            self.atom_decoder = ['H', 'C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'I']
            self.valencies = [1, 4, 3, 2, 1, 2, 1, 1, 1]
            self.num_atom_types = 9
            self.max_n_nodes = 100
            self.max_weight = 1000
            self.atom_weights = {0: 1, 1: 12, 2: 14, 3: 16, 4: 19, 5: 32, 6: 35, 7: 80, 8: 127}

        super().__init__()

        if recompute_statistics:
            self._compute_statistics(datamodule)
        else:
            self._set_default_distributions()

    def _set_default_distributions(self):
        self.n_nodes = torch.zeros(self.max_n_nodes + 1)
        self.n_nodes[10:41] = torch.tensor([
            0.01, 0.015, 0.02, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06,
            0.06, 0.06, 0.055, 0.05, 0.045, 0.04, 0.035, 0.03, 0.025, 0.02,
            0.018, 0.016, 0.014, 0.012, 0.01, 0.008, 0.006, 0.005, 0.004, 0.003,
            0.002,
        ])
        self.n_nodes = self.n_nodes / self.n_nodes.sum()

        if self.remove_h:
            self.node_types = torch.tensor([0.56, 0.15, 0.17, 0.03, 0.03, 0.03, 0.02, 0.01])
        else:
            self.node_types = torch.tensor([0.45, 0.31, 0.09, 0.10, 0.015, 0.015, 0.01, 0.007, 0.003])

        self.edge_types = torch.tensor([0.88, 0.07, 0.02, 0.0, 0.03])
        self.valency_distribution = torch.zeros(3 * self.max_n_nodes - 2)
        self.valency_distribution[0:6] = torch.tensor([0.08, 0.34, 0.24, 0.18, 0.11, 0.05])
        super().complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)

    def _compute_statistics(self, datamodule):
        print('Computing PDBbind dataset statistics...')
        np.set_printoptions(suppress=True, precision=5)

        self.n_nodes = datamodule.node_counts()
        print('Distribution of number of nodes:', self.n_nodes)

        self.node_types = datamodule.node_types()
        print('Distribution of node types:', self.node_types)

        self.edge_types = datamodule.edge_counts()
        print('Distribution of edge types:', self.edge_types)

        valencies = datamodule.valency_count(self.max_n_nodes)
        print('Distribution of valencies:', valencies)
        self.valency_distribution = valencies

        super().complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)


def get_train_smiles(cfg, train_dataloader, dataset_infos, evaluate_dataset=False):
    """Extract training SMILES for novelty computation during sampling."""
    print('Extracting training SMILES from PDBbind dataset...')

    train_smiles = []
    for batch in train_dataloader:
        if hasattr(batch, 'smiles'):
            if isinstance(batch.smiles, list):
                train_smiles.extend(batch.smiles)
            else:
                train_smiles.append(batch.smiles)

    if evaluate_dataset:
        print(f'Extracted {len(train_smiles)} training SMILES strings')

    return train_smiles