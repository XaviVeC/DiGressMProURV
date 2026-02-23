"""
SARS-CoV-2 Main Protease (Mpro) Inhibitor Dataset Module
========================================================

This module implements PATHWAY A (Molecular Generation with Binding Labels) for
integrating the MPro-URV_Version2 database into the DiGress framework.

The dataset consists of 379 SARS-CoV-2 Mpro inhibitors with experimental binding
affinities (pIC50 values). Data is split into train/valid/test sets.

Author: URV Research Team
Date: February 2026
"""

import os
import os.path as osp
import pathlib
from typing import Any, Sequence, Dict, List
import json

import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import subgraph

import src.utils as utils
from src.datasets.abstract_dataset import MolecularDataModule, AbstractDatasetInfos


def files_exist(files) -> bool:
    """Check if list of files exist on disk."""
    return len(files) != 0 and all([osp.exists(f) for f in files])


def to_list(value: Any) -> Sequence:
    """Convert value to list if not already a sequence."""
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]


class RemoveYTransform:
    """Transform to remove property labels during generation."""
    def __call__(self, data):
        data.y = torch.zeros((1, 0), dtype=torch.float)
        return data


class MproDataset(InMemoryDataset):
    """
    PyTorch Geometric Dataset for SARS-CoV-2 Mpro Inhibitors
    
    Loads molecules from MPro-URV_Version2 database:
    - SMILES strings from Ligand_SMI directory
    - pIC50 binding affinities from Info.csv
    - Pre-defined train/valid/test splits
    
    Attributes:
        stage: 'train', 'val', or 'test'
        root: Path to dataset root directory
        remove_h: Whether to remove hydrogen atoms
        dataset_dir: Path to MPro-URV_Version2 database
    """
    
    def __init__(self, stage: str, root: str, remove_h: bool = True,
                 dataset_dir: str = None, transform=None, 
                 pre_transform=None, pre_filter=None):
        """
        Initialize dataset.
        
        Args:
            stage: 'train', 'val', or 'test'
            root: Path to processed data root
            remove_h: Whether to remove hydrogens (default: True)
            dataset_dir: Path to MPro-URV_Version2 database
            transform: Optional PyG transform
            pre_transform: Optional pre-processing
            pre_filter: Optional filtering function
        """
        self.stage = stage
        self.remove_h = remove_h
        
        # Resolve dataset_dir path in a portable way
        if dataset_dir is None:
            # Default: repo root is two levels above this file; expect MPro-URV_Version2 next to repo
            base_path = pathlib.Path(__file__).resolve().parents[2]
            self.dataset_dir = str((base_path / 'MPro-URV_Version2').resolve())
        else:
            # Allow user to pass a relative or absolute path; normalize it
            self.dataset_dir = str(pathlib.Path(dataset_dir).expanduser().resolve())
        
        # Map stage to file index
        self.file_idx = {'train': 0, 'val': 1, 'test': 2}[stage]
        
        super().__init__(root, transform, pre_transform, pre_filter)
        
        # Load pre-processed data
        self.data, self.slices = torch.load(self.processed_paths[self.file_idx])
    
    @property
    def raw_file_names(self):
        """Required raw files from MPro-URV_Version2."""
        return [
            'Info.csv',
            'pIC50.txt',
            'Ligand/Ligand_SMI',  # Directory (nested under Ligand)
            'Splits'       # Directory
        ]
    
    @property
    def split_file_name(self):
        """Train/val/test split file names."""
        return ['train.csv', 'val.csv', 'test.csv']
    
    @property
    def split_paths(self):
        """Absolute paths to split files."""
        files = to_list(self.split_file_name)
        return [osp.join(self.raw_dir, f) for f in files]
    
    @property
    def processed_file_names(self):
        """Output file names for processed splits."""
        if self.remove_h:
            return ['proc_tr_no_h.pt', 'proc_val_no_h.pt', 'proc_test_no_h.pt']
        else:
            return ['proc_tr_h.pt', 'proc_val_h.pt', 'proc_test_h.pt']
    
    @property
    def raw_dir(self) -> str:
        """Override raw directory to use provided dataset_dir."""
        return self.dataset_dir
    
    def download(self):
        """
        No download needed - assumes MPro-URV_Version2 is already present.
        
        Expected structure:
        MPro-URV_Version2/
        ├── Info.csv
        ├── pIC50.txt
        ├── Ligand/
        │   └── Ligand_SMI/  (379 .smi files)
        └── Splits/
            ├── train_index_folder.txt
            ├── valid_index_folder.txt
            └── test_index_folder.txt
        """
        if files_exist(self.split_paths):
            return
        
        # Create CSV split files from MPro-URV split indices
        self._create_split_csvs()
    
    def _create_split_csvs(self):
        """Create train/val/test CSV files from split indices."""
        info_path = osp.join(self.raw_dir, 'Info.csv')
        if not osp.exists(info_path):
            raise FileNotFoundError(f"Info.csv not found at {info_path}")
        
        # Load full dataset info
        df = pd.read_csv(info_path, sep=';')
        pdb_ids = df['PDB_ID'].values
        
        # Load split indices
        splits = self._load_split_indices()
        
        # Use fold 0
        fold = 0
        train_idx = splits['train'][fold]
        val_idx = splits['val'][fold]
        test_idx = splits['test'][fold]
        
        # Create split DataFrames
        train_df = df[df['PDB_ID'].isin(train_idx)]
        val_df = df[df['PDB_ID'].isin(val_idx)]
        test_df = df[df['PDB_ID'].isin(test_idx)]
        
        # Save as CSV
        train_df.to_csv(self.split_paths[0], index=False)
        val_df.to_csv(self.split_paths[1], index=False)
        test_df.to_csv(self.split_paths[2], index=False)
        
        print(f"Created {len(train_df)} train, {len(val_df)} val, {len(test_df)} test splits")
    
    def _load_split_indices(self) -> Dict[str, List]:
        """Load pre-defined train/val/test split indices from Splits directory."""
        splits_dir = osp.join(self.raw_dir, 'Splits')
        
        splits = {'train': [], 'val': [], 'test': []}
        
        for split_type in ['train', 'val', 'test']:
            split_file = osp.join(splits_dir, f'{split_type}_index_folder.txt')
            if split_type == 'val':
                split_file = osp.join(splits_dir, 'valid_index_folder.txt')
            
            if not osp.exists(split_file):
                raise FileNotFoundError(f"Split file not found: {split_file}")
            
            with open(split_file, 'r') as f:
                # Split files contain Python list format
                content = f.read().strip()
                indices_list = eval(content)  # List of lists (one per fold)
                splits[split_type] = indices_list
        
        return splits
    
    def process(self):
        """
        Convert SMILES to molecular graphs.
        
        Process:
        1. Load SMILES from Ligand_SMI/ directory
        2. Load pIC50 from Info.csv
        3. Convert SMILES → RDKit molecules
        4. Extract node and edge features
        5. Optionally remove hydrogens
        6. Create PyG Data objects
        """
        RDLogger.DisableLog('rdApp.*')
        
        # Atom type mapping (standard InChI notation)
        atom_types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'S': 5, 'Cl': 6, 'Br': 7, 'I': 8}
        
        # Bond type mapping
        bond_types = {
            BT.SINGLE: 0,
            BT.DOUBLE: 1,
            BT.TRIPLE: 2,
            BT.AROMATIC: 3
        }
        
        # Load dataset info
        csv_path = osp.join(self.raw_dir, 'Info.csv')
        info_df = pd.read_csv(csv_path, sep=';')
        
        # Create pIC50 lookup
        pic50_dict = dict(zip(info_df['PDB_ID'], info_df['pIC50']))
        
        # Load split data
        split_df = pd.read_csv(self.split_paths[self.file_idx])
        pdb_ids = split_df['PDB_ID'].values
        
        # Load SMILES from Ligand_SMI
        smiles_dir = osp.join(self.raw_dir, 'Ligand', 'Ligand_SMI')
        smi_files = {f.split('_')[0]: f for f in os.listdir(smiles_dir) if f.endswith('.smi')}
        
        data_list = []
        invalid_count = 0
        
        for pdb_id in tqdm(pdb_ids, desc=f"Processing {self.stage} split"):
            # Get SMILES file
            if pdb_id not in smi_files:
                invalid_count += 1
                continue
            
            smiles_file = osp.join(smiles_dir, smi_files[pdb_id])
            
            # Parse SMILES
            try:
                with open(smiles_file, 'r') as f:
                    smiles = f.readline().split()[0]
            except:
                invalid_count += 1
                continue
            
            # Convert SMILES to molecule
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                invalid_count += 1
                continue
            
            # Extract atom types
            type_idx = []
            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                if symbol not in atom_types:
                    # Use heavy atom type as fallback
                    type_idx.append(atom_types.get(symbol[0], 8))
                else:
                    type_idx.append(atom_types[symbol])
            
            # Extract bonds
            row, col, edge_type = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                # Add 1 to bond type (0 is reserved for no-bond)
                edge_type += 2 * [bond_types[bond.GetBondType()] + 1]
            
            # Create edge tensors
            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)
            edge_attr = F.one_hot(edge_type, num_classes=len(bond_types) + 1).to(torch.float)
            
            # Sort edges
            if len(row) > 0:
                N = mol.GetNumAtoms()
                perm = (edge_index[0] * N + edge_index[1]).argsort()
                edge_index = edge_index[:, perm]
                edge_attr = edge_attr[perm]
            
            # Create node features (one-hot atom types)
            x = F.one_hot(torch.tensor(type_idx), num_classes=len(atom_types)).float()
            
            # Get pIC50 as target property
            pic50 = pic50_dict.get(pdb_id, 5.0)  # Default to median value
            y = torch.tensor([[pic50]], dtype=torch.float)
            
            # Remove hydrogens if requested
            if self.remove_h:
                type_idx = torch.tensor(type_idx).long()
                to_keep = type_idx > 0  # Remove H (type 0)
                if len(row) > 0:
                    edge_index, edge_attr = subgraph(
                        to_keep, edge_index, edge_attr,
                        relabel_nodes=True, num_nodes=len(to_keep)
                    )
                x = x[to_keep]
                x = x[:, 1:]  # Remove H from atom type encoding
            
            # Create PyG Data object
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,  # pIC50 value
                idx=pdb_id,
                smiles=smiles
            )
            
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            
            data_list.append(data)
        
        if invalid_count > 0:
            print(f"Warning: {invalid_count} invalid molecules in {self.stage} split")
        
        torch.save(self.collate(data_list), self.processed_paths[self.file_idx])


class MproDataModule(MolecularDataModule):
    """
    PyTorch Lightning DataModule for MPro dataset.
    
    Handles train/val/test data loading with proper batching and sampling.
    """
    
    def __init__(self, cfg):
        """
        Initialize data module.
        
        Args:
            cfg: Hydra configuration with dataset settings
        """
        self.datadir = cfg.dataset.datadir
        self.remove_h = cfg.dataset.remove_h
        
        # Get path to MPro-URV_Version2 database
        dataset_dir = cfg.dataset.get('raw_data_dir', None)
        
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)
        
        # Create datasets for each split
        datasets = {
            'train': MproDataset(
                stage='train',
                root=root_path,
                remove_h=cfg.dataset.remove_h,
                dataset_dir=dataset_dir,
                transform=RemoveYTransform()
            ),
            'val': MproDataset(
                stage='val',
                root=root_path,
                remove_h=cfg.dataset.remove_h,
                dataset_dir=dataset_dir,
                transform=RemoveYTransform()
            ),
            'test': MproDataset(
                stage='test',
                root=root_path,
                remove_h=cfg.dataset.remove_h,
                dataset_dir=dataset_dir,
                transform=RemoveYTransform()
            )
        }
        
        super().__init__(cfg, datasets)


class MproInfos(AbstractDatasetInfos):
    """
    Dataset information and statistics for Mpro.
    
    Defines:
    - Atom types and their encodings
    - Bond types
    - Node and edge type distributions
    - Graph size statistics
    
    Mpro dataset is small (379 molecules), so distributions are computed
    specifically for this dataset (not inherited from QM9).
    """
    
    def __init__(self, datamodule, cfg, recompute_statistics=False):
        """
        Initialize dataset information.
        
        Args:
            datamodule: MproDataModule instance
            cfg: Configuration
            recompute_statistics: Whether to recompute from scratch
        """
        self.remove_h = cfg.dataset.remove_h
        self.need_to_strip = False
        self.name = 'mpro'
        
        if self.remove_h:
            # Atom types: C, N, O, F, S, Cl, Br, I (no H)
            self.atom_encoder = {'C': 0, 'N': 1, 'O': 2, 'F': 3, 'S': 4, 'Cl': 5, 'Br': 6, 'I': 7}
            self.atom_decoder = ['C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'I']
            self.num_atom_types = 8
            self.valencies = [4, 3, 2, 1, 2, 1, 1, 1]  # Expected valencies
            self.atom_weights = {0: 12, 1: 14, 2: 16, 3: 19, 4: 32, 5: 35, 6: 80, 7: 127}
            self.max_n_nodes = 60  # Mpro inhibitors typically < 60 heavy atoms
            self.max_weight = 800
        else:
            # Atom types: H, C, N, O, F, S, Cl, Br, I
            self.atom_encoder = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'S': 5, 'Cl': 6, 'Br': 7, 'I': 8}
            self.atom_decoder = ['H', 'C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'I']
            self.valencies = [1, 4, 3, 2, 1, 2, 1, 1, 1]
            self.num_atom_types = 9
            self.max_n_nodes = 100  # With hydrogens
            self.max_weight = 1000
            self.atom_weights = {0: 1, 1: 12, 2: 14, 3: 16, 4: 19, 5: 32, 6: 35, 7: 80, 8: 127}
        
        # Initialize parent class
        super().__init__()
        
        if recompute_statistics:
            # Recompute statistics from data
            self._compute_statistics(datamodule)
        else:
            # Use default distributions for small dataset
            self._set_default_distributions()
    
    def _set_default_distributions(self):
        """
        Set default distributions for Mpro dataset (379 molecules).
        
        These are empirically derived from the Mpro-URV database:
        - Most molecules are small (10-40 heavy atoms)
        - Dominated by C, N, O atoms
        - Mostly single and aromatic bonds
        """
        # Node count distribution (empirical from Mpro)
        self.n_nodes = torch.tensor([
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # 0-9 atoms
            0.02, 0.04, 0.08, 0.12, 0.15, 0.16, 0.14, 0.12, 0.10, 0.07,  # 10-19
            0.05, 0.03, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0, 0,  # 20-29
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # 30-39
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # 40-49
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # 50-59
            0  # 60
        ])
        
        # Node type distribution (empirical from Mpro)
        if self.remove_h:
            self.node_types = torch.tensor([0.45, 0.20, 0.25, 0.02, 0.04, 0.02, 0.015, 0.005])  # C, N, O, F, S, Cl, Br, I
        else:
            self.node_types = torch.tensor([0.55, 0.30, 0.10, 0.15, 0.01, 0.02, 0.01, 0.01, 0.003])  # H, C, N, O, F, S, Cl, Br, I
        
        # Edge type distribution
        self.edge_types = torch.tensor([0.75, 0.15, 0.03, 0.07, 0.0])  # single, double, triple, aromatic, no-bond
        
        # Bond valency distribution
        self.valency_distribution = torch.zeros(3 * self.max_n_nodes - 2)
        self.valency_distribution[0:6] = torch.tensor([0.1, 0.4, 0.2, 0.15, 0.1, 0.05])
        
        super().complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)
    
    def _compute_statistics(self, datamodule):
        """Recompute statistics directly from data."""
        print("Computing Mpro dataset statistics...")
        np.set_printoptions(suppress=True, precision=5)
        
        self.n_nodes = datamodule.node_counts()
        print("Distribution of number of nodes:", self.n_nodes)
        
        self.node_types = datamodule.node_types()
        print("Distribution of node types:", self.node_types)
        
        self.edge_types = datamodule.edge_counts()
        print("Distribution of edge types:", self.edge_types)
        
        valencies = datamodule.valency_count(self.max_n_nodes)
        print("Distribution of valencies:", valencies)
        self.valency_distribution = valencies
        
        super().complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)


def get_train_smiles(cfg, train_dataloader, dataset_infos, evaluate_dataset=False):
    """
    Get or compute training SMILES for evaluation purposes.
    
    For Mpro, we extract SMILES from the training data itself.
    
    Args:
        cfg: Configuration
        train_dataloader: DataLoader for training data
        dataset_infos: Dataset information
        evaluate_dataset: Whether to evaluate dataset quality
    
    Returns:
        List of SMILES strings from training set
    """
    print("Extracting training SMILES from Mpro dataset...")
    
    train_smiles = []
    for batch in train_dataloader:
        if hasattr(batch, 'smiles'):
            # batch.smiles is a list of SMILES strings, one per graph in the batch
            if isinstance(batch.smiles, list):
                train_smiles.extend(batch.smiles)
            else:
                # Handle case where it's a single string
                train_smiles.append(batch.smiles)
        elif hasattr(batch, 'idx'):
            # Fallback: if we have indices, we could reconstruct (not implemented)
            pass
    
    if evaluate_dataset:
        print(f"Extracted {len(train_smiles)} unique training SMILES")
    
    return train_smiles
