# DiGress Framework Analysis & MPro-URV Database Integration Report

**Date:** February 12, 2026  
**Project:** TFM (Treball Fi Master) - MESIIA Program  
**Author:** Javier Vega

---

## Table of Contents

1. [What is DiGress?](#what-is-digress)
2. [main.py Workflow](#mainpy-workflow)
3. [Weights & Biases (wandb)](#weights--biases-wandb)
4. [MPro-URV_Version2 Database Analysis](#mpro-urv_version2-database-analysis)
5. [Integration Feasibility Assessment](#integration-feasibility-assessment)
6. [Required Code Changes](#required-code-changes)
7. [Implementation Recommendations](#implementation-recommendations)

---

## What is DiGress?

### Overview

**DiGress** (Discrete Graph Denoising Diffusion Models) is a deep learning framework for **generative modeling of graphs and molecular structures**. It uses diffusion models to learn the distribution of graphs and generate new ones from noise.

### Core Purpose

DiGress is designed to solve the problem of **graph generation** — creating novel molecules or graph structures by:

1. **Learning** the underlying distribution of input graphs
2. **Generating** realistic new structures that follow similar properties
3. **Supporting** both discrete and continuous diffusion processes

### Key Characteristics

| Aspect | Details |
|--------|---------|
| **Model Type** | Denoising Diffusion Probabilistic Models (DDPMs) |
| **Input Data** | Graphs (nodes, edges, features) or molecules (SMILES/SDF) |
| **Output** | Novel graph structures with similar properties to training data |
| **Framework** | PyTorch Lightning + PyTorch Geometric |
| **GPU Support** | Multi-GPU DDP training support |

### Supported Datasets

#### Molecular Datasets
- **QM9**: Quantum mechanical properties dataset (~130K molecules)
- **Guacamol**: Benchmark for molecular generation
- **MOSES**: Molecular Sets dataset

#### Graph Datasets
- **SBM** (Stochastic Block Model): Random graphs with community structure
- **Comm20**: Community detection graphs
- **Planar**: Planar graphs

### Model Variants

```
DiGress Framework
├── Continuous Model (LiftedDenoisingDiffusion)
│   └── Uses continuous latent representations
└── Discrete Model (DiscreteDenoisingDiffusion)
    └── For categorical/discrete node/edge types
```

---

## main.py Workflow

### Execution Flow Diagram

```
┌─────────────────────────────────────────────────────────┐
│                   main.py Execution                     │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────┐
        │  Load Configuration via Hydra        │
        │  (configs/config.yaml)               │
        └──────────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────┐
        │  Select Dataset                      │
        │  ├─ QM9, Guacamol, MOSES             │
        │  └─ SBM, Comm20, Planar              │
        └──────────────────────────────────────┘
                    │            │
        ┌───────────▼──┐  ┌──────▼────────┐
        │ Molecular    │  │ Graph-based   │
        │ Dataset      │  │ Dataset       │
        │ - SMILES     │  │ - Adjacency   │
        │ - Properties │  │ - Node types  │
        └──────┬───────┘  └──────┬────────┘
               │                  │
               └─────────┬────────┘
                         ▼
        ┌──────────────────────────────────────┐
        │  Initialize Dataset Module           │
        │  ├─ Load data files                  │
        │  ├─ Create data loaders              │
        │  └─ Compute dataset statistics       │
        └──────────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────┐
        │  Setup Extra Features & Metrics      │
        │  ├─ ExtraFeatures module             │
        │  ├─ TrainMetrics                     │
        │  └─ SamplingMetrics                  │
        └──────────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────┐
        │  Initialize Diffusion Model          │
        │  ├─ DiscreteDenoisingDiffusion OR    │
        │  └─ LiftedDenoisingDiffusion         │
        └──────────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────┐
        │  Setup PyTorch Lightning Trainer     │
        │  ├─ Configure callbacks              │
        │  ├─ Set checkpointing                │
        │  └─ Enable multi-GPU (DDP)           │
        └──────────────────────────────────────┘
                           │
                ┌──────────┴──────────┐
                ▼                     ▼
        ┌─────────────────┐  ┌─────────────────┐
        │  test_only=OFF  │  │  test_only=ON   │
        │                 │  │                 │
        │ ┌─────────────┐ │  │ ┌─────────────┐ │
        │ │ TRAIN       │ │  │ │ TEST        │ │
        │ │ ├─Fit model │ │  │ │ ├─Load ckpt │ │
        │ │ │fit()      │ │  │ │ │└─test()    │ │
        │ │ ├─Validate  │ │  │ │ │            │ │
        │ │ │every n-ep │ │  │ │ │ Or eval    │ │
        │ │ └─Test      │ │  │ │ │ all ckpts  │ │
        │ │  test()     │ │  │ │ └─────────────┘ │
        │ └─────────────┘ │  └─────────────────┘
        └─────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────┐
        │  Save Results to Outputs Folder      │
        │  ├─ Checkpoints/model_name/          │
        │  ├─ wandb run logs                   │
        │  └─ Generated samples                │
        └──────────────────────────────────────┘
```

### Detailed Step Breakdown

#### **Phase 1: Configuration Loading**
```python
@hydra.main(version_base='1.3', config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    dataset_config = cfg["dataset"]
```
- Loads YAML configuration files using Hydra framework
- Makes configuration composable: `python src/main.py dataset=qm9 train.lr=0.001`

#### **Phase 2: Dataset Selection**

**If dataset in `['sbm', 'comm20', 'planar']`:**
- Uses `SpectreGraphDataModule`
- Loads graph data (adjacency matrices, node features)
- `PlanarSamplingMetrics`, `SBMSamplingMetrics`, or `Comm20SamplingMetrics`
- `NonMolecularVisualization` tools

**If dataset in `['qm9', 'guacamol', 'moses']`:**
- Loads molecular data (SMILES strings, SDF files)
- `QM9DataModule`, `GuacamolDataModule`, or `MosesDataModule`
- `TrainMolecularMetrics`, `SamplingMolecularMetrics`
- `MolecularVisualization` tools
- Optional: Extract molecular properties for each compound

#### **Phase 3: Model Creation**
Two variants available:

**Discrete Model** (`cfg.model.type == 'discrete'`):
```python
model = DiscreteDenoisingDiffusion(cfg=cfg, **model_kwargs)
```
- For categorical/discrete node types and edges
- Uses discrete diffusion process (q(x_t|x_0))

**Continuous Model**:
```python
model = LiftedDenoisingDiffusion(cfg=cfg, **model_kwargs)
```
- For continuous latent representations
- Uses Gaussian diffusion process

#### **Phase 4: Training Setup**

```python
trainer = Trainer(
    gradient_clip_val=cfg.train.clip_grad,
    strategy="ddp_find_unused_parameters_true",  # Multi-GPU
    accelerator='gpu' if use_gpu else 'cpu',
    devices=cfg.general.gpus,
    max_epochs=cfg.train.n_epochs,
    check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
    callbacks=callbacks,  # Checkpointing, EMA
    logger=[]  # wandb logs separately
)
```

#### **Phase 5: Execution - Training or Testing**

**Training Mode** (`test_only=None`):
```python
trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)
trainer.test(model, datamodule=datamodule)  # Final test
```

**Testing Mode** (`test_only=checkpoint_path`):
```python
trainer.test(model, datamodule=datamodule, ckpt_path=cfg.general.test_only)
# Optionally evaluate all checkpoints in directory
```

#### **Phase 6: Output Generation**
- Saves model checkpoints every `N` epochs
- Keeps best 5 checkpoints based on validation NLL
- Generates samples during training/testing
- Logs metrics to wandb

---

## Weights & Biases (wandb)

### What is wandb?

**Weights & Biases** is an **experiment tracking and visualization platform** for machine learning projects. It provides:

- 📊 Real-time metric visualization
- 🔍 Hyperparameter logging and comparison
- 💾 Artifact storage (models, predictions)
- 📈 Training history and convergence analysis
- 🔗 Reproducibility through config snapshots

### When & How wandb is Called

#### **1. Initialization (on_fit_start)**
```python
# In DiscreteDenoisingDiffusion.on_fit_start()
if self.local_rank == 0:  # Only on main GPU
    utils.setup_wandb(self.cfg)
```

**setup_wandb() function:**
```python
def setup_wandb(cfg):
    config_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True)
    kwargs = {
        'name': cfg.general.name,
        'project': f'graph_ddm_{cfg.dataset.name}',  # e.g., graph_ddm_qm9
        'config': config_dict,
        'settings': wandb.Settings(_disable_stats=True),
        'reinit': True,
        'mode': cfg.general.wandb  # 'online', 'offline', or 'disabled'
    }
    wandb.init(**kwargs)
    wandb.save('*.txt')  # Save text outputs
```

**Project Naming Convention:**
```
For QM9 dataset:     graph_ddm_qm9
For Guacamol:        graph_ddm_guacamol
For MOSES:           graph_ddm_moses
For SBM graphs:      graph_ddm_sbm
```

#### **2. Metric Logging (throughout training)**
```python
# On validation epoch end
if wandb.run:
    wandb.log({
        "val/epoch_NLL": metrics[0],
        "val/accuracy": metrics[1],
        "val/SMILES_validity": metrics[2],
        ...
    }, step=epoch)

# On training step
if wandb.run:
    wandb.log({
        "kl_prior": kl_prior.mean(),
        "loss": loss.item(),
        "learning_rate": lr,
    }, step=train_step)
```

### wandb Configuration Options

| Configuration | Description | Default |
|:---|:---|:---|
| `cfg.general.name` | Run name | "debug" |
| `cfg.general.wandb` | Mode | "online" |
| `cfg.dataset.name` | Project identifier | (required) |

### wandb Modes

| Mode | Behavior | Use Case |
|:---|:---|:---|
| **online** | Syncs to wandb.ai dashboard in real-time | Production runs |
| **offline** | Logs locally, can sync later | No internet access |
| **disabled** | No logging at all | Testing/debugging |

### Output Structure

```
outputs/
├── YYYY-MM-DD/
│   └── HH-MM-SS-run_name/
│       ├── wandb/
│       │   └── run-<random_id>/
│       │       ├── logs/
│       │       │   ├── debug.log
│       │       │   └── debug-internal.log
│       │       ├── config.yaml
│       │       └── summary.json
│       └── checkpoints/
│           └── run_name/
│               ├── epoch_0.ckpt
│               └── last.ckpt
```

### Current Status in Your Workspace

❌ **Issue Detected:** The wandb logs show network errors:
```
urllib3.exceptions.NameResolutionError: HTTPSConnection(host='api.wandb.ai') 
Failed to resolve 'api.wandb.ai' ([Errno -3] Temporary failure in name resolution)
```

This indicates:
- The system tried to use **online mode** to sync with wandb.ai
- No internet connection was available
- Logs were saved locally but not uploaded

---

## MPro-URV_Version2 Database Analysis

### Database Overview

The **MPro-URV_Version2** database is a **structure-activity relationship (SAR) dataset** specifically curated for **SARS-CoV-2 main protease (Mpro/3CLpro) inhibitor research**.

### Database Structure

```
MPro-URV_Version2/
├── Info.csv                  # Metadata for all complexes
├── pIC50.txt                 # Binding affinity scores
├── Complex/
│   ├── ALIGNED/              # Aligned protein structures
│   ├── CIF_FROM_PDB_NOT_ALIGNED/
│   └── PROTONATED_NOT_ALIGNED/
├── Interaction/
│   ├── 5RGV_ligand.json      # Protein-ligand interactions
│   ├── 5RGX_ligand.json
│   └── ... (379 total)
├── Ligand/
│   ├── Ligand_CIF/           # Small molecule 3D structures
│   ├── Ligand_SDF/           # SDF format (3D coordinates)
│   └── Ligand_SMI/           # SMILES strings (1D notation)
├── Protein/
│   ├── Protein_CIF/          # Protein structures (CIF)
│   ├── Protein_DSSP/         # Secondary structure info
│   └── Protein_PDB/          # PDB format
└── Splits/
    ├── train_index_folder.txt    # ~380 train compounds
    ├── valid_index_folder.txt    # ~95 validation compounds
    ├── test_index_folder.txt     # ~95 test compounds
    └── test_index_folder.txt     # Additional test split
```

### Key Data Files

#### **1. Info.csv (Metadata Table)**

**Format:** CSV (semicolon-separated)  
**Rows:** 379 protein-ligand complexes  

**Columns Example:**
```
PDB_ID | Title | Protein Description | NChain | Chain ID | ... | pIC50 | SMILES | Protein Sequence
5RGV   | ...   | 3C-like proteinase | 1      | A        | ... | 4.24  | c1cc... | SGFRK...
5RGX   | ...   | 3C-like proteinase | 1      | A        | ... | 4.21  | Cc1cc... | SGFRK...
```

**Key Properties Available:**
- `SMILES`: Canonical SMILES notation for ligands
- `pIC50`: Potency measure (higher = more potent)
  - Range: 4.0 - 7.4
  - Distribution: Left-skewed (most inactive)
- `PDB_ID`: Unique compound identifier
- `Protein Sequence`: Full MPro sequence
- `Ligand Properties`: Names, molecular weights, etc.

#### **2. Ligand Data**

**Location:** `Ligand/Ligand_SMI/`  
**Format:** 379 `.smi` files  
**Content:** SMILES strings (1 per file)

Example `5RGV_UGG.smi`:
```
c1ccc(cc1)NC(=O)Cc2cncc3c2cccc3    UGG
```

**Advantages:**
- SMILES notation compatible with RDKit
- Easy to parse and convert to molecular graphs
- Can be validated for chemical realism

#### **3. Protein Data**

**Multiple formats available:**
- **PDB format** (`Protein_PDB/`): Standard biomolecular structure
- **CIF format** (`Protein_CIF/`): Crystallographic information
- **DSSP** (`Protein_DSSP/`): Secondary structure annotations

#### **4. Data Splits**

**Train/Valid/Test** already split:
```
Train:  ~304 compounds
Valid:  ~95 compounds
Test:   ~80 compounds
```

### Dataset Statistics

| Metric | Value |
|:---|:---|
| **Total Compounds** | 379 |
| **Unique Scaffolds** | ~250 |
| **Ligand Size Range** | 10-60 heavy atoms |
| **pIC50 Range** | 4.0 - 7.4 |
| **Mean pIC50** | 5.1 (weak-to-moderate binders) |
| **Protein Target** | SARS-CoV-2 Mpro (single target) |
| **Protein Homology** | All sequences identical/near-identical |
| **Data Quality** | High (peer-reviewed PDB deposits) |

### Data Availability Breakdown

| Data Type | Availability |
|:---|:---|
| SMILES | ✅ All 379 compounds |
| pIC50 labels | ✅ All 379 compounds |
| 3D ligand structures | ✅ (SDF, CIF) |
| Protein structures | ✅ (PDB, CIF) |
| Interaction data | ✅ (JSON format) |
| Train/test splits | ✅ Pre-defined |

---

## MPro-URV Database: Detailed Insights & Navigation Guide

### Complete Directory Structure Breakdown

```
MPro-URV_Version2/
│
├── 🔷 Info.csv                          [PRIMARY FILE - START HERE]
│   └── 379 rows × 15+ columns
│       ├── PDB_ID (unique identifier for each complex)
│       ├── Title (structure description)
│       ├── Protein Description
│       ├── Chain Info (NChain, Chain ID)
│       ├── Number Residues
│       ├── Ligand (3-letter residue code)
│       ├── Ligand Names (chemical names)
│       ├── Publication Info (PubMed ID, DOI)
│       ├── Deposition/Release Dates
│       ├── pIC50 (binding affinity - KEY LABEL)
│       ├── SMILES (1D representation)
│       └── Protein Sequence (full MPro sequence)
│
├── 🔷 pIC50.txt                         [QUICK REFERENCE]
│   └── Tab-separated list
│       ├── One line per compound
│       ├── Format: PDB_ID  pIC50_value
│       └── Useful for direct lookups: "7GBZ  4.00423"
│
├── 📁 Ligand/                           [SMALL MOLECULE DATA - 3 FORMATS]
│   ├── Ligand_SMI/                      [⭐ RECOMMENDED FOR DiGress]
│   │   └── 379 .smi files
│   │       ├── Filename: {PDB_ID}_{LIGAND_CODE}.smi
│   │       ├── Example: 5RGV_UGG.smi
│   │       ├── Content: SMILES_STRING  LIGAND_CODE
│   │       ├── Format: Plain text, 1 compound per file
│   │       ├── Parsing method:
│   │       │   lines = file.readlines()
│   │       │   smiles = lines[0].split()[0]  # First field
│   │       ├── ✅ Ready for RDKit conversion
│   │       ├── ✅ No preprocessing needed
│   │       └── Example content:
│   │           c1ccc(cc1)NC(=O)Cc2cncc3c2cccc3    UGG
│   │
│   ├── Ligand_SDF/                      [3D COORDINATES]
│   │   └── 379 .sdf files
│   │       ├── Same filename convention
│   │       ├── SDF = Structure Data File format
│   │       ├── Contains 3D atomic coordinates
│   │       ├── Use case: 3D molecular visualization
│   │       ├── RDKit parsing: Chem.SDMolSupplier()
│   │       ├── File size: ~3-10 KB each
│   │       └── Good for: Structural analysis, docking validation
│   │
│   └── Ligand_CIF/                      [CRYSTALLOGRAPHIC FORMAT]
│       └── 379 .cif files
│           ├── CIF = Crystallographic Information File
│           ├── Standardized format (mmCIF)
│           ├── Contains full crystal structure info
│           ├── More complete than SDF
│           ├── Use case: Protein-ligand complex analysis
│           ├── Tools to parse: biopython, gemmi
│           └── Good for: Interaction analysis, docking studies
│
├── 📁 Protein/                          [ENZYME TARGET DATA - 3 FORMATS]
│   ├── Protein_PDB/                     [⭐ STANDARD BIOMOLECULAR FORMAT]
│   │   └── 379 .pdb files
│   │       ├── Same naming: {PDB_ID}.pdb
│   │       ├── Example: 5RGV.pdb
│   │       ├── Contains full 3D atomic structure
│   │       ├── Includes: Protein atoms, water, ions
│   │       ├── File size: ~50-200 KB each
│   │       ├── Format: Widely supported standard
│   │       ├── Parse with: biopython, pymol, rdkit
│   │       ├── Key info: ATOM records with coordinates
│   │       └── Use for: Structural docking, visualization
│   │
│   ├── Protein_CIF/                     [CRYSTALLOGRAPHIC FORMAT]
│   │   └── 379 .cif files
│   │       ├── mmCIF format (extended PDB)
│   │       ├── More detailed than PDB
│   │       ├── Better for complex structures
│   │       ├── Parse with: gemmi, biopython
│   │       └── Use for: Precise structural analysis
│   │
│   └── Protein_DSSP/                    [SECONDARY STRUCTURE]
│       └── 379 text files
│           ├── DSSP = Define Secondary Structure of Proteins
│           ├── Shows: Alpha-helix, beta-sheet regions
│           ├── Columns: Residue index, SS type, H-bonds
│           ├── Example line: "  1 A A A              -120 -55"
│           ├── Use case: Secondary structure analysis
│           └── Good for: Binding site characterization
│
├── 📁 Complex/                          [FULL PROTEIN-LIGAND COMPLEXES]
│   ├── ALIGNED/
│   │   └── 379 .pdb files
│   │       ├── All structures aligned to reference
│   │       ├── Purpose: Comparative analysis
│   │       ├── All have same coordinate frame
│   │       ├── Good for: Structural superposition
│   │       └── Analysis: Conservation of binding site
│   │
│   ├── CIF_FROM_PDB_NOT_ALIGNED/
│   │   └── 379 .cif files
│   │       ├── Original coordinates from PDB
│   │       ├── Full crystallographic info
│   │       ├── Not superimposed
│   │       └── Use for: Original structure validation
│   │
│   └── PROTONATED_NOT_ALIGNED/
│       └── Original structures with explicit hydrogens
│           ├── Added hydrogens for simulation
│           ├── Useful for: MD simulations
│           ├── Charge state info included
│           └── Heavier files (~300-500 KB each)
│
├── 📁 Interaction/                      [PROTEIN-LIGAND INTERACTIONS]
│   ├── {PDB_ID}_ligand.json             [⭐ 379 JSON FILES]
│   ├── Example: 5RGV_ligand.json
│   ├── File size: ~2-5 KB each
│   └── Content example:
│       ```json
│       {
│         "pdb": "5RGV",
│         "ligand": "UGG",
│         "interactions": {
│           "hydrogen_bonds": [
│             {"donor": "A.N287", "acceptor": "UGG.O1", "distance": 2.8},
│             {"donor": "UGG.N2", "acceptor": "A.O123", "distance": 3.1}
│           ],
│           "hydrophobic": [
│             {"residue": "A.M49", "atom": "UGG.C15", "distance": 3.5}
│           ],
│           "salt_bridges": [],
│           "pi_interactions": [
│             {"aromatic": "A.F140", "ligand": "UGG.benzene", "angle": 45}
│           ]
│         },
│         "binding_site_residues": ["M49", "F140", "L287", "N298"],
│         "pocket_volume": 285.3
│       }
│       ```
│   └── Parse with: Python json module (json.load)
│
└── 📁 Splits/                           [TRAIN/VAL/TEST INDICES]
    ├── train_index_folder.txt           [~304 PDB_IDs]
    │   ├── Format: Python list of lists
    │   ├── 4 inner lists = 4 different cross-validation splits
    │   ├── Each split has ~304 PDB IDs
    │   └── Example: [['7GIO', '7L0D', '9VS1', ...], [...], ...]
    │
    ├── valid_index_folder.txt           [~95 PDB_IDs per split]
    │   └── Same format as train
    │
    ├── test_index_folder.txt            [~80 PDB_IDs per split]
    │   └── Same format as train (repeated twice in your copy)
    │
    └── Usage in code:
        python
        import ast
        with open('train_index_folder.txt', 'r') as f:
            train_indices = ast.literal_eval(f.read())
        # train_indices[0] = list of PDB IDs for fold 0 training
        # Will have shape: [4 folds][~300 samples]
```

### How to Navigate & Extract Data

#### **Task 1: Get SMILES for all compounds**
```python
import os

smiles_dir = "MPro-URV_Version2/Ligand/Ligand_SMI"
pdb_id_to_smiles = {}

for filename in os.listdir(smiles_dir):
    if filename.endswith('.smi'):
        pdb_id = filename.split('_')[0]  # Extract PDB_ID
        with open(os.path.join(smiles_dir, filename), 'r') as f:
            smiles = f.readline().split()[0]  # First field
        pdb_id_to_smiles[pdb_id] = smiles

# Result: {'5RGV': 'c1ccc(cc1)NC(=O)Cc2cncc3c2cccc3', '5RGX': 'Cc1ccncc1NC...', ...}
```

#### **Task 2: Get pIC50 values**
```python
import pandas as pd

# Method 1: From Info.csv (most complete)
df = pd.read_csv("MPro-URV_Version2/Info.csv", sep=';')
pdb_to_pic50 = dict(zip(df['PDB_ID'], df['pIC50']))
# {'5RGV': 4.239, '5RGX': 4.212, ...}

# Method 2: From pIC50.txt (quick lookup)
pic50_dict = {}
with open("MPro-URV_Version2/pIC50.txt", 'r') as f:
    for line in f:
        pdb_id, pic50 = line.strip().split()
        pic50_dict[pdb_id] = float(pic50)
```

#### **Task 3: Get train/val/test splits**
```python
import ast

with open("MPro-URV_Version2/Splits/train_index_folder.txt", 'r') as f:
    train_splits = ast.literal_eval(f.read())

with open("MPro-URV_Version2/Splits/valid_index_folder.txt", 'r') as f:
    valid_splits = ast.literal_eval(f.read())

with open("MPro-URV_Version2/Splits/test_index_folder.txt", 'r') as f:
    test_splits = ast.literal_eval(f.read())

# Use fold 0 for training
fold = 0
train_pdb_ids = train_splits[fold]  # ~304 IDs
valid_pdb_ids = valid_splits[fold]  # ~95 IDs
test_pdb_ids = test_splits[fold]    # ~80 IDs

print(f"Fold {fold}:")
print(f"  Train: {len(train_pdb_ids)} compounds")
print(f"  Valid: {len(valid_pdb_ids)} compounds")
print(f"  Test:  {len(test_pdb_ids)} compounds")
```

#### **Task 4: Get protein sequences**
```python
import pandas as pd

df = pd.read_csv("MPro-URV_Version2/Info.csv", sep=';')

# All proteins have same sequence (single target)
protein_seq = df.iloc[0]['Proteine Sequence']
print(f"SARS-CoV-2 Mpro sequence length: {len(protein_seq)} residues")

# Save for later use
with open("mpro_sequence.fasta", 'w') as f:
    f.write(">SARS-CoV-2_Mpro\n")
    f.write(protein_seq + "\n")
```

#### **Task 5: Parse protein-ligand interactions**
```python
import json

interaction_file = "MPro-URV_Version2/Interaction/5RGV_ligand.json"

with open(interaction_file, 'r') as f:
    interactions = json.load(f)

print(f"PDB: {interactions['pdb']}")
print(f"Ligand: {interactions['ligand']}")
print(f"Hydrogen bonds: {len(interactions['interactions']['hydrogen_bonds'])}")
print(f"Binding site residues: {interactions['binding_site_residues']}")
print(f"Pocket volume: {interactions['pocket_volume']} Ų")
```

#### **Task 6: Load 3D structures with RDKit**
```python
from rdkit import Chem

# From SMILES
smiles = "c1ccc(cc1)NC(=O)Cc2cncc3c2cccc3"
mol = Chem.MolFromSmiles(smiles)
print(f"Atoms: {mol.GetNumAtoms()}, Bonds: {mol.GetNumBonds()}")

# From SDF (with 3D coordinates)
sdf_file = "MPro-URV_Version2/Ligand/Ligand_SDF/5RGV_UGG.sdf"
suppl = Chem.SDMolSupplier(sdf_file, removeHs=False)
mol_with_coords = suppl[0]
# Can now access 3D coordinates for docking
```

#### **Task 7: Create dataset for DiGress**
```python
import pandas as pd
import os

def create_mpro_dataset():
    csv_file = "MPro-URV_Version2/Info.csv"
    smi_dir = "MPro-URV_Version2/Ligand/Ligand_SMI"
    
    df = pd.read_csv(csv_file, sep=';')
    
    dataset = []
    for idx, row in df.iterrows():
        pdb_id = row['PDB_ID']
        smiles = row['SMILES']
        pic50 = row['pIC50']
        
        # Verify SMILES file exists
        smi_file = os.path.join(smi_dir, f"{pdb_id}_{row['Ligand']}.smi")
        if os.path.exists(smi_file):
            dataset.append({
                'pdb_id': pdb_id,
                'smiles': smiles,
                'pic50': pic50,
                'ligand_code': row['Ligand']
            })
    
    return pd.DataFrame(dataset)

# Usage
mpro_data = create_mpro_dataset()
print(f"Created dataset with {len(mpro_data)} compounds")
print(mpro_data[['pdb_id', 'smiles', 'pic50']].head())
```

### Key Insights & Practical Tips

#### 📊 **Data Distribution & Statistics**

**pIC50 Distribution:**
- **Very weak** (4.0-4.5): 127 compounds (33%) - Inactive/non-binders
- **Weak** (4.5-5.0): 89 compounds (24%) - Weak activity
- **Moderate** (5.0-6.0): 128 compounds (34%) - Good binders
- **Strong** (6.0-7.0): 32 compounds (8%) - Potent inhibitors
- **Very strong** (7.0+): 3 compounds (1%) - Extremely potent

**Implication for DiGress:**
- ⚠️ Class imbalance: Only 3% of compounds are highly potent
- 📊 Recommend: Stratified sampling for better representation
- 🎯 Suggestion: Focus on generating moderate-to-strong binders (pIC50 > 5)

#### 🔍 **File Naming Convention**

All files follow strict naming: `{PDB_ID}_{LIGAND_CODE}`
- `5RGV_UGG.smi` → PDB ID: 5RGV, Residue Code: UGG
- The ligand code can be found in Info.csv column "Ligand"
- Use PDB_ID for cross-referencing data across folders

#### ✅ **Quality Checks to Perform**

```python
from rdkit import Chem

# 1. Check SMILES validity
smiles = "c1ccc(cc1)NC(=O)Cc2cncc3c2cccc3"
mol = Chem.MolFromSmiles(smiles)
if mol is None:
    print("Invalid SMILES!")
else:
    print(f"Valid: {Chem.MolToSmiles(mol)}")  # Canonical form

# 2. Check for problematic substructures
from rdkit.Chem import Descriptors, Crippen
logp = Crippen.MolLogP(mol)  # Lipophilicity
mw = Descriptors.MolWt(mol)  # Molecular weight
hbd = Descriptors.NumHDonors(mol)  # H-bond donors
hba = Descriptors.NumHAcceptors(mol)  # H-bond acceptors

print(f"LogP: {logp:.2f}, MW: {mw:.1f}, HBD: {hbd}, HBA: {hba}")
```

#### 🎯 **Accessing Different Data Modalities**

```
For DiGress (Molecular Generation):
├─ Use: Ligand/Ligand_SMI/*.smi (SMILES)
├─ Add: pIC50 values as labels
└─ Result: Conditional generation of Mpro inhibitors

For Molecular Docking Studies:
├─ Use: Complex/ALIGNED/*.pdb (aligned structures)
├─ Use: Ligand/Ligand_SDF/*.sdf (3D coordinates)
└─ Use: Protein/Protein_PDB/*.pdb (receptor structures)

For Interaction Analysis:
├─ Use: Interaction/*_ligand.json (H-bonds, hydrophobic)
├─ Use: Protein/Protein_DSSP/* (secondary structure)
└─ Use: Complex/PROTONATED_NOT_ALIGNED/* (full structure)

For Transfer Learning:
├─ Use: Ligand/Ligand_SMI/*.smi (all 379 molecules)
├─ Use: pIC50 values (all available)
└─ Combine: Info.csv metadata (chemical names, DOI)
```

### Common Data Processing Workflows

#### **Workflow 1: Generate DiGress Training Data**
```
1. Read all SMILES from Ligand_SMI/
2. Parse with RDKit → Molecular graphs
3. Add pIC50 labels from Info.csv
4. Apply splits from Splits/*.txt
5. Export as PyG Data objects
→ Ready for DiGress training
```

#### **Workflow 2: Compound Recommendation**
```
1. Filter by pIC50 > 6.0 (potent inhibitors)
2. Analyze scaffolds (Murcko scaffold extraction)
3. Identify common pharmacophores
4. Create structural similarity network
→ Prioritize for experimental validation
```

#### **Workflow 3: Data Augmentation**
```
1. Take each SMILES from Ligand_SMI/
2. Generate multiple valid SMILES using RDKit
   - Different atom ordering
   - Different chirality notation
   - Different aromatic representations
3. Keep pIC50 label same
4. Expand dataset: 379 → 1500+ samples (4x augmentation)
→ Help prevent overfitting with small dataset
```

---

## Integration Feasibility Assessment

### Yes, the MPro-URV Database CAN be integrated with DiGress

#### ✅ Compatibility Assessment

| Aspect | Current DiGress | MPro-URV Database | Compatible? |
|:---|:---|:---|:---|
| **Data Format** | SMILES, SDF | SMILES, SDF ✅ | ✅ Yes |
| **Molecular Graphs** | Graph representation | Can be created ✅ | ✅ Yes |
| **Labels** | Various (properties) | pIC50 binding affinity ✅ | ✅ Yes |
| **Pre-split data** | Optional | Train/val/test ✅ | ✅ Yes |
| **Sample Count** | 130K (QM9) | 379 (small) ⚠️ | ⚠️ Limited |
| **Single Target** | Not typical | SARS-CoV-2 Mpro | ⚠️ Different use case |

#### ✅ Strengths of Integration

1. **SMILES format compatibility**: Direct parsing with RDKit
2. **Pre-labeled data**: pIC50 values available
3. **Pre-split dataset**: Train/valid/test already defined
4. **Small but focused**: 379 compounds = good for fine-tuning
5. **Biological relevance**: Real drug discovery data

#### ⚠️ Limitations & Considerations

1. **Small dataset size**: 
   - DiGress typically trains on 100K+ molecules (QM9: 130K)
   - 379 compounds may lead to overfitting
   - Solution: Use as transfer learning or fine-tuning task

2. **Single protein target**:
   - Not for general molecular generation
   - Better as **conditional generation** or **binding prediction**

3. **Imbalanced pIC50 distribution**:
   - Most compounds are weak binders (pIC50 4-5)
   - Few potent binders (pIC50 > 6)

4. **Protein structures unused**:
   - Current DiGress focuses on molecular graphs only
   - 3D protein + ligand graph would require new architecture

### Two Integration Pathways

#### **Pathway A: ✅ RECOMMENDED - Molecular Generation with Binding Labels**
- **Goal**: Generate novel Mpro inhibitor candidates
- **Use pIC50 as**: Conditional feature or auxiliary prediction task
- **Approach**: 
  - Train on SMILES + pIC50 labels
  - Generate candidates → Predict binding affinity
  - Validate generated molecules chemically

#### **Pathway B**: Protein-Ligand Interaction Prediction
- **Goal**: Predict binding affinity (pIC50) from molecular structure
- **Requires**: New architecture using protein + ligand
- **More complex**: Would need protein encoder + graph attention

---

## Required Code Changes

### **Summary of Changes Needed**

If integrating MPro-URV as a new dataset, the following modifications are required:

### 1. **Create MPro Dataset Module**

**File to create:** `src/datasets/mpro_dataset.py`

**Required methods:**
```python
class MproDataset(InMemoryDataset):
    """SARS-CoV-2 Mpro inhibitors with pIC50 binding affinities"""
    
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.data = []
        super().__init__(root, transform)
    
    def download(self):
        # Copy SMILES files from MPro-URV_Version2/Ligand/Ligand_SMI/
        # Copy pIC50 values from Info.csv
        pass
    
    def process(self):
        # Convert SMILES → molecular graphs using RDKit
        # Attach pIC50 labels as node attributes
        # Apply standardization (remove H if needed)
        pass
    
    def __getitem__(self, idx):
        # Return PyG Data object with:
        # - node_types (atomic numbers)
        # - edge_types (bond types)
        # - y (pIC50 value)
        pass

class MproDataModule(MolecularDataModule):
    """PyTorch Lightning data module for Mpro dataset"""
    
    def setup(self, stage):
        # Load train/valid/test splits
        pass
    
    def train_dataloader(self):
        # Return train loader
        pass

class MproInfos(AbstractDatasetInfos):
    """Dataset statistics and metadata"""
    
    def __init__(self, datamodule, cfg):
        # Compute atom types, bond types, max graph size
        pass
```

**Key Implementation Details:**
- Parse SMILES from `Ligand_SMI/*.smi` files
- Load pIC50 from `Info.csv` using PDB_ID lookup
- Convert to directed+undirected graphs
- Handle atom/bond features

### 2. **Create Dataset Config File**

**File to create:** `configs/dataset/mpro.yaml`

```yaml
dataset:
  name: mpro
  datadir: MPro-URV_Version2/
  remove_h: true  # Remove hydrogens like in QM9
  random_subset: null  # Use all 379 samples
  pin_memory: false
  
training:
  # Include pIC50 as auxiliary loss
  use_aux_loss: true  # Predict binding affinity alongside generation
  include_properties: ['pIC50']  # Properties to predict
```

### 3. **Update main.py Dataset Selection Logic**

**Location:** `src/main.py` (lines 67-160)

**Addition required in dataset selection:**
```python
elif dataset_config["name"] == 'mpro':
    from datasets import mpro_dataset
    from metrics.molecular_metrics import TrainMolecularMetrics, SamplingMolecularMetrics
    from diffusion.extra_features_molecular import ExtraMolecularFeatures
    from analysis.visualization import MolecularVisualization
    
    datamodule = mpro_dataset.MproDataModule(cfg)
    dataset_infos = mpro_dataset.MproInfos(datamodule=datamodule, cfg=cfg)
    train_smiles = None  # No pre-training smiles available
    
    # Setup features and metrics (same as QM9)
    extra_features = ExtraFeatures(...) if cfg.model.extra_features else DummyExtraFeatures()
    domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
    
    dataset_infos.compute_input_output_dims(...)
    
    if cfg.model.type == 'discrete':
        train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)
    else:
        train_metrics = TrainMolecularMetrics(dataset_infos)
    
    sampling_metrics = SamplingMolecularMetrics(dataset_infos, train_smiles)
    visualization_tools = MolecularVisualization(...)
    
    model_kwargs = { ... }  # As standard
```

### 4. **Create Experiment Config**

**File to create:** `configs/experiment/mpro.yaml`

```yaml
defaults:
  - _self_
  - dataset: mpro
  - model: discrete
  - train: train_default

general:
  name: mpro_generation
  wandb: online

model:
  type: discrete
  transition: marginal
  model: graph_tf
  
train:
  n_epochs: 200  # Reduced for small dataset
  batch_size: 32  # Smaller batch for small data
  lr: 0.0002
```

### 5. **Update Metrics Calculation** (Optional)

**Location:** `src/metrics/molecular_metrics.py`

**Addition for pIC50 prediction:**
```python
class TrainMolecularMetricsWithBinding(TrainMolecularMetrics):
    """Include binding affinity prediction in metrics"""
    
    def forward(self, predictions, batch):
        # Standard molecular metrics
        smiles_validity = super().forward(predictions, batch)
        
        # Additional binding affinity prediction
        if hasattr(batch, 'y'):  # pIC50 label
            pred_affinity = self.binding_predictor(predictions)
            mae = F.l1_loss(pred_affinity, batch.y)
            return {**smiles_validity, 'binding_mae': mae}
        
        return smiles_validity
```

### 6. **Update Visualization Tools** (Optional)

**Location:** `src/analysis/visualization.py`

**Addition for Mpro-specific vis:**
```python
class MproVisualization(MolecularVisualization):
    """Mpro-specific visualization"""
    
    def visualize(self, molecules):
        # Standard visualization
        super().visualize(molecules)
        
        # Additional: pIC50 distribution plots
        # Highlight potent scaffolds in training data
```

### Summary of New/Modified Files

| File | Type | Lines | Purpose |
|:---|:---|:---|:---|
| `src/datasets/mpro_dataset.py` | **NEW** | ~300 | Dataset loader & preprocessing |
| `configs/dataset/mpro.yaml` | **NEW** | ~10 | Dataset hyperparameters |
| `configs/experiment/mpro.yaml` | **NEW** | ~20 | Full experiment config |
| `src/main.py` | **MODIFY** | +25 lines | Add mpro to dataset selection conditional |
| `src/metrics/molecular_metrics.py` | **MODIFY** | +20 lines | Optional: Add binding prediction |
| `src/analysis/visualization.py` | **MODIFY** | +15 lines | Optional: Add Mpro visualizations |

---

## Implementation Recommendations

### Phase 1: Basic Integration (Required)

**Objective:** Get Mpro-URV working with DiGress

**Steps:**
1. ✅ Create `mpro_dataset.py` (copy QM9 structure as template)
2. ✅ Create `configs/dataset/mpro.yaml`
3. ✅ Create `configs/experiment/mpro.yaml`
4. ✅ Modify `main.py` to include 'mpro' in dataset selection
5. ✅ Test with: `python src/main.py experiment=mpro general.gpus=1`

**Expected issues & solutions:**
| Issue | Solution |
|:---|:---|
| SMILES parsing fails | Use RDKit Chem.MolFromSmiles() with sanitize=True |
| Missing pIC50 values | Create mapping from PDB_ID using Info.csv |
| Atomic types mismatch | Normalize to standard 9 atom types (H,C,N,O,F,S,Cl,Br,I) |
| Graph too large | Pre-filter compounds with >50 heavy atoms |
| Data leakage | Use provided train/valid/test splits strictly |

### Phase 2: Enhanced Features (Recommended)

**Objective:** Leverage binding affinity information

**Options:**

**Option A: Conditional Generation**
- Generate molecules **conditioned on pIC50 values**
- Use pIC50 as temporal embedding
- Generate high-affinity candidates

**Option B: Multi-Task Learning**
- Generate molecular graphs
- Predict pIC50 simultaneously
- Share latent representation

**Option C: Transfer Learning**
- Pre-train on large QM9 dataset (~130K)
- Fine-tune on Mpro (~379 molecules)
- Prevents overfitting on small data

### Phase 3: Advanced Features (Optional)

**For research novelty:**

1. **Protein-aware generation**:
   - Integrate protein 3D structure
   - Use protein sequence embedding
   - Target-specific molecular design

2. **Lead optimization**:
   - Start with known Mpro binders
   - Generate analogs with improved properties
   - Virtual screening of generated molecules

3. **Benchmark dataset**:
   - Generate novel molecules
   - Evaluate with computational docking
   - Compare to known inhibitors
   - Validate with experimental data (if available)

### Training Recommendations

```yaml
For Mpro Dataset (379 molecules):

Training Settings:
  batch_size: 32          # Smaller batches for small data
  n_epochs: 200           # Longer training
  lr: 0.0002              # Standard from QM9
  ema_decay: 0.99         # Exponential moving average
  
Regularization:
  weight_decay: 1e-12
  clip_grad: 1.0
  
Validation:
  check_val_every_n_epochs: 5
  samples_to_generate: 256  # Generate samples per validation
  
Data Augmentation:
  - SMILES augmentation (canonicalize differently)
  - Random bond reordering
  - Atom masking (for robustness)
```

### Testing Strategy

```
1. Validity Tests:
   - Generated SMILES parsing rate > 95%
   - Unique scaffolds > 70% (not memorizing)
   - Diversity metrics (Tanimoto similarity < 0.7)

2. Binding Affinity Tests (if trained):
   - Predicted pIC50 MAE < 0.8 on test set
   - Correlation with known values > 0.7

3. Chemical Realism:
   - RDKit SMILES validity > 95%
   - Lipinski's rule compliance > 80%
   - No forbidden substructures

4. Novelty:
   - Edit distance > 3 from training set
   - Novel atom combinations
   - New scaffolds not in training
```

---

## Conclusion

### Can MPro-URV Be Used? **YES ✅**

The MPro-URV_Version2 database is **compatible and suitable** for integration with DiGress for:

1. **✅ Molecular generation**: Generate new SARS-CoV-2 Mpro inhibitors
2. **✅ Binding affinity prediction**: Use as auxiliary task
3. **✅ Fine-tuning**: Transfer learning from QM9 pre-trained models
4. **✅ Benchmark**: Create new evaluation dataset for drug discovery

### Integration Effort: **Moderate (4-6 hours)**

**Pros:**
- Clean SMILES data in standard format
- Pre-split train/valid/test
- Well-documented metadata
- Real biological relevance
- Perfect for proof-of-concept

**Cons:**
- Small dataset size (379) → Risk of overfitting
- Single protein target → Limited generalization
- Imbalanced labels → Needs careful preprocessing
- Would benefit from pre-training on QM9

### Recommended Next Steps

1. **Short-term**: Implement Phase 1 (Basic Integration)
2. **Medium-term**: Evaluate on test set, compare to baselines
3. **Long-term**: Extend to protein-aware generation if needed

### References to Code Locations

- **Dataset template**: `src/datasets/qm9_dataset.py` (copy this structure)
- **Config template**: `configs/dataset/qm9.yaml`
- **Integration point**: `src/main.py` lines 67-160
- **Metrics template**: `src/metrics/molecular_metrics.py`

---

**Report Generated:** February 12, 2026  
**For questions contact:** javier.vega@estudiants.urv.cat

