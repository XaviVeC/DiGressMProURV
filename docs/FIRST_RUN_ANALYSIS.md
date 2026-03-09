# DiGressMProURV: First Run Analysis & Comprehensive Explanation

**Date:** March 2, 2026  
**Project:** TFM (Treball Fi Master) – MESIIA Program, URV  
**Author:** Xavi Vega  
**Run analyzed:** `outputs/2026-02-23/17-14-14-test/`

---

## Table of Contents

1. [Project Context & Motivation](#1-project-context--motivation)
2. [Code Changes: What Was Modified and Why](#2-code-changes-what-was-modified-and-why)
3. [The Model: What It Is and How It Works](#3-the-model-what-it-is-and-how-it-works)
4. [Training Pipeline: Step by Step](#4-training-pipeline-step-by-step)
5. [Dataset: Molecules Being Used](#5-dataset-molecules-being-used)
6. [Output Files: What Gets Produced and by Which Code](#6-output-files-what-gets-produced-and-by-which-code)
7. [First Run Results Interpretation](#7-first-run-results-interpretation)
8. [Model Parameters & Tuning Guide](#8-model-parameters--tuning-guide)

---

## 1. Project Context & Motivation

### Background

This project is a **Master's Thesis (TFM)** at the Universitat Rovira i Virgili (URV) under the MESIIA program. The goal is to apply **deep generative models for drug discovery** targeting the **SARS-CoV-2 Main Protease (Mpro/3CLpro)**, a critical enzyme for viral replication and a validated target for antiviral therapeutics.

### Original Proposal

The original TFM proposal described using a Graph Convolutional Policy Network (GCPN) for molecular generation. This was **replaced by DiGress** (Discrete Denoising Diffusion for Graph Generation), a more modern approach that uses **discrete diffusion models** instead of reinforcement learning. Both serve the same purpose: generating novel, chemically valid molecules that could act as Mpro inhibitors.

### Why DiGress?

DiGress is advantageous because:
- It operates directly on discrete graph representations (atom types, bond types) without needing continuous relaxation
- It has shown state-of-the-art results on molecular benchmarks (QM9, MOSES, GuacaMol)
- It naturally handles the categorical nature of chemical graphs (distinct atom/bond types)
- The diffusion framework provides a principled probabilistic generation process
- And most importantly, it runs in `Python3`

---

## 2. Code Changes: What Was Modified and Why

### Overview

The DiGressMProURV fork adapts the original DiGress codebase to accept the **MPro-URV_Version2** database. The changes follow "Pathway A" — treating it as a molecular generation task with binding affinity labels. The integration is **minimal and backward-compatible**.

### New Files Created (3 files, ~870 lines)

#### 2.1 `src/datasets/mpro_dataset.py` (~550 lines)

This is the core addition. It implements three classes:

| Class | Purpose |
|-------|---------|
| **`MproDataset`** | PyTorch Geometric `InMemoryDataset` that reads SMILES from the MPro-URV database, converts them to molecular graphs using RDKit, extracts atom/bond features, and optionally removes hydrogens |
| **`MproDataModule`** | PyTorch Lightning `DataModule` wrapping train/val/test datasets with appropriate batching |
| **`MproInfos`** | Dataset metadata class defining atom types (C, N, O, F, S, Cl, Br, I), bond types (single, double, triple, aromatic), node count distributions, and valency information |

Key implementation details:
- **SMILES parsing**: Reads `.smi` files from `Ligand/Ligand_SMI/` directory, matched to PDB IDs
- **pIC50 loading**: Reads binding affinities from `Info.csv` and attaches as graph-level label `y`
- **Hydrogen removal**: When `remove_h=True` (default), strips H atoms and re-indexes, reducing from 9 to 8 atom types
- **Split handling**: Uses pre-defined train/val/test indices from `Splits/` folder (fold 0)
- **`RemoveYTransform`**: A transform applied at load time that zeros out the `y` property labels so the model trains purely on graph structure (unconditional generation)

**Important**: The `RemoveYTransform` is applied to all three splits in `MproDataModule.__init__()`. This means the pIC50 values are **read and stored** during processing (in the `.pt` files) but **zeroed out** during training. The model currently does **unconditional** molecular generation — it does not condition on or predict binding affinity.

#### 2.2 `configs/dataset/mpro.yaml` (~60 lines)

Dataset configuration specifying:
- `name: 'mpro'` — dataset identifier
- `datadir: 'data/mpro/'` — where processed `.pt` files are stored
- `raw_data_dir: null` — overridable path to MPro-URV_Version2 database
- `remove_h: True` — remove hydrogens (recommended for small dataset)

#### 2.3 `configs/experiment/mpro.yaml` (~205 lines)

Full experiment configuration with:
- Model settings (discrete, 500 diffusion steps, cosine noise schedule, 5-layer graph transformer)
- Training hyperparameters (300 epochs, batch size 16, lr 0.0002, AdamW)
- Sampling settings (256 samples during validation, 1000 at test)
- Mpro-specific options (binding affinity, conditional generation — all disabled)
- Comprehensive documentation comments

### Modified Files (1 file, +8 lines)

#### 2.4 `src/main.py`

Added `'mpro'` to the molecular dataset conditional branch:

```python
elif dataset_config["name"] in ['qm9', 'guacamol', 'moses', 'mpro']:
    ...
    elif dataset_config['name'] == 'mpro':
        from datasets import mpro_dataset
        datamodule = mpro_dataset.MproDataModule(cfg)
        dataset_infos = mpro_dataset.MproInfos(datamodule=datamodule, cfg=cfg)
        train_smiles = mpro_dataset.get_train_smiles(...)
```

### Modified Default Config

#### 2.5 `configs/config.yaml`

The default dataset was changed from `qm9` to `mpro`:
```yaml
defaults:
    - dataset : mpro
```

---

## 3. The Model: What It Is and How It Works

### 3.1 Discrete Denoising Diffusion (DDPM for Graphs)

DiGress implements a **Discrete Denoising Diffusion Probabilistic Model (D3PM)** adapted for graphs. The core idea:

1. **Forward process (noise)**: Starting from a real molecular graph, gradually corrupt atom types and bond types over $T$ timesteps by randomly flipping them toward a limiting distribution (marginal distribution of the training set)
2. **Reverse process (denoising)**: A neural network learns to predict the original clean graph from any noisy intermediate state
3. **Generation**: Start from pure noise (sampled from the limiting distribution) and iteratively denoise for $T$ steps to produce a new molecular graph

Formally, for discrete variables $x$ (atom types) and $e$ (edge types):

$$q(z_t | z_0) = z_0 \cdot \overline{Q}_t$$

where $\overline{Q}_t$ is a transition matrix that increasingly mixes categories. The model learns $p_\theta(z_0 | z_t)$, and the reverse step is computed via Bayes' theorem:

$$p(z_{t-1} | z_t) \propto p_\theta(z_0 | z_t) \cdot q(z_t | z_{t-1}) \cdot q(z_{t-1} | z_0)$$

### 3.2 Transition Model: Marginal

The run uses `transition: marginal`, meaning the limiting (fully noised) distribution matches the **empirical marginal distribution** of atom and bond types in the training data. This biases the noising process toward the data distribution rather than uniform noise, making learning easier.

From the log:
```
Marginal distribution of the classes:
  Nodes: [0.45, 0.20, 0.25, 0.02, 0.04, 0.02, 0.015, 0.005]
         (C,    N,    O,    F,    S,    Cl,   Br,    I)
  Edges: [0.75, 0.15, 0.03, 0.07, 0.0]
         (none, single, double, triple, aromatic)
```

### 3.3 Neural Network Architecture: Graph Transformer

The neural network backbone is a **Graph Transformer** (`src/models/transformer_model.py`) with:

- **5 layers** of `XEyTransformerLayer`
- Each layer jointly updates **node features** ($X$), **edge features** ($E$), and **global features** ($y$)
- Uses multi-head self-attention (8 heads) where edge features modulate attention weights
- Feedforward networks with ReLU activation after each attention block
- Layer normalization and dropout (0.1) for regularization

**Input/Output dimensions:**
- Nodes: 8 atom types + extra features → 8 predicted atom type logits
- Edges: 5 bond types (including no-bond) + extra features → 5 predicted bond type logits
- Global: timestep embedding + extra features → global features

**Extra features** (`cfg.model.extra_features: 'all'`): Includes eigenvalues of the graph Laplacian, cycle counts, and other structural features. These help the model understand graph topology beyond local neighborhoods.

### 3.4 Noise Schedule

The model uses a **cosine noise schedule** (`diffusion_noise_schedule: cosine`) with 500 timesteps. The cosine schedule provides nearly linear signal-to-noise ratio decay, which is gentler early in the process and more aggressive later, helping preserve structural information during early denoising steps.

### 3.5 Training Objective

The training loss is a **weighted cross-entropy** between predicted and true atom/bond types:

$$\mathcal{L} = \lambda_X \cdot \text{CE}(X_{pred}, X_{true}) + \lambda_E \cdot \text{CE}(E_{pred}, E_{true})$$

With `lambda_train: [5, 0]`:
- $\lambda_X = 5$: Node (atom type) cross-entropy weight
- $\lambda_E = 0$: Edge (bond type) cross-entropy weight is **zero** in the training loss (but edges still contribute through the diffusion loss/NLL during validation)

The `y_CE: -1.000` in logs indicates that the global property loss is not active (since `RemoveYTransform` zeroes it out).

---

## 4. Training Pipeline: Step by Step

### 4.1 Launch Command

```bash
python src/main.py dataset=mpro \
    +dataset.raw_data_dir=/path/to/MPro-URV_Version2 \
    train.batch_size=32 \
    general.name=test
```

### 4.2 Execution Flow

The complete pipeline, governed by `src/main.py`:

```
1. Hydra loads configs/config.yaml + overrides
   ↓
2. Dataset selection: 'mpro' → loads mpro_dataset module
   ↓
3. MproDataModule(cfg) → creates MproDataset for train/val/test
   ├── Reads MPro-URV_Version2/Info.csv (379 molecules)
   ├── Reads SMILES from Ligand/Ligand_SMI/*.smi
   ├── Converts SMILES → RDKit mol → PyG Data graphs
   ├── Removes hydrogens, creates one-hot features
   └── Saves as proc_tr_no_h.pt, proc_val_no_h.pt, proc_test_no_h.pt
   ↓
4. MproInfos → defines atom/bond encodings, distributions
   ↓
5. ExtraFeatures('all') + ExtraMolecularFeatures → structural features
   ↓
6. compute_input_output_dims() → calculates network I/O sizes
   ↓
7. DiscreteDenoisingDiffusion(cfg, ...) → initializes model
   ├── GraphTransformer (5 layers, 8 heads, dx=256, de=64, dy=64)
   ├── PredefinedNoiseScheduleDiscrete (cosine, 500 steps)
   └── MarginalUniformTransition → transition matrices
   ↓
8. Trainer setup (PyTorch Lightning)
   ├── GPU acceleration, DDP strategy
   ├── ModelCheckpoint (best 5 by val NLL + last)
   └── Weights & Biases logging
   ↓
9. trainer.fit(model, datamodule)
   ├── For each epoch:
   │   ├── training_step(): Add noise → predict clean graph → CE loss
   │   ├── Every 5 epochs: validation_step() → compute NLL
   │   └── Every 20 epochs (sample_every_val=4): Generate 512 molecules
   │       ├── Sample node counts from learned distribution
   │       ├── Initialize from noise (marginal distribution)
   │       ├── 500 reverse diffusion steps
   │       └── Decode to molecules, compute validity/uniqueness/novelty
   ↓
10. trainer.test(model, datamodule)
    ├── Compute test NLL
    ├── Generate 10,000 final molecules
    └── Save SMILES, chains, visualizations
```

### 4.3 Key Code Files and Their Roles

| File | Role |
|------|------|
| `src/main.py` | Orchestrator: config loading, dataset/model creation, training loop launch |
| `src/diffusion_model_discrete.py` | Core model: training/validation/test steps, noise application, sampling, loss computation |
| `src/models/transformer_model.py` | Neural network: Graph Transformer with joint node-edge-global attention |
| `src/diffusion/noise_schedule.py` | Noise schedule: cosine beta schedule for forward diffusion process |
| `src/diffusion/diffusion_utils.py` | Utilities: posterior distributions, sampling from discrete distributions |
| `src/datasets/mpro_dataset.py` | Data pipeline: SMILES → graphs, splits, statistics |
| `src/metrics/molecular_metrics.py` | Training metrics: per-atom/bond CE tracking, sampling metrics orchestration |
| `src/analysis/rdkit_functions.py` | Chemical validation: graph → RDKit mol → SMILES, validity/uniqueness/novelty |
| `src/analysis/visualization.py` | Visualization: molecule images, diffusion chain GIFs |
| `src/utils.py` | General utilities: PlaceHolder, dense conversion, WandB setup |

---

## 5. Dataset: Molecules Being Used

### 5.1 MPro-URV_Version2 Database

The dataset contains **379 experimentally characterized SARS-CoV-2 Mpro inhibitors** from crystallographic studies (PDB co-crystal structures). Each entry includes:

| Field | Description | Example |
|-------|-------------|---------|
| PDB_ID | Protein Data Bank identifier | 5RGV, 7EN8 |
| Ligand | Ligand code | UGG, J7R |
| SMILES | Molecular structure string | `c1ccc(cc1)NC(=O)Cc2cncc3c2cccc3` |
| pIC50 | Binding affinity (-log₁₀ IC₅₀) | 4.24 to 7.14 |
| Protein Sequence | Mpro amino acid sequence | 306 residues typically |

### 5.2 Dataset Splits

The data is divided using pre-defined indices from the `Splits/` folder:

| Split | Approximate Size | Purpose |
|-------|------------------|---------|
| Train | ~304 molecules | Model training |
| Validation | ~95 molecules | Hyperparameter tuning, early stopping |
| Test | ~80 molecules | Final evaluation |

### 5.3 Molecular Characteristics

- **Size range**: 10–50 heavy atoms (mostly 15–25)
- **Atom types** (no H): C (45%), O (25%), N (20%), S (4%), F (2%), Cl (2%), Br (1.5%), I (0.5%)
- **Bond types**: Single (75%), Aromatic (7%), Double (15%), Triple (3%)
- **pIC50 range**: 4.0–7.4 (median ~5.0)
  - Weak binders (<5.0): ~57%
  - Moderate binders (5.0–6.0): ~34%
  - Potent binders (>6.0): ~9%

### 5.4 What the Model Learns From

The model learns the **structural distribution** of Mpro inhibitors:
- Which atom types appear and how often
- How atoms are connected (bond patterns)
- Typical molecular sizes
- Ring structures, functional groups common in Mpro inhibitors (e.g., pyridine, chlorophenyl, amide bonds visible in the training data SMILES)

The model does **not** currently use pIC50 values during training (due to `RemoveYTransform`). It generates molecules that look structurally similar to known Mpro inhibitors but without explicit binding affinity guidance.

---

## 6. Output Files: What Gets Produced and by Which Code

### 6.1 Output Directory Structure

```
outputs/2026-02-23/17-14-14-test/
├── .hydra/                          ← Hydra configuration snapshots
│   ├── config.yaml                  ← Full resolved config
│   ├── hydra.yaml                   ← Hydra internal config
│   └── overrides.yaml               ← Command-line overrides used
├── main.log                         ← Minimal logger output
├── wandb/                           ← Weights & Biases logs
│   └── run-.../files/
│       ├── output.log               ← FULL training log (55,302 lines)
│       ├── wandb-summary.json       ← Final metrics snapshot
│       └── config.yaml              ← WandB copy of config
├── checkpoints/test/                ← Saved model weights
│   ├── epoch=569.ckpt               ← Best checkpoint by val NLL
│   ├── epoch=714.ckpt
│   ├── epoch=744.ckpt
│   ├── epoch=814.ckpt               ← Best val NLL: 230.12
│   ├── epoch=894.ckpt
│   └── last-v1.ckpt                 ← Last epoch checkpoint
├── chains/test/                     ← Diffusion chain visualizations
│   ├── epoch14/chains/              ← GIF animations of denoising process
│   ├── epoch34/chains/
│   ├── ...                          ← One folder per validation sampling epoch
│   └── epoch994/chains/
└── graphs/test/                          ← Generated molecule outputs
    ├── epoch14_b0/ ... epoch14_b448/           ← PNG images of individual molecules
    ├── valid_unique_molecules_e14_b-1.txt      ← Valid SMILES at epoch 14
    ├── valid_unique_molecules_e34_b-1.txt
    ├── ...
    └── valid_unique_molecules_e994_b-1.txt     ← Valid SMILES at epoch 994
```

### 6.2 File Generation by Code Module

| Output | Generated by | Code file |
|--------|-------------|-----------|
| `.hydra/` configs | Hydra framework | `main.py` (decorator) |
| `main.log` | Python logging | Standard logger |
| `wandb/` logs | WandB integration | `diffusion_model_discrete.py` (in hooks) |
| `output.log` (training metrics) | `on_train_epoch_end()`, `on_validation_epoch_end()` | `diffusion_model_discrete.py` lines 138–145, 172–184 |
| `checkpoints/*.ckpt` | `ModelCheckpoint` callback | `main.py` lines 177–184 |
| `chains/*/molecule_*.png/.gif` | `visualization_tools.visualize_chain()` | `diffusion_model_discrete.py` lines 521–534 |
| `graphs/epoch*_b*/` (PNGs) | `visualization_tools.visualize()` | `diffusion_model_discrete.py` lines 537–540 |
| `valid_unique_molecules_*.txt` | `compute_molecular_metrics()` | `src/analysis/rdkit_functions.py` → called from `molecular_metrics.py` |
| `wandb-summary.json` | WandB auto-summary | Framework |

### 6.3 How Molecules Get Saved

During validation sampling (every `sample_every_val × check_val_every_n_epochs = 4 × 5 = 20` epochs):

1. `on_validation_epoch_end()` calls `sample_batch()` to generate 512 molecules
2. `sample_batch()` runs 500 reverse diffusion steps, producing `(atom_types, edge_types)` tuples
3. `visualization_tools.visualize()` renders molecules as PNG images in `graphs/test/epoch{N}_b{B}/`
4. `sampling_metrics.forward()` calls `compute_molecular_metrics()` which:
   - Converts graph representations to RDKit molecules
   - Converts to SMILES strings
   - Checks validity, uniqueness, novelty
   - Saves valid unique SMILES to `valid_unique_molecules_e{epoch}_b-1.txt`

---

## 7. First Run Results Interpretation

### 7.1 Training Configuration Used

| Parameter | Value | Note |
|-----------|-------|------|
| Epochs | 1000 | Full training run |
| Batch size | 32 | Overridden from default 512 |
| Learning rate | 0.0002 | AdamW optimizer |
| Diffusion steps | 500 | Cosine schedule |
| Transformer layers | 5 | 8-head attention |
| Extra features | all | Eigenvalues + cycles |
| EMA decay | 0 | Disabled |
| Training time per epoch | ~2.3s (training) + ~420s (when sampling) | Single GPU |
| Total estimated time | ~8.5 hours | Including sampling epochs |

### 7.2 Loss Progression

#### Training Loss (Cross-Entropy)

| Epoch | X_CE (atoms) | E_CE (bonds) | y_CE |
|-------|-------------|-------------|------|
| 0 | 0.960 | 0.562 | -1.0 |
| 10 | 0.635 | 0.235 | -1.0 |
| 50 | 0.583 | 0.212 | -1.0 |
| 100 | 0.534 | 0.194 | -1.0 |
| 500 | 0.490 | 0.176 | -1.0 |
| 999 | 0.464 | 0.176 | -1.0 |

**Interpretation:**
- `X_CE` (atom type CE) decreases from 0.96 → ~0.46 — model learns atom patterns
- `E_CE` (edge type CE) decreases from 0.56 → ~0.18 — model learns bond patterns
- `y_CE = -1.0` throughout: global property loss is inactive (pIC50 zeroed by `RemoveYTransform`)
- Loss plateaus around epoch 200–300, with noisy oscillations afterwards — signs of **overfitting** on the small dataset

#### Validation NLL

| Epoch | Val NLL | Atom KL | Edge KL | Status |
|-------|---------|---------|---------|--------|
| 0 | 1469.65 | 45.71 | 831.03 | Initial (before training) |
| 24 | 379.16 | 24.84 | 315.44 | Rapid improvement |
| 54 | 343.47 | 33.25 | 280.67 | Still improving |
| 100 | ~330 | ~25 | ~270 | Slow improvement |
| 500 | ~290 | ~20 | ~240 | Fluctuating |
| **814** | **230.12** | **16.78** | **188.25** | **Best** |
| 999 | 288.98 | 26.50 | 237.15 | End of training |

**Interpretation:**
- Best val NLL of **230.12** achieved at epoch 814
- After epoch ~300, val NLL fluctuates significantly (230–370 range) — strong **overfitting** signal
- Edge KL dominates the loss (188 vs 17 for atoms at best), meaning the model struggles more with learning correct bond patterns than atom types
- This is expected for a small dataset (only ~304 training molecules)

### 7.3 Generation Quality Metrics Over Training

| Period | Validity | Relaxed Validity | Uniqueness | Novelty |
|--------|----------|------------------|------------|---------|
| Epoch 14 | 0% | 0% | - | - |
| Epoch 54 | 0.20% | 0.20% | - | - |
| Epoch 154 | 5.86% | 6.45% | 100% | 100% |
| Epoch 274 | 14.06% | 15.04% | 100% | 100% |
| Epoch 474 | 18.75% | 19.34% | 100% | 100% |
| Epoch 714 | 31.64% | 32.42% | 99.38% | 100% |
| **Epoch 974** | **32.03%** | **33.01%** | **99.41%** | **100%** |
| Epoch 994 | 27.15% | 27.93% | 100% | 100% |

**Interpretation:**

- **Validity peaks at ~32%**: Out of 512 generated molecules, ~164 are chemically valid RDKit molecules. This is **low** compared to QM9 benchmarks (>90%) but **expected for a small dataset** of only 304 training molecules with diverse, drug-like structures
- **Uniqueness near 100%**: The model is not mode-collapsing — nearly all valid molecules are distinct
- **Novelty = 100%**: All generated molecules are novel (not copies of training data), confirming the model generalizes
- **Relaxed validity** (allowing sanitization fixes) is only slightly higher than strict validity, suggesting most failures are fundamental structural issues (broken valences, impossible ring closure, disconnected fragments)

### 7.4 Example Generated Molecules (Epoch 994)

From `valid_unique_molecules_e994_b-1.txt`, examples of valid generated SMILES:

```
O=C(NC1=CN=CC=C(Cl)C=C1)C1CC2CC21
O=C1Cc2c(Cl)cncc2N1
O=C1NC2=CC=CC=C(Cl)C1=C2
O=C1Cc2ccc(Cl)ccncc(c2)N1
```

These show the model has learned to generate:
- Amide bonds (`NC(=O)`, `C(=O)N`)
- Chlorinated aromatic rings (`C(Cl)`)
- Heterocyclic nitrogen-containing rings (`cncc`, `ccnc`)
- Lactam/cyclic structures

These are structurally consistent with the Mpro inhibitor chemical space (chlorophenyl amides, pyridine derivatives).

### 7.5 Checkpoints Saved

The top 5 checkpoints by validation NLL + last:

| Checkpoint | Epoch | Val NLL |
|-----------|-------|---------|
| `epoch=814.ckpt` | 814 | 230.12 (best) |
| `epoch=894.ckpt` | 894 | 243.14 |
| `epoch=744.ckpt` | 744 | 242.37 |
| `epoch=714.ckpt` | 714 | 241.35 |
| `epoch=569.ckpt` | 569 | ~240-250 |
| `last-v1.ckpt` | 999 | 288.98 |

### 7.6 WandB Summary (Final Epoch)

Key final metrics from `wandb-summary.json`:

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Validity | 27.15% | Low but non-trivial for small dataset |
| Uniqueness | 100% | No duplicates |
| Novelty | 100% | All novel molecules |
| Val NLL | 288.98 | Not at best, fluctuating |
| Best Val NLL | 230.12 | Achieved at epoch 814 |
| nc_mu (mean components) | 1.27 | Average ~1.3 fragments per molecule |
| nc_max (max components) | 3 | Some molecules have disconnected parts |
| C_dist | +0.31 | Generates more C than training data |
| O_dist | -0.19 | Generates less O than training data |
| N_dist | -0.07 | Slightly less N than training data |

---

## 8. Model Parameters & Tuning Guide

### 8.1 All Tunable Parameters

#### Architecture Parameters (in `configs/model/discrete.yaml`)

| Parameter | Current Value | What it controls | Tuning advice |
|-----------|--------------|------------------|---------------|
| `n_layers` | 5 | Depth of graph transformer | 3–8; more layers for complex structures, fewer to reduce overfitting |
| `hidden_dims.dx` | 256 | Node feature dimension | 128–512; smaller for small datasets |
| `hidden_dims.de` | 64 | Edge feature dimension | 32–128 |
| `hidden_dims.dy` | 64 | Global feature dimension | 32–128 |
| `hidden_dims.n_head` | 8 | Attention heads | 4–16; dx must divide by n_head |
| `hidden_dims.dim_ffX` | 256 | Node feedforward hidden dim | Match or 2× dx |
| `hidden_dims.dim_ffE` | 128 | Edge feedforward hidden dim | Match or 2× de |
| `hidden_dims.dim_ffy` | 128 | Global feedforward hidden dim | Match or 2× dy |
| `hidden_mlp_dims.X` | 256 | Input MLP hidden dim (nodes) | 128–512 |
| `hidden_mlp_dims.E` | 128 | Input MLP hidden dim (edges) | 64–256 |
| `hidden_mlp_dims.y` | 128 | Input MLP hidden dim (global) | 64–256 |
| `diffusion_steps` | 500 | Number of noising/denoising steps | 200–1000; more steps = better quality, slower generation |
| `diffusion_noise_schedule` | cosine | Shape of the noise schedule | cosine or polynomial_2 |
| `extra_features` | all | Extra structural features fed to model | all, cycles, eigenvalues, or null |
| `lambda_train` | [5, 0] | Loss weights [node_weight, edge_weight] | [5, 0] means only nodes count in training CE; try [5, 1] or [1, 1] |
| `transition` | marginal | Transition model type | marginal (recommended) or uniform |

#### Training Parameters (in `configs/train/train_default.yaml` or experiment)

| Parameter | Current Value | What it controls | Tuning advice |
|-----------|--------------|------------------|---------------|
| `n_epochs` | 1000 | Total training epochs | 300–2000; watch for overfitting |
| `batch_size` | 32 | Batch size | 16–64 for small dataset; 32 is reasonable |
| `lr` | 0.0002 | Learning rate | 1e-4 to 5e-4; lower for small datasets |
| `weight_decay` | 1e-12 | L2 regularization | Increase to 1e-6 or 1e-4 to reduce overfitting |
| `ema_decay` | 0 | Exponential moving average | Set to 0.999 for smoother models; strongly recommended |
| `clip_grad` | null | Gradient clipping | Set to 1.0 for stability |
| `optimizer` | adamw | Optimizer choice | adamw (default), nadamw for large batch |

#### General/Sampling Parameters (in `configs/general/general_default.yaml`)

| Parameter | Current Value | What it controls | Tuning advice |
|-----------|--------------|------------------|---------------|
| `check_val_every_n_epochs` | 5 | Validation frequency | 5–10; more frequent = slower but better monitoring |
| `sample_every_val` | 4 | Sample every Nth validation | 2–4; every 2nd validation for closer monitoring |
| `samples_to_generate` | 512 | Molecules generated per sampling | 256–1024 |
| `final_model_samples_to_generate` | 10000 | Final molecules to generate | 1000–50000 |

#### Dataset Parameters (in `configs/dataset/mpro.yaml`)

| Parameter | Current Value | What it controls | Tuning advice |
|-----------|--------------|------------------|---------------|
| `remove_h` | True | Remove hydrogen atoms | True recommended for smaller graphs |
| `pin_memory` | True | Faster GPU data transfer | Keep True |

### 8.2 How to Override Parameters

Using Hydra command-line overrides:

```bash
# Change learning rate and batch size
python src/main.py train.lr=0.0001 train.batch_size=16

# Use full experiment config
python src/main.py experiment=mpro

# Enable EMA
python src/main.py train.ema_decay=0.999

# Reduce model size
python src/main.py model.n_layers=3 model.hidden_dims.dx=128

# More diffusion steps
python src/main.py model.diffusion_steps=1000
```

### 8.3 Summary of First Run

| Aspect | Assessment |
|--------|-----------|
| **Training convergence** | Loss decreased well initially, then plateaued with noise |
| **Overfitting** | Significant — val NLL fluctuates widely after epoch 300 |
| **Generation validity** | ~27–32% — low but expected for 304-molecule dataset |
| **Uniqueness/Novelty** | Excellent (100%/100%) |
| **Chemical realism** | Generated SMILES show Mpro-relevant scaffolds |
| **Recommended best checkpoint** | epoch=814 (val NLL 230.12) |
