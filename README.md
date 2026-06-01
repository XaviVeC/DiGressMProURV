# DiGress-MPro-URV: Molecular Generation and Activity Prediction for SARS-CoV-2 Main Protease Inhibitors

This repository extends the original DiGress discrete graph diffusion framework to the domain of
SARS-CoV-2 main protease (Mpro) inhibitor discovery. It covers three stages: (1) training and
fine-tuning DiGress generative models on Mpro and PDBbind data, (2) scoring the generated molecules
with a GINE-based pIC50 regressor, and (3) running data-augmentation experiments to measure whether
adding DiGress-generated pseudo-labeled molecules improves regression performance.

This work was developed as part of a Master's thesis (TFM) in the MESIIA programme at Universitat
Rovira i Virgili (URV).

---

## Authors

- **Javier Vega** — MPro-URV dataset integration, PDBbind pretraining pipeline, MPro fine-tuning,
  GINE regressor design, data-augmentation experiments, multi-seed evaluation (TFM MESIIA URV, 2026)
- **Clement Vignac, Igor Krawczuk, Antoine Siraudin, Bohan Wang, Volkan Cevher, Pascal Frossard** —
  Original DiGress framework (ICLR 2023)

---

## Overview of the workflow

Two independent pipelines were evaluated. The primary pipeline asks whether a domain-specific
generative model (trained only on MPro data) can produce molecules that, when added to the
training set, leave regressor quality unchanged — which would suggest the generated molecules
are chemically plausible MPro inhibitor candidates. The secondary pipeline asks the same
question for a generator that was first pre-trained on a large general-chemistry dataset
(PDBbind v2020) before being fine-tuned on MPro.

```
══════════════════════════════════════════════════════════════════════════
PRIMARY PIPELINE  (MPro-only DiGress → Upgraded GINE augmentation)
══════════════════════════════════════════════════════════════════════════

MPro-URV_Version2
        │
        ▼
[1] DiGress trained exclusively on MPro data
    config: mpro_upgraded  (epoch 320, run: 17-56-48-mpro_upgraded)
        │
        ▼
[2] SMILES generation
    → outputsDEF/28Apr2026/.../valid_unique_molecules_e320_b65.txt
        │
        ▼
[3] mpro_gine_pic50_regression.ipynb
    Baseline GINE trained on MPro only
    → external_predictions_28Apr2026_mpro_upgraded.csv  (446 molecules)
        │
        ▼
[4] mpro_digress_upgraded_gine_regression.ipynb
    GINE retrained on MPro + 200 filtered DiGress molecules (N_DIGRESS_FIXED=200)
    Multi-seed evaluation (seeds 42–46)
        │
        ▼
    upgraded_digress_test_predictions.csv
    RESULT: No significant quality degradation → generated molecules
            are plausible candidates for MPro inhibition


══════════════════════════════════════════════════════════════════════════
SECONDARY PIPELINE  (PDBbind-pretrained DiGress → Filtered/Unfiltered GINE)
══════════════════════════════════════════════════════════════════════════

PDBbind v2020
        │
        ▼
[1] DiGress pretrained on PDBbind
    config: pdbbind_pretrain
        │
        ▼
[2] DiGress fine-tuned on MPro
    config: mpro_finetune
    runs: mpro_finetune_20260414_163410_resume  (14Apr, epoch 75)
          mpro_finetune_smallbs_resume          (21Apr, epoch 80)
        │
        ▼
[3] SMILES generation
    → outputsDEF/14Apr2026/.../valid_unique_molecules_e75_b1.txt   (1802 mol.)
    → outputsDEF/21Apr2026/.../valid_unique_molecules_e80_b1.txt   (1780 mol.)
        │
        ▼
[4] mpro_gine_pic50_regression.ipynb
    Baseline GINE scores all generated molecules
    → external_predictions_14Apr2026.csv
    → external_predictions_21Apr2026.csv
        │
        ├──────────────────────────────────────┐
        ▼                                      ▼
[5a] mpro_digress_gine_pic50_regression    [5b] mpro_digress_filtered_gine_regression
     All 14+21 Apr molecules, no filter         14+21 Apr, filtered to MPro chemical
     (3785 augmented training graphs)           space (pIC50 ± buffer, atom count ±
                                                buffer), DIGRESS_APPEND_PERCENT = 50
                                                Multi-seed evaluation (seeds 42–46)
        │                                      │
        ▼                                      ▼
  digress_augmented_test_predictions.csv  filtered_digress_test_predictions.csv
```

---

## Repository structure

```
DiGressMProURV/
|
├── configs/
|   ├── config.yaml                    Main Hydra config entry point
|   ├── dataset/
|   |   ├── mpro.yaml                  MPro-URV_Version2 dataset config (~379 compounds, no-H)
|   |   ├── pdbbind.yaml               PDBbind v2020 dataset config (pretraining)
|   |   └── (digress defaults).yaml    Configs for QM9, MOSES, GuacaMol, etc.
|   ├── experiment/
|   |   ├── mpro.yaml                  Train DiGress from scratch on MPro
|   |   ├── mpro_finetune.yaml         Fine-tune from a PDBbind checkpoint onto MPro
|   |   ├── mpro_upgraded.yaml         Train DiGress on MPro only with tuned hyperparams
|   |   ├── pdbbind_pretrain.yaml      Pretrain DiGress on PDBbind v2020
|   |   ├── debug.yaml                 Quick smoke-test run
|   |   └── (digress defaults).yaml    Configs for other benchmark datasets
|   ├── general/general_default.yaml   Shared general settings
|   ├── model/
|   |   ├── discrete.yaml              Discrete diffusion model (used here)
|   |   └── continuous.yaml            Continuous diffusion model (original DiGress)
|   └── train/train_default.yaml       Default training hyperparams
|
├── data/
|   ├── mpro/processed/                PyG-processed MPro graphs (auto-generated)
|   └── pdbbind/processed/             PyG-processed PDBbind graphs (auto-generated)
|
├── MPro-URV_Version2/                 MPro inhibitor dataset (URV curation)
|   ├── Info.csv                       Full compound metadata
|   ├── train.csv / val.csv / test.csv Pre-defined splits
|   ├── pIC50.txt                      Binding affinity labels
|   ├── Complex/                       Protein-ligand complex structures (PDB/CIF)
|   ├── Interaction/                   Per-complex interaction fingerprints (JSON)
|   ├── Ligand/                        Ligand files (SDF, CIF, SMILES)
|   ├── Protein/                       Protein structures (PDB, CIF, DSSP)
|   └── Splits/                        Train/val/test index files
|
├── notebooks/
|   └── mpro_gine_regression/
|       ├── mpro_gine_pic50_regression.ipynb          Baseline GINE + pseudo-labeling
|       ├── mpro_digress_gine_pic50_regression.ipynb  Augmentation: all 14+21 Apr molecules
|       ├── mpro_digress_filtered_gine_regression.ipynb  Augmentation: filtered 14+21 Apr (50%)
|       ├── mpro_digress_upgraded_gine_regression.ipynb  Augmentation: filtered 28 Apr (N=200)
|       └── artifacts/                                Shared checkpoints, CSVs, JSON logs
|
├── PDBbindv2020/                      PDBbind v2020 processed splits
|   ├── Total.csv                      Full dataset metadata
|   ├── pdbbind_train.csv / pdbbind_val.csv / pdbbind_test.csv
|   └── Ligand_SMILES/                 Per-complex SMILES files
|
├── scripts/
|   └── run_pdbbind_pretrain_then_mpro_finetune.sh    Two-stage training script
|
├── src/
|   ├── main.py                        Hydra entry point for all training/generation runs
|   ├── diffusion_model_discrete.py    Core discrete DiGress model (used in all experiments)
|   ├── diffusion_model.py             Continuous variant (original DiGress)
|   ├── utils.py                       Shared utilities
|   ├── analysis/                      Graph metrics, RDKit helpers, ORCA orbit counts
|   ├── datasets/
|   |   ├── mpro_dataset.py            MPro-URV_Version2 dataset loader
|   |   ├── pdbbind_dataset.py         PDBbind v2020 dataset loader
|   |   └── (other datasets)           QM9, MOSES, GuacaMol, SBM, Planar
|   ├── diffusion/                     Noise schedule, transition matrices, extra features
|   ├── metrics/                       Molecular validity, uniqueness, train metrics
|   └── models/
|       ├── transformer_model.py       Graph Transformer backbone (DiGress architecture)
|       └── layers.py                  Attention and MLP layers
|
└── outputs/                           Local training run outputs (date-stamped)

outputsDEF/                            DiGress generation outputs used for regression experiments
├── 14Apr2026/  (PDBbind pretrain → MPro fine-tune)
|   └── mpro_finetune_20260414_163410_resume/
|       └── valid_unique_molecules_e75_b1.txt   1802 valid unique SMILES
├── 21Apr2026/  (PDBbind pretrain → MPro fine-tune, small batch)
|   └── mpro_finetune_smallbs_resume/
|       └── valid_unique_molecules_e80_b1.txt   1780 valid unique SMILES
└── 28Apr2026/  (MPro-only training, upgraded hyperparams)
    └── 17-56-48-mpro_upgraded/
        └── graphs/mpro_upgraded/
            └── valid_unique_molecules_e320_b65.txt   446 valid unique SMILES
```

---

## Experiments

### Stage 1 — DiGress generative model

Three DiGress runs were conducted, all using the discrete diffusion model with a Graph Transformer
backbone (3 layers, hidden dims dx=128, de=32, dy=32, 4 attention heads):

| Label | Config | Training | Checkpoint used |
|-------|--------|----------|-----------------|
| 14Apr2026 | `pdbbind_pretrain` → `mpro_finetune` | PDBbind pretrain (200 epochs) then MPro fine-tune | epoch 75 |
| 21Apr2026 | `pdbbind_pretrain` → `mpro_finetune` (small batch) | Same two-stage, smaller batch | epoch 80 |
| 28Apr2026 | `mpro_upgraded` | MPro only from scratch, tuned hyperparams | epoch 320 |

The two-stage runs (14Apr, 21Apr) can be reproduced with:

```bash
bash scripts/run_pdbbind_pretrain_then_mpro_finetune.sh <PDBBIND_DIR> <MPRO_DIR>
```

Single-stage MPro-only run:

```bash
python3 src/main.py +experiment=mpro_upgraded
```

### Stage 2 — GINE pIC50 regressor and pseudo-labeling

The notebook `notebooks/mpro_gine_regression/mpro_gine_pic50_regression.ipynb` trains a baseline
GINERegressor (6 GINE layers, hidden dim 256, dropout 0.05, lr 5e-4) on the MPro training split
only. Hyperparameters were found by Bayesian search and stored in
`artifacts/gine_best_hparams.json` (val RMSE = 0.525, val R² = 0.653). The trained model is then
used to score all DiGress-generated SMILES, producing pseudo-label CSVs:

- `artifacts/external_predictions_14Apr2026.csv` — 1802 molecules
- `artifacts/external_predictions_21Apr2026.csv` — 1780 molecules
- `artifacts/external_predictions_28Apr2026_mpro_upgraded.csv` — 446 molecules

### Stage 3 — Data augmentation experiments

Three augmentation notebooks retrain the GINE on MPro + pseudo-labeled DiGress molecules and
compare test-set performance against the baseline. All share the same val/test split and the
same hyperparameters from `gine_best_hparams.json`.

| Notebook | Source CSVs | Filter | Size control | Checkpoint |
|----------|-------------|--------|--------------|------------|
| `mpro_digress_gine_pic50_regression.ipynb` | 14Apr + 21Apr | None | All available (3785 train graphs) | `gine_digress_pic50_regressor.pt` |
| `mpro_digress_filtered_gine_regression.ipynb` | 14Apr + 21Apr | pIC50 + atom count | `DIGRESS_APPEND_PERCENT = 50` | `gine_filtered_digress_pic50_regressor.pt` |
| `mpro_digress_upgraded_gine_regression.ipynb` | 28Apr only | pIC50 + atom count | `N_DIGRESS_FIXED = 200` | `gine_upgraded_digress_pic50_regressor.pt` |

The filtered and upgraded notebooks include a five-seed robustness evaluation (seeds 42–46).
Summary statistics (mean ± std of RMSE, MAE, R², Pearson r) are printed alongside ROC and
scatter plots for the best seed. Per-seed checkpoints are saved as
`gine_filtered_digress_seed{43..46}.pt` and `gine_upgraded_digress_seed{43..46}.pt`.

---

## Environment installation

This project was developed with Python 3.9, PyTorch 2.x (CPU), and RDKit 2025.x inside a conda
environment named `digress`.

```bash
conda create -c conda-forge -n digress rdkit python=3.9
conda activate digress
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .
```

To use GPU acceleration, replace the PyTorch install with the appropriate CUDA wheel, e.g.:

```bash
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda
```

Compile the ORCA orbit-counting binary (required for graph metrics):

```bash
cd src/analysis/orca
g++ -O2 -std=c++11 -o orca orca.cpp
```

---

## Running the code

All DiGress training and generation runs are launched via Hydra through `src/main.py`:

```bash
# Quick smoke test
python3 src/main.py +experiment=debug

# Pretrain on PDBbind
python3 src/main.py +experiment=pdbbind_pretrain +dataset.raw_data_dir=<PDBBIND_DIR>

# Fine-tune on MPro (resume from a PDBbind checkpoint)
python3 src/main.py +experiment=mpro_finetune general.resume=<CHECKPOINT_PATH>

# Train on MPro from scratch (upgraded config)
python3 src/main.py +experiment=mpro_upgraded +dataset.raw_data_dir=<MPRO_DIR>

# Run generation only on a saved checkpoint
python3 src/main.py +experiment=mpro_finetune general.test_only=<CHECKPOINT_PATH>
```

After generation, run the four regression notebooks in order:
1. `mpro_gine_pic50_regression.ipynb` — must be run first to produce the pseudo-label CSVs and `gine_best_hparams.json`
2. The three augmentation notebooks can be run in any order afterwards

---

## Troubleshooting

- `PermissionError: .../orca/orca`: compile orca as described above.
- `FileNotFoundError: Checkpoint not found`: run the preceding notebook first to generate the `.pt` file.
- `KeyError` in dataset loading: ensure `raw_data_dir` points to the correct data root, or leave it `null` to use the default relative path resolution.

---

## Cite the original DiGress paper

```bibtex
@inproceedings{
vignac2023digress,
title={DiGress: Discrete Denoising diffusion for graph generation},
author={Clement Vignac and Igor Krawczuk and Antoine Siraudin and Bohan Wang and Volkan Cevher and Pascal Frossard},
booktitle={The Eleventh International Conference on Learning Representations},
year={2023},
url={https://openreview.net/forum?id=UaAD-Nu86WX}
}
```
