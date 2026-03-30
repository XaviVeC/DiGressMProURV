# PDBbindv2020 Integration for DiGress

This document explains the implemented two-stage workflow:

1. Pretrain DiGress on PDBbindv2020 so the graph diffusion model learns broad ligand chemistry.
2. Fine-tune the pretrained checkpoint on the existing MPro dataset so generation shifts toward SARS-CoV-2 MPro inhibitors.

## What Was Added

- A new dataset module: `src/datasets/pdbbind_dataset.py`
- A new dataset config: `configs/dataset/pdbbind.yaml`
- A pretraining experiment config: `configs/experiment/pdbbind_pretrain.yaml`
- A fine-tuning experiment config: `configs/experiment/mpro_finetune.yaml`
- Dataset registration in `src/main.py`

## Data Assumptions

The code expects a local folder:

`../PDBbindv2020`

relative to `DiGressMProURV`, unless `dataset.raw_data_dir` is overridden.

The folder must contain:

- `Total.csv`
- `Ligand_SMILES/`

## Filtering Rules

PDBbind records are filtered before split creation and graph processing:

- Keep ligands whose `.smi` file exists
- Keep ligands with RDKit-parseable SMILES
- Keep only molecules with atom types compatible with MPro fine-tuning: `C, N, O, F, S, Cl, Br, I` (hydrogens removed after parsing)
- Keep only molecules with `<= 60` heavy atoms so the graph-size regime stays compatible with MPro
- Keep both covalent and non-covalent ligands

The diffusion training remains unconditional, matching the current MPro setup. PDBbind `[-log k]` values are loaded into the graph objects as metadata but removed from `y` during training by `RemoveYTransform()`.

## Why PDBbind Needs Its Own Split

Yes, PDBbind needs a train/val/test split.

DiGress relies on:

- validation for early stopping and checkpoint ranking (`val/epoch_NLL`)
- test for final evaluation and sample reporting

The implementation creates deterministic splits with seed `42`:

- `pdbbind_train.csv`
- `pdbbind_val.csv`
- `pdbbind_test.csv`

using an `80 / 10 / 10` split over the filtered ligand set.

## Training Workflow

### 1. Pretrain on PDBbind

```bash
python src/main.py +experiment=pdbbind_pretrain
```

Outputs follow the existing DiGress structure:

- `outputs/<date>/<time>-pdbbind_pretrain/`
- `checkpoints/pdbbind_pretrain/`
- `graphs/pdbbind_pretrain/`
- `chains/pdbbind_pretrain/`

WandB project naming also stays aligned with the current code:

- `graph_ddm_pdbbind`

### 2. Fine-tune on MPro

```bash
python src/main.py +experiment=mpro_finetune general.resume='outputs/<date>/<time>-pdbbind_pretrain/checkpoints/pdbbind_pretrain/last.ckpt'
```

This reuses the pretrained GraphTransformer weights and fine-tunes with the existing MPro train/val/test split.

Outputs stay in the same structure:

- `outputs/<date>/<time>-mpro_finetune/`
- `checkpoints/mpro_finetune/`
- `graphs/mpro_finetune/`
- `chains/mpro_finetune/`

WandB project:

- `graph_ddm_mpro`

## Important Compatibility Constraint

PDBbind pretraining and MPro fine-tuning must keep the same model feature space and architecture:

- same atom vocabulary
- same bond encoding
- same hidden dimensions
- same number of layers

That is why the new experiment configs intentionally share the same core model dimensions.

## Override Examples

Use a custom PDBbind location:

```bash
python src/main.py +experiment=pdbbind_pretrain dataset.raw_data_dir='C:/path/to/PDBbindv2020'
```

Reduce epochs for a smoke test:

```bash
python src/main.py +experiment=pdbbind_pretrain general.name=debug train.n_epochs=2 general.wandb=disabled
```

Fine-tune from a specific checkpoint:

```bash
python src/main.py +experiment=mpro_finetune general.resume='outputs/2026-03-30/12-00-00-pdbbind_pretrain/checkpoints/pdbbind_pretrain/last.ckpt'
```