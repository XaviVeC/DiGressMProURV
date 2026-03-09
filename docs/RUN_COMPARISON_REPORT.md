# DiGressMProURV: Run Output vs. Documentation Comparison Report

**Date:** Generated automatically  
**Run analyzed:** `outputs/2026-02-23/17-14-14-test/`  
**Documents compared:** `FIRST_RUN_ANALYSIS.md`, `PROPOSED_IMPROVEMENTS.md`  
**Data source:** `MPro-URV_Version2/`

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Dataset & Split Comparison](#2-dataset--split-comparison)
3. [Missing Molecules: Root Cause & Inventory](#3-missing-molecules-root-cause--inventory)
4. [Training Output vs. Documentation](#4-training-output-vs-documentation)
5. [SMILES Output File Bug](#5-smiles-output-file-bug)
6. [Proposed Improvements Analysis](#6-proposed-improvements-analysis)
7. [Recommendations & Action Items](#7-recommendations--action-items)

---

## 1. Executive Summary

This report compares the **actual training output** from run `17-14-14-test` against the documentation in `FIRST_RUN_ANALYSIS.md` and evaluates the `PROPOSED_IMPROVEMENTS.md` recommendations in light of the true dataset size and model configuration.

### Key Findings

| Finding | Severity | Summary |
|---------|----------|---------|
| **6 molecules silently dropped** | **Critical** | Corrupted PDB_IDs in `Info.csv` cause 6 molecules to be excluded from training/validation/test |
| **Document overstates dataset size** | Major | Doc claims "379 molecules" and "~304 train / ~95 valid / ~80 test"; reality is 378 in CSV, 372 used, 245/57/76 in fold-0 splits |
| **SMILES output files are corrupted** | Major | `writelines()` bug concatenates all SMILES into a single unreadable string |
| **Training metrics match** | OK | Validity (27–32%), best NLL (230.12 at epoch 814), uniqueness/novelty (100%) are accurately documented |
| **Proposed improvements reference wrong numbers** | Minor | All improvements use "304 training molecules" instead of actual 242 |

---

## 2. Dataset & Split Comparison

### 2.1 Total Molecule Count

| Source | Claimed | Actual | Match? |
|--------|---------|--------|--------|
| `FIRST_RUN_ANALYSIS.md` Section 5.1 | 379 molecules | — | — |
| `Info.csv` rows (excluding header) | — | **378** rows | **NO** (off by 1) |
| `.smi` files in `Ligand/Ligand_SMI/` | — | **378** files | Matches Info.csv |
| Split files fold-0 (train+valid+test) | — | 245+57+76 = **378** | Matches Info.csv |
| **Molecules actually used after `isin()` filtering** | — | 242+55+75 = **372** | **6 lost** |

The documentation claims **379** molecules. The actual `Info.csv` has **378** data rows. Of those, only **372** survive the code's `isin()` filtering due to 6 corrupted PDB_IDs.

### 2.2 Split Size Comparison

#### Document Claims vs. Reality (Fold 0)

| Split | `FIRST_RUN_ANALYSIS.md` claims | Split file (fold 0) | After `isin()` filtering | Discrepancy |
|-------|-------------------------------|---------------------|--------------------------|-------------|
| Train | ~304 | **245** | **242** | Doc is **+62 too high** vs. split file |
| Validation | ~95 | **57** | **55** | Doc is **+38 too high** vs. split file |
| Test | ~80 | **76** | **75** | Doc is **+4 too high** vs. split file |
| **Total** | **~479** | **378** | **372** | Doc total (~479) is **impossible** for a 378-molecule dataset |

The document's split sizes (~304/~95/~80 = ~479 total) are mathematically impossible for a 378-molecule dataset. No fold of the 5-fold split matches these numbers.

#### All 5 Folds (From Split Files)

| Fold | Train | Valid | Test | Total |
|------|-------|-------|------|-------|
| 0 | 245 | 57 | 76 | 378 |
| 1 | 245 | 57 | 76 | 378 |
| 2 | 245 | 57 | 76 | 378 |
| 3 | 246 | 57 | 75 | 378 |
| 4 | 246 | 57 | 75 | 378 |

Every fold totals 378, confirming all molecules are assigned. The code uses **fold 0** (hardcoded at line 175 of `mpro_dataset.py`).

---

## 3. Missing Molecules: Root Cause & Inventory

### 3.1 Root Cause

Six PDB_IDs in `Info.csv` are **corrupted by locale-specific formatting** — likely from opening the CSV in a spreadsheet application (e.g., Excel) with a European locale that interpreted the leading digits as numbers:

| Correct PDB_ID (in Split files) | Corrupted form in `Info.csv` | Corruption type |
|--------------------------------|------------------------------|-----------------|
| `7GEK` | `'7,00\xa0GEK'` | Number `7` → `7,00` + non-breaking space |
| `7GEL` | `'7,00\xa0GEL'` | Number `7` → `7,00` + non-breaking space |
| `7GIP` | `'7,00\xa0GIP'` | Number `7` → `7,00` + non-breaking space |
| `7GNF` | `'7\xa0GNF'` | Non-breaking space inserted |
| `7GNS` | `'7,00\xa0GNS'` | Number `7` → `7,00` + non-breaking space |
| `9LVR` | `'9,00\xa0LVR'` | Number `9` → `9,00` + non-breaking space |

The `\xa0` character is a **non-breaking space** (Unicode U+00A0). The `,00` suffix is European decimal formatting (e.g., `7,00` = `7.00`). These corrupted IDs fail to match the clean IDs in the split files when `df[df['PDB_ID'].isin(train_idx)]` is called, causing **silent data loss**.

### 3.2 Per-Split Inventory of Lost Molecules

| PDB_ID | Split | Has `.smi` file? | SMILES file name | Recoverable? |
|--------|-------|-------------------|------------------|--------------|
| **7GIP** | **Train** | Yes | `7GIP_QI7.smi` | Yes — fix Info.csv |
| **7GNS** | **Train** | Yes | `7GNS_S6K.smi` | Yes — fix Info.csv |
| **9LVR** | **Train** | Yes | `9LVR_A1L7P.smi` | Yes — fix Info.csv |
| **7GEK** | **Validation** | Yes | `7GEK_NOI.smi` | Yes — fix Info.csv |
| **7GEL** | **Validation** | Yes | `7GEL_NQ3.smi` | Yes — fix Info.csv |
| **7GNF** | **Test** | Yes | `7GNF_RWT.smi` | Yes — fix Info.csv |

**Impact:** 3 training molecules, 2 validation molecules, and 1 test molecule are silently excluded. All 6 have valid SMILES files and can be recovered by fixing the PDB_IDs in `Info.csv`.

### 3.3 Recovery

To fix `Info.csv`, replace the 6 corrupted PDB_IDs:

```
7,00 GEK  →  7GEK
7,00 GEL  →  7GEL
7,00 GIP  →  7GIP
7 GNF     →  7GNF
7,00 GNS  →  7GNS
9,00 LVR  →  9LVR
```

(Where the spaces shown above include the invisible non-breaking space character `\xa0`.)

After fixing, the dataset will use all 378 molecules: **245 train, 57 valid, 76 test**.

---

## 4. Training Output vs. Documentation

### 4.1 Configuration — Matches

| Parameter | Doc claims | Actual (`.hydra/config.yaml`) | Match? |
|-----------|-----------|-------------------------------|--------|
| Epochs | 1000 | 1000 | Yes |
| Batch size | 32 | 32 | Yes |
| Learning rate | 0.0002 | 0.0002 | Yes |
| Diffusion steps | 500 | 500 | Yes |
| Transformer layers | 5 | 5 | Yes |
| EMA decay | 0 (disabled) | 0 | Yes |
| Noise schedule | cosine | cosine | Yes |
| Transition | marginal | marginal | Yes |
| `lambda_train` | [5, 0] | [5, 0] | Yes |

### 4.2 Validation NLL Progression — Matches

| Epoch | Doc NLL | Actual NLL | Match? |
|-------|---------|------------|--------|
| 0 | 1469.65 | 1469.65 | Yes |
| 24 | 379.16 | 379.16 | Yes |
| Best (814) | 230.12 | 230.12 | Yes |
| 999 | 288.98 | 288.98 | Yes |

### 4.3 Generation Quality — Matches

| Metric | Doc claims | `wandb-summary.json` | Match? |
|--------|-----------|---------------------|--------|
| Peak validity | ~32% (epoch 974) | 32.03% epoch 974 | Yes |
| Final validity | 27.15% | 0.271484375 (27.15%) | Yes |
| Uniqueness | ~100% | 1.0 (100%) | Yes |
| Novelty | 100% | 1.0 (100%) | Yes |
| nc_mu (mean components) | 1.27 | 1.269 | Yes |
| nc_max | 3 | 3 | Yes |

### 4.4 Summary of Documentation Accuracy

| Section | Accuracy |
|---------|----------|
| Training config & hyperparameters | **Accurate** |
| Architecture description | **Accurate** |
| Code file descriptions | **Accurate** |
| NLL / Loss progression | **Accurate** |
| Validity / Uniqueness / Novelty | **Accurate** |
| Checkpoint listing | **Accurate** |
| **Molecule count (379 vs 378)** | **Inaccurate** |
| **Split sizes (~304/~95/~80)** | **Significantly inaccurate** |
| **6 lost molecules** | **Not mentioned at all** |
| **SMILES output file bug** | **Not mentioned** |

---

## 5. SMILES Output File Bug

### The Bug

In `src/metrics/molecular_metrics.py` (lines 153–155):

```python
valid_unique_molecules = rdkit_metrics[1]
textfile = open(f'graphs/{name}/valid_unique_molecules_e{current_epoch}_b{val_counter}.txt', "w")
textfile.writelines(valid_unique_molecules)
```

`writelines()` writes a list of strings **without any separator**. If `valid_unique_molecules` is `['CC(=O)O', 'c1ccccc1', 'CCCC']`, the output file contains:

```
CC(=O)Oc1ccccc1CCCC
```

instead of the expected:

```
CC(=O)O
c1ccccc1
CCCC
```

### Impact

- All `valid_unique_molecules_e*_b*.txt` files in `graphs/test/` are **unparseable** — you cannot extract individual SMILES from them
- The validity/uniqueness/novelty metrics computed during training are still correct (computed from the in-memory list before writing)
- The bug only affects the **saved output files**, not the training itself
- **Neither `FIRST_RUN_ANALYSIS.md` nor `PROPOSED_IMPROVEMENTS.md` mention this bug**

### Fix

```python
textfile.writelines([smi + '\n' for smi in valid_unique_molecules])
```

---

## 6. Proposed Improvements Analysis

The `PROPOSED_IMPROVEMENTS.md` proposes improvements organized by priority. Below is an assessment of each in light of the **actual** dataset size (242 training molecules, 372 total) and model configuration.

### Baseline Correction

The document consistently uses "304 training molecules" — the actual number is **242** (or 245 in the split file, 242 after corruption loss). This makes the overfitting and data-scarcity arguments even **stronger** than stated.

### 6.1 Critical Issues (P0) — All Valid

| # | Improvement | Assessment | Comments |
|---|------------|------------|----------|
| 1.1 | **Enable EMA** (`ema_decay: 0.999`) | **Strongly recommended** | With just 242 training molecules, weight averaging is critical. The 58-epoch gap between best NLL (epoch 814) and final epoch confirms unstable weights. Expected impact: noticeable smoothing of val metrics. |
| 1.2 | **Fix `lambda_train: [5, 1]`** | **Strongly recommended** | Edge KL dominates the validation NLL (237 out of 289 at final epoch). Giving zero weight to edge CE in training is the single biggest bottleneck. The doc correctly identifies this as the highest-impact change. |
| 1.3 | **Add early stopping** (patience=50) | **Recommended** | Best checkpoint at epoch 814 vs. 1000 total means 186 epochs of wasted compute. With 242 training samples, overfitting is severe and early stopping will help. Patience=50 (i.e. 250 actual epochs since val computed every 5) seems generous — patience=30 may be more practical. |
| 1.4 | **Gradient clipping** (`clip_grad: 1.0`) | **Recommended** | Standard good practice for small, noisy datasets. Low risk, moderate benefit. |
| 1.5 | **LR scheduler** (cosine annealing) | **Recommended** | Constant LR=0.0002 for 1000 epochs leaves no room for fine-grained convergence. Cosine annealing to `eta_min=1e-6` is a sensible choice. |

### 6.2 Model Architecture (P1) — Mostly Valid, One Caveat

| # | Improvement | Assessment | Comments |
|---|------------|------------|----------|
| 2.1 | **Reduce model size** (3 layers, dx=128) | **Recommended with caution** | The current model (5 layers, dx=256, 8 heads) is designed for QM9 (130K molecules) and is indeed severely overparameterized for 242 molecules. Reducing to 3 layers, dx=128, 4 heads is reasonable. However, **test both configurations** — sometimes larger models with strong regularization (EMA + dropout + early stopping) outperform smaller models. |
| 2.2 | **Increase dropout** (0.1 → 0.2) | **Recommended** | Standard regularization for small datasets. 0.2 is conservative; 0.3 may be worth trying. |
| 2.3 | **Fewer diffusion steps** (500 → 200) | **For development only** | The doc correctly notes this speeds up iteration. However, for final results, 500 steps should be kept. The quality difference between 200 and 500 steps can be significant for complex molecular structures. |

### 6.3 Data Pipeline & Augmentation (P1) — Mixed

| # | Improvement | Assessment | Comments |
|---|------------|------------|----------|
| 3.1 | **Compute stats from data** | **Strongly recommended** | The hardcoded distributions in `MproInfos._set_default_distributions()` may not match the actual 242-molecule training set (they may have been estimated from the original 378-molecule total). Since the marginal transition model uses these distributions as the noise target, inaccurate statistics directly degrade generation quality. |
| 3.2 | **SMILES augmentation** | **Not applicable** | The doc itself acknowledges this: "Since DiGress operates on graphs (not SMILES directly), SMILES augmentation doesn't help directly." The suggested graph augmentation alternatives (subgraph sampling, functional group perturbation) are research-grade ideas with uncertain payoff on a 242-molecule set. **Skip this.** |
| 3.3 | **Transfer learning from QM9** | **Promising but non-trivial** | Pre-training on QM9 (130K molecules) then fine-tuning on Mpro is theoretically sound — both use the same atom types (C, N, O, F, S, Cl, Br, I). However: (a) QM9 molecules are smaller (≤9 heavy atoms) while Mpro ligands are 15–50 heavy atoms — the size distribution mismatch is significant; (b) QM9 lacks aromatic bonds entirely while Mpro data has ~7% aromatic edges; (c) the dimensional compatibility depends on matching `extra_features` configurations. **Worth attempting but expect limited transfer.** A better pre-training corpus would be MOSES or GuacaMol (drug-like molecules). |
| 3.4 | **Replace `eval()` with `ast.literal_eval()`** | **Recommended** | Trivial fix, eliminates a security risk. The data format (list of lists of strings) is fully supported by `ast.literal_eval()`. |

### 6.4 Evaluation & Metrics (P1) — All Valid

| # | Improvement | Assessment | Comments |
|---|------------|------------|----------|
| 4.1 | **Drug-likeness metrics** (QED, Lipinski, SA) | **Recommended** | Validity alone is insufficient for a drug discovery project. QED, SA, and Lipinski compliance are computationally cheap and highly informative. |
| 4.2 | **Tanimoto similarity to training set** | **Recommended** | With 100% novelty, knowing *how different* the generated molecules are from training data is essential. A mean Tanimoto of 0.3 vs. 0.7 tells very different stories about generation quality. |
| 4.3 | **Scaffold analysis** | **Recommended** | Murcko scaffold comparison is standard in molecular generation literature and provides insight into whether the model learns meaningful chemical scaffolds vs. random valid structures. |
| 4.4 | **Consistent WandB logging** | **Good practice** | Ensures all metrics are trackable across runs. |

### 6.5 Code Quality (P2) — All Valid

All 5 code quality improvements (remove CUDA memory dumps, add type hints, config validation, seed propagation, replace `eval()`) are sound engineering practices with no downside.

### 6.6 Advanced Research (P2-P3) — Context-Dependent

| # | Improvement | Assessment | Comments |
|---|------------|------------|----------|
| 6.1 | **Conditional generation on pIC50** | **Very risky with 242 molecules** | The doc suggests discretized affinity bins, which is the right instinct. However, even with 3 bins (low/medium/high), the "potent" bin would have only ~22 molecules (9% of 242). The model cannot learn a meaningful conditional distribution from 22 examples. **Defer until dataset is expanded.** |
| 6.2 | **Multi-objective optimization** | **Premature** | Requires conditional generation to work first. Same concerns about data scarcity. |
| 6.3 | **Docking score integration** | **Valuable as post-processing** | Using AutoDock Vina or GNINA to rank generated molecules is independent of the generative model and should be done regardless. **Recommended as a downstream validation step.** |
| 6.4 | **Use `fixed_bug` branch** | **Investigate immediately** | If the DiGress `fixed_bug` branch fixes neural network layer bugs, it could affect all training results. This should have been P0, not P1. |
| 6.5 | **Data enrichment** | **The most impactful long-term change** | 242 training molecules is fundamentally too few for a diffusion model. ChEMBL has thousands of additional Mpro inhibitors. Curriculum learning (pre-train on MOSES/GuacaMol, fine-tune on Mpro) could dramatically improve results. |

---

## 7. Recommendations & Action Items

### Immediate Fixes (Before Next Run)

| Priority | Action | Effort |
|----------|--------|--------|
| **1** | **Fix 6 corrupted PDB_IDs in `Info.csv`** — restore `7GEK`, `7GEL`, `7GIP`, `7GNF`, `7GNS`, `9LVR` | 5 minutes |
| **2** | **Fix `writelines()` bug** in `molecular_metrics.py` line 155 — add `\n` separators | 1 minute |
| **3** | **Replace `eval()` with `ast.literal_eval()`** in `mpro_dataset.py` line 210 | 1 minute |
| **4** | **Check the DiGress `fixed_bug` branch** for applicable fixes | 30 minutes |

### Configuration Changes (Next Run)

| Priority | Change | Config parameter |
|----------|--------|-----------------|
| **5** | Enable EMA | `train.ema_decay=0.999` |
| **6** | Fix edge weight | `model.lambda_train=[5,1]` |
| **7** | Enable gradient clipping | `train.clip_grad=1.0` |
| **8** | Add early stopping | Code change in `main.py` |
| **9** | Add LR scheduler | Code change in `diffusion_model_discrete.py` |

### Model Tuning (After Validating Fixes)

| Priority | Change | Notes |
|----------|--------|-------|
| **10** | Experiment with 3-layer model (dx=128) | Compare against 5-layer with EMA+dropout |
| **11** | Increase dropout to 0.2 | Test both model sizes |
| **12** | Compute marginal distributions from actual data | Replace hardcoded stats |
| **13** | Add drug-likeness + Tanimoto + scaffold metrics | Better evaluation |

### Strategic (Parallel Track)

| Priority | Action | Notes |
|----------|--------|-------|
| **14** | Expand dataset via ChEMBL Mpro data | Most impactful long-term improvement |
| **15** | Explore transfer learning from MOSES/GuacaMol | Better source than QM9 for drug-like molecules |
| **16** | Correct `FIRST_RUN_ANALYSIS.md` — molecule count (378 not 379) and split sizes | Documentation accuracy |

---

## Appendix A: Validity Progression (Full Log)

| Epoch | Validity | Relaxed Validity |
|-------|----------|------------------|
| 14 | 0.00% | 0.00% |
| 34 | 0.00% | 0.00% |
| 54 | 0.20% | 0.20% |
| 74 | 1.95% | 2.15% |
| 94 | 1.76% | 2.15% |
| 114 | 3.52% | 3.71% |
| 134 | 3.52% | 3.71% |
| 154 | 5.86% | 6.45% |
| 174 | 4.49% | 4.88% |
| 194 | 5.66% | 6.05% |
| 214 | 7.81% | 8.01% |
| 234 | 10.35% | 10.55% |
| 254 | 13.48% | 13.87% |
| 274 | 14.06% | 15.04% |
| 294 | 13.48% | 14.06% |
| 314 | 18.16% | 18.95% |
| 334 | 16.60% | 17.38% |
| 354 | 15.43% | 16.41% |
| 374 | 15.04% | 15.82% |
| 394 | 18.95% | 19.73% |
| 414 | 18.36% | 19.14% |
| 434 | 17.58% | 18.36% |
| 454 | 16.41% | 17.19% |
| 474 | 18.75% | 19.34% |
| 494 | 20.51% | 21.09% |
| 514 | 16.80% | 17.38% |
| 534 | 21.29% | 22.07% |
| 554 | 20.31% | 20.90% |
| 574 | 22.27% | 22.66% |
| 594 | 26.76% | 27.54% |
| 614 | 26.37% | 27.15% |
| 634 | 22.46% | 23.24% |
| 654 | 27.34% | 28.32% |
| 674 | 23.24% | 24.02% |
| 694 | 24.41% | 25.98% |
| 714 | 31.64% | 32.42% |
| 734 | 28.32% | 29.49% |
| 754 | 22.46% | 23.24% |
| 774 | 21.48% | 22.27% |
| 794 | 28.52% | 29.30% |
| 814 | 26.95% | 28.32% |
| 834 | 25.59% | 26.76% |
| 854 | 26.56% | 27.34% |
| 874 | 22.07% | 22.85% |
| 894 | 26.37% | 27.34% |
| 914 | 20.70% | 21.09% |
| 934 | 23.63% | 24.41% |
| 954 | 28.71% | 29.10% |
| 974 | 32.03% | 33.01% |
| 994 | 27.15% | 27.93% |

## Appendix B: Val NLL Progression (Endpoints)

| Epoch | Val NLL | Atom KL | Edge KL |
|-------|---------|---------|---------|
| 0 | 1469.65 | 45.71 | 831.03 |
| 24 | 379.16 | 24.84 | 315.44 |
| 100 | ~330 | ~25 | ~270 |
| 500 | ~290 | ~20 | ~240 |
| **814** | **230.12** | **16.78** | **188.25** |
| 974 | 258.31 | 17.09 | 216.15 |
| 999 | 288.98 | 26.50 | 237.15 |
