# DiGressMProURV: Proposed Improvements

**Date:** March 2, 2026  
**Project:** TFM (Treball Fi Master) – MESIIA Program, URV  
**Status:** Recommendations based on first run analysis  
**Priority levels:** P0 (Critical), P1 (High), P2 (Medium), P3 (Nice-to-have)

---

## Table of Contents

1. [Critical Training Issues](#1-critical-training-issues-p0)
2. [Model Architecture Improvements](#2-model-architecture-improvements-p1)
3. [Data Pipeline & Augmentation](#3-data-pipeline--augmentation-p1)
4. [Evaluation & Metrics](#4-evaluation--metrics-p1)
5. [Code Quality & Engineering](#5-code-quality--engineering-p2)
6. [Advanced Research Directions](#6-advanced-research-directions-p2-p3)
7. [Implementation Roadmap](#7-implementation-roadmap)

---

## 1. Critical Training Issues (P0)

### 1.1 Enable EMA (Exponential Moving Average)

**Problem:** `ema_decay: 0` — EMA is disabled. The final model weights are the raw last-step weights, which are noisy due to SGD variance. This is especially harmful for small datasets.

**Evidence:** Val NLL fluctuates wildly (230–370) in late training. The best checkpoint (epoch 814) is far from the last (epoch 999), indicating unstable weights.

**Fix:**
```yaml
# configs/train/train_default.yaml or experiment override
ema_decay: 0.999   # or 0.9999 for slower averaging
```

**Impact:** EMA averages model weights over recent epochs, producing much smoother and more stable models. This alone could improve validity by 5–15%.

---

### 1.2 Fix `lambda_train` Edge Weight

**Problem:** `lambda_train: [5, 0]` means edge prediction has **zero weight** in the training cross-entropy loss. The model only receives gradient signal for atom types, not bond types, during the training step loss computation.

**Evidence:** Edge KL dominates the validation NLL (188 out of 230 at best). The model learns bond patterns only through the full diffusion NLL computed at validation, not through the per-step training loss.

**Fix:**
```yaml
# configs/model/discrete.yaml
lambda_train: [5, 1]   # Add edge weight > 0
# or
lambda_train: [5, 2]   # Stronger edge signal
```

**Impact:** Directly training on edge prediction should significantly reduce edge KL and improve structural validity. This is likely the **single highest-impact change**.

---

### 1.3 Add Early Stopping

**Problem:** Training runs for 1000 epochs with no early stopping. Best val NLL (230.12) is at epoch 814, but the model continues training for 186 more epochs with degrading validation performance.

**Fix in `src/main.py`:**
```python
from pytorch_lightning.callbacks import EarlyStopping

if cfg.train.save_model:
    early_stop_callback = EarlyStopping(
        monitor='val/epoch_NLL',
        patience=50,       # Stop after 50 epochs without improvement
        mode='min',
        verbose=True
    )
    callbacks.append(early_stop_callback)
```

**Also add to config:**
```yaml
train:
  early_stopping_patience: 50  # New parameter
```

---

### 1.4 Enable Gradient Clipping

**Problem:** `clip_grad: null` — no gradient clipping. On a small, noisy dataset, gradient spikes can destabilize training.

**Fix:**
```yaml
train:
  clip_grad: 1.0
```

---

### 1.5 Use a Learning Rate Scheduler

**Problem:** No LR scheduler is configured. The constant LR of 0.0002 is fine early but prevents fine-grained convergence later.

**Fix in `src/diffusion_model_discrete.py`:**
```python
def configure_optimizers(self):
    optimizer = torch.optim.AdamW(
        self.parameters(), lr=self.cfg.train.lr,
        amsgrad=True, weight_decay=self.cfg.train.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=self.cfg.train.n_epochs, eta_min=1e-6
    )
    return {"optimizer": optimizer, "lr_scheduler": scheduler}
```

---

## 2. Model Architecture Improvements (P1)

### 2.1 Reduce Model Size to Prevent Overfitting

**Problem:** The model has ~256-dim node features with 5 layers and 8 heads — this is the default QM9 configuration designed for 130K molecules. With only 304 training molecules, this model is severely overparameterized.

**Proposed reduced config:**
```yaml
model:
  n_layers: 3                    # Was 5 → fewer layers
  hidden_mlp_dims:
    X: 128                       # Was 256
    E: 64                        # Was 128
    'y': 64                      # Was 128
  hidden_dims:
    dx: 128                      # Was 256
    de: 32                       # Was 64
    dy: 32                       # Was 64
    n_head: 4                    # Was 8
    dim_ffX: 128                 # Was 256
    dim_ffE: 64                  # Was 128
    dim_ffy: 64                  # Was 128
```

**Estimated param reduction:** ~75% fewer parameters. This should reduce overfitting substantially.

### 2.2 Increase Dropout

**Problem:** Default dropout in the transformer is 0.1. For a 304-molecule dataset, this is insufficient regularization.

**Fix:** Modify `src/models/transformer_model.py` to accept a configurable dropout:
```python
class GraphTransformer(nn.Module):
    def __init__(self, n_layers, input_dims, hidden_mlp_dims, hidden_dims,
                 output_dims, act_fn_in, act_fn_out, dropout=0.2):  # Higher default
```

**And pass from config:**
```yaml
model:
  dropout: 0.2   # or 0.3 for aggressive regularization
```

### 2.3 Fewer Diffusion Steps for Faster Iteration

**Problem:** 500 diffusion steps makes sampling slow (~420s per sampling epoch). During development, faster iteration is more valuable.

**Fix:**
```yaml
model:
  diffusion_steps: 200   # Faster sampling; increase for final runs
```

---

## 3. Data Pipeline & Augmentation (P1)

### 3.1 Compute Dataset Statistics From Data (Don't Hardcode)

**Problem:** `MproInfos._set_default_distributions()` uses **hardcoded** approximate distributions:
```python
self.n_nodes = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0.02, 0.04, 0.08, ...])  # Manually estimated
```

These may not match the actual data, causing the marginal transition model to have incorrect limiting distributions, which degrades generation quality.

**Fix:** Always compute from data by passing `recompute_statistics=True`:
```python
# In mpro_dataset.py, MproInfos.__init__()
super().__init__()
self._compute_statistics(datamodule)  # Always recompute
```

Or better, **cache the computed statistics** to a file:
```python
def __init__(self, datamodule, cfg, recompute_statistics=False):
    stats_path = os.path.join(cfg.dataset.datadir, 'dataset_stats.pt')
    if os.path.exists(stats_path) and not recompute_statistics:
        stats = torch.load(stats_path)
        self.n_nodes = stats['n_nodes']
        self.node_types = stats['node_types']
        self.edge_types = stats['edge_types']
        self.valency_distribution = stats['valency_distribution']
    else:
        self._compute_statistics(datamodule)
        torch.save({
            'n_nodes': self.n_nodes,
            'node_types': self.node_types,
            'edge_types': self.edge_types,
            'valency_distribution': self.valency_distribution
        }, stats_path)
    super().complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)
```

### 3.2 SMILES Augmentation

**Problem:** Only 304 training molecules — severe data scarcity. SMILES strings have multiple valid representations for the same molecule (non-canonical orderings).

**Fix:** Add SMILES augmentation in `MproDataset.process()`:
```python
from rdkit.Chem import MolToSmiles, MolFromSmiles

def augment_smiles(smiles, num_augmentations=10):
    """Generate multiple random SMILES orderings for the same molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [smiles]
    augmented = set()
    for _ in range(num_augmentations * 3):  # Over-sample to account for duplicates
        aug = Chem.MolToSmiles(mol, doRandom=True)
        augmented.add(aug)
        if len(augmented) >= num_augmentations:
            break
    return list(augmented)
```

**Note:** Since DiGress operates on graphs (not SMILES directly), SMILES augmentation doesn't help directly. Instead, consider **graph augmentation**:
- Random atom reordering (node permutation) — already invariant in the model
- Subgraph sampling for larger molecules
- Add/remove random hydrogens or functional groups as training noise

### 3.3 Transfer Learning from QM9

**Problem:** Training from scratch on 304 molecules is data-inefficient. QM9 has 130K molecules with the same atom types.

**Fix:** Pre-train on QM9, then fine-tune on Mpro:
```bash
# Step 1: Train on QM9
python src/main.py dataset=qm9 general.name=qm9_pretrain train.n_epochs=500

# Step 2: Fine-tune on Mpro with lower LR
python src/main.py dataset=mpro \
    general.resume=outputs/.../checkpoints/qm9_pretrain/last.ckpt \
    train.lr=0.00005 \
    train.n_epochs=200
```

The `get_resume_adaptive()` function in `main.py` already supports this workflow. The only caveat is that Mpro uses the same atom types as QM9 minus a few, so the network dimensions should be compatible if `remove_h=True`.

### 3.4 Fix Split Index Loading Safety

**Problem:** `_load_split_indices()` uses `eval(content)` which is a **security risk**:
```python
indices_list = eval(content)  # Dangerous!
```

**Fix:** Replace with `ast.literal_eval` or `json.loads`:
```python
import ast
indices_list = ast.literal_eval(content)
```

---

## 4. Evaluation & Metrics (P1)

### 4.1 Add Drug-Likeness Metrics

**Problem:** Current metrics (validity, uniqueness, novelty) measure general chemical quality but not **drug-likeness** or **Mpro relevance**.

**Fix:** Add to `src/metrics/molecular_metrics.py`:
```python
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors

def compute_drug_metrics(smiles_list):
    """Compute drug-likeness metrics for generated molecules."""
    metrics = {
        'MW': [],       # Molecular weight
        'LogP': [],     # Lipophilicity
        'HBD': [],      # H-bond donors
        'HBA': [],      # H-bond acceptors
        'TPSA': [],     # Topological polar surface area
        'QED': [],      # Quantitative Estimate of Drug-likeness
        'SA': [],       # Synthetic accessibility
        'Lipinski_pass': 0,  # Lipinski's Rule of Five compliance
    }
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        tpsa = Descriptors.TPSA(mol)
        
        metrics['MW'].append(mw)
        metrics['LogP'].append(logp)
        metrics['HBD'].append(hbd)
        metrics['HBA'].append(hba)
        metrics['TPSA'].append(tpsa)
        
        # Lipinski Rule of 5
        if mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10:
            metrics['Lipinski_pass'] += 1
    
    return metrics
```

### 4.2 Add Tanimoto Similarity to Training Set

**Problem:** "Novelty = 100%" doesn't tell us how _similar_ generated molecules are to the training data. They could be completely irrelevant structures.

**Fix:**
```python
from rdkit.Chem import AllChem
from rdkit import DataStructs

def compute_similarity_to_training(generated_smiles, train_smiles, top_k=5):
    """Compute average Tanimoto similarity to nearest training molecules."""
    train_fps = [AllChem.GetMorganFingerprintAsBitVect(
        Chem.MolFromSmiles(s), 2, 1024) for s in train_smiles if Chem.MolFromSmiles(s)]
    
    similarities = []
    for smi in generated_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)
        sims = [DataStructs.TanimotoSimilarity(fp, tfp) for tfp in train_fps]
        sims.sort(reverse=True)
        similarities.append(np.mean(sims[:top_k]))
    
    return {
        'mean_similarity': np.mean(similarities),
        'median_similarity': np.median(similarities),
        'max_similarity': np.max(similarities),
    }
```

### 4.3 Add Scaffold Analysis

**Problem:** No analysis of whether generated molecules share scaffolds with known Mpro inhibitors.

**Fix:**
```python
from rdkit.Chem.Scaffolds import MurckoScaffold

def compute_scaffold_metrics(generated_smiles, train_smiles):
    """Compare scaffolds of generated vs. training molecules."""
    def get_scaffold(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    
    gen_scaffolds = set(filter(None, [get_scaffold(s) for s in generated_smiles]))
    train_scaffolds = set(filter(None, [get_scaffold(s) for s in train_smiles]))
    
    return {
        'num_gen_scaffolds': len(gen_scaffolds),
        'scaffold_overlap': len(gen_scaffolds & train_scaffolds),
        'novel_scaffolds': len(gen_scaffolds - train_scaffolds),
        'scaffold_diversity': len(gen_scaffolds) / max(len(generated_smiles), 1),
    }
```

### 4.4 Log Metrics Consistently to WandB

**Problem:** Many computed metrics are printed to stdout but not all are systematically logged to WandB (e.g., per-epoch drug metrics).

**Fix:** Ensure all custom metrics pass through `wandb.log()` with clear namespacing:
```python
if wandb.run:
    wandb.log({
        "drug_metrics/mean_MW": np.mean(metrics['MW']),
        "drug_metrics/mean_LogP": np.mean(metrics['LogP']),
        "drug_metrics/lipinski_pass_rate": metrics['Lipinski_pass'] / len(smiles_list),
        "similarity/mean_tanimoto": sim_metrics['mean_similarity'],
    })
```

---

## 5. Code Quality & Engineering (P2)

### 5.1 Remove Unnecessary CUDA Memory Logging

**Problem:** Every training epoch prints a full CUDA memory summary (~30 lines of tables), creating a 55,302-line log file. This makes logs unreadable.

**Fix in `src/diffusion_model_discrete.py`:**
```python
def on_train_epoch_end(self) -> None:
    to_log = self.train_loss.log_epoch_metrics()
    self.print(
        f"Epoch {self.current_epoch}: X_CE: {to_log['train_epoch/x_CE']:.3f}"
        f" -- E_CE: {to_log['train_epoch/E_CE']:.3f}"
        f" -- y_CE: {to_log['train_epoch/y_CE']:.3f}"
        f" -- {time.time() - self.start_epoch_time:.1f}s"
    )
    epoch_at_metrics, epoch_bond_metrics = self.train_metrics.log_epoch_metrics()
    self.print(f"Epoch {self.current_epoch}: {epoch_at_metrics} -- {epoch_bond_metrics}")
    # Remove or gate behind debug flag:
    # if self.cfg.get('debug', {}).get('verbose', False) and torch.cuda.is_available():
    #     print(torch.cuda.memory_summary())
```

### 5.2 Add Type Hints and Docstrings

**Problem:** Many functions in the original codebase lack type hints. The new `mpro_dataset.py` has good docstrings but some methods could be clearer.

**Example fix:**
```python
def process(self) -> None:
    """Convert SMILES to molecular graphs and save as .pt files.
    
    Reads from MPro-URV_Version2 database, creates PyG Data objects
    with one-hot node features (atom types) and edge features (bond types),
    with optional hydrogen removal.
    
    Saves:
        self.processed_paths[self.file_idx]: Collated dataset .pt file
    """
```

### 5.3 Configuration Validation

**Problem:** No validation that config parameters are within expected ranges. Invalid combinations (e.g., `dx=256, n_head=7`) silently crash.

**Fix — add a validation function in `main.py`:**
```python
def validate_config(cfg):
    """Validate configuration parameters before training."""
    assert cfg.model.hidden_dims.dx % cfg.model.hidden_dims.n_head == 0, \
        f"dx ({cfg.model.hidden_dims.dx}) must be divisible by n_head ({cfg.model.hidden_dims.n_head})"
    assert cfg.train.batch_size > 0, "Batch size must be positive"
    assert 0 <= cfg.train.ema_decay < 1, f"EMA decay must be in [0, 1), got {cfg.train.ema_decay}"
    if cfg.dataset.name == 'mpro':
        assert cfg.train.batch_size <= 64, \
            f"Batch size {cfg.train.batch_size} too large for Mpro dataset (304 samples)"
```

### 5.4 Reproducibility: Set All Seeds

**Problem:** `train.seed: 0` is set but may not be propagated to all random number generators.

**Fix:**
```python
import pytorch_lightning as pl
pl.seed_everything(cfg.train.seed, workers=True)
```

This should be called at the top of `main()` before any data loading.

### 5.5 Replace `eval()` with Safe Alternative

**Problem:** In `mpro_dataset.py`, `_load_split_indices()` uses `eval()` to parse split files, which is a security vulnerability.

**Fix:**
```python
import ast
indices_list = ast.literal_eval(content)
```

---

## 6. Advanced Research Directions (P2-P3)

### 6.1 Conditional Generation on pIC50 (P2)

**Current state:** pIC50 values are loaded but zeroed out by `RemoveYTransform`. The experiment config has `conditional_generation: false`.

**Proposed implementation:**

1. Remove `RemoveYTransform` from the datamodule:
```python
datasets = {
    'train': MproDataset(stage='train', root=root_path,
                         remove_h=cfg.dataset.remove_h,
                         dataset_dir=dataset_dir),  # No transform
    ...
}
```

2. The `y` tensor already carries pIC50 values as the graph-level feature. DiGress's architecture already accepts `y` through the transformer's global feature pathway.

3. At generation time, provide target pIC50 to guide toward high-affinity molecules:
```python
# In sample_batch(), replace y initialization:
target_pic50 = 6.5  # Generate potent inhibitors
y = torch.full((batch_size, 1), target_pic50, device=self.device)
```

**Challenge:** With only 304 molecules and a continuous pIC50 range, conditional generation may need discretized affinity bins (e.g., low/medium/high) rather than exact values.

### 6.2 Multi-Objective Optimization (P3)

**Goal:** Generate molecules that are simultaneously:
- Chemically valid
- Drug-like (Lipinski compliant)
- Structurally similar to known Mpro inhibitors
- Predicted to have high binding affinity

**Approach:** Use the `guidance` branch of DiGress (mentioned in README) or implement classifier-free guidance:
```python
# During sampling, blend conditional and unconditional predictions
pred_conditional = self.forward(noisy_data, extra_data, node_mask)
pred_unconditional = self.forward(noisy_data_no_y, extra_data, node_mask)
pred_guided = pred_unconditional + guidance_scale * (pred_conditional - pred_unconditional)
```

### 6.3 Docking Score Integration (P3)

**Goal:** Validate generated molecules by computing predicted binding to Mpro active site.

**Approach:** Post-generation filtering pipeline:
1. Generate N molecules with DiGress
2. Filter by validity and drug-likeness
3. Convert to 3D conformers (RDKit ETKDG)
4. Dock against Mpro crystal structure (AutoDock Vina or GNINA)
5. Rank by predicted binding energy
6. (Optional) Use docking scores as reward signal for guided generation

### 6.4 Use the `fixed_bug` Branch (P1)

The DiGress README recommends using the `fixed_bug` branch for new training from scratch:

> "If you are training new models from scratch, we recommend to use the `fixed_bug` branch in which some neural network layers have been fixed."

This should be investigated and merged if applicable.

### 6.5 Data Enrichment (P2)

**Problem:** 379 molecules is very small for a generative model.

**Options:**
- **ChEMBL Mpro data**: Search ChEMBL for additional SARS-CoV-2 Mpro inhibitors (hundreds more exist)
- **Related protease inhibitors**: Include inhibitors of similar proteases (e.g., 3CLpro from other coronaviruses, HIV protease inhibitors with similar binding pockets)
- **Curriculum learning**: Pre-train on a larger drug-like dataset (MOSES: 1.9M molecules), then fine-tune on Mpro

---

## 7. Implementation Roadmap

### Phase 1: Quick Wins (1-2 days)

| # | Change | Expected Impact | Files |
|---|--------|----------------|-------|
| 1 | Enable EMA (`ema_decay: 0.999`) | Smoother, more stable model | `configs/train/train_default.yaml` |
| 2 | Fix `lambda_train: [5, 1]` | Better bond prediction | `configs/model/discrete.yaml` |
| 3 | Enable gradient clipping (`clip_grad: 1.0`) | Training stability | `configs/train/train_default.yaml` |
| 4 | Add early stopping (patience=50) | Prevent overfitting | `src/main.py` |
| 5 | Remove CUDA memory dump from logs | Cleaner logs | `src/diffusion_model_discrete.py` |

### Phase 2: Architecture Tuning (3-5 days)

| # | Change | Expected Impact | Files |
|---|--------|----------------|-------|
| 6 | Reduce model size (3 layers, dx=128) | Reduce overfitting | `configs/model/discrete.yaml` |
| 7 | Compute statistics from data | Correct marginal distributions | `src/datasets/mpro_dataset.py` |
| 8 | Add LR scheduler (cosine annealing) | Better convergence | `src/diffusion_model_discrete.py` |
| 9 | Increase dropout to 0.2–0.3 | Regularization | `src/models/transformer_model.py` |
| 10 | Replace `eval()` with `ast.literal_eval()` | Security fix | `src/datasets/mpro_dataset.py` |

### Phase 3: Evaluation & Analysis (1 week)

| # | Change | Expected Impact | Files |
|---|--------|----------------|-------|
| 11 | Add drug-likeness metrics (QED, Lipinski, SA) | Better evaluation | `src/metrics/molecular_metrics.py` |
| 12 | Add Tanimoto similarity analysis | Understand generation quality | `src/metrics/molecular_metrics.py` |
| 13 | Add scaffold analysis | Chemical diversity assessment | `src/metrics/molecular_metrics.py` |
| 14 | Systematic hyperparameter search | Find optimal config | New script |

### Phase 4: Advanced Features (2-4 weeks)

| # | Change | Expected Impact | Files |
|---|--------|----------------|-------|
| 15 | Transfer learning from QM9 | Better generalization | Training pipeline |
| 16 | Conditional generation on pIC50 | Targeted molecule design | Multiple files |
| 17 | Docking score validation | Binding prediction | New script |
| 18 | Data enrichment (ChEMBL) | More training data | `src/datasets/mpro_dataset.py` |

### Recommended Next Run Configuration

```bash
python src/main.py dataset=mpro \
    +dataset.raw_data_dir=/path/to/MPro-URV_Version2 \
    general.name=mpro_v2 \
    train.batch_size=32 \
    train.n_epochs=500 \
    train.ema_decay=0.999 \
    train.clip_grad=1.0 \
    model.lambda_train='[5,1]' \
    model.n_layers=3 \
    model.hidden_dims.dx=128 \
    model.hidden_dims.de=32 \
    model.hidden_dims.dy=32 \
    model.hidden_dims.n_head=4 \
    model.hidden_dims.dim_ffX=128 \
    model.hidden_dims.dim_ffE=64 \
    model.hidden_dims.dim_ffy=64 \
    model.hidden_mlp_dims.X=128 \
    model.hidden_mlp_dims.E=64 \
    'model.hidden_mlp_dims.y=64'
```

This configuration:
- Enables EMA for stable weights
- Enables gradient clipping for training stability
- Reduces model size by ~75% to match dataset size
- Adds edge weight to training loss
- Uses only 500 epochs (with early stopping to be added in code)
- Should produce higher validity with less overfitting
