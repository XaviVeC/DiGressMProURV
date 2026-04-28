# outputsAndromeda: PDBbind Pretrain and MPro Fine-Tune Artifact Review

## Executive Summary

- `outputsAndromeda` does contain real PDBbind pretraining artifacts. The two valuable pretrain runs are:
  - `outputsAndromeda/2026-04-14/16-34-12-pdbbind_pretrain_20260414_163410`
  - `outputsAndromeda/2026-04-21/16-20-53-pdbbind_pretrain_smallbs`
- The best overall PDBbind run in this folder is the small-batch run from `2026-04-21`. It has better validity and more usable generated SMILES than the `2026-04-14` run.
- The MPro fine-tune artifacts are real too, but the useful outputs are not stored in the top-level fine-tune launch folders. They were written inside the resumed pretrain folder `outputsAndromeda/2026-04-14/16-34-12-pdbbind_pretrain_20260414_163410/` under the subfolders `checkpoints/`, `graphs/`, and `chains/`.
- The most reusable DiGress-generated molecule files are the `valid_unique_molecules_*.txt` files in the `graphs/` folders. These are already valid and deduplicated.
- The `final_smiles.txt` files are also useful, but they contain `None` lines and duplicates, so they need filtering before downstream use.
- The `generated_samples*.txt` files are not SMILES files. They are raw graph dumps and are not suitable as notebook input.

## What Is Actually Valuable

### 1. Best PDBbind Pretrain Run

Run:

- `outputsAndromeda/2026-04-21/16-20-53-pdbbind_pretrain_smallbs`

Why this is the best pretrain artifact:

- Dataset: `pdbbind`
- Batch size: `8`
- Model preset: upgraded 3-layer discrete GraphTransformer with dropout `0.2`
- Checkpoints available:
  - `checkpoints/pdbbind_pretrain_smallbs/last.ckpt`
  - `checkpoints/pdbbind_pretrain_smallbs/epoch=19.ckpt`
  - `checkpoints/pdbbind_pretrain_smallbs/epoch=44.ckpt`
  - `checkpoints/pdbbind_pretrain_smallbs/epoch=49.ckpt`
  - `checkpoints/pdbbind_pretrain_smallbs/epoch=59.ckpt`
  - `checkpoints/pdbbind_pretrain_smallbs/epoch=74.ckpt`

W&B metrics worth keeping:

- Validation summary:
  - `val/epoch_NLL = 356.3183`
  - `Validity = 0.2891`
  - `Relaxed Validity = 0.3086`
  - `Uniqueness = 1.0`
  - `Novelty = 1.0`
- Test summary:
  - `test/epoch_NLL = 348.5024`
  - `Validity = 0.3070`
  - `Relaxed Validity = 0.3201`
  - `Uniqueness = 0.9938`
  - `Novelty = 0.9994`

Most valuable molecule outputs:

- `final_smiles.txt`
  - 10,000 total sampled entries
  - 3,070 valid SMILES
  - 3,050 unique valid SMILES
- `graphs/pdbbind_pretrain_smallbs/valid_unique_molecules_e75_b16.txt`
  - 3,181 lines
  - already valid and unique
- Earlier filtered snapshots also exist:
  - `valid_unique_molecules_e14_b-1.txt` with 106 molecules
  - `valid_unique_molecules_e34_b-1.txt` with 135 molecules
  - `valid_unique_molecules_e54_b-1.txt` with 182 molecules
  - `valid_unique_molecules_e74_b-1.txt` with 158 molecules

Recommendation:

- If you want the best checkpoint to resume from, use `checkpoints/pdbbind_pretrain_smallbs/last.ckpt`.
- If you want the cleanest ready-to-use generated molecules, start from `graphs/pdbbind_pretrain_smallbs/valid_unique_molecules_e75_b16.txt`.
- If you want the full 10k sample pool, use `final_smiles.txt`, but filter out `None` and deduplicate.

### 2. Earlier PDBbind Pretrain Run

Run:

- `outputsAndromeda/2026-04-14/16-34-12-pdbbind_pretrain_20260414_163410`

What it contains:

- Dataset: `pdbbind`
- Batch size: `32`
- Checkpoints available for the pretrain subrun:
  - `checkpoints/pdbbind_pretrain_20260414_163410/epoch=34.ckpt`
  - `checkpoints/pdbbind_pretrain_20260414_163410/epoch=44.ckpt`
  - `checkpoints/pdbbind_pretrain_20260414_163410/epoch=54.ckpt`
  - `checkpoints/pdbbind_pretrain_20260414_163410/epoch=69.ckpt`
  - `checkpoints/pdbbind_pretrain_20260414_163410/epoch=74.ckpt`
- This pretrain subrun does not contain a `last.ckpt` file.

W&B metrics worth keeping:

- Validation summary:
  - `val/epoch_NLL = 350.0269`
  - `Validity = 0.2715`
  - `Relaxed Validity = 0.2871`
  - `Uniqueness = 1.0`
  - `Novelty = 1.0`
- Test summary:
  - `test/epoch_NLL = 357.0888`
  - `Validity = 0.2689`
  - `Relaxed Validity = 0.2828`
  - `Uniqueness = 0.9972`
  - `Novelty = 0.9993`

Most valuable molecule outputs:

- `final_smiles.txt`
  - 10,000 total sampled entries
  - 1,460 valid SMILES
  - 1,454 unique valid SMILES
- `graphs/pdbbind_pretrain_20260414_163410/valid_unique_molecules_e75_b16.txt`
  - 2,820 valid unique molecules
- Additional filtered snapshots:
  - `valid_unique_molecules_e14_b-1.txt` with 75 molecules
  - `valid_unique_molecules_e34_b-1.txt` with 105 molecules
  - `valid_unique_molecules_e54_b-1.txt` with 132 molecules
  - `valid_unique_molecules_e74_b-1.txt` with 147 molecules

Interpretation:

- This is a real and useful pretrain run.
- It is weaker than the `2026-04-21` small-batch run in terms of usable final SMILES yield.
- It is still valuable as a backup source of checkpoints and generated molecules.

### 3. Real MPro Fine-Tune Outputs

Important storage detail:

- The top-level fine-tune launch folders under `outputsAndromeda/2026-04-15`, `2026-04-17`, and `2026-04-21` mostly contain `.hydra` plus a tiny `main.log`.
- The real fine-tune outputs were written into the resumed pretrain folder:
  - `outputsAndromeda/2026-04-14/16-34-12-pdbbind_pretrain_20260414_163410/`

This folder contains three meaningful fine-tune subruns.

#### 3.1 `mpro_finetune_20260414_163410_resume`

- Checkpoint:
  - `checkpoints/mpro_finetune_20260414_163410_resume/epoch=74.ckpt`
- Generated valid unique molecules:
  - `graphs/mpro_finetune_20260414_163410_resume/valid_unique_molecules_e75_b1.txt`
  - 1,802 valid unique molecules

#### 3.2 `finetune_resume`

- Checkpoint:
  - `checkpoints/finetune_resume/epoch=74.ckpt`
- Generated valid unique molecules:
  - `graphs/finetune_resume/valid_unique_molecules_e75_b1.txt`
  - 1,802 valid unique molecules

Important note:

- `graphs/finetune_resume/valid_unique_molecules_e75_b1.txt` and
  `graphs/mpro_finetune_20260414_163410_resume/valid_unique_molecules_e75_b1.txt`
  are identical files in content.
- Their intersection is 1,802 molecules and the Jaccard similarity is 1.0.
- In practice, this means they are duplicates and you only need to keep one of them.

#### 3.3 `mpro_finetune_smallbs_resume`

- Checkpoints:
  - `checkpoints/mpro_finetune_smallbs_resume/epoch=79.ckpt`
  - `checkpoints/mpro_finetune_smallbs_resume/epoch=79-v1.ckpt`
  - `checkpoints/mpro_finetune_smallbs_resume/last.ckpt`
  - `checkpoints/mpro_finetune_smallbs_resume/last-v1.ckpt`
- Generated valid unique molecules:
  - `graphs/mpro_finetune_smallbs_resume/valid_unique_molecules_e80_b1.txt`
  - 1,780 valid unique molecules

W&B metrics worth keeping for this small-batch fine-tune:

- Validation summary:
  - `val/epoch_NLL = 558.2033`
- Test summary:
  - `test/epoch_NLL = 575.5796`
  - `Validity = 0.1460`
  - `Relaxed Validity = 0.1787`
  - `Uniqueness = 0.9961`
  - `Novelty = 1.0`

Interpretation:

- This fine-tune lineage is real because it produced checkpoints, graph folders, and a filtered valid-unique SMILES file.
- But its logged validity is much worse than the PDBbind pretrains.
- The generated set is also almost completely different from the earlier 1,802-molecule fine-tune set:
  - intersection with the 1,802-molecule set is only 7 molecules
  - Jaccard similarity is about `0.002`

Practical recommendation:

- If you want MPro-targeted generated molecules, the safest starting point is one of the duplicated 1,802-molecule files.
- Use the small-batch fine-tune outputs only if you specifically want an alternative MPro-biased set and are comfortable with the lower validity statistics.

## Folders That Are Mostly Noise, Launch Stubs, or Debug Traces

These folders do not contain meaningful output artifacts by themselves:

- `outputsAndromeda/2026-04-14/16-29-49-debugpretrain`
  - short debug run
  - no checkpoints
  - no SMILES outputs
- `outputsAndromeda/2026-04-15/16-58-26-mpro_finetune_20260414_163410`
- `outputsAndromeda/2026-04-17/14-34-26-finetune`
- `outputsAndromeda/2026-04-21/16-18-49-finetune`
- `outputsAndromeda/2026-04-21/16-19-40-mpro_finetune_smallbs`
- `outputsAndromeda/2026-04-21/16-30-39-mpro_finetune_smallbs`
- `outputsAndromeda/2026-04-21/16-34-29-mpro_finetune_smallbs`
- `outputsAndromeda/2026-04-21/16-34-42-mpro_finetune_smallbs`

Why these are not the main artifacts:

- They contain only `.hydra` and a minimal `main.log`.
- They do not contain their own checkpoint, graph, chain, or final SMILES outputs.
- Several of them are clearly failed or malformed launch attempts. Examples:
  - `16-18-49-finetune` points to `last.ckptaa`
  - `16-34-29-mpro_finetune_smallbs` has `general.wandb: onlineasfasf`

Also low value:

- `outputsAndromeda/2026-03-09/17-00-26-mpro_upgraded_1`
  - effectively empty in the current workspace snapshot

## Which Files Matter for Molecules

### Best Ready-to-Use SMILES Files

If the goal is to consume DiGress-generated molecules downstream, these are the best files:

1. `outputsAndromeda/2026-04-21/16-20-53-pdbbind_pretrain_smallbs/graphs/pdbbind_pretrain_smallbs/valid_unique_molecules_e75_b16.txt`
   - 3,181 valid unique molecules
   - cleanest general-purpose output in this folder

2. `outputsAndromeda/2026-04-14/16-34-12-pdbbind_pretrain_20260414_163410/graphs/pdbbind_pretrain_20260414_163410/valid_unique_molecules_e75_b16.txt`
   - 2,820 valid unique molecules
   - useful backup set from the earlier pretrain

3. `outputsAndromeda/2026-04-14/16-34-12-pdbbind_pretrain_20260414_163410/graphs/mpro_finetune_20260414_163410_resume/valid_unique_molecules_e75_b1.txt`
   - 1,802 valid unique molecules
   - MPro-oriented fine-tune output

4. `outputsAndromeda/2026-04-14/16-34-12-pdbbind_pretrain_20260414_163410/graphs/mpro_finetune_smallbs_resume/valid_unique_molecules_e80_b1.txt`
   - 1,780 valid unique molecules
   - alternative MPro-oriented fine-tune output

### `final_smiles.txt` Files

These files are real and useful, but they are rawer:

- `outputsAndromeda/2026-04-21/16-20-53-pdbbind_pretrain_smallbs/final_smiles.txt`
  - 10,000 lines
  - 3,070 valid
  - 3,050 unique valid
- `outputsAndromeda/2026-04-14/16-34-12-pdbbind_pretrain_20260414_163410/final_smiles.txt`
  - 10,000 lines
  - 1,460 valid
  - 1,454 unique valid

Meaning:

- `final_smiles.txt` includes invalid generations represented as `None`.
- It may also include duplicates.
- These files are useful if you want the full sample pool, but they need cleanup before direct downstream use.

### `generated_samples*.txt`

These files are not valid downstream molecule lists.

- They store graph-structured raw outputs (`N=`, `X:`, `E:` blocks).
- They are useful only if you want to inspect the raw DiGress graph samples.
- They are not appropriate as SMILES input for notebooks.

## Can These Molecules Be Used in the GINE Notebooks?

Yes, but with one important caveat.

What exists:

- There are valid DiGress-generated SMILES files in `outputsAndromeda`.
- The cleanest ones are the `valid_unique_molecules_*.txt` files listed above.

What the notebook expects:

- The notebook `notebooks/mpro_gine_regression/mpro_digress_gine_pic50_regression.ipynb` is built around a pseudo-label CSV, not a bare SMILES text file.
- It expects columns named `SMILES` and `pIC50`.

So the practical answer is:

- These DiGress SMILES files can be used as molecular inputs for the notebook pipeline.
- They are not immediately usable as supervised GINE training data by themselves, because they do not contain `pIC50` labels.
- To use them in the DiGress-augmented GINE notebook, you first need to assign pseudo-labels or predicted `pIC50` values and convert them into a CSV with at least:
  - `SMILES`
  - `pIC50`

Best choices for that workflow:

- If you want broad PDBbind-derived diversity, start from:
  - `outputsAndromeda/2026-04-21/16-20-53-pdbbind_pretrain_smallbs/graphs/pdbbind_pretrain_smallbs/valid_unique_molecules_e75_b16.txt`
- If you want MPro-closer molecules, start from:
  - `outputsAndromeda/2026-04-14/16-34-12-pdbbind_pretrain_20260414_163410/graphs/mpro_finetune_20260414_163410_resume/valid_unique_molecules_e75_b1.txt`
  - or the equivalent duplicate file under `graphs/finetune_resume/`

## Bottom Line

- Yes, there are real PDBbind pretrain checkpoints in `outputsAndromeda`.
- Yes, there are real MPro fine-tune outputs too, but the meaningful ones live inside the `2026-04-14/16-34-12-pdbbind_pretrain_20260414_163410/` folder.
- The highest-value artifacts are:
  - pretrain checkpoints
  - fine-tune checkpoints under the resumed pretrain folder
  - `valid_unique_molecules_*.txt`
  - `final_smiles.txt` after filtering
- The single best ready-to-use generated molecule file in this tree is:
  - `outputsAndromeda/2026-04-21/16-20-53-pdbbind_pretrain_smallbs/graphs/pdbbind_pretrain_smallbs/valid_unique_molecules_e75_b16.txt`
- The best MPro-targeted generated molecule file is:
  - `outputsAndromeda/2026-04-14/16-34-12-pdbbind_pretrain_20260414_163410/graphs/mpro_finetune_20260414_163410_resume/valid_unique_molecules_e75_b1.txt`