# Feb 23 Outputs And PDBbind Pretrain Scan

Date: 2026-04-28

## Scope

This note explains the saved DiGress output directories associated with the February 23 MPro runs and scans the repository for any saved PDBbind pretraining results.

The analysis is based on the actual artifacts saved under `outputs/`, the Hydra configs in each run folder, and the available logs.

## Executive Summary

There are three relevant MPro run folders dated 2026-02-23:

1. `DiGressMProURV/outputs/2026-02-23/16-21-25-graph-tf-model/`
2. `DiGressMProURV/outputs/2026-02-23/16-24-28-test/`
3. `outputs/2026-02-23/17-14-14-test/`

Only the third one is a real long training run with usable outputs.

The PDBbind pretraining scan found four April 14 directories:

1. `outputs/2026-04-14/15-50-47-debug_pdbbind/`
2. `outputs/2026-04-14/16-01-26-debug_pdbbind/`
3. `outputs/2026-04-14/16-16-12-debug_pdbbind/`
4. `outputs/2026-04-14/16-23-20-debug_pdbbind/`

None of those four directories contains a usable pretrained model, checkpoints, saved sample images, saved chains, or metric summaries. They are startup traces or aborted debug attempts, not completed pretraining results.

## Why The Feb 23 `test` Run Does Not Contain Final Test Outputs

In this codebase revision, `src/main.py` only runs `trainer.test(...)` after training if the run name is neither `debug` nor `test`.

Relevant logic:

```python
if not cfg.general.test_only:
    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)
    if cfg.general.name not in ['debug', 'test']:
        trainer.test(model, datamodule=datamodule)
```

That means the successful Feb 23 run named `test` produced:

- training checkpoints
- validation-time sampling artifacts
- W&B logs

but it did not execute the separate final `trainer.test(...)` stage. So these are training outputs plus periodic validation samples, not the full final 10,000-sample test export described by the config defaults.

## Feb 23 Runs

### 1. `16-21-25-graph-tf-model`: config-only or very early aborted launch

Saved artifacts:

- `.hydra/config.yaml`
- empty `main.log`

What the config says:

- dataset: `mpro`
- run name: `graph-tf-model`
- batch size: `512`
- epochs: `1000`
- `dataset.raw_data_dir` is not set in the saved config

What is missing:

- no W&B run folder
- no `graphs/`
- no `chains/`
- no `checkpoints/`
- no metric logs

Interpretation:

This folder is not a completed training run. It only proves Hydra created the run directory and saved the configuration. There is no evidence that the model reached meaningful execution beyond startup.

### 2. `16-24-28-test`: interrupted MPro attempt

Saved artifacts:

- `.hydra/`
- `graphs/test/` and `chains/test/`, both empty
- one W&B run folder
- no checkpoints

Saved overrides:

- `dataset=mpro`
- `+dataset.raw_data_dir=/mnt/c/Users/xaviv/URV/.MESIIA/TFM/repo/MPro-URV_Version2`
- `general.name=test`

What happened:

- Initial validation ran once.
- W&B summary recorded only step 0 metrics.
- Then training was interrupted by `KeyboardInterrupt` during the first training epoch.

Recorded metrics before interruption:

- validation NLL: `1675.5034`
- validation atom KL: `57.03`
- validation edge KL: `960.58`

Where it stopped:

- The traceback ends inside `diffusion_utils.assert_correctly_masked(...)` during the first training step.
- The log shows a `KeyboardInterrupt`, so this was stopped manually rather than finishing normally.

Interpretation:

This was a real launch, but not a successful run. It only reached the initial validation pass. Because it was also named `test`, even a successful completion would still not have triggered the separate `trainer.test(...)` stage.

### 3. `17-14-14-test`: the real Feb 23 training run

Saved overrides:

- `dataset=mpro`
- `+dataset.raw_data_dir=/home/andromeda/Documentos/TFM_Xavi/DiGressMProURV/MPro-URV_Version2`
- `train.batch_size=32`
- `general.name=test`

This is the only Feb 23 folder that contains all the expected training artifacts:

- `.hydra/`
- `wandb/`
- `graphs/test/`
- `chains/test/`
- `checkpoints/test/`

It completed the full fit stage:

- training reached `max_epochs=1000`
- the log ends with `Trainer.fit stopped: max_epochs=1000 reached`

#### What the checkpoints mean

Saved checkpoints:

- `epoch=569.ckpt`
- `epoch=714.ckpt`
- `epoch=744.ckpt`
- `epoch=814.ckpt`
- `epoch=894.ckpt`
- `last-v1.ckpt`

Interpretation:

- the numbered checkpoint files are the best `save_top_k=5` checkpoints according to `val/epoch_NLL`
- `last-v1.ckpt` is the latest model state at the end of training
- the best validation checkpoint is `epoch=814.ckpt`

Best and last validation losses:

- best validation loss: `230.1224` at epoch `814`
- last validation loss: `288.9807` at epoch `999`

So the model kept training until epoch 1000, but its best validation point happened earlier.

#### What the `graphs/` folders mean

Inside `graphs/test/` there are folders such as:

- `epoch14_b0/`
- `epoch34_b0/`
- ...
- `epoch994_b448/`

Interpretation:

- sampling was performed periodically during validation, not only at the end
- because `check_val_every_n_epochs=5` and `sample_every_val=4`, sampling happened every 20 epochs
- that is why you see epochs `14, 34, 54, ... , 994`
- the `_b0`, `_b64`, `_b128`, ... suffixes correspond to sampling batches within the same sampling event
- those folders contain PNG visualizations of generated molecules

In practical terms, `graphs/test/` is a visual history of how generated molecules evolved during training.

#### What the `chains/` folders mean

Inside `chains/test/` there are folders such as:

- `epoch14/`
- `epoch34/`
- ...
- `epoch994/`

Each one stores diffusion chain visualizations, typically one GIF per saved chain. These files show how a sampled graph evolves during reverse diffusion from noise to molecule.

#### What the `valid_unique_molecules_*.txt` files mean

Files such as:

- `valid_unique_molecules_e814_b-1.txt`
- `valid_unique_molecules_e994_b-1.txt`

contain the valid and unique molecules saved from a sampling event.

Important caveat:

- in this Feb 23 run, those files are malformed as one long concatenated line rather than one SMILES per line
- that matches the later repo fix for the SMILES export bug
- so the content is still evidence of generated molecules, but it is not cleanly formatted for downstream use

#### How sample quality changed during training

The periodic sampling metrics show clear improvement over time.

Early training:

- epoch 14 sample: `0.00%` validity
- epoch 34 sample: `0.20%` validity
- epoch 54 sample: `3.32%` validity

Mid training:

- validity rises into the `10%` to `20%` range
- uniqueness stays near `100%`
- novelty stays at `100%`, meaning valid unique molecules were not simply copied from the training set

Late training:

- one late sample logged `27.93%` validity, `28.71%` relaxed validity, `98.64%` uniqueness, `100%` novelty
- the final W&B summary stores `val/epoch_NLL = 288.9807` and a last recorded sampling validity around `27.15%`

Interpretation:

- the model learned to generate noticeably more chemically valid molecules than at the beginning
- novelty remained excellent
- uniqueness stayed very high
- but validity is still moderate rather than strong, so the generator was improving but not yet producing highly reliable chemistry

#### Why this run matters

This folder is the meaningful Feb 23 result because it proves:

- the MPro dataset integration worked well enough to train for 1000 epochs
- the model improved steadily in generative quality
- the best checkpoint was reached before the end of training
- the saved outputs are training-time validation artifacts, not a separate final test export

## PDBbind Pretraining Scan

### What was found

The repository contains four output folders for PDBbind pretraining attempts, all dated 2026-04-14 and all configured from `+experiment=pdbbind_pretrain`.

All four share the same main debug-style overrides:

- `general.name=debug_pdbbind`
- `train.n_epochs=2`
- `general.wandb=disabled`

Their saved configs also show:

- dataset: `pdbbind`
- model reduced to 3 transformer layers
- smaller hidden dimensions than the Feb 23 MPro run
- dropout `0.2`
- EMA enabled with `0.999`
- cosine learning-rate scheduler
- `lambda_train = [5, 1]`

### Important context: the PDBbind dataset cache exists

The local processed dataset files are present in:

- `DiGressMProURV/data/pdbbind/processed/pdbbind_proc_tr_no_h.pt`
- `DiGressMProURV/data/pdbbind/processed/pdbbind_proc_val_no_h.pt`
- `DiGressMProURV/data/pdbbind/processed/pdbbind_proc_test_no_h.pt`

Also, the dataset code defaults to the local `PDBbindv2020` folder when `raw_data_dir` is null.

So `raw_data_dir: null` in these run configs does not by itself prove a dataset-path failure.

### Run-by-run interpretation

#### `15-50-47-debug_pdbbind`

Artifacts found:

- `.hydra/`
- empty `main.log`

Missing:

- no `graphs/`
- no `chains/`
- no `checkpoints/`
- no W&B artifacts

Interpretation:

This is only a Hydra run directory, not a usable pretraining result.

#### `16-01-26-debug_pdbbind`

Artifacts found:

- `.hydra/`
- `graphs/debug_pdbbind/` exists but is empty
- `chains/debug_pdbbind/` exists but is empty
- empty `main.log`

Missing:

- no checkpoints
- no PNGs
- no GIFs
- no TXT sample exports

Interpretation:

This run started far enough for folder creation, but no actual training or sampling results were saved.

#### `16-16-12-debug_pdbbind`

Artifacts found:

- `.hydra/`
- `graphs/debug_pdbbind/` exists but is empty
- `chains/debug_pdbbind/` exists but is empty
- empty `main.log`

Interpretation:

Same situation as `16-01-26`: initialization traces only, no usable pretraining output.

#### `16-23-20-debug_pdbbind`

Artifacts found:

- `.hydra/`
- `graphs/debug_pdbbind/` exists
- `chains/debug_pdbbind/` exists
- `main.log` contains only two DDP startup lines

Missing:

- no checkpoints
- no saved sample images
- no saved diffusion chains
- no saved text exports

Interpretation:

This run reached distributed initialization, but the artifact set still shows that it never produced a completed epoch or any saved pretraining result.

### Bottom line on PDBbind

The repository does contain evidence that PDBbind pretraining was attempted, but it does not contain evidence of a successful or even minimally completed pretraining run.

What is not present anywhere under `outputs/2026-04-14/`:

- `.ckpt` files
- generated molecule PNGs
- diffusion-chain GIFs
- exported valid molecule text files
- W&B summaries

Therefore:

- there is no reusable pretrained PDBbind checkpoint in the scanned output directories
- there are no saved sampling metrics to interpret
- the April 14 folders should be treated as failed or aborted debug attempts, not as actual pretraining results

## Final Interpretation

If you want to explain "the outputs from February 23", the important answer is:

- only `outputs/2026-02-23/17-14-14-test/` is the real result
- it is an MPro training run that finished 1000 epochs
- its best checkpoint is `epoch=814.ckpt`
- its graph and chain folders are periodic validation samples collected during training
- its molecule export text files are affected by the old SMILES formatting bug
- because the run name is `test`, it never executed the separate final test stage

If you want to explain "the PDBbind pretrain results", the important answer is:

- pretraining was attempted on April 14
- no completed or usable pretraining outputs were saved
- there is nothing in the scanned directories that can be interpreted as a successful pretrained model
