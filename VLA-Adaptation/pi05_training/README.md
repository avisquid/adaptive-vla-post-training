# pi0.5 cooking training

Trains pi0.5 from `pi05_base` on the RSM cooking demonstrations with the recipe of
sunshine's `pi05-kitchen-sft-v5` **h50/canonical** run, with one deliberate change:
`discrete_state_input=True`. OpenPI is upstream at a pinned revision with no patches.
Everything task-specific is in `cooking.py`, and `train.py` runs OpenPI's own
`scripts/train.py` with that configuration.

## Setup

From `VLA-Adaptation/`, on the training machine:

```bash
pi05_training/setup.sh                                # upstream OpenPI 215abfb + its uv env
PY=pi05_training/openpi/.venv/bin/python
$PY pi05_training/prepare.py                          # pinned dataset, ~0.9 GB
JAX_PLATFORMS=cpu $PY pi05_training/train.py --exp-name check --preflight
```

`setup.sh` clones OpenPI to `pi05_training/openpi` (set `OPENPI_DIR` to put it
elsewhere) and refuses to continue if that checkout has local changes. The dataset
goes to `$HF_LEROBOT_HOME` (by default `~/.cache/huggingface/lerobot`). The base
weights and tokenizer are downloaded from `gs://openpi-assets` into
`$OPENPI_DATA_HOME` on first use.

The preflight runs on CPU and checks the following:
- the 163,850 training frames
- segment clamping
- the canonical prompts with the discretized state
- a multi-worker batch
- the optimizer's encoder/rest split

## Train

```bash
$PY pi05_training/train.py --exp-name h50_canonical_dstate
```

The defaults are the full recipe on one 8-GPU node (FSDP over 8). Checkpoints go to
`pi05_training/checkpoints/pi05_cooking_h50_canonical_dstate/<exp-name>/<step>/`, or to
`--checkpoint-dir` (use it for cluster scratch). Pass `--resume` to continue a run and
`--no-wandb` to turn off logging. For fewer GPUs, set `--fsdp-devices` and keep
`--batch-size` divisible by the GPU count.

## Data

The data is
[`xinyunsunshine/rsm_cooking_v5_language`](https://huggingface.co/datasets/xinyunsunshine/rsm_cooking_v5_language)
@ `b92545b`, the exact dataset sunshine's runs used. It contains byte-identical
episodes of `jmarangola/rsm_cooking_v5` @ `ea7caa6` (248 episodes, 50 Hz) and adds:

- **Seven task labels.** `task_index` maps each frame to its task's canonical
  instruction, such as "Place the tomato in the pan." No paraphrases are used.
- **The SFT split** in `meta/language_split.json`: 223 training episodes (163,850
  frames) and 25 validation episodes. Training uses only the 223.
- **Segment boundaries** in `annotation.language_remaining`. A 50-step action chunk
  that would run into the next task repeats the current segment's last action
  instead.

## Recipe

| Setting | Value | Source |
|---|---|---|
| Model | pi0.5, `action_horizon=50`, actions padded 10 → 32 | sunshine's checkpoint card |
| State input | **discrete prompt tokens** (`discrete_state_input=True`) | the change; sunshine used `False` |
| Initialization | `gs://openpi-assets/checkpoints/pi05_base/params` | recipe |
| Batch, steps, seed | 256, 30,000, 0 | `training_config.json` |
| GPUs | 8, FSDP 8 | `training_config.json` |
| Data workers | 16 | `training_config.json` |
| Checkpoints | save every 250, keep every 1,000 | `training_config.json` |
| EMA | 0.999 (checkpoints hold EMA weights) | checkpoint card, recipe |
| Learning rate | cosine: 1,000 warmup, peak 5e-5, decays to 5e-6 at 30k | recipe |
| Optimizer | AdamW, global clip 1.0; SigLIP encoder at 0.2 × lr | recipe |
| Image augmentation | `rise_real`: 1 % crop, colour jitter with hue 0, both cameras, no rotation | recipe |
| Normalization | quantile, `assets/local/rsm_cooking_v5_language/norm_stats.json` | copied from sunshine's checkpoint |

Sources:
- **training_config.json** is sunshine's `h50/canonical/training_config.json`.
- **recipe** is zwhong's `pi05_cooking_stock` OpenPI config, which sunshine's run used.
  Its evaluation metric names (`xyz_l2_first_cm`, `train_eval`) come from that code.

`training_config.json` does not record the learning rate, augmentation, or encoder
scaling. They are taken from that config, where the cosine schedule already spans
30,000 steps; there it ran at batch 32. `--lr` overrides the peak.

**Discrete state.** pi0.5 has no continuous state input. With
`discrete_state_input=True`, each of the 10 normalized state values is binned into
256 levels and written into the prompt, for example
`Task: Place the tomato in the pan on the stove., State: 95 110 … 254;\nAction: `.
The prompt is about 60 tokens, well under the limit of 200. A policy trained this way
must be served with the same setting.

## Not included

There are no in-training validation metrics: upstream's training loop has none.
Sunshine selected step 22,000 by validation XYZ error, measured on the 25 held-out
episodes at the retained checkpoints.
