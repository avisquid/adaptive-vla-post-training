# DAgger fine-tuning and union comparison

These scripts reproduce the executed pi0.5 cooking experiments: DAgger-only
fine-tuning and a fixed-compute comparison against the union of DAgger and the
original language-annotated demonstrations. They are supervised updates on the
recorded actions, not online rollout collection or an entire DAgger algorithm.

## Dependencies

Set `OPENPI_ROOT` to the **patched OpenPI cooking checkout** and `OPENPI_PYTHON`
to its Python environment. The experiments used upstream revision
`215abfb217dbac7d5f1273282331b9b1866c0479` plus local cooking/training extensions:
`pi05_cooking_stock_ft50_s0`, `LeRobotCookingDataConfig`, `cooking_policy`,
episode-subset fixes, validation helpers, encoder LR scaling, and image
augmentation. Checking out the upstream revision alone is **not sufficient**.
This transfer contains the experiment drivers, not a replacement for that
patched dependency. The original source lives in
`/proj/dmfexp/zwhong/rsm_rise/third_party/openpi` on Blue Vela.

The training environment must provide that checkout and its dependencies,
including JAX/CUDA, PyTorch, LeRobot v2.1, PyAV, PyArrow, and NumPy.
`OPENPI_DATA_HOME` optionally selects the tokenizer/assets cache. It defaults to
`dagger_training/openpi-assets`; use the existing cache on the cluster.
The download/publish commands need `huggingface-hub>=1.33,<2`. Credentials come
from the user's HF login/environment; none are embedded in this package.

The tested training dependency files are fingerprinted in
`openpi-training-source.json`. On the existing cluster, select them with:

```bash
export OPENPI_ROOT=/proj/dmfexp/zwhong/rsm_rise/third_party/openpi
export OPENPI_PYTHON="$OPENPI_ROOT/.venv/bin/python"
export OPENPI_DATA_HOME=/proj/dmfexp/zwhong/rsm_rise/.openpi_assets
```

## Prepare inputs

From `VLA-Adaptation/` (the Python project root):

```bash
python dagger_training/prepare.py --with-baseline
"$OPENPI_PYTHON" dagger_training/union-comparison/audit_union.py
```

The source manifests pin both datasets and the starting checkpoint:

- `avisquid/DAgger_1`: `03ea0096d806adcfcede64be40ff2e10a6479cf5`
- `xinyunsunshine/rsm_cooking_v5_language`: `b92545b777ee572d65382ee080b07b7c2ebf2f1c`
- `xinyunsunshine/pi05-kitchen-sft-v5`: `1fff07dbbd46b3ce9395ba15e9019a93b3402c5e`,
  subdirectory `h50/canonical/checkpoints/22000`

`--with-baseline` downloads the completed DAgger-only checkpoint to
`dagger_training/dagger-1/export`. Omit it if retraining that baseline.
`DAGGER_CHECKPOINT` can override its path for comparison evaluation.

## Validate before submission

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu "$OPENPI_PYTHON" dagger_training/dagger-1/train_dagger.py --preflight
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu "$OPENPI_PYTHON" dagger_training/union-comparison/train_union.py --preflight
```

Both checks exercise the real eight-worker loader. The union check also verifies
that a language action chunk repeats its final segment action rather than
crossing a task boundary. Both entry points have a multiprocessing main guard.
Training loaders use a 180-second timeout to fail instead of hanging on workers.

## Train on Slurm

```bash
sbatch --partition=gpu-mid --account=grp_runtime --qos=grp_runtime_medium \
  dagger_training/slurm.sh dagger
sbatch --partition=gpu-mid --account=grp_runtime --qos=grp_runtime_medium \
  dagger_training/slurm.sh dagger-eval
sbatch --partition=gpu-mid --account=grp_runtime --qos=grp_runtime_medium \
  dagger_training/slurm.sh union
```

Run `dagger-eval` after the baseline is downloaded or training completes.
Submit each mode once; do not run duplicate jobs against the same mode's output.
Each mode uses four GPUs on one node. The trainers resume from their persistent
checkpoint directory after a scheduler requeue. They save every 100 steps and
keep the latest optimizer state locally. The recipe uses 1,000 updates, batch
32, seed 0, peak LR 1e-5, 100-step warmup, and the parent's normalization assets.
Horizon is 50; absolute 10-dimensional actions are padded to 32; global and wrist
cameras are used. The observer camera is excluded, matching the starting policy.

`dagger-1/train_dagger.py` exports inference weights to `dagger-1/export` (or
`WEB_TRAIN_OUTPUT`). `union-comparison/train_union.py` writes to `exports/union`
and `exports/dagger_eval` under its directory. Both trainers record live
`progress.json`. Training logs and `eval_history.jsonl` are the authoritative
progress and validation records for both modes.

## Comparison protocol

The union is a concatenated loader over the two pinned source datasets; it does
not rewrite episode/task IDs. It contains 193,506 training frames: 29,656 DAgger
frames from 28 episodes and 163,850 language frames from 223 episodes. Sampling
is uniform over frames. All recorded DAgger actions are supervised, including
policy actions; intervention-only sampling was not used in either experiment.

Four DAgger episodes (2, 11, 21, 25) and the language dataset's published 25
validation episodes are excluded from training. Evaluation uses identical fixed
128-chunk subsets per source. Canonical prompts are retained, and language
chunks are clamped to annotated task boundaries. `union-manifest.json` records
source membership, episode hashes, and split assignments. The audit rejects
nonfinite state/action values, duplicate complete episodes, and inconsistent
segment boundaries.

Both fine-tunes use the same base checkpoint and optimizer budget. Union sees
fewer DAgger examples (~15.3% of samples); this is not an exposure-matched study.
`comparison.json` contains the measured results:

| Policy | DAgger XYZ chunk (cm) | Language XYZ chunk (cm) | DAgger loss | Language loss |
|---|---:|---:|---:|---:|
| Starting checkpoint | 3.499 | 2.214 | 0.046609 | 0.027208 |
| DAgger-only | 1.524 | 2.333 | 0.008076 | 0.018998 |
| Union | 1.688 | 2.209 | 0.009769 | 0.021761 |

Single-seed offline diagnostics do not establish robot rollout success.

## Upload and existing artifacts

Upload a new run under a new destination prefix; do not overwrite these published
results:

```bash
hf buckets sync dagger_training/union-comparison/exports/union \
  hf://buckets/rsm-bucket/rsm-shared/YOUR_NEW_RUN/checkpoint
```

- DAgger-only: `hf://buckets/rsm-bucket/rsm-shared/dagger_1_pi05_h50/step_1000`
- Union + report: `hf://buckets/rsm-bucket/rsm-shared/dagger1_union_comparison/20261105_1000steps_s0`

Keep every Orbax `params/` file and the matching `assets/` directory together.
These are EMA inference exports, not Transformers `from_pretrained` models.

## Integration and validation

This workflow lives alongside `dagger/train.py`. That trainer selects only full
intervention chunks; this experiment intentionally trains on all recorded actions
and holds out episodes for the matched comparison. The collection and inference
environments are unchanged. Run training in a separate OpenPI training environment.

The cluster checkout named above supplies the verified training extensions.
`real.scripts.setup_bc` configures inference; it is not by itself a guarantee of
training compatibility. Both preflight commands must pass before submitting jobs
with another OpenPI checkout. Data and checkpoints remain outside version control.

Submit from `VLA-Adaptation/`, or set `DAGGER_TRAINING_DIR` to the absolute path of
this directory. Slurm runs a spool copy of the launcher, so it resolves the source
via `SLURM_SUBMIT_DIR` rather than the spool location.

Verified in this checkout against the pinned local datasets and fingerprinted
OpenPI dependency: DAgger preflight (29,656 frames, eight-worker batch of shape
`(4, 50, 32)`); union preflight (29,656 DAgger + 163,850 language training frames,
eight workers per source, segment-tail repetition); both existing
`tests/test_dagger_frames.py` tests; Slurm source-path dispatch; and progress-hook
initialization. These are integration checks, not a new GPU training run. The
reported model results come from the completed runs linked above.
