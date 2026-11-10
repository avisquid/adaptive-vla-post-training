# pi0.5 setup and rollouts

pi0.5 inference runs in the **`rsm` conda environment**. The collection dashboard
(robot control, SpaceMouse, recording) runs in `adaptive` (aiofranka 0.9.0; see
[single_arm_collection.md](single_arm_collection.md)) and starts the `rsm` model
worker itself. The worker does not import aiofranka.

## Installation

From this checkout, with Conda and Git on PATH:

```bash
python -m real.scripts.setup_bc
conda activate rsm
python -m pip check
```

The installer creates Python 3.11.13, installs this project editable, installs
the pinned OpenPI source and dependencies, and downloads the approximately 12 GB
stock checkpoint. Git access to this repository is needed to retrieve the exact
cooking-source handoff. Re-running setup reuses the checkout and downloaded files.
Use `--no-checkpoint` to install software without downloading weights.

Sources and checkpoints live under `~/.local/share/rsm-real`; override that base
with `--root`. `installation.json` records the source revisions, dependency hashes,
and checkpoint location. Environment pins are in
[`requirements/bc-inference.txt`](../requirements/bc-inference.txt); native build
tool pins are in [`requirements/bc-build.txt`](../requirements/bc-build.txt).
The installer uses uv as a fast pip installer inside Conda, without creating
another virtual environment.

The OpenPI source is pinned to `215abfb217dbac7d5f1273282331b9b1866c0479`, with
Richard's cooking handoff from `a373756093472e145072bb9e3cfef3151022e0d8`.
[`openpi-real.patch`](../requirements/openpi-real.patch) only moves `gym-aloha`
to an optional simulation extra. This keeps `rsm` on the robot's MuJoCo 3.12.0;
ALOHA simulation requires MuJoCo below 3. The model code,
normalization, and cooking transforms are unchanged by this packaging adjustment.
Do not install the OpenPI simulation extra or run its original `uv sync` in `rsm`.

`rsm` is the model environment. Keep running the dashboard in `adaptive`, whose
LeRobot version differs from the pinned OpenPI training source.

## Roll out pi0.5 from the dashboard

Rollouts that the operator can pause and resume run from the collection dashboard; see
[Policy rollouts, pauses and resumption](single_arm_collection.md#policy-rollouts-pauses-and-resumption)
for the keys and dataset format. The preset
[`real/config/rollout/pi05_stock.yaml`](../real/config/rollout/pi05_stock.yaml)
selects the installed stock checkpoint (`pi05_cooking_stock`, 10-action chunks),
`execute_k: 10`, and the worker settings: the `rsm` Python,
`JAX_PLATFORMS=cuda`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8` and the
`--xla_gpu_enable_triton_gemm=false` workaround below. These apply only to the
worker subprocess; no shell exports are needed. When setup used a custom `--root`,
override `rollout.checkpoint`.

Check real checkpoint inference with no robot, cameras or recording:

```bash
cd /home/ankile/rsm_real/VLA-Adaptation
conda run --no-capture-output -n adaptive python -m real.collect_sft_data task=tabletop \
  rollout=pi05_stock collect.simulate=true collect.dry_run=true
```

Open `http://localhost:8766`, press **H**, then **P**: the pill reports decisions and
planning time (about 0.1 s per plan on the RTX 4090). **Space** takes over.

On the robot, unlock it (`aiofranka unlock`) and drop `collect.simulate=true`; keep
`collect.dry_run=true` for an unrecorded first trial. The dashboard loads and warms the
model, then takes torque control and connects the gripper and cameras. Ctrl+C gives
torque control back, closes the worker and frees the GPU.

Policy actions are executed exactly as predicted: the first `rollout.execute_k`
actions of each chunk go to `env.step` with no clipping or smoothing (at 50 Hz,
ten actions take 0.2 s plus inference). Robot-driver constraints and the
controller's joint-limit stop still apply, and a limit stop interrupts the
attempt. The policy sees only the task's `global` and `wrist` cameras.

## Kitchen SFT checkpoint (per-edge prompts)

`rollout=pi05_kitchen_sft_v5` uses
[`xinyunsunshine/pi05-kitchen-sft-v5`](https://huggingface.co/xinyunsunshine/pi05-kitchen-sft-v5/tree/main/h50/canonical)
`h50/canonical` step 22000, the latest published step: pi0.5 fine-tuned on
`rsm_cooking_v5` with 50-action chunks and one canonical instruction per route edge.
It lives in the git-ignored `VLA-Adaptation/checkpoints/` folder. From `VLA-Adaptation/`,
download only the inference files (12 GB):

```bash
conda run -n rsm python -c "from huggingface_hub import snapshot_download; snapshot_download('xinyunsunshine/pi05-kitchen-sft-v5', allow_patterns=['h50/canonical/checkpoints/22000/**', 'h50/canonical/*.json', 'h50/canonical/README.md'], local_dir='checkpoints/pi05-kitchen-sft-v5')"
C=checkpoints/pi05-kitchen-sft-v5/h50/canonical/checkpoints/22000
mkdir -p $C/assets/jmarangola && ln -sfn ../local/rsm_cooking_v5_language $C/assets/jmarangola/rsm_cooking_v5
```

The link lets the installed `pi05_cooking_stock_ft50_s0` config (`pi05_cooking_stock`
with `action_horizon=50`: same architecture and transforms) find its normalization
stats. The dashboard sends the canonical instruction for the current edge from
`language_prompts.json` (e.g. 0→5 "Place the tomato on the stove.") and switches it
when **G** marks a goal. Routes containing an edge it was not trained on cannot be
rolled out. The preset executes all 50 actions of each chunk
(`rollout.execute_k=50`, 1 s at 50 Hz) before replanning. `rollout.extra_prompts` adds
named prompts to the dashboard's **Policy prompt** menu: `full_task` and
`full_task_stepwise` chain the edge instructions into one prompt for the whole task
(0 → 3 → 4). The model was not trained on them, and a pinned prompt stays in effect
until another one is selected.

## Other pi0.5 checkpoints: 50-action fine-tunes

Selected deployment information from [PR #12 at `702e874`](https://github.com/jmarangola/rsm_real/blob/702e8744687b3301fe4469c51cbf4f16c020019c/rise_stack/CHECKPOINTS.md).
These fine-tunes predict **50 actions (1 second at 50 Hz)**. Their parent checkpoints
predict 10 actions; for example, `pi05_cooking_stock/stock_s0/29999` is the parent
of `stock_s0_ft50`, not the fine-tuned checkpoint itself. At this source revision
the fine-tunes were available on the training cluster and had not been mirrored
to Hugging Face.

| Run directory | OpenPI config | W&B |
|---|---|---|
| `stock_s0_ft50` | `pi05_cooking_stock_ft50_s0` | [4sls4xv4](https://wandb.ai/richardrl/rsm_rise/runs/4sls4xv4) |
| `stock_s1_ft50` | `pi05_cooking_stock_ft50_s1` | [sa309a54](https://wandb.ai/richardrl/rsm_rise/runs/sa309a54) |
| `rise_s0_ft50` | `pi05_cooking_rise_ft50_s0` | [ffc3e5o0](https://wandb.ai/richardrl/rsm_rise/runs/ffc3e5o0) |
| `rise_s1_ft50` | `pi05_cooking_rise_ft50_s1` | [k22pmyen](https://wandb.ai/richardrl/rsm_rise/runs/k22pmyen) |
| `riseadv_s0_ft50` | `pi05_cooking_riseadv_ft50_s0` | [3ipijvt3](https://wandb.ai/richardrl/rsm_rise/runs/3ipijvt3) |
| `sn1_knn0p03_ft50` | `pi05_cooking_sn1_knn0p03_ft50_s0` | [8zxd9bti](https://wandb.ai/richardrl/rsm_rise/runs/8zxd9bti) |
| `sn0p1_knn0p03_ft50` | `pi05_cooking_sn0p1_knn0p03_ft50_s0` | [0cjoeuwz](https://wandb.ai/richardrl/rsm_rise/runs/0cjoeuwz) |

Cluster checkpoint layout:
`/proj/dmfexp/zwhong/rsm_rise/runs/openpi/checkpoints/<config>/<run>/<step>/`.
Keep the same relative layout locally under
`~/.local/share/rsm-real/checkpoints/<config>/<run>/<step>/`.
For stock seed 0 at step 50,000 this is
`~/.local/share/rsm-real/checkpoints/pi05_cooking_stock_ft50_s0/stock_s0_ft50/50000/`.

Copy `params/`, `assets/`, and checkpoint metadata; optimizer `train_state/` is
unnecessary for inference. Preserve the checkpoint's
`assets/jmarangola/rsm_cooking_v5/norm_stats.json` (published MD5
`da20b66d4beb1dae800603b6745623c3`). These fine-tunes reuse the parent's per-frame
statistics; the from-base `_h50` control uses different statistics.

The seven configuration definitions come directly from
[PR #13 at `fbf76ec`](https://github.com/jmarangola/rsm_real/blob/fbf76ec93c8ec55d3538384a37b18989bbaa83ab/rise_stack/openpi/rise_on_pi05.patch)
through [`requirements/openpi-ft50.patch`](../requirements/openpi-ft50.patch).
`python -m real.scripts.setup_bc` applies that additive patch and records its hash.
Roll one out from the dashboard by overriding `rollout.checkpoint` and `rollout.config_name` (for example `rollout.config_name=pi05_cooking_stock_ft50_s0`); `rollout.execute_k` chooses how many of the 50 actions run before replanning.

The complete checkpoint catalog, including RISE-recipe actors and scorer bundles, was
`Leftovers/rise_stack/CHECKPOINTS.md`; it is in Git history (`git show 5a76cf8:Leftovers/rise_stack/CHECKPOINTS.md`).

## Native warmup errors

An `Inference worker failed during plan` message with an exit signal means the
subprocess died; read the native error above it. The observed warmup failure was
`ptxas exited with non-zero error code 139` during XLA GEMM autotuning, using
JAX 0.5.3 and the environment's CUDA 12.9.41 compiler. This is a compiler crash;
the following pipe EOF was only its consequence. It does not establish an OOM.

Several subsequent operator launches passed warmup after the two JAX settings
above were exported, but the same compiler crash recurred at 18:01:19 on
2026-09-23. The stock preset now supplies these settings automatically; they
are not a demonstrated fix for the intermittent compiler crash. No XLA compiler
flags or installed packages were changed. Keep the robot collector stopped during fresh model checks,
and run only one stock BC worker on this GPU at a time.

On 2026-09-25 the transferred
`pi05_cooking_ft50_best/pi05_cooking_stock_s0_ft50_143000` checkpoint loaded
successfully but reproduced the `ptxas` exit-139 crash during warmup. A retry with
the worker-only setting `XLA_FLAGS=--xla_gpu_enable_triton_gemm=false` passed both
warmup calls and a hardware-free rollout. The `pi05_stock` rollout preset now sets it
by default (`rollout.inference.environment.XLA_FLAGS`).

This [XLA compiler option](https://openxla.org/xla/flags_guidance) disables
Triton-based GEMM compilation. It changes kernel selection and may affect speed;
it does not replace weights or normalization assets. The check produced finite
`[1, 50, 10]` action chunks; it does not establish real-world policy performance.
The older local `pi05_cooking_stock_ft50_s0/stock_s0_ft50/50000` copy was truncated
and requires transfer repair independently of this compiler setting.

## Validation on this machine

On 2026-10-17, after moving robot control in process (aiofranka 0.9.0, `adaptive`),
the dashboard loaded the stock checkpoint in the `rsm` worker and ran a simulated
rollout: 41 decisions and 408 policy frames without error, then takeover, a goal
marker and a clean Ctrl+C exit. No physical robot was driven by this check.

On 2026-10-17 the dashboard (`vla`) started the stock checkpoint in an `rsm` worker,
warmed it, and ran a simulated rollout: about 0.095 s per plan, 10 actions executed
per decision, and takeover stopped policy actions on the next control tick.

The notes below are from installation, when rollouts used the terminal `real.deploy`
runner (since removed).

The installed stock checkpoint produced finite `[1, 10, 10]` action chunks on
the RTX 4090. Two subsequent proposal calls took 88 and 84 ms after the first
JAX compilation, using synthetic observations; these are not real-world episode
performance measurements. The complete five-second simulated deployment command
above passed using `rsm` for both parent and worker, with no evaluation files.
Robot, OSC, camera, and CAN module imports passed. No physical robot rollout was
performed during installation. Hardware startup subsequently loaded BC, connected
the gripper and both task cameras, and exited cleanly at the operator prompt with
reset disabled. The restarted OSC controller remained running with fresh state
updates; no reset or policy trial was executed during that check.

`python -m pip check` passes in `rsm`. The installer was rerun successfully,
including checkpoint retrieval from the local download. The existing `vla`
aiofranka 0.4.0 installation was verified unchanged by hashing its installed files.
The deployment regression suite passed with **97 tests and 2 skips** in `rsm`;
36 additional unittest cases covering recording and mocked robot control passed.
The 27 runtime tests also passed under Python 3.10 in `vla` and Python 3.11 in `rsm`.
