# VLA-Adaptation

One operator tool for the single-arm **Franka FR3 + AgileX gripper** cooking rig: the
**collection dashboard** (`http://localhost:8766`). It records SpaceMouse demonstrations and
rolls out pi0.5 checkpoints that you can pause, teleoperate and resume, saving both as LeRobot
datasets. Run every command below from this directory.

## Teleoperation (conda env `adaptive`)

Unlock the robot (`aiofranka unlock`, or Franka Desk), then start the dashboard. It takes torque
control itself; nothing else, such as gravcomp, may control the robot.

```bash
conda run --no-capture-output -n adaptive python -m real.collect_sft_data task=tabletop \
  collect.dataset_root=/home/ankile/lrds collect.repo_id=rsm_cooking_v5 \
  collect.task_prompt="Follow the displayed cooking state sequence"
```

Create `adaptive` once with `python -m real.scripts.setup_adaptive` (a clone of `vla` plus
aiofranka 0.9.0).

**H** reset · **T** setup teleop · **S** record · **G** mark goal · **U** undo goal ·
**E** save · **R** discard. Add `collect.dry_run=true` to practise without writing data.
Details: [docs/single_arm_collection.md](docs/single_arm_collection.md).

## pi0.5 rollouts with pause and resume

Add `rollout=pi05_kitchen_sft_v5` and a separate rollout dataset:

```bash
conda run --no-capture-output -n adaptive python -m real.collect_sft_data task=tabletop rollout=pi05_kitchen_sft_v5 \
  collect.dataset_root=/home/ankile/lrds collect.repo_id=rsm_cooking_rollouts_canonical \
  collect.task_prompt="Follow the displayed cooking state sequence"
```

It loads pi0.5 in an `rsm` worker process before taking torque control. **H** reset → **P** run
the policy (or **S** to start with the SpaceMouse) → within the same recording, **T** takes over
with the SpaceMouse, **P** hands back to the policy and **Space** pauses, in any order and as often
as needed → **E** save. Every frame is labelled `intervention` (0 policy, 1 human or paused). `rollout=pi05_kitchen_sft_v5` uses the kitchen SFT
checkpoint with per-edge instructions. Other checkpoints: override `rollout.checkpoint` and
`rollout.config_name`. Setup and checkpoints: [docs/bc_setup.md](docs/bc_setup.md);
keys and data format: [docs/single_arm_collection.md](docs/single_arm_collection.md#policy-rollouts-pauses-and-resumption).

## Layout

* `real/` — robot stack: `FR3RealEnv` (10-d absolute EE + gripper targets at 50 Hz), driving
  `real/robot.py` (aiofranka's native 1 kHz loop, in process) with the joint-limit nullspace
  law in `real/nullspace_law.py`; RealSense cameras, SpaceMouse, AgileX gripper and camera arm.
* `real/collection/` — the dashboard: control loop (`runtime.py`), policy driver (`policy.py`),
  LeRobot recorder, HTTP server and UI.
* `real/policies/`, `real/deployment/{worker,models}.py` — pi0.5 (OpenPI) inference in a
  separate worker process; `bc/proposer.py` is the policy interface it implements.
* `dagger/` — fine-tunes pi0.5 on the SpaceMouse corrections in a rollout dataset
  ([dagger/README.md](dagger/README.md)).
* `real/scripts/` — `setup_adaptive` (creates `adaptive`), `setup_bc` (installs `rsm` +
  checkpoint), `check_collection_dataset`,
  `push_dataset_to_hub`.

## Tests

```bash
conda run -n adaptive python -m pytest -q $(ls tests/test_*.py | grep -v deployment)   # dashboard + robot stack
conda run -n rsm python -m pytest -q tests/test_deployment_*.py                    # model worker
```

## Cluster training experiments

See [DAgger and union training](dagger_training/README.md) for the pinned
DAgger_1 baseline and union comparison, Slurm launch instructions, offline
comparison results, and Hugging Face checkpoints.
