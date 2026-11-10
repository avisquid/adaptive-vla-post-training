# Repository layout

- `VLA-Adaptation/` is the maintained project: one operator tool, the collection
  dashboard (`real.collect_sft_data`), for SpaceMouse demonstrations and pi0.5
  rollouts with key-press takeover, plus DAgger fine-tuning on its corrections in
  `dagger/`. It is the Python project root: run commands, tests and
  `pip install -e .` from there. All paths below are relative to it.

# Deployment environment

- Run the collection dashboard, which also controls the robot, in the `adaptive`
  conda environment: a clone of `vla` (Python 3.10, LeRobot 0.3.3) plus
  `aiofranka==0.9.0` and Numba. Create it with `python -m real.scripts.setup_adaptive`.
  Preserve the existing `vla` environment.
- Use the `rsm` conda environment (Python 3.11.13) for stock pi0.5 BC inference in the
  model worker.
- Keep `aiofranka==0.9.0`, `pylibfranka==0.21.3`, and `mujoco==3.12.0` pinned: the
  native loop is built for exactly that pylibfranka. Do not upgrade robot drivers
  while changing model dependencies.
- Reproduce the `rsm` environment with `python -m real.scripts.setup_bc` from
  `VLA-Adaptation/`. It creates `rsm`, installs the project editable, and downloads the
  pinned stock BC checkpoint. `--no-checkpoint` installs software only.
- The project is pip installable through `pyproject.toml`. Use its `hardware`
  extra for robot dependencies. Exact deployment dependencies and native build
  constraints are in `requirements/bc-inference.txt` and `requirements/bc-build.txt`.
- OpenPI is an external editable checkout at
  `~/.local/share/rsm-real/openpi`, pinned to `215abfb217dbac7d5f1273282331b9b1866c0479`.
  Setup obtains Richard's cooking changes from the pinned Git revision recorded
  in the installer, verifies their hash, and applies `requirements/openpi-real.patch`.
  This last patch only makes ALOHA simulation an optional dependency; model code
  and trained transforms stay unchanged. Do not install OpenPI's simulation extra
  in `rsm`: it requires MuJoCo <3, which conflicts with the robot driver.
- Keep checkpoints and external sources out of this repository. Setup records
  their paths, source revisions, and dependency hashes in
  `~/.local/share/rsm-real/installation.json`.
- LeRobot's version in `rsm` matches the OpenPI inference source. Data collection
  keeps LeRobot 0.3.3 (in `adaptive`); do not replace that collection stack.

# Validation and launch

- Read `docs/bc_setup.md` and `docs/single_arm_collection.md` for installation,
  launch commands, and operator controls.
- The dashboard is one process with a synchronous control loop. It owns the arm
  through `real.robot.FR3`: `aiofranka.Controller` runs the 1 kHz torque loop in C++
  on its own real-time thread (`robot.realtime_cpu`) and its watcher on an event loop
  in a thread of its own, so the collector's blocking calls, threads and GIL cannot
  delay a torque command. There is no separate OSC server. Joint-limit avoidance is
  the Numba control law in `real/nullspace_law.py`, a port of aiofranka's native OSC.
- The dashboard runs in `adaptive`; with `rollout=<preset>` it starts the model in a
  `real.deployment.worker.ModelWorker` subprocess using `rollout.inference.python`
  (the `rsm` Python). JAX/XLA settings live in `rollout.inference.environment` and
  apply only to the worker. Keep them in configuration, not shell exports.
- Rollout startup order: load/warm the model, then take torque control
  (`FR3.start`), then connect the other devices. Preserve it.
  Never block the control loop on inference; takeover must drop queued actions at once.
- Rollouts write only to datasets with the per-frame `intervention` feature; never
  mix them into teleop datasets such as `rsm_cooking_v5`.
- Automated checks must use `collect.simulate=true collect.dry_run=true`.
  Dry-run alone uses real hardware.
- Check model/device imports without constructing robot or camera controllers.
  Keep real robot movement under the operator's deployment command.
- Run the relevant tests after changing the control loop, policy driver, worker
  communication, or recording. A mock rollout does not validate a trained
  checkpoint; acceptance also requires a simulated dashboard rollout with the real
  checkpoint through the actual worker.
  The test runner is `pytest==9.0.3`; the main checks are
  `python -m pytest -q tests/test_deployment_*.py` in `rsm`. Collection tests
  (the other `tests/test_*.py`) run in `adaptive`.
- Rollouts execute one policy chunk per decision; there is no candidate scoring.
  Preserve the shared task/camera configuration, configurable execution prefix
  (`rollout.execute_k`), and recorded checkpoint provenance in episode metadata.
- Execute the policy chunk prefix directly. Translation and
  rotation bounds default to null; do not add implicit clipping or smoothing to
  deployment actions. Keep driver constraints and joint-limit stops separate.
