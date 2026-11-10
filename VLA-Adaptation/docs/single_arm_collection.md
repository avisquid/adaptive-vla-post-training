The collector supports one FR3, one AgileX gripper, and one SpaceMouse. The browser shows reference photos for the active route, accepts goal markers, and saves whole routes as LeRobot episodes with directed-edge annotations. Finish the current route's quota before advancing to the next.

Use the `adaptive` conda environment: a clone of `vla` with aiofranka 0.9.0 and Numba, created once with `python -m real.scripts.setup_adaptive`. The writer is tested with `lerobot==0.3.3` and `datasets==3.6.0`, pinned in `requirements-collection.txt`. The latter was corrected in this machine's environment after finding an incompatible 4.x installation.

**Reset poses** for every state are already captured in `real/config/task/*.yaml`. The
pose-capture tool (gravcomp + `collection_gui --mode poses`) was removed with `Leftovers/`;
recover it from Git history (commit `5a76cf8`) if a task needs new poses.

From your workstation, forward the dashboard port:

```bash
ssh -N -L 8766:127.0.0.1:8766 ankile@improbable014
```

**Unlock the robot** with `aiofranka unlock` (or in Franka Desk). Nothing else (such as gravcomp) may control the robot: the collector takes torque control itself. Start it on the robot computer:

```bash
conda run --no-capture-output -n adaptive python -m real.collect_sft_data \
  collect.dataset_root=/home/ankile/lrds \
  collect.repo_id=tabletop_v1 \
  collect.task_prompt="Follow the tabletop route"
```

Visit **http://localhost:8766** on your workstation. Dataset collection checks all state poses and reference images before connecting devices. Hardware dry runs allow uncaptured poses for setup teleop; an explicit reset still requires a valid target. The arm holds its measured pose when the collector takes torque control; it moves only on an explicit reset.

| Key | Operator action |
| --- | --- |
| **H** | During setup, move the arm to the current route's first state's qpos. Clear the movement path first; this joint move is not a collision planner. |
| **T** | Toggle unrecorded setup teleoperation to arrange the starting pose. |
| **S** | Start one full-route recording; requires a completed reset. |
| **G** | Mark arrival at the highlighted next state. Works while stationary. Every edge must contain at least one taken transition. |
| **U** | Undo the most recent arrival, including a mistaken final arrival. |
| **E** | After the last goal, preserve the staged episode, reset to the next attempt's first state, then finalize the dataset in the background. Clear the arm's path before pressing E. |
| **R** | Discard and reset to the same route's start after confirming the dialog. Clear the movement paths first. The draft stays on disk. |

## Policy rollouts, pauses and resumption

Launch the same dashboard with a rollout preset to let pi0.5 drive. You can pause it at any
time, continue with the SpaceMouse, and hand back to the policy, as often as you like. Rollouts need their **own dataset**: every frame carries an `intervention` label
(0 = policy, 1 = human), so the collector refuses to mix rollouts into a teleop dataset such as
`rsm_cooking_v5`, and vice versa.

```bash
conda run --no-capture-output -n adaptive python -m real.collect_sft_data task=tabletop rollout=pi05_kitchen_sft_v5 \
  collect.dataset_root=/home/ankile/lrds collect.repo_id=rsm_cooking_rollouts_canonical \
  collect.task_prompt="Follow the displayed cooking state sequence"
```

The collector loads pi0.5 in a worker process that uses the `rsm` environment (about a minute,
including JAX compilation), and only then takes torque control and connects the cameras.

| Key | Operator action |
| --- | --- |
| **P** | After **H**, start a recorded rollout: the policy drives from the reset pose. The top-right pill shows the decision count and planning time. **S** instead starts the same kind of recording with you driving. |
| **Space** | Pause immediately, whether the policy or you are driving: queued policy actions are dropped, an in-flight plan is ignored, and the arm holds its current pose with the last gripper command. |
| **T** (recording) | Drive with the SpaceMouse from the current pose, taking over from the policy (its queued actions are dropped at once) or from a pause. |
| **P** (recording) | Hand the arm to the policy, from the SpaceMouse or a pause: it plans afresh from the current observation under the prompt now selected, holding the pose until the plan arrives. |

**G**, **U**, **E** and **R** work as in teleop collection; mark goals while the policy drives.

**Policy prompt** (dropdown under the robot controls) chooses the instruction. *Auto* follows the
route edge (the next edge's prompt after each **G**); choosing an edge prompt or *Default*
(`rollout.prompt`) pins it, including across attempts and on edges without a trained prompt.
It can change at any time, including mid-rollout: the change applies from the next plan, while
already queued actions (at most `rollout.execute_k`) finish. Each rollout dataset frame stores
the prompt that generated its action as its LeRobot `task` (so `task_index` changes mid-episode);
human and paused frames store the prompt in effect. Teleop datasets keep
`collect.task_prompt`. Prompt changes are logged as `prompt` events, and each decision in the
`rollout.decision_log` records its `prompt` and `prompt_choice`. The worker echoes the prompt
each plan was conditioned on (`model_prompt`, shown in the top-right pill as *model:*, with
*next plan:* while a change is pending); a mismatch with the requested prompt stops the rollout.
**S** still records a pure teleop demonstration, labelled `intervention=1` throughout.
Each plan is sent to the worker without blocking the control loop; the arm holds its last target
while waiting, then executes `rollout.execute_k` actions (10 by default) unchanged before
replanning. A plan slower than `rollout.plan_timeout_s` (2 s), a worker error, a joint-limit stop,
or closing the page interrupts the attempt and holds the arm.

Frames recorded while you drive or while paused are labelled `intervention=1`. A pause records
no frames unless the arm moves. Episode metadata records `collection_mode: rollout`, `handoffs`
(each `controller` change to `paused`, `human` or `policy`, with its `frame` and `time_s`),
`takeover_frame` and `takeover_time_s` (the first pause), and a `rollout` block (checkpoint, `config_name`, `execute_k`, per-decision seeds and planning times across all resumptions,
worker versions and GPU). Other checkpoints: override `rollout.checkpoint` and
`rollout.config_name`, e.g.
`rollout.checkpoint=$HOME/.local/share/rsm-real/checkpoints/pi05_cooking_ft50_best/pi05_cooking_rise_s0_ft50_137500 rollout.config_name=pi05_cooking_rise_ft50_s0`.
Dry runs (`collect.dry_run=true`) never time out between the dashboard and the policy: model
load, warmup, reset and each plan wait indefinitely, and a lost page heartbeat does not interrupt
teleop or the policy, so either process can sit at a debugger breakpoint. On hardware this means
closing the page does not stop the arm; use Space or the robot's stop. Recorded runs keep
`rollout.plan_timeout_s` and the heartbeat.
`collect.simulate=true collect.dry_run=true` exercises the whole flow with the real model and no robot.

## Resets

Each H reset and automatic reset on E or discard samples independent joint offsets using the task's
`qpos_noise_deg: 2.0`: `nominal + deg2rad(2.0) * uniform(-1, 1, size=7)`.
Use `qpos_noise_deg: 0` for exact nominal resets, or a seven-number list for
per-joint half-ranges. Near a limit, the uniform interval is restricted to the
existing 0.05 rad reset buffer. The saved nominal poses are unchanged; each
episode's reset metadata includes nominal joints, sampled target, actual offset,
configured ranges and measured post-reset joints. Noise is sampled once per
reset, never on G or each control tick. It changes the EE starting
pose and is not collision planning. Collection uses the shared
`real.reset_sampling.sample_reset_qpos` helper; other evaluation/deployment
callers can use it explicitly before their exact-qpos reset.

The task also sets `agilex_qpos` (six radians) and `agilex_can: can_piper` for
the arm holding the global camera. Each H/automatic reset holds the FR3,
positions the camera mount, then resets the FR3 and refreshes observations.
Camera positioning uses the controller ported from MDPO: the same joint
commands, speed setting and one-second settling wait. It reads the joints back
and refuses to continue until the arm is within 1 degree of its target (10 s
limit), recording the measured pose in the episode metadata. Enabling the arm
times out after five seconds instead of hanging.

The camera arm ignores commands while its controller is still in teaching mode,
which persists after hand-guiding even when the teach button is off. The reset
then stops with "still in teaching mode". Hold the camera arm (its joints can
briefly lose torque) and run:

```bash
conda run --no-capture-output -n adaptive python -m real.agilex reset
```

It prints the control mode, which should be CAN control, and holds the current
pose. Then press H again.
The camera target receives no FR3 qpos noise and is saved in task/episode
metadata. `task.agilex_qpos=null` disables camera-arm control; simulation never
connects to it. Shutdown disconnects CAN without moving to a parking pose.
The global image stream remains controlled separately by `task.perception.global`
(currently commented out). Clear both arms' paths before H, E, or confirming discard.

Either SpaceMouse button toggles the gripper. Starting recording preserves its
commanded grip; every reset opens it. E first seals the staged frames and goal metadata,
then resets the arm to the next attempt's first state with fresh qpos noise.
Parquet creation and video encoding start in the background after the reset.
Arrange the physical scene while this finishes; S and setup teleop unlock after
the save succeeds. The quota advances only on successful save. This repeats the
current route until its quota is met, then uses the next route. There is no reset
after the final quota. Recording stays off: arrange the physical scene and press
S when ready. The robot reset does not restore the objects. Discard automatically
resets to the same route's start; G does not reset the arm. Set
`collect.auto_reset_after_save=false` to require H after saving instead.
If staging fails, the arm does not reset. If finalization fails, the arm may
already have reset; staged data is preserved and the collector stops without
advancing the quota. If the reset fails, the completed episode is still saved
and counted, then the collector stops and reports the reset failure. Shutdown
waits for a running save. Camera previews pause while resetting/saving.
Quotas include requested routes ending in failure states 5 and 6.
Task outcome is separate from acceptance; unspecified outcomes remain `unspecified`.

Videos use H.264 with `collect.video_encoding.preset=veryfast`, `crf=30`,
`gop=2`, and `threads=2`. All video frames are encoded directly from the staged
NPZ files; only the PNGs sampled by LeRobot for raw-image statistics are written.
CRF and GOP retain the previous settings. Each episode records its encoding
settings and save timings. `preset=ultrafast` is available for further speed at
the cost of larger files; changing these settings does not require a new dataset.

Translation and rotation use the original SpaceMouse sensitivity. The collector
drives the arm with aiofranka 0.9's native 1 kHz loop, in its own process: the
loop runs in C++ on a real-time thread pinned to `robot.realtime_cpu` (31, an
isolated core), so nothing else in the collector can delay a torque command.
Its OSC adds a bounded joint-limit avoidance torque in the dynamically
consistent nullspace (`real/nullspace_law.py`, a Numba port of aiofranka's
native OSC). It increases smoothly within 0.60 rad of a physical joint
limit, with the projected addition capped at 1.5 Nm per joint. The Cartesian
gains, existing posture spring, and torque-rate limiter are preserved. Continuous
target scaling and acceleration projection remain removed. To use the stock
controller, launch the collector with `robot.nullspace_avoidance.enabled=false`.
The first start compiles the law (a few seconds) while the arm holds its pose.
Override `collect.spacemouse_translation_scale=0.5` for half translation sensitivity.

Nullspace avoidance uses redundant arm motion to reduce proximity to limits
while preserving the Cartesian task in the model. It cannot guarantee clearance
for every requested gripper motion, especially near singularities or when
several joints compete for the available redundancy. The addition has passed
offline controller parity, projection, torque bound, and startup ownership
tests. A supervised state-2 reset and 37-second hardware test with 3 mm / 1 degree
excursions passed without a robot fault or limit pause. Minimum clearance stayed
above 0.300 rad and ended at 0.311 rad; joint 7 stayed near 0.49 rad. This local
test does not prove prevention of limits during full-speed teleoperation or
isolate improvement over the stock controller. See the
[hardware report](../.collection-diagnostics/nullspace-hardware-test-report.md).

The command boundary now checks measured joint positions and a 0.25-second
constant-velocity prediction. If either enters the existing 0.05 rad joint
buffer, it requests the measured end-effector pose, pauses collection, and
shows the joint number, velocity toward the limit, and predicted clearance in
the browser. The draft is preserved; the rejected
input is not recorded. Use **R** to discard, reset, and return to setup.
Commands otherwise pass through unchanged. This check runs at the command
rate and cannot guarantee prevention of hardware faults or instantaneous
stopping. Saved task poses have been updated using IK to preserve their modeled
end-effector poses with at least 0.30 rad of joint clearance. State 0 passed a
bounded hardware test with the previous stock OSC. State 2 subsequently passed
the bounded test with avoidance enabled. The other state resets and full-speed
teleoperation remain unverified.

Keep the browser page open and focused for shortcuts. One browser tab owns collection. Losing its heartbeat for three seconds during teleoperation interrupts the attempt and requests a hold at the measured arm pose. Device errors preserve the draft and require restarting. The GUI is an operator interface, not an emergency stop; use the physical robot stop when needed.

To practice on real hardware without writing data:

```bash
conda run --no-capture-output -n adaptive python -m real.collect_sft_data collect.dry_run=true
```

In the browser, press **T** to enable SpaceMouse control from the current pose and **T** again to disable it. This works before state qpos values have been captured. To practice the full route workflow, first capture the start state's qpos, then use H → S → G at each goal → E. Dry runs write no dataset and do not advance quotas. The SpaceMouse, configured cameras and CAN gripper must be connected to the robot computer.

For a hardware-free browser simulation:

```bash
conda run --no-capture-output -n adaptive python -m real.collect_sft_data \
  collect.simulate=true collect.dry_run=true
```

In simulation, **M** generates a short movement for exercising each goal boundary. Camera views are labeled synthetic. A dry run does not increment quotas. Simulation with a dataset writer is supported for tests and its metadata is marked `simulation: true`; it cannot append to a hardware dataset.

**What is saved.** The example command creates `/home/ankile/lrds/tabletop_v1`. Each accepted full route is one episode. Observations and actions have 10 values: XYZ, the first two rotation-matrix columns, and gripper width. Camera images or videos use the configured global and wrist streams.

The original measured-state selection rule is preserved exactly: `taken = not np.allclose(previous_measured, current_measured, atol=1e-4)`, including NumPy's default relative tolerance. Previous measured state updates on every control tick. Stationary ticks produce no training rows. LeRobot's nominal timestamps are therefore retained-row index / FPS; real elapsed times and original control ticks are separate columns.

| Location | Contents |
| --- | --- |
| Dataset columns `annotation.edge_from`, `annotation.edge_to` | Directed edge being traversed, such as `(0, 1)` |
| `annotation.route_index`, `annotation.segment_index` | Full route and its current segment |
| `annotation.goal_reached`, `annotation.reached_state_id` | Arrival on the final retained row of each segment; `-1` when no arrival |
| `next.reward` | Float32 sparse reward: 1 on a goal row arriving at a state in `task.reward_states`; 0 on every other row |
| `timing.capture_elapsed_s`, `timing.control_tick_index` | Original timing before stationary ticks were omitted |
| `meta/collection/task.json` and `states/` beside it | Frozen state labels, qpos, routes, outcomes and portable reference photos with hashes |
| `meta/collection/episodes/<uuid>.json` | Goal receipt/application times, correction history, `[start, end)` segment boundaries, reset target/measurement and outcome |
| `meta/collection/arrivals/<uuid>/` | Arrival observations captured on G, including stationary arrivals, outside the training timeline |

Keep complete episodes together when splitting train/validation data, then derive per-edge views if desired. Accepted counts resume from committed episode metadata. Changes to the task definition, including qpos, require a new dataset ID. A bounded background queue stages frames under `/home/ankile/lrds/.tabletop_v1.collection-staging/`; staged frames remain after save/discard for debugging and consume disk space. Remove old staging only after checking the saved dataset. If a commit is interrupted, `pending_commit.json` blocks automatic append; inspect the dataset and staged episode before recovery. Automatic crash repair is not implemented.

The task config currently sets `reward_states: [4]`. For route 2 → 3 → 4,
the last row of 2 → 3 has reward 0 and the last row of 3 → 4 has reward 1;
all non-goal rows have reward 0. State IDs refer to the task's `states` list,
not positions within a route. Rewarded intermediate subgoals also receive 1.
Rewards are assigned during commit after goal corrections/undo, and the rule
and state IDs are frozen in `meta/collection/task.json`. This is operator-labeled
arrival, not automatic visual verification. Existing datasets without the reward
column are not rewritten; use a new dataset ID when changing the reward schema
or the configured reward states.

Each new dataset gets a descriptive Hugging Face `README.md`, generated from
its frozen task, camera schema, simulation flag and recording settings. It
documents the state/route tables, reset noise, pose conventions, sparse rewards,
timing, goal snapshots, loading examples and episode-level splitting guidance.
Counts are read from `meta/info.json` so the card does not become stale during
collection. Existing user-edited cards are not overwritten on resume.

After saving finishes, check annotation consistency without connecting to hardware:

```bash
conda run --no-capture-output -n adaptive python -m real.scripts.check_collection_dataset \
  /home/ankile/lrds/rsm_cooking_ik
```

To exercise the full collection/writer/reader path without hardware:

```bash
conda run --no-capture-output -n adaptive python -m real.scripts.smoke_collection_dataset
```

This creates a separate, clearly marked simulated dataset under
`.collection-diagnostics/lerobot-smoke-<timestamp>`, with one episode per route.
It checks stationary-tick omission, goal rewards, video decoding through LeRobot,
reset metadata, and resume/append. It never connects to either robot. The output
includes `README.md` and `smoke-test-result.json`; synthetic images do not validate
physical camera alignment or goal achievement.

The checker prints each episode's route, segment row ranges, goal rows, state
labels, goal rewards and last action per edge. It checks metadata against saved Parquet rows,
finite 10D state/action vectors, timestamps, and arrival snapshots. It does not
establish that the camera image visually matches the annotated goal, or verify
the `taken` predicate from omitted ticks. For that, inspect the saved camera
stream and the `meta/collection/arrivals/<uuid>/arrival-XX.npz` snapshots. The
goal flag belongs to the last retained row before G; the arrival snapshot is a
separate observation captured when the goal event is processed.

The resulting dataset is ready for Hugging Face publication. Upload explicitly after stopping collection:

```bash
conda run --no-capture-output -n adaptive python -m real.scripts.push_dataset_to_hub \
  --root /home/ankile/lrds/tabletop_v1 --repo-id YOUR_HF_USERNAME/tabletop_v1 --private
```

Authentication is required. Publication is separate from collection, and includes the `meta/collection` annotations and reference assets. LeRobot 0.3.3 commits episodes with `save_episode()`; it does not require the newer `finalize()` API.

Validation covers the pose recorder, browser/backend flows, empty/stationary edges, quota progression, reset request validation, and actual LeRobot image/video create/save/reopen/append round trips. The hardware control path still needs a supervised pilot with the robot, gripper, SpaceMouse, and cameras.
