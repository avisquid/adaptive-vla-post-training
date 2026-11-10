"""Descriptive dataset cards generated from the frozen collection task."""


def dataset_card(repo_id, task, fps, camera_shapes, videos, simulation):
    def cell(value):
        return str(value).replace('|', '\\|').replace('\n', ' ')

    state_rows = '\n'.join(
        f'| {s["id"]} | {cell(s["label"])} | {s["terminal"]} | {cell(s["outcome"])} | '
        f'{s["id"] in task.get("reward_states", [])} |'
        for s in task['states'])
    route_rows = '\n'.join(
        f'| {r["id"]} | {" → ".join(map(str, r["states"]))} | {r["quota"]} |'
        for r in task['routes'])
    camera_rows = '\n'.join(
        f'| `observation.images.{cell(name)}` | {list(shape)} | '
        f'{"video" if videos else "image"}; RGB |' for name, shape in camera_shapes.items())
    return f'''---
tags:
  - robotics
  - lerobot
  - imitation-learning
  - reinforcement-learning
---
# {cell(repo_id)}

Single-arm FR3 / AgileX-gripper demonstrations for **{cell(task['name'])}**.
Collection mode: **{'SIMULATED — not physical robot demonstrations' if simulation else 'physical robot with SpaceMouse teleoperation'}**.
Each accepted episode traverses one ordered route through the states below.
The operator marks subgoal arrival with G and accepts the completed route with E.
An accepted episode may intentionally end in a failure state; acceptance means
the requested route was recorded, not that the task succeeded.

This card describes the collection schema. Read `meta/info.json` for current
episode/frame counts and feature definitions; route quotas below are targets,
not claims about how many demonstrations have been collected.

## Task states and routes

State IDs are indexes in the task's states list, not positions within a route.
The frozen source of truth is [meta/collection/task.json](meta/collection/task.json),
including seven-joint reset qpos values in radians, portable reference-image
paths, image hashes, task fingerprint, and reward definition. Reference images
under `meta/collection/states/` illustrate goals, rather than recorded observations.

| State ID | Description | Terminal in task | Outcome | Rewarded arrival |
| --- | --- | --- | --- | --- |
{state_rows}

| Route ID | Ordered state IDs | Target accepted episodes |
| --- | --- | ---: |
{route_rows}

Camera mount: the separate six-joint AgileX arm has configured target
`{task.get('agilex_qpos')}` radians on `{cell(task.get('agilex_can', 'can_piper'))}`
(`None` disables positioning). It is positioned before FR3 resets, with no
FR3 reset noise applied to the camera target. The episode reset metadata's
`camera_arm` record includes the commanded target and whether the run was
simulated. The ported MDPO controller waits one second after sending the target;
`arrival_verified: false` means camera-arm arrival was not verified by feedback.
Camera-arm joints are not part of the 10D policy action/observation. Configuring
the mount does not enable a camera stream; recorded streams are listed below.

## Observations and actions

Explicit state resets randomize the nominal seven-joint qpos with independent
uniform offsets. Configured half-ranges in degrees are `{task.get('qpos_noise_deg', [2.0] * 7)}`.
Away from limits, the formula is `nominal + radians(range) * uniform(-1, 1)`
per joint. The sampling interval is intersected with the existing 0.05 rad reset
joint-limit buffer; it is not Gaussian noise or post-sample clipping. A zero
range disables noise for that joint. This changes the starting end-effector pose;
it does not preserve the nominal Cartesian pose or perform collision planning.
The episode metadata's `reset` record stores `nominal`, `target`, `noise_rad`,
`qpos_noise_deg`, and measured post-reset joints. Its `trigger` distinguishes
`manual`, `after_discard`, and automatic `before_save` resets (older episodes may use
`after_save`). Every reset opens the gripper. The collector seals the preceding episode's staged frames before
resetting to the next attempt's first state, then finalizes that episode in the
background. Reset motion is outside recordings; S explicitly starts the next
episode after saving succeeds. G markers never trigger resets or perturbations during a route.
Nominal state qposes remain unchanged.

Video episodes can include `encoding` settings and `save_timing_s` in their
episode metadata. The collector defaults to H.264 `veryfast`, CRF 30, GOP 2,
and two encoder threads. Settings can differ by episode; older records may
omit them. Videos use all retained frames, and image statistics use LeRobot's
sampling of original, uncompressed frames. Encoding runs after recording ends.

| Column | Shape | Meaning |
| --- | --- | --- |
| `observation.state` | [10] | Measured absolute end-effector pose and gripper width |
| `action` | [10] | Absolute target accepted by the arm controller, plus commanded gripper width |
{camera_rows}

The 10D vector is `[x, y, z, R00, R10, R20, R01, R11, R21, gripper_width]`.
XYZ and gripper width are meters. Rotation is the first two rotation-matrix
columns, concatenated column by column; it is not Euler angles or a quaternion.
Pose is expressed in the robot base frame. Reconstruct the third column with
the cross product after orthonormalizing the first two if needed.
Actions are absolute references, not SpaceMouse deltas, torques, measured joint
angles, or proof the commanded pose was achieved. The joint reset qpos values
in task metadata are distinct from the 10D observations.

A row pairs the cached observation obtained before dispatch with the command
accepted on that control tick. The following observation is obtained after
dispatch. A limit-stop command is omitted and the attempt is interrupted.
There is no explicit `next.observation` column, and adjacent retained rows need
not be adjacent control ticks. Images and state are sampled sequentially;
per-camera timestamps are not currently saved in the training rows.

## Edges, goals, and rewards

| Column | Meaning |
| --- | --- |
| `annotation.route_index` | Configured route ID |
| `annotation.segment_index` | Zero-based edge position within the route |
| `annotation.edge_from`, `annotation.edge_to` | Directed edge's source/destination state IDs |
| `annotation.goal_reached` | 1 on the last retained row before each accepted G marker, otherwise 0 |
| `annotation.reached_state_id` | Destination state ID on that row, otherwise -1 |
| `next.reward` | Float32 sparse transition reward, described below |

**Reward state IDs: `{task.get('reward_states', [])}`.** The exact rule is:

```python
reward = float(goal_reached == 1 and reached_state_id in reward_states)
```

Every other row receives 0, including arrivals at unlisted subgoals. The rule
also applies to rewarded intermediate subgoals. Rewards and edge annotations
are assigned on save using the final goal boundaries, after any undo/correction.
Arrival is operator-labeled, not automatically verified from images.

G works while stationary: no extra training row is added. Its marker/reward
attaches to the last retained row, which may precede the actual arrival image.
The separately captured arrival observation is saved as
`meta/collection/arrivals/<episode_uuid>/arrival-XX.npz` (one-based goal number),
with `state` and `images.<camera_name>` arrays. Inspect these snapshots to review
visual goal achievement. Goal receipt/application times are in episode metadata.

The writer does not add `next.done` or `next.truncated`. Do not infer task
termination from positive reward, zero reward, or every goal marker. Episode
boundaries delimit collected routes, which can end at nonterminal task states.
If a learning pipeline needs done masks, define them explicitly using its
task/edge formulation and the frozen task metadata.

## Timing and frame selection

Nominal FPS is **{fps}**. Only ticks where consecutive measured 10D states differ
under `not np.allclose(previous, current, atol=1e-4)` are retained during recording
(NumPy's default relative tolerance also applies). The measurement comparison
occurs before dispatch of that tick's action. It does not certify that the
newly saved action itself caused motion. Gripper movement also counts.

- `timestamp` is retained-row index / FPS, so omitted pauses compress playback.
- `timing.capture_elapsed_s` is the observation capture's elapsed monotonic time
  since recording started; use it for physical elapsed-time analysis.
- `timing.control_tick_index` is the collector tick counter, not an episode-local
  contiguous row index. Gaps represent omitted ticks.

The omitted ticks are not available to recompute the original selection rule.
Account for irregular elapsed time when deriving velocities or temporal discounts.

## Files and loading

- `data/`: LeRobot Parquet episode tables. Resolve paths using `meta/info.json`.
- `videos/`: camera streams when video storage is enabled; image mode embeds
  image data in the dataset instead.
- `meta/tasks.jsonl`: task text mapped from each row's `task_index`.
- `meta/collection/task.json`: frozen states, routes, rewards and task fingerprint.
- `meta/collection/episodes/<uuid>.json`: LeRobot episode index, reset record,
  goal events, accepted/outcome flags, and segments with `[start, end)` row bounds.
- `meta/collection/arrivals/`: separately captured goal observations.

The writer uses LeRobot 0.3.3 and datasets 3.6.0. To load observations/videos:

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(repo_id={repo_id!r}, root="/path/to/local/dataset",
                        download_videos=False, video_backend="pyav")
row = dataset[0]
print(row["action"], row["annotation.edge_to"], row["next.reward"])
```

To inspect labels without decoding videos:

```python
import json
from pathlib import Path
import pyarrow.parquet as pq

root = Path("/path/to/local/dataset")
info = json.loads((root / "meta/info.json").read_text())
i = 0  # episode index
path = root / info["data_path"].format(
    episode_chunk=i // info["chunks_size"], episode_index=i)
rows = pq.read_table(path, columns=["frame_index", "annotation.edge_from",
    "annotation.edge_to", "annotation.goal_reached", "next.reward"]).to_pandas()
print(rows)
```

In the RSM repository, run the read-only structural checker after saving finishes:

```bash
python -m real.scripts.check_collection_dataset /path/to/local/dataset
```

It checks annotations, rewards, episode/segment boundaries, timestamps and
state/action shapes. It cannot verify visual goal achievement or camera alignment.
Split train/validation/test by complete episodes before deriving per-edge samples
to avoid leaking neighboring frames. Preserve state IDs and consult this dataset's
frozen metadata rather than a later edited task config. Do not silently combine
datasets with different reward rules, pose conventions or simulation status.
'''
