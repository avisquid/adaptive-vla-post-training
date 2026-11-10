"""Read-only validation of saved collection annotations; no robot connection."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def require(condition, message):
    if not condition:
        raise ValueError(message)


def task_without_reset_poses(task):
    """Compare annotation semantics while retaining historical reset provenance."""
    task = deepcopy(task)
    task.pop('fingerprint', None)
    for state in task['states']:
        for key in ('qpos', 'qpos_ready', 'qpos_error'):
            state.pop(key, None)
    return task


def compatible_task_hashes(meta, task):
    hashes = {task['fingerprint']}
    semantics = task_without_reset_poses(task)
    for path in (meta / 'revisions').glob('*/task.json'):
        previous = json.loads(path.read_text())
        # Accept an archived fingerprint only when reset poses are the sole
        # difference. Changes to routes, rewards, outcomes or sensors still fail.
        if task_without_reset_poses(previous) == semantics:
            hashes.add(previous['fingerprint'])
    return hashes


def check_dataset(root):
    root = Path(root)
    meta = root / 'meta/collection'
    require((meta / 'task.json').is_file(),
            f'No saved collection metadata at {meta}. Save a complete episode with E first.')
    pending = root.parent / f'.{root.name}.collection-staging/pending_commit.json'
    require(not pending.exists(), 'A commit is in progress or interrupted; inspect again after saving finishes.')
    task = json.loads((meta / 'task.json').read_text())
    task_hashes = compatible_task_hashes(meta, task)
    info = json.loads((root / 'meta/info.json').read_text())
    has_rewards = 'next.reward' in info['features']
    if not has_rewards:
        require('reward_contract' not in task, 'Reward metadata exists but next.reward is missing')
        print('Legacy dataset: no next.reward column; reward validation skipped.')
    else:
        require('reward_states' in task and 'reward_contract' in task, 'Reward metadata is missing')
        print(f'Reward state IDs: {task["reward_states"]}')
    records = [json.loads(p.read_text()) for p in sorted((meta / 'episodes').glob('*.json'))]
    require(len(records) == info['total_episodes'], 'Episode metadata count disagrees with LeRobot metadata')
    require(sorted(r['episode_index'] for r in records) == list(range(len(records))),
            'Episode indexes are missing or duplicated')
    print(f'{root}: {len(records)} saved episodes; simulation={task["simulation"]}')
    total = 0
    for record in sorted(records, key=lambda r: r['episode_index']):
        index = record['episode_index']
        require(record['accepted'] and record['route_completed'], f'Episode {index}: incomplete route')
        require(record['task_hash'] in task_hashes,
                f'Episode {index}: task hash differs without a compatible archived reset-pose revision')
        route_index = record['route_index']
        route = task['routes'][route_index]['states']
        require(record['route'] == route, f'Episode {index}: route differs from task metadata')
        path = root / info['data_path'].format(episode_chunk=index // info['chunks_size'], episode_index=index)
        rows = pq.read_table(path).to_pydict()
        n = len(rows['episode_index'])
        require(n == record['frames'] and n > 0, f'Episode {index}: frame count differs')

        def scalar(name):
            values = np.asarray(rows[name])
            require(values.size == n, f'Episode {index}: malformed {name}')
            return values.reshape(n)

        require(np.all(scalar('episode_index') == index), f'Episode {index}: wrong episode index')
        require(np.array_equal(scalar('frame_index'), np.arange(n)), f'Episode {index}: wrong frame indexes')
        require(np.allclose(scalar('timestamp'), np.arange(n) / info['fps'], atol=1e-5),
                f'Episode {index}: nominal timestamps differ from retained frame index / fps')
        for key in ['observation.state', 'action']:
            values = np.asarray(rows[key])
            require(values.shape == (n, 10) and np.isfinite(values).all(),
                    f'Episode {index}: {key} must contain finite 10D vectors')
        ticks = scalar('timing.control_tick_index')
        elapsed = scalar('timing.capture_elapsed_s')
        require(np.isfinite(elapsed).all() and np.all(np.diff(elapsed) >= 0),
                f'Episode {index}: invalid capture times')
        require(np.all(np.diff(ticks) > 0), f'Episode {index}: control ticks do not increase')
        require(np.all(scalar('annotation.route_index') == route_index), f'Episode {index}: wrong route labels')
        require(len(record['segments']) == len(record['goals']) == len(route) - 1,
                f'Episode {index}: wrong number of segments/goals')
        print(f'\nEpisode {index}: route {route}, {n} rows, '
              f'{elapsed[-1] - elapsed[0]:.2f}s between first/last retained captures')
        expected_start = 0
        for segment_index, (segment, goal) in enumerate(zip(record['segments'], record['goals'])):
            start, end = segment['start'], segment['end']
            edge = route[segment_index:segment_index + 2]
            require(start == expected_start and start < end <= n, f'Episode {index}: invalid segment bounds')
            require(segment['edge'] == goal['edge'] == edge and goal['boundary_index'] == end
                    and goal['segment_index'] == segment_index and goal['reached_state_id'] == edge[1],
                    f'Episode {index}: goal event disagrees with segment')
            for key, value in [('segment_index', segment_index), ('edge_from', edge[0]), ('edge_to', edge[1])]:
                require(np.all(scalar(f'annotation.{key}')[start:end] == value),
                        f'Episode {index}, segment {segment_index}: wrong {key}')
            flags = np.zeros(end - start, dtype=int); flags[-1] = 1
            reached = np.full(end - start, -1); reached[-1] = edge[1]
            require(np.array_equal(scalar('annotation.goal_reached')[start:end], flags)
                    and np.array_equal(scalar('annotation.reached_state_id')[start:end], reached),
                    f'Episode {index}, segment {segment_index}: incorrect goal row')
            if has_rewards:
                expected_rewards = flags * int(edge[1] in task['reward_states'])
                require(np.array_equal(scalar('next.reward')[start:end], expected_rewards),
                        f'Episode {index}, segment {segment_index}: incorrect reward')
            arrival = meta / 'arrivals' / record['attempt_id'] / f'arrival-{segment_index + 1:02d}.npz'
            require(arrival.is_file(), f'Episode {index}: missing arrival snapshot {arrival.name}')
            with np.load(arrival, allow_pickle=False) as snapshot:
                require(snapshot['state'].shape == (10,) and np.isfinite(snapshot['state']).all(),
                        f'Episode {index}: malformed arrival state')
            print(f'  {edge[0]} -> {edge[1]}: rows [{start}, {end}), goal at row {end - 1}')
            if has_rewards:
                print(f'    Goal reward: {scalar("next.reward")[end - 1]:g}')
            print(f'    {task["states"][edge[0]]["label"]} -> {task["states"][edge[1]]["label"]}')
            print(f'    Last action: {np.array2string(np.asarray(rows["action"][end - 1]), precision=4)}')
            expected_start = end
        require(expected_start == n, f'Episode {index}: segments do not cover all rows')
        total += n
    require(total == info['total_frames'], 'Total row count disagrees with LeRobot metadata')
    print(f'\nPASS: {len(records)} episodes / {total} rows. Annotation structure is consistent.')
    print('This does not verify visual goal achievement, camera synchronization, or replay omitted ticks.')
    return len(records), total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    try:
        check_dataset(args.root)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f'FAIL: {exc}\n')


if __name__ == '__main__':
    main()
