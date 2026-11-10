"""Round-trip sparse rewards through the real LeRobot writer and checker."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from real.collection.guide import load_task
from real.collection.recorder import Recorder
from real.collection.session import Session
from real.scripts.check_collection_dataset import check_dataset


TASK_PATH = Path(__file__).resolve().parents[1] / 'real/config/task/tabletop.yaml'


class RewardTests(unittest.TestCase):
    def test_reward_state_validation(self):
        data = yaml.safe_load(TASK_PATH.read_text())
        for bad in [[True], [-1], [7], [4, 4], [4.0], '4', None]:
            with self.subTest(value=bad), self.assertRaisesRegex(ValueError, 'reward_states'):
                load_task(TASK_PATH, {**data, 'reward_states': bad})
        without = dict(data); without.pop('reward_states')
        self.assertEqual(load_task(TASK_PATH, without)['reward_states'], [])

    def test_intermediate_final_and_unrewarded_goals_round_trip(self):
        data = yaml.safe_load(TASK_PATH.read_text())
        data['edge_combinations'] = [[[0, 4, 3], 1], [[0, 3, 4], 1], [[0, 5], 1]]
        task = load_task(TASK_PATH, data)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'dataset'
            recorder = Recorder(root, 'reward_test', task, 50, videos=False, simulation=True)
            session = Session(task)
            state = np.array([.4, 0, .3, 1, 0, 0, 0, 1, 0, .04], dtype=np.float32)
            image = np.zeros((16, 16, 3), dtype=np.uint8)
            try:
                for route in task['routes']:
                    session.reset_ready = True
                    session.begin()
                    staged = recorder.start(session.attempt_id)
                    try:
                        for segment in range(len(route['states']) - 1):
                            for _ in range(3):
                                tick = session.frames
                                staged.add({'observation.state': state, 'action': state,
                                            'observation.images.wrist': image,
                                            'timing.control_tick_index': np.array([tick * 3]),
                                            'timing.capture_elapsed_s': np.array([tick * .06])})
                                session.frames += 1
                            session.goal({'id': f'goal-{segment}'})
                            np.savez(staged.path / f'arrival-{segment + 1:02d}.npz',
                                     state=state, **{'images.wrist': image})
                        recorder.commit(staged, session.metadata(), {'wrist': image.shape}, 'Reward test')
                    finally:
                        staged.close()
                    session.accepted()
            finally:
                recorder.close()
            files = sorted((root / 'data').rglob('*.parquet'))
            expected = [[0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0]]
            for path, rewards in zip(files, expected):
                saved = pq.read_table(path).to_pydict()
                np.testing.assert_array_equal(np.asarray(saved['next.reward']).ravel(), rewards)
            snapshot = json.loads((root / 'meta/collection/task.json').read_text())
            self.assertEqual(snapshot['reward_states'], [4])
            self.assertIn('reward_contract', snapshot)
            card = (root / 'README.md').read_text()
            for text in ['SIMULATED', '0 → 4 → 3', 'Reward state IDs: `[4]`',
                         'next.reward', 'noise_rad', 'not an episode-local', 'first two rotation-matrix']:
                self.assertIn(text, card)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(check_dataset(root), (3, 15))
                original = pq.read_table(files[0])
                for key in ['next.reward', 'annotation.edge_to', 'annotation.goal_reached']:
                    field = original.schema.field(key)
                    bad = pa.array([99] * 6, type=field.type)
                    pq.write_table(original.set_column(original.column_names.index(key), field, bad), files[0])
                    with self.subTest(corrupt=key), self.assertRaisesRegex(ValueError, 'incorrect|wrong'):
                        check_dataset(root)
                pq.write_table(original, files[0])
            changed = load_task(TASK_PATH, {**data, 'reward_states': [3]})
            with self.assertRaisesRegex(ValueError, 'Task definition'):
                Recorder(root, 'reward_test', changed, 50, videos=False, simulation=True)


if __name__ == '__main__':
    unittest.main()
