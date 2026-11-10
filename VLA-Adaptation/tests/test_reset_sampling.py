"""Reset randomization bounds and collector metadata, without robot access."""
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf

from real.collection.guide import load_task
from real.collection.runtime import Collection
from real.collection.teleop import SimulatedEnv
from real.joint_limits import joint_limits, RESET_MARGIN_RAD
from real.reset_sampling import noise_degrees, sample_reset_qpos


class ResetSamplingTests(unittest.TestCase):
    def setUp(self):
        self.lower, self.upper = np.asarray(joint_limits()).T
        self.nominal = (self.lower + self.upper) / 2

    def test_uniform_formula_and_zero_noise(self):
        sampled = sample_reset_qpos(self.nominal.tolist(), 2, np.random.default_rng(41))
        expected = self.nominal + np.deg2rad(2) * np.random.default_rng(41).uniform(-1, 1, 7)
        np.testing.assert_allclose(sampled['target'], expected, atol=1e-15)
        np.testing.assert_array_equal(sample_reset_qpos(self.nominal.tolist(), 0)['target'], self.nominal)
        np.testing.assert_allclose(np.array(sampled['nominal']) + sampled['noise_rad'], sampled['target'])

    def test_per_joint_ranges_and_conditioning_near_limits(self):
        q = self.lower + RESET_MARGIN_RAD + .001
        q[1::2] = self.upper[1::2] - RESET_MARGIN_RAD - .001
        rng = np.random.default_rng(17)
        ranges = [0, 1, 2, 3, 4, 5, 6]
        targets = np.array([sample_reset_qpos(q.tolist(), ranges, rng)['target'] for _ in range(100)])
        self.assertTrue(np.all(targets >= self.lower + RESET_MARGIN_RAD))
        self.assertTrue(np.all(targets <= self.upper - RESET_MARGIN_RAD))
        self.assertTrue(np.all(np.abs(targets - q) <= np.deg2rad(ranges) + 1e-15))
        self.assertTrue(np.all(targets[:, 0] == q[0]))
        self.assertGreater(np.unique(targets[:, 2]).size, 95)  # No clipping-induced point mass.

    def test_invalid_inputs_rejected(self):
        for value in [-1, float('nan'), float('inf'), True, '2', [2, 2], [2] * 6 + [-1]]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                noise_degrees(value)
        with self.assertRaisesRegex(ValueError, 'placeholder'):
            sample_reset_qpos([0] * 7)

    def test_collector_dispatches_sampled_target_once_and_records_it(self):
        task = load_task(Path(__file__).resolve().parents[1] / 'real/config/task/tabletop.yaml')
        cfg = OmegaConf.create({'collect': {'simulate': True, 'dry_run': True,
            'gripper_open_width': .069, 'gripper_closed_width': .004, 'ui': {'heartbeat_timeout': 3}}})
        env = SimulatedEnv(50, [])
        run = Collection(cfg, task, env)
        run.reset_rng = np.random.default_rng(3)
        def command(action, identity):
            run.command({'action': action, 'client_id': 'test', 'event_id': identity,
                         'revision': run.session.revision})
            run.process_command()
        with patch.object(env, 'reset', wraps=env.reset) as reset:
            command('reset', 'reset-1')
            self.assertTrue(run.session.reset_ready)
            record = run.session.reset_info
            self.assertEqual(reset.call_count, 1)
            self.assertEqual(reset.call_args.kwargs['options']['qpos'], record['target'])
            self.assertEqual(record['nominal'], task['states'][0]['qpos'])
            np.testing.assert_allclose(np.array(record['target']) - record['nominal'], record['noise_rad'])
            command('start', 'start-1')
            run.session.frames = 1
            command('goal', 'goal-1')
            self.assertEqual(reset.call_count, 1)  # Goal annotation does not randomize/reset.
            self.assertEqual(run.session.metadata()['reset'], record)


if __name__ == '__main__':
    unittest.main()
