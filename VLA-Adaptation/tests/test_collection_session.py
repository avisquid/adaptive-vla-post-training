from pathlib import Path
import unittest
from unittest.mock import patch, Mock

import numpy as np
from omegaconf import OmegaConf

from real.collection.guide import load_task, Conflict
from real.collection.runtime import Collection, run as launch_collection
from real.collection.session import Session
from real.collection.teleop import SimulatedEnv, taken
from real.rotation_utils import homogeneous_to_pose_10d, pose_10d_to_homogeneous

TASK_PATH = Path(__file__).resolve().parents[1] / 'real/config/task/tabletop.yaml'


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.task = load_task(TASK_PATH)
        self.task['routes'] = [{'id': 0, 'states': [0, 1, 3], 'quota': 2}, {'id': 1, 'states': [0, 5], 'quota': 1}]
        self.session = Session(self.task)

    def begin(self):
        self.session.reset_ready = True
        self.session.begin()

    def test_paused_markers_empty_segments_and_undo(self):
        with self.assertRaises(ValueError):
            self.session.begin()
        self.begin()
        with self.assertRaises(ValueError):
            self.session.goal({'id': 'empty'})
        self.session.frames = 10
        self.session.goal({'id': 'first', 'at': 100.0})
        with self.assertRaises(ValueError):
            self.session.goal({'id': 'empty-next'})
        self.session.frames = 14
        self.session.goal({'id': 'second', 'at': 125.0})
        self.assertEqual(self.session.mode, 'review')
        self.assertEqual(self.session.frames, 14)
        self.assertEqual(self.session.metadata()['segments'], [{'edge': [0,1], 'start': 0, 'end': 10}, {'edge':[1,3], 'start':10, 'end':14}])
        self.session.undo()
        self.assertEqual(self.session.frames, 14)
        self.assertEqual(self.session.mode, 'recording')
        self.session.goal({'id': 'corrected'})
        self.assertEqual(self.session.goals[-1]['boundary_index'], 14)

    def test_quota_counts_only_accepted_whole_routes_and_dry_run_does_not_count(self):
        for count in [False, True, True]:
            self.begin()
            self.session.frames = 1
            self.session.goal({'id':'one'})
            self.session.frames = 2
            self.session.goal({'id':'two'})
            self.session.accepted(count=count)
        self.assertEqual(self.session.counts, [2,0])
        self.assertEqual(self.session.route_index, 1)
        self.begin()
        self.session.frames = 3
        self.session.goal({'id':'failure-destination'})
        self.session.accepted()
        self.assertEqual(self.session.mode, 'complete')

    def test_motion_check_preserves_default_relative_tolerance_and_gripper(self):
        previous = np.zeros(10)
        self.assertFalse(taken(previous, previous))
        tiny = previous.copy(); tiny[0] = .00009
        self.assertFalse(taken(previous, tiny))
        grip = previous.copy(); grip[9] = .001
        self.assertTrue(taken(previous, grip))
        large = np.ones(10) * 100
        self.assertFalse(taken(large, large + .0002))  # Existing rtol matters.

    def test_dry_run_allows_uncaptured_poses_without_relaxing_recording_preflight(self):
        from hydra import compose, initialize_config_dir
        with initialize_config_dir(version_base=None, config_dir=str(TASK_PATH.parents[1])):
            cfg = compose(config_name='collect', overrides=['collect.dry_run=true'])
        for state in cfg.task.states:
            state.qpos = [0] * 7
        # Stop at device construction: this proves the preflight succeeds without
        # connecting the test to robot, CAN, SpaceMouse or camera hardware.
        with patch('real.env.FR3RealEnv', side_effect=RuntimeError('device construction reached')) as env:
            with self.assertRaisesRegex(RuntimeError, 'device construction reached'):
                launch_collection(cfg)
            env.assert_called_once()
            env.reset_mock()
            cfg.collect.dry_run = False
            with self.assertRaisesRegex(ValueError, "Capture state 0's qpos first"):
                launch_collection(cfg)
            env.assert_not_called()

    def test_setup_teleop_works_without_reset_but_rejects_placeholder_reset(self):
        self.task['states'][0]['qpos'] = [0] * 7
        cfg = OmegaConf.create({'collect': {'simulate':False, 'dry_run':True, 'gripper_open_width':.069,
                        'gripper_closed_width':.004, 'ui':{'heartbeat_timeout':3}}})
        env = SimulatedEnv(50, [])
        delta = np.eye(4); delta[0,3] = .001
        mouse = Mock()
        mouse.snapshot.return_value = (delta, ())
        runtime = Collection(cfg, self.task, env, mouse=mouse)
        def action(name, identity):
            runtime.command({'action':name, 'client_id':'operator', 'event_id':identity, 'revision':runtime.session.revision})
            runtime.process_command()
        initial = env.get_ee_pose()
        action('teleop','enable')
        with patch('time.sleep'):
            runtime.step()
        self.assertGreater(env.get_ee_pose()[0,3], initial[0,3])
        self.assertEqual(runtime.session.frames, 0)
        self.assertIsNone(runtime.staged)
        action('teleop','disable')
        self.assertFalse(runtime.session.teleop_enabled)
        with patch.object(env, 'reset') as reset:
            action('reset','invalid-reset')
            reset.assert_not_called()
        self.assertIn('placeholder', runtime.session.error)
        action('start','invalid-start')
        self.assertEqual(runtime.session.mode,'setup')
        self.assertFalse(runtime.session.reset_ready)

    def test_runtime_stationary_ticks_do_not_record_and_empty_goal_does_not_advance(self):
        cfg = OmegaConf.create({'collect': {'simulate':True, 'dry_run':True, 'gripper_open_width':.069,
                        'gripper_closed_width':.004, 'ui':{'heartbeat_timeout':3}}})
        env = SimulatedEnv(50, [])
        run = Collection(cfg, self.task, env)
        def action(name, identity):
            run.command({'action':name, 'client_id':'operator', 'event_id':identity, 'revision':run.session.revision})
            run.process_command()
        action('reset','reset'); action('start','start')
        with patch('time.sleep'):
            for _ in range(5): run.step()
            self.assertEqual(run.session.frames,0)
            action('goal','empty')
            self.assertEqual(len(run.session.goals),0)
            action('simulate_motion','move')
            for _ in range(12): run.step()
            frames = run.session.frames
            self.assertGreater(frames,0)
            for _ in range(5): run.step()
            action('goal','goal')
            self.assertEqual(run.session.frames, frames)
            self.assertEqual(run.session.goals[-1]['boundary_index'], frames)
            with self.assertRaises(Conflict):
                run.command({'action':'goal','client_id':'other','event_id':'other','revision':run.session.revision})
            run.last_heartbeat = 0
            run.step()
            self.assertEqual(run.session.mode,'recording')  # Dry runs ignore the heartbeat.
            run.dry_run = False
            run.step()
            self.assertEqual(run.session.mode,'interrupted')
            self.assertFalse(run.session.teleop_enabled)

    def test_pose_convention_matches_first_two_columns(self):
        transform = np.eye(4)
        transform[:3,:3] = [[0,-1,0],[1,0,0],[0,0,1]]
        transform[:3,3] = [.4,.1,.2]
        packed = homogeneous_to_pose_10d(transform,.04)
        np.testing.assert_array_equal(packed[3:9],[0,1,0,-1,0,0])
        restored,width = pose_10d_to_homogeneous(packed)
        np.testing.assert_allclose(restored, transform, atol=1e-7)
        self.assertAlmostEqual(width,.04)
