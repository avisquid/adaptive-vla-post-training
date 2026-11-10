"""Policy rollouts with pause, teleop and policy resumption in the collection dashboard; no hardware."""
from concurrent.futures import Future
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from mock_rollout import mock_rollout
from omegaconf import OmegaConf

from real.collection.guide import load_task
from real.collection.policy import PolicyDriver, start_worker
from real.collection.recorder import Recorder, features
from real.collection.runtime import Collection
from real.collection.teleop import SimulatedEnv

TASK_PATH = Path(__file__).resolve().parents[1] / 'real/config/task/tabletop.yaml'


class FakeWorker:
    """Plans complete only when the test says so."""
    def __init__(self):
        self.metadata = {'proposer_kind': 'fake', 'trained_action_horizon': 6}
        self.requests, self.resets, self.closed = [], 0, False
        self.future = None

    def submit_plan(self, observation, seed):
        assert self.future is None or self.future.done(), 'one plan in flight'
        self.requests.append((observation, seed))
        self.future = Future()
        return self.future

    def finish(self, chunk):
        self.future.set_result({'chosen_actions': chunk, 'timings': {'planning_s': .05, 'worker_planning_s': .04}})

    def reset(self):
        self.resets += 1

    def submit_reset(self):
        """Completes at once unless hold_reset is set; the test then finishes it."""
        assert self.future is None or self.future.done(), 'one request in flight'
        self.resets += 1
        self.future = Future()
        if not getattr(self, 'hold_reset', False):
            self.future.set_result(None)
        return self.future

    def close(self):
        self.closed = True


def chunk_from(state, steps=6, dx=.002):
    chunk = np.repeat(np.asarray(state, np.float32)[None], steps, 0)
    chunk[:, 0] += dx * np.arange(1, steps + 1, dtype=np.float32)
    chunk[:, 9] = .004  # Close the gripper.
    return chunk


class RolloutTests(unittest.TestCase):
    def setUp(self):
        self.task = load_task(TASK_PATH)
        self.task['routes'] = [{'id': 0, 'states': [0, 1, 3], 'quota': 2}]
        self.cfg = OmegaConf.create({'collect': {'simulate': True, 'dry_run': True, 'gripper_open_width': .069,
                                                 'gripper_closed_width': .004, 'ui': {'heartbeat_timeout': 3},
                                                 'task_prompt': 'fixture'}})
        self.env = SimulatedEnv(50, [])
        self.worker = FakeWorker()
        self.policy = PolicyDriver(self.worker, checkpoint='/ckpt/pi05', config_name='pi05_cooking_stock',
                                   prompt='Follow the displayed cooking state sequence', execute_k=4, plan_timeout_s=2.)
        self.run = Collection(self.cfg, self.task, self.env, policy=self.policy)
        self.sleep = patch('time.sleep')
        self.sleep.start()
        self.addCleanup(self.sleep.stop)

    def action(self, name, identity=None):
        self.run.command({'action': name, 'client_id': 'operator', 'event_id': identity or f'{name}-{self.run.session.revision}',
                          'revision': self.run.session.revision})
        self.run.process_command()

    def start_rollout(self):
        self.action('reset')
        self.action('rollout')
        self.assertEqual(self.run.session.mode, 'recording')
        self.assertEqual(self.run.session.controller, 'policy')
        self.assertFalse(self.run.session.teleop_enabled)

    def test_policy_reset_never_blocks_the_collector_loop(self):
        self.worker.hold_reset = True
        self.start_rollout()  # Returns while the worker is still resetting.
        self.assertEqual(self.worker.resets, 1)
        held = self.run.desired.copy()
        for _ in range(3):
            self.run.step()  # Ticks keep running, holding the pose; no plan yet.
        self.assertEqual(self.worker.requests, [])
        np.testing.assert_allclose(self.run.desired, held)
        self.worker.future.set_result(None)
        self.run.step()  # The reset finished: the first plan goes out.
        self.assertEqual(len(self.worker.requests), 1)

    def test_a_slow_policy_reset_times_out_like_a_plan(self):
        self.worker.hold_reset = True
        clock = [0.]
        self.policy.clock = lambda: clock[0]
        self.start_rollout()
        clock[0] = 10.
        self.run.step()
        self.assertIn('Policy reset exceeded', self.run.session.error)
        self.assertEqual(self.worker.requests, [])

    def test_rollout_requires_reset_and_a_loaded_policy(self):
        self.action('rollout')
        self.assertEqual(self.run.session.mode, 'setup')
        self.assertIn('Reset', self.run.session.error)
        plain = Collection(self.cfg, self.task, SimulatedEnv(50, []))
        plain.command({'action': 'reset', 'client_id': 'o', 'event_id': 'r', 'revision': plain.session.revision})
        plain.process_command()
        plain.command({'action': 'rollout', 'client_id': 'o', 'event_id': 'p', 'revision': plain.session.revision})
        plain.process_command()
        self.assertIn('No policy is loaded', plain.session.error)
        self.assertIsNone(plain.snapshot()['rollout'])

    def test_holds_while_planning_then_executes_prefix_in_order_unchanged(self):
        self.start_rollout()
        start = self.env.get_state_10d()
        self.run.step()  # Submits the first plan.
        self.assertEqual(len(self.worker.requests), 1)
        observation, _ = self.worker.requests[0]
        self.assertEqual(observation['task'], 'Follow the displayed cooking state sequence')
        self.assertEqual(len(observation['history']), 1)
        for _ in range(3):
            self.run.step()
            np.testing.assert_allclose(self.env.get_state_10d()[:3], start[:3], atol=1e-9)
        self.assertEqual(self.run.snapshot()['rollout']['status'], 'planning')
        chunk = chunk_from(start)
        self.worker.finish(chunk)
        executed = []
        for _ in range(4):
            self.run.step()
            executed.append(self.env.get_state_10d())
        np.testing.assert_allclose(np.array(executed)[:, :3], chunk[:4, :3], atol=1e-6)
        self.assertAlmostEqual(self.env.get_gripper_width(), .004)
        self.assertEqual(len(self.worker.requests), 1)
        self.run.step()  # Queue drained: the next plan is requested from the new pose.
        self.assertEqual(len(self.worker.requests), 2)
        self.assertEqual(self.run.session.policy_frames, self.run.session.frames)
        self.assertGreater(self.run.session.frames, 0)

    def test_pause_drops_queue_holds_then_teleop_continues(self):
        self.start_rollout()
        start = self.env.get_state_10d()
        self.run.step()
        self.worker.finish(chunk_from(start))
        for _ in range(2):
            self.run.step()
        self.run.step()  # Executes the third action.
        self.action('pause')
        session = self.run.session
        self.assertEqual(session.controller, 'paused')
        self.assertFalse(session.teleop_enabled)
        self.assertEqual(session.takeover_frame, session.frames)
        held = self.env.get_state_10d()
        for _ in range(3):
            self.run.step()
        # The fourth queued action never runs; the arm holds.
        np.testing.assert_allclose(self.env.get_state_10d(), held, atol=1e-9)
        self.assertEqual(len(self.worker.requests), 1)
        self.assertAlmostEqual(self.env.get_gripper_width(), .004)  # Policy's grip is kept.
        self.assertEqual(self.run.snapshot()['rollout']['status'], 'paused')
        self.action('teleop')
        self.assertEqual((session.controller, session.teleop_enabled), ('human', True))
        self.assertEqual(self.run.snapshot()['rollout']['status'], 'human')
        self.action('simulate_motion')
        for _ in range(12):
            self.run.step()
        self.assertGreater(session.frames, session.takeover_frame)
        self.assertEqual(session.policy_frames, session.takeover_frame)
        self.action('rollout')  # The policy takes over from the SpaceMouse, without a pause.
        self.assertIsNone(session.error)
        self.assertEqual((session.controller, session.teleop_enabled), ('policy', False))
        self.action('rollout')
        self.assertIn('already drives', session.error)

    def test_spacemouse_takes_over_from_the_policy_without_a_pause(self):
        self.start_rollout()
        session = self.run.session
        self.run.step()
        self.worker.finish(chunk_from(self.env.get_state_10d()))
        self.run.step()  # One of four queued actions runs.
        self.action('teleop')
        self.assertIsNone(session.error)
        self.assertEqual((session.controller, session.teleop_enabled), ('human', True))
        self.assertEqual(session.takeover_frame, session.frames)
        held = self.env.get_state_10d()
        self.run.step()
        np.testing.assert_allclose(self.env.get_state_10d(), held, atol=1e-9)  # The queue was dropped.
        self.assertEqual([h['controller'] for h in session.handoffs], ['human'])

    def test_a_spacemouse_recording_can_hand_to_the_policy(self):
        self.action('reset'); self.action('start')  # S with a policy loaded: a rollout recording.
        session = self.run.session
        self.assertEqual((session.collection_mode, session.controller), ('rollout', 'human'))
        self.action('rollout')
        self.assertIsNone(session.error)
        self.assertEqual(session.controller, 'policy')
        self.run.step()
        self.assertEqual(len(self.worker.requests), 1)  # It plans from the current pose.
        self.action('pause'); self.action('teleop')
        self.assertEqual([h['controller'] for h in session.handoffs], ['policy', 'paused', 'human'])

    def test_each_spacemouse_started_attempt_has_its_own_decision_log(self):
        logs = []
        for _ in range(2):
            self.action('reset'); self.action('start'); self.action('rollout')
            self.run.step()
            self.worker.finish(chunk_from(self.env.get_state_10d()))
            self.run.step()
            self.action('discard')
            logs.append((len(self.policy.decisions), self.policy.seed_base))
        self.assertEqual([count for count, _ in logs], [1, 1])  # Not carried over between attempts.
        self.assertNotEqual(logs[0][1], logs[1][1])

    def test_pause_and_resume_alternate_any_number_of_times(self):
        self.start_rollout()
        session = self.run.session

        def drive_policy(steps=3):
            self.run.step()  # Submits a plan from the current pose.
            self.worker.finish(chunk_from(self.env.get_state_10d()))
            for _ in range(steps):
                self.run.step()

        drive_policy()
        self.action('pause'); self.action('teleop')
        self.action('simulate_motion')
        for _ in range(5):
            self.run.step()
        self.action('pause')
        self.assertEqual(session.controller, 'paused')
        self.policy.select_prompt('default')
        self.action('rollout')  # Resume the policy, now under the chosen prompt.
        self.assertIsNone(session.error)
        self.assertEqual((session.controller, session.teleop_enabled), ('policy', False))
        drive_policy()
        self.assertEqual(self.worker.requests[-1][0]['task'], 'Follow the displayed cooking state sequence')
        self.action('pause'); self.action('rollout')
        drive_policy()
        self.assertEqual([h['controller'] for h in session.handoffs],
                         ['paused', 'human', 'paused', 'policy', 'paused', 'policy'])
        self.assertEqual(session.takeover_frame, session.handoffs[0]['frame'])
        self.assertEqual(len(self.policy.decisions), 3)  # One decision log across resumptions.
        self.assertEqual(len({d['seed'] for d in self.policy.decisions}), 3)
        self.assertEqual(self.worker.resets, 3)  # A fresh policy state at each start or resume.
        self.assertEqual(self.run.export()['session']['handoffs'], session.handoffs)
        self.action('pause'); self.action('pause')  # Already paused.
        self.assertIn('Pause is available', session.error)

    def test_frames_carry_intervention_labels_and_metadata_records_handoffs(self):
        staged = []

        class Staged:
            path = Path(tempfile.mkdtemp())
            def add(self, frame):
                staged.append(frame)
            def checkpoint(self, metadata):
                pass
            def close(self):
                pass
            def flush(self):
                pass

        class FakeRecorder:
            counts = [0]
            def start(self, identity):
                return Staged()
            def commit(self, episode, metadata, camera_shapes, prompt):
                self.metadata = metadata
                return metadata

        self.run.recorder = FakeRecorder()
        self.start_rollout()
        self.run.step()
        self.worker.finish(chunk_from(self.env.get_state_10d()))
        for _ in range(4):
            self.run.step()
        self.action('goal')
        self.action('pause')
        self.action('teleop')
        self.action('simulate_motion')
        for _ in range(12):
            self.run.step()
        self.action('goal')
        self.assertEqual(self.run.session.mode, 'review')
        self.action('accept')
        self.run.finish_save(wait=True)
        labels = [int(frame['intervention'][0]) for frame in staged]
        takeover = self.run.recorder.metadata['takeover_frame']
        self.assertEqual(labels, [0] * takeover + [1] * (len(labels) - takeover))
        self.assertGreater(takeover, 0)
        self.assertLess(takeover, len(labels))
        metadata = self.run.recorder.metadata
        self.assertEqual(metadata['collection_mode'], 'rollout')
        self.assertEqual(metadata['rollout']['checkpoint'], '/ckpt/pi05')
        self.assertEqual(metadata['rollout']['execute_k'], 4)
        self.assertEqual(metadata['rollout']['decisions'], 1)
        self.assertIsNotNone(metadata['takeover_time_s'])
        self.assertEqual([h['controller'] for h in metadata['handoffs']], ['paused', 'human'])
        self.assertEqual(metadata['handoffs'][0]['frame'], takeover)

    def test_plan_timeout_and_worker_errors_interrupt_and_hold(self):
        clock = [0.]
        self.policy.clock = lambda: clock[0]
        self.start_rollout()
        self.run.step()
        clock[0] = 3.
        self.run.step()
        self.assertEqual(self.run.session.mode, 'interrupted')
        self.assertIn('plan_timeout_s', self.run.session.error)
        self.assertFalse(self.run.session.policy_driving)
        self.action('discard')
        self.assertEqual(self.run.session.mode, 'setup')
        # The cancelled plan must finish before the next rollout can start.
        self.action('rollout')
        self.assertIn('still finishing', self.run.session.error)
        self.worker.finish(chunk_from(self.env.get_state_10d()))
        self.action('rollout')
        self.assertTrue(self.run.session.policy_driving)
        self.run.step()
        self.worker.future.set_exception(RuntimeError('Inference worker: boom'))
        self.run.step()
        self.assertEqual(self.run.session.mode, 'interrupted')
        self.assertIn('boom', self.run.session.error)

    def test_lost_connection_interrupts_policy_control(self):
        self.run.dry_run = False  # Recorded runs keep the page heartbeat safeguard.
        self.start_rollout()
        self.run.last_heartbeat = 0
        self.run.step()
        self.assertEqual(self.run.session.mode, 'interrupted')
        self.assertIn('connection lost', self.run.session.error)

    def test_dry_run_ignores_lost_heartbeat_and_slow_plans(self):
        cfg = OmegaConf.merge(self.cfg, {'rollout': {'checkpoint': '/ckpt', 'prompt': 'p', 'plan_timeout_s': 2.}})
        policy = PolicyDriver.from_config(cfg, self.worker)
        self.assertIsNone(policy.plan_timeout_s)
        clock = [0.]
        policy.clock = lambda: clock[0]
        self.run.policy = policy
        self.start_rollout()
        self.run.last_heartbeat = 0
        self.run.step()  # Submits the first plan.
        clock[0] = 3600.
        for _ in range(3):
            self.run.step()
        self.assertEqual(self.run.session.mode, 'recording')
        self.assertEqual(self.run.session.controller, 'policy')
        self.assertIsNone(self.run.session.error)
        self.worker.finish(chunk_from(self.env.get_state_10d()))
        self.run.step()
        self.assertEqual(policy.status, 'executing')

    def test_recorded_runs_keep_the_plan_timeout(self):
        cfg = OmegaConf.merge(self.cfg, {'collect': {'dry_run': False},
                                         'rollout': {'checkpoint': '/ckpt', 'prompt': 'p', 'plan_timeout_s': 2.}})
        self.assertEqual(PolicyDriver.from_config(cfg, self.worker).plan_timeout_s, 2.)

    def test_attempt_video_spans_each_rollout_and_teleop_attempt(self):
        self.run.video = video = Mock()
        self.start_rollout()
        attempt, info = video.begin.call_args.args
        self.assertEqual(attempt, self.run.session.attempt_id)
        self.assertEqual((info['collection_mode'], info['checkpoint'], info['route'], info['dry_run']),
                         ('rollout', '/ckpt/pi05', [0, 1, 3], True))
        self.action('discard')
        self.assertEqual(video.end.call_args.args[0], 'discarded')
        self.worker.future = None
        self.action('start')
        self.assertEqual(video.begin.call_args.args[1]['collection_mode'], 'teleop')
        for _ in range(2):
            self.action('simulate_motion')
            for _ in range(3):
                self.run.step()
            self.action('goal')
        self.assertEqual(self.run.session.mode, 'review')
        video.end.reset_mock()
        self.action('accept')
        video.end.assert_called_once()
        self.assertEqual(video.end.call_args.args[0], 'accepted')
        self.assertEqual(video.end.call_args.args[1]['goals'], 2)

    def test_undo_after_policy_reached_review_resumes_policy(self):
        self.start_rollout()
        self.run.step()
        self.worker.finish(chunk_from(self.env.get_state_10d()))
        for _ in range(4):
            self.run.step()
        self.action('goal')
        self.run.step()
        self.worker.finish(chunk_from(self.env.get_state_10d()))
        for _ in range(4):
            self.run.step()
        self.action('goal')
        self.assertEqual(self.run.session.mode, 'review')
        self.assertFalse(self.run.session.policy_driving)
        self.action('undo')
        self.assertTrue(self.run.session.policy_driving)


class EdgePromptTests(unittest.TestCase):
    def setUp(self):
        self.task = load_task(TASK_PATH)
        self.cfg = OmegaConf.create({'collect': {'simulate': True, 'dry_run': True, 'gripper_open_width': .069,
                                                 'gripper_closed_width': .004, 'ui': {'heartbeat_timeout': 3},
                                                 'task_prompt': 'fixture'}})
        self.worker = FakeWorker()
        prompts = {'0_1': 'Place the tomato in the pan.', '1_3': 'Place the pan on the stove.'}
        self.policy = PolicyDriver(self.worker, checkpoint='/ckpt', prompt='fallback', execute_k=2, prompts=prompts)
        self.sleep = patch('time.sleep'); self.sleep.start(); self.addCleanup(self.sleep.stop)

    def collection(self, route):
        task = dict(self.task, routes=[{'id': 0, 'states': route, 'quota': 1}])
        self.run = Collection(self.cfg, task, SimulatedEnv(50, []), policy=self.policy)
        return self.run

    def action(self, name):
        self.run.command({'action': name, 'client_id': 'o', 'event_id': f'{name}-{self.run.session.revision}',
                          'revision': self.run.session.revision})
        self.run.process_command()

    def test_extra_prompt_is_selectable_and_stays_pinned_across_goals(self):
        full = 'Place the tomato in the pan on the stove, then finish.'
        self.policy = PolicyDriver(self.worker, checkpoint='/ckpt', prompt='fallback', execute_k=2,
                                   prompts=self.policy.prompts, extra_prompts={'full_task': full})
        run = self.collection([0, 1, 3])
        self.assertEqual(run.snapshot()['rollout']['prompt_options']['full_task'], full)
        self.choose('full_task')
        self.action('reset'); self.action('rollout')
        run.step()
        self.assertEqual(self.worker.requests[-1][0]['task'], full)
        self.worker.finish(chunk_from(run.env.get_state_10d()))
        for _ in range(2):
            run.step()
        self.action('goal')
        run.step()  # Pinned: the next edge keeps the full-task prompt.
        self.assertEqual(self.worker.requests[-1][0]['task'], full)
        self.assertEqual(self.policy.summary()['extra_prompts'], {'full_task': full})
        with self.assertRaisesRegex(ValueError, 'clash'):
            PolicyDriver(self.worker, checkpoint='/ckpt', prompt='p', prompts={'0_1': 'a'}, extra_prompts={'0_1': 'b'})

    def test_prompt_follows_the_current_edge(self):
        run = self.collection([0, 1, 3])
        self.action('reset'); self.action('rollout')
        run.step()
        self.assertEqual(self.worker.requests[-1][0]['task'], 'Place the tomato in the pan.')
        self.assertEqual(run.snapshot()['rollout']['prompt'], 'Place the tomato in the pan.')
        self.worker.finish(chunk_from(run.env.get_state_10d()))
        for _ in range(2):
            run.step()
        self.action('goal')
        run.step()  # Queue drained: the next plan uses the new edge's instruction.
        self.assertEqual(self.worker.requests[-1][0]['task'], 'Place the pan on the stove.')
        self.worker.finish(chunk_from(run.env.get_state_10d()))
        run.step()
        self.assertEqual([d['prompt'] for d in self.policy.decisions],
                         ['Place the tomato in the pan.', 'Place the pan on the stove.'])
        self.assertEqual(self.policy.decisions[1]['edge'], (1, 3))

    def choose(self, choice):
        self.run.command({'action': 'prompt', 'prompt_choice': choice, 'client_id': 'o',
                          'event_id': f'prompt-{self.run.session.revision}', 'revision': self.run.session.revision})
        self.run.process_command()

    def test_prompt_switch_mid_rollout_labels_each_frame_with_its_plans_prompt(self):
        staged = []

        class Staged:
            path = Path(tempfile.mkdtemp())
            add = staticmethod(staged.append)
            def checkpoint(self, metadata): pass
            def close(self): pass

        run = self.collection([0, 1, 3])
        run.recorder = Mock(counts=[0], start=Mock(return_value=Staged()))
        self.action('reset'); self.action('rollout')
        run.step()
        self.worker.finish(chunk_from(run.env.get_state_10d()))
        run.step()  # First of two queued 0 -> 1 actions.
        self.choose('1_3')
        self.assertIsNone(run.session.error)
        self.assertEqual(run.snapshot()['rollout']['prompt_choice'], '1_3')
        run.step()  # The queued action still belongs to the first plan.
        run.step()  # Queue drained: the next plan uses the chosen prompt.
        self.assertEqual(self.worker.requests[-1][0]['task'], 'Place the pan on the stove.')
        self.worker.finish(chunk_from(run.env.get_state_10d(), dx=.004))
        for _ in range(2):
            run.step()
        self.assertEqual([(d['prompt'], d['prompt_choice']) for d in self.policy.decisions],
                         [('Place the tomato in the pan.', 'auto'), ('Place the pan on the stove.', '1_3')])
        self.choose('default')  # Human frames carry the prompt in effect.
        self.action('pause'); self.action('teleop'); self.action('simulate_motion')
        for _ in range(3):
            run.step()
        labels = [(str(f['task']), int(f['intervention'][0])) for f in staged]
        runs = [key for key, _ in itertools.groupby(labels)]
        self.assertEqual(runs, [('Place the tomato in the pan.', 0), ('Place the pan on the stove.', 0), ('fallback', 1)])
        self.assertEqual([e['prompt_choice'] for e in run.session.events if e['action'] == 'prompt'], ['1_3', 'default'])

    def test_pinned_prompt_allows_untrained_edges_and_auto_rechecks_the_route(self):
        run = self.collection([0, 5])
        self.choose('0_1')
        self.action('reset'); self.action('rollout')
        self.assertIsNone(run.session.error)
        run.step()
        self.assertEqual(self.worker.requests[-1][0]['task'], 'Place the tomato in the pan.')
        self.choose('auto')
        self.assertIn('no prompt for edge 0 -> 5', run.session.error)
        self.assertEqual(self.policy.prompt_choice, '0_1')
        self.choose('9_9')
        self.assertIn('Unknown prompt', run.session.error)

    def test_operator_chooses_the_start_state_and_the_policy_follows_that_route(self):
        self.policy.prompts.update({'2_3': 'Place the tomato in the pan on the stove.', '3_4': 'Finish.'})
        run = Collection(self.cfg, self.task, SimulatedEnv(50, []), policy=self.policy)
        self.run = run
        index = next(i for i, r in enumerate(self.task['routes']) if r['states'] == [2, 3, 4])
        self.action('reset')
        self.assertEqual(run.session.reset_info['state_id'], 0)
        run.command({'action': 'route', 'route_index': index, 'client_id': 'o', 'event_id': 'route',
                     'revision': run.session.revision})
        run.process_command()
        self.assertIsNone(run.session.error)
        self.assertFalse(run.session.reset_ready)  # The arm is still at state 0.
        self.action('rollout')
        self.assertIn('Reset to the starting state', run.session.error)
        self.action('reset')
        self.assertEqual(run.session.reset_info['state_id'], 2)
        self.action('rollout')
        self.assertIsNone(run.session.error)
        run.step()
        self.assertEqual(self.worker.requests[-1][0]['task'], 'Place the tomato in the pan on the stove.')
        self.action('discard')  # The choice persists for the next attempt.
        self.assertEqual(run.session.route, [2, 3, 4])
        self.assertEqual(run.session.reset_info['state_id'], 2)

    def test_recorded_runs_cannot_choose_a_route_whose_quota_is_done(self):
        run = self.collection([0, 1, 3])
        run.dry_run = False
        run.session.counts[0] = 1
        with self.assertRaisesRegex(ValueError, 'quota is complete'):
            run.session.select_route(0)
        run.session.mode = 'setup'
        run.session.select_route(0, allow_complete=True)  # Dry runs pass allow_complete.

    def test_route_with_an_untrained_edge_is_refused(self):
        run = self.collection([0, 5])
        self.action('reset'); self.action('rollout')
        self.assertEqual(run.session.mode, 'setup')
        self.assertIn('no prompt for edge 0 -> 5', run.session.error)
        self.assertEqual(self.worker.requests, [])


class RecorderSchemaTests(unittest.TestCase):
    def test_intervention_feature_only_for_rollout_datasets(self):
        self.assertNotIn('intervention', features({'wrist': (4, 4, 3)}, True))
        self.assertEqual(features({'wrist': (4, 4, 3)}, True, intervention=True)['intervention']['dtype'], 'int64')

    def test_rollouts_cannot_append_to_a_teleop_dataset_and_vice_versa(self):
        task = load_task(TASK_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            for has_label, rollout, message in [(False, True, 'choose a new repo_id for rollouts'),
                                                (True, False, 'This is a rollout dataset')]:
                root = Path(tmp) / f'dataset-{has_label}'
                (root / 'meta/collection/episodes').mkdir(parents=True)
                info = {'fps': 50, 'total_episodes': 0, 'features': features({}, True, intervention=has_label)}
                (root / 'meta/info.json').write_text(json.dumps(info))
                (root / 'meta/collection/task.json').write_text(json.dumps({'fingerprint': task['fingerprint'], 'simulation': True}))
                with self.assertRaisesRegex(ValueError, message):
                    Recorder(root, 'fixture', task, 50, simulation=True, intervention=rollout)


class WorkerIntegrationTests(unittest.TestCase):
    def test_real_worker_subprocess_with_mock_policy(self):
        cfg = OmegaConf.create({
            'collect': {'simulate': True},
            'task': {'perception': {'global': {'resolution': [32, 24]}, 'wrist': {'resolution': [32, 24]}}},
            'rollout': mock_rollout(),
        })
        worker = start_worker(cfg)
        try:
            policy = PolicyDriver.from_config(cfg, worker)
            policy.begin(7)
            env = SimulatedEnv(50, ['global', 'wrist'])
            obs = env._get_obs()
            action = None
            for _ in range(200):
                action = policy.next_action(obs, None)
                if action is not None:
                    break
                import time; time.sleep(.01)
            self.assertIsNotNone(action)
            self.assertEqual(action.shape, (10,))
            self.assertEqual(policy.summary()['worker']['proposer_kind'], 'mock')
            self.assertEqual(len(policy.queue), 3)  # execute_k=4 from the mock preset.
        finally:
            worker.close()

    def test_dashboard_prompt_change_reaches_the_worker_mid_rollout(self):
        cfg = OmegaConf.create({
            'collect': {'simulate': True, 'dry_run': True, 'gripper_open_width': .069, 'gripper_closed_width': .004,
                        'ui': {'heartbeat_timeout': 30}, 'task_prompt': 'fixture'},
            'task': {'perception': {'global': {'resolution': [32, 24]}, 'wrist': {'resolution': [32, 24]}}},
            'rollout': mock_rollout(),
        })
        worker = start_worker(cfg)
        try:
            prompts = {'0_1': 'Place the tomato in the pan.', '1_3': 'Place the pan on the stove.'}
            policy = PolicyDriver(worker, checkpoint='mock', prompt='fallback', execute_k=2, prompts=prompts)
            task = dict(load_task(TASK_PATH), routes=[{'id': 0, 'states': [0, 1, 3], 'quota': 1}])
            run = Collection(cfg, task, SimulatedEnv(50, ['global', 'wrist']), policy=policy)

            def act(name, **extra):
                run.command({'action': name, 'client_id': 'o', 'event_id': f'{name}-{run.session.revision}',
                             'revision': run.session.revision, **extra})
                run.process_command()
                self.assertIsNone(run.session.error)

            def drive_until(done):
                for _ in range(2000):
                    run.step()
                    if done():
                        return
                self.fail('the policy did not plan in time')

            act('reset'); act('rollout')
            drive_until(lambda: len(policy.decisions) >= 2)
            act('prompt', prompt_choice='1_3')
            switched = len(policy.decisions)
            drive_until(lambda: len(policy.decisions) >= switched + 3)
            used = [d['model_prompt'] for d in policy.decisions]
            self.assertEqual(set(used[:2]), {'Place the tomato in the pan.'})
            self.assertEqual(set(used[switched + 1:]), {'Place the pan on the stove.'})
            self.assertEqual(run.snapshot()['rollout']['model_prompt'], 'Place the pan on the stove.')
        finally:
            worker.close()


if __name__ == '__main__':
    unittest.main()
