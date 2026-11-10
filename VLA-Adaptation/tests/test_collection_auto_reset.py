"""Collection reset ordering and failures, using simulated devices only."""
from pathlib import Path
import threading
import unittest
from unittest.mock import Mock, patch

from omegaconf import OmegaConf

from real.collection.guide import load_task
from real.collection.runtime import Collection
from real.collection.teleop import SimulatedEnv


class AutoResetTests(unittest.TestCase):
    def collector(self, counts=None, enabled=True, dry_run=False):
        task = load_task(Path(__file__).resolve().parents[1] / 'real/config/task/tabletop.yaml')
        task['routes'] = [{'id': 0, 'states': [0, 1, 3], 'quota': 2},
                          {'id': 1, 'states': [2, 3, 4], 'quota': 1}]
        cfg = OmegaConf.create({'collect': {'simulate': True, 'dry_run': dry_run,
            'auto_reset_after_save': enabled, 'task_prompt': 'test',
            'gripper_open_width': .069, 'gripper_closed_width': .004,
            'ui': {'heartbeat_timeout': 3}}})
        recorder = Mock(counts=counts or [0, 0])
        run = Collection(cfg, task, SimulatedEnv(50, []), recorder=recorder)
        self.addCleanup(run.close)
        run.session.reset_ready = True
        run.session.begin()
        run.session.frames = 1
        run.session.goal({'id': 'first'})
        run.session.frames = 2
        run.session.goal({'id': 'second'})
        if not dry_run:
            run.staged = Mock()
        return run

    def save(self, run, wait=True):
        run.command({'action': 'accept', 'event_id': 'save', 'client_id': 'test',
                     'revision': run.session.revision})
        run.process_command()
        if wait:
            run.finish_save(wait=True)

    def discard(self, run):
        run.command({'action': 'discard', 'event_id': 'discard', 'client_id': 'test',
                     'revision': run.session.revision})
        run.process_command()

    def test_staging_is_sealed_then_reset_precedes_encoding(self):
        run = self.collector()
        run.env.width = run.gripper.target = run.gripper.closed_width
        order = []
        run.recorder.commit.side_effect = lambda *args: order.append('commit')
        run.staged.flush.side_effect = lambda: order.append('flush')
        run.staged.close.side_effect = lambda: order.append('close')
        original_reset = run.env.reset
        def reset(*args, **kwargs):
            order.append('reset')
            self.assertEqual(run.session.counts, [0, 0])
            return original_reset(*args, **kwargs)
        self.assertEqual(run.snapshot()['save_reset_state_id'], 0)
        with patch.object(run.env, 'reset', side_effect=reset) as move:
            self.save(run)
        self.assertEqual(order, ['flush', 'close', 'reset', 'commit'])
        self.assertEqual(move.call_count, 1)
        self.assertTrue(move.call_args.kwargs['options']['open_gripper'])
        self.assertEqual(run.env.width, run.gripper.open_width)
        self.assertEqual(run.gripper.target, run.gripper.open_width)
        self.assertIsNone(run.staged)
        self.assertEqual(run.session.mode, 'setup')
        self.assertTrue(run.session.reset_ready)
        self.assertFalse(run.session.teleop_enabled)
        self.assertEqual(run.session.reset_info['trigger'], 'before_save')
        self.assertEqual(run.session.reset_info['state_id'], 0)
        self.assertEqual(move.call_args.kwargs['options']['qpos'], run.session.reset_info['target'])

    def test_manual_reset_opens_gripper_and_keeps_it_open_when_recording_resumes(self):
        run = self.collector(dry_run=True)
        run.session.discard()
        run.env.width = run.gripper.target = run.gripper.closed_width
        run._reset_to_start()
        self.assertEqual(run.env.width, run.gripper.open_width)
        run.session.begin()
        with patch('time.sleep'):
            run.step()
        self.assertAlmostEqual(run.env.width, run.gripper.open_width)
        self.assertEqual(run.session.frames, 0)

    def test_discard_closes_draft_then_resets_same_route_without_counting(self):
        for mode in ('recording', 'review', 'interrupted'):
            with self.subTest(mode=mode):
                run = self.collector(counts=[2, 0], enabled=False)
                run.session.mode = mode
                run.env.width = run.gripper.target = run.gripper.closed_width
                staged, order = run.staged, []
                staged.checkpoint.side_effect = lambda *args: order.append('checkpoint')
                staged.close.side_effect = lambda: order.append('close')
                reset = run.env.reset
                def move(*args, **kwargs):
                    order.append('reset')
                    self.assertIsNone(run.staged)
                    self.assertFalse(run.session.teleop_enabled)
                    return reset(*args, **kwargs)
                with patch.object(run.env, 'reset', side_effect=move) as reset_call:
                    self.discard(run)
                self.assertEqual(order, ['checkpoint', 'close', 'reset'])
                self.assertTrue(reset_call.call_args.kwargs['options']['open_gripper'])
                self.assertEqual(run.session.counts, [2, 0])
                self.assertEqual(run.session.route_index, 1)
                self.assertEqual(run.session.mode, 'setup')
                self.assertTrue(run.session.reset_ready)
                self.assertEqual(run.session.reset_info['state_id'], 2)
                self.assertEqual(run.session.reset_info['trigger'], 'after_discard')
                self.assertEqual(run.gripper.target, run.gripper.open_width)
                self.assertEqual(run.env.width, run.gripper.open_width)
                run.recorder.commit.assert_not_called()

    def test_failed_discard_reset_leaves_recording_disabled(self):
        run = self.collector()
        with patch.object(run.env, 'reset', side_effect=RuntimeError('reset fault')):
            self.discard(run)
        self.assertTrue(run.fatal)
        self.assertEqual(run.session.mode, 'interrupted')
        self.assertFalse(run.session.reset_ready)
        self.assertFalse(run.session.teleop_enabled)
        self.assertEqual(run.session.counts, [0, 0])
        self.assertIn('reset fault', run.session.error)
        run.recorder.commit.assert_not_called()

    def test_quota_rollover_uses_next_route_start(self):
        run = self.collector(counts=[1, 0])
        self.assertEqual(run.snapshot()['save_reset_state_id'], 2)
        self.save(run)
        self.assertEqual(run.session.route_index, 1)
        self.assertEqual(run.session.counts, [2, 0])
        self.assertEqual(run.session.reset_info['state_id'], 2)
        self.assertEqual(run.session.reset_info['nominal'], run.task['states'][2]['qpos'])

    def test_final_quota_does_not_move(self):
        run = self.collector(counts=[2, 0])
        self.assertIsNone(run.snapshot()['save_reset_state_id'])
        with patch.object(run.env, 'reset') as move:
            self.save(run)
            move.assert_not_called()
        self.assertEqual(run.session.mode, 'complete')
        self.assertEqual(run.session.counts, [2, 1])

    def test_failed_staging_never_resets_or_counts(self):
        run = self.collector()
        run.staged.flush.side_effect = OSError('disk full')
        with patch.object(run.env, 'reset') as move:
            self.save(run)
            move.assert_not_called()
        self.assertEqual(run.session.counts, [0, 0])
        self.assertTrue(run.fatal)
        self.assertIn('disk full', run.session.error)
        self.assertIsNotNone(run.staged)

    def test_failed_commit_after_reset_preserves_staged_source_and_counts(self):
        run = self.collector()
        run.recorder.commit.side_effect = OSError('encode failed')
        with patch.object(run.env, 'reset', wraps=run.env.reset) as reset:
            self.save(run)
            reset.assert_called_once()
        self.assertEqual(run.session.counts, [0, 0])
        self.assertTrue(run.fatal)
        self.assertIn('Staged episode preserved', run.session.error)
        self.assertIn('encode failed', run.session.error)

    def test_writer_runs_in_background_and_quota_changes_only_after_success(self):
        run = self.collector(counts=[1, 0])
        started, finish = threading.Event(), threading.Event()
        caller = threading.get_ident()
        def commit(*args):
            self.assertNotEqual(threading.get_ident(), caller)
            started.set()
            self.assertTrue(finish.wait(5))
        run.recorder.commit.side_effect = commit
        try:
            with patch.object(run.env, 'reset', wraps=run.env.reset) as reset:
                self.save(run, wait=False)
                reset.assert_called_once()
            self.assertTrue(started.wait(2))
            self.assertEqual(run.session.mode, 'saving')
            self.assertEqual(run.session.counts, [1, 0])
            self.assertTrue(run.snapshot()['save_status']['reset_complete'])
            self.assertEqual(run.snapshot()['save_status']['next_state_id'], 2)
            with self.assertRaisesRegex(Exception, 'still saving'):
                run.command({'action': 'start', 'event_id': 'too-early', 'client_id': 'test',
                             'revision': run.session.revision})
        finally:
            finish.set()
            run.finish_save(wait=True)
        self.assertEqual(run.session.counts, [2, 0])
        self.assertTrue(run.session.reset_ready)
        self.assertEqual(run.session.reset_info['state_id'], 2)
        saved = run.recorder.commit.call_args.args[1]
        self.assertEqual(saved['route_index'], 0)
        self.assertEqual(saved['route'], [0, 1, 3])

    def test_reset_failure_preserves_saved_episode_even_if_hold_fails(self):
        run = self.collector()
        with patch.object(run.env, 'reset', side_effect=RuntimeError('reset fault')), \
                patch.object(run, '_hold', side_effect=RuntimeError('hold also failed')):
            self.save(run)
        run.recorder.commit.assert_called_once()
        self.assertEqual(run.session.counts, [1, 0])
        self.assertIsNone(run.staged)
        self.assertFalse(run.session.reset_ready)
        self.assertFalse(run.session.teleop_enabled)
        self.assertTrue(run.fatal)
        self.assertIn('Episode saved successfully', run.session.error)
        self.assertIn('reset fault', run.session.error)
        with patch.object(run.env, 'step') as step:
            run.step()
            step.assert_not_called()
        self.assertIn('Episode saved successfully', run.session.error)

    def test_close_drains_a_running_save_before_publishing_its_quota(self):
        run = self.collector()
        release = threading.Event()
        run.recorder.commit.side_effect = lambda *args: release.wait(5)
        self.save(run, wait=False)
        self.assertEqual(run.session.counts, [0, 0])
        timer = threading.Timer(.05, release.set)
        timer.start()
        try:
            run.close()
        finally:
            release.set()
            timer.join()
        self.assertEqual(run.session.counts, [1, 0])
        self.assertIsNone(run._save_future)
        self.assertIsNone(run._save_executor)

    def test_interrupt_during_reset_never_counts_an_uncommitted_episode(self):
        run = self.collector()
        with patch.object(run.env, 'reset', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.save(run)
        run.close()
        run.recorder.commit.assert_not_called()
        self.assertEqual(run.session.counts, [0, 0])
        self.assertTrue(run.fatal)
        self.assertIn('Staged episode preserved', run.session.error)

    def test_disabled_setting_and_dry_run(self):
        run = self.collector(enabled=False)
        self.assertIsNone(run.snapshot()['save_reset_state_id'])
        with patch.object(run.env, 'reset') as move:
            self.save(run)
            move.assert_not_called()
        self.assertFalse(run.session.reset_ready)
        run = self.collector(dry_run=True)
        self.save(run)
        run.recorder.commit.assert_not_called()
        self.assertEqual(run.session.counts, [0, 0])
        self.assertEqual(run.session.reset_info['state_id'], 0)
        self.assertTrue(run.session.reset_ready)


if __name__ == '__main__':
    unittest.main()
