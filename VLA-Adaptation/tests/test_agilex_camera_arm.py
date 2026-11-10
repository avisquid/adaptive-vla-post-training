"""Camera positioning integration with mocked SDK/devices; never opens CAN."""
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from omegaconf import OmegaConf

from real.agilex import AgileXPositionController, _AGILEX_FACTOR, validate_agilex_qpos
from real.collection.guide import load_task
from real.collection.runtime import Collection
from real.collection.teleop import SimulatedEnv


TASK = Path(__file__).resolve().parents[1] / 'real/config/task/tabletop.yaml'


class CameraArmTests(unittest.TestCase):
    @staticmethod
    def piper_at(q, mode=1):
        """A fake SDK arm that reports mode and, after each command, the joints q."""
        piper = Mock()
        piper.EnablePiper.return_value = True
        piper.GetArmStatus.return_value.arm_status.ctrl_mode = mode
        units = np.rint(np.asarray(q) * _AGILEX_FACTOR).astype(int)
        piper.GetArmJointMsgs.return_value.joint_state = SimpleNamespace(
            **{f'joint_{i + 1}': int(u) for i, u in enumerate(units)})
        return piper

    def test_mdpo_command_parity_lazy_connection_and_disconnect(self):
        q = np.array([.30199822, .97517933, -1.05531911, .032499, .689858, -.21507256])
        piper = self.piper_at(q)
        factory = Mock(return_value=piper)
        with patch.dict('sys.modules', {'piper_sdk': SimpleNamespace(C_PiperInterface_V2=factory)}), \
                patch('real.agilex.time.sleep') as sleep:
            controller = AgileXPositionController()
            factory.assert_not_called()
            np.testing.assert_allclose(controller.move(q), q, atol=1e-5)
            factory.assert_called_once_with('can_piper')
            piper.ConnectPort.assert_called_once()
            piper.MotionCtrl_2.assert_called_once_with(1, 1, 100, 0)
            piper.JointCtrl.assert_called_once_with(*np.rint(q * _AGILEX_FACTOR).astype(int).tolist())
            sleep.assert_called_once_with(1)
            controller.move(q)
            piper.ConnectPort.assert_called_once()
            controller.close(); controller.close()
            piper.DisconnectPort.assert_called_once()
            self.assertEqual(piper.JointCtrl.call_count, 2)  # Closing never parks the arm.

    def test_teaching_mode_is_refused_before_any_command(self):
        piper = self.piper_at(np.zeros(6), mode=2)
        with patch.dict('sys.modules', {'piper_sdk': SimpleNamespace(C_PiperInterface_V2=Mock(return_value=piper))}):
            with self.assertRaisesRegex(RuntimeError, 'teaching mode'):
                AgileXPositionController().move([.1] * 6)
        piper.JointCtrl.assert_not_called()

    def test_an_arm_that_does_not_arrive_raises(self):
        piper = self.piper_at(np.zeros(6))
        with patch.dict('sys.modules', {'piper_sdk': SimpleNamespace(C_PiperInterface_V2=Mock(return_value=piper))}), \
                patch('real.agilex.time.sleep'), patch('real.agilex.time.monotonic', side_effect=[0., 0., 11.]):
            with self.assertRaisesRegex(RuntimeError, 'deg from its target'):
                AgileXPositionController().move([.1] * 6)

    def test_enable_timeout_disconnects_without_commanding_motion(self):
        piper = Mock(); piper.EnablePiper.return_value = False
        with patch.dict('sys.modules', {'piper_sdk': SimpleNamespace(C_PiperInterface_V2=Mock(return_value=piper))}), \
                patch('real.agilex.time.monotonic', side_effect=[0., 6.]):
            controller = AgileXPositionController()
            with self.assertRaisesRegex(TimeoutError, 'did not enable'):
                controller.move([0] * 6)
            piper.DisconnectPort.assert_called_once()
            piper.JointCtrl.assert_not_called()
            self.assertIsNone(controller.piper)

    def test_invalid_targets_rejected_before_connecting(self):
        controller = AgileXPositionController()
        for q in [[0] * 7, [0] * 5, [0] * 5 + [np.nan], [np.inf] * 6]:
            with self.subTest(target=q), patch.object(controller, '_connect') as connect:
                with self.assertRaises(ValueError):
                    controller.move(q)
                connect.assert_not_called()
        np.testing.assert_array_equal(validate_agilex_qpos([0] * 6), np.zeros(6))

    def collection(self, simulation=False):
        task = load_task(TASK)
        cfg = OmegaConf.create({'collect': {'simulate': simulation, 'dry_run': True,
            'gripper_open_width': .069, 'gripper_closed_width': .004,
            'ui': {'heartbeat_timeout': 3}}})
        camera = Mock()
        run = Collection(cfg, task, SimulatedEnv(50, []), camera_arm=camera)
        return run, camera

    def test_camera_moves_before_fr3_for_manual_and_automatic_resets(self):
        run, camera = self.collection()
        order = []
        camera.move.side_effect = lambda q: order.append('camera')
        reset = run.env.reset
        def move_fr3(*args, **kwargs):
            order.append('fr3')
            return reset(*args, **kwargs)
        with patch.object(run.env, 'reset', side_effect=move_fr3):
            run._reset_to_start()
            run._reset_to_start(trigger='after_save')
        self.assertEqual(order, ['camera', 'fr3', 'camera', 'fr3'])
        for call in camera.move.call_args_list:
            np.testing.assert_array_equal(call.args[0], run.task['agilex_qpos'])
        record = run.session.reset_info['camera_arm']
        self.assertEqual(record['target_qpos'], run.task['agilex_qpos'])
        self.assertFalse(record['arrival_verified'])
        self.assertTrue(run.session.reset_ready)

    def test_camera_failure_prevents_fr3_reset(self):
        run, camera = self.collection()
        camera.move.side_effect = TimeoutError('camera enable failed')
        with patch.object(run.env, 'reset') as reset:
            with self.assertRaisesRegex(TimeoutError, 'camera enable failed'):
                run._reset_to_start()
            reset.assert_not_called()
        self.assertFalse(run.session.reset_ready)

    def test_simulation_and_disabled_mount_do_not_move_camera(self):
        run, camera = self.collection(simulation=True)
        run._reset_to_start()
        camera.move.assert_not_called()
        self.assertTrue(run.session.reset_info['camera_arm']['simulation'])
        run, camera = self.collection()
        run.task['agilex_qpos'] = None
        run._reset_to_start()
        camera.move.assert_not_called()
        self.assertNotIn('camera_arm', run.session.reset_info)


if __name__ == '__main__':
    unittest.main()
