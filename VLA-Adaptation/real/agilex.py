"""AgileX hardware over CAN: the FR3's gripper and the camera arm.

Camera-arm positioning is ported from mdpo/real/real_util.py and uses the same
CAN commands, speed and one-second settling wait as MDPO. Construction/import
does not connect or enable the camera arm; move() does.
"""
import time

import numpy as np


# Keep the conversion used to capture existing camera-arm calibration poses.
_AGILEX_FACTOR = 57324.840764

# Physical gripper range; wider commands can crash the fingers.
GRIPPER_OPEN_WIDTH = 0.069

# Camera-arm control modes reported by GetArmStatus().arm_status.ctrl_mode.
CAN_CONTROL, TEACHING = 0x01, 0x02
ARRIVAL_TOLERANCE_RAD = np.radians(1.0)


class AgileXGripper:
    """Width commands are single CAN frames, so step() returns at once."""

    def __init__(self, can, torque_nm):
        from piper_sdk import C_PiperInterface_V2
        self._gripper = C_PiperInterface_V2(can)
        try:
            self._gripper.ConnectPort()
        except Exception as exc:
            raise RuntimeError(f"Gripper did not connect on CAN interface {can}: {exc}") from exc
        self._torque_nm = float(torque_nm)

    def step(self, width_m):
        width_m = float(np.clip(width_m, 0, GRIPPER_OPEN_WIDTH))
        self._gripper.GripperCtrl(int(width_m * 1_000_000), int(self._torque_nm * 1_000), 0x01, 0)

    def get_state(self):
        return self._gripper.GetArmGripperMsgs().gripper_state.grippers_angle / 1_000_000.0

    def close(self):
        """Disconnect only; the gripper keeps its last width."""
        gripper, self._gripper = self._gripper, None
        if gripper is not None:
            gripper.DisconnectPort()


def validate_agilex_qpos(value):
    q = np.asarray(value, dtype=float)
    if q.shape != (6,) or not np.isfinite(q).all():
        raise ValueError('agilex_qpos must contain six finite joint angles in radians')
    return q


class AgileXPositionController:
    def __init__(self, can='can_piper', enable_timeout=5.0, move_timeout=10.0):
        if not isinstance(can, str) or not can:
            raise ValueError('AgileX CAN interface must be a nonempty string')
        if not np.isfinite(enable_timeout) or enable_timeout <= 0:
            raise ValueError('AgileX enable timeout must be finite and positive')
        self.can = can
        self.enable_timeout = enable_timeout
        self.move_timeout = move_timeout
        self.piper = None

    def _connect(self):
        if self.piper is not None:
            return
        from piper_sdk import C_PiperInterface_V2
        self.piper = C_PiperInterface_V2(self.can)
        try:
            self.piper.ConnectPort()
            deadline = time.monotonic() + self.enable_timeout
            while not self.piper.EnablePiper():
                if time.monotonic() >= deadline:
                    raise TimeoutError(f'AgileX arm did not enable on {self.can}')
                time.sleep(.01)
        except BaseException:
            self.close()
            raise

    def joints(self):
        """Measured joint angles [rad]."""
        state = self.piper.GetArmJointMsgs().joint_state
        units = [state.joint_1, state.joint_2, state.joint_3, state.joint_4, state.joint_5, state.joint_6]
        return np.asarray(units, dtype=float) / _AGILEX_FACTOR

    def move(self, q):
        """Move to q and return the measured joints once it arrives.

        Raises instead of silently leaving the camera elsewhere: the arm ignores
        commands while its controller is still in teaching mode.
        """
        q = validate_agilex_qpos(q)
        units = [int(u) for u in np.rint(q * _AGILEX_FACTOR)]
        self._connect()
        if int(self.piper.GetArmStatus().arm_status.ctrl_mode) == TEACHING:
            raise RuntimeError('The camera arm is still in teaching mode and ignores move commands. '
                               'Hold the arm and run `python -m real.agilex reset`, then reset again.')
        deadline = time.monotonic() + self.move_timeout
        while True:
            self.piper.MotionCtrl_2(CAN_CONTROL, 0x01, 100, 0x00)
            self.piper.JointCtrl(*units)
            error = float(np.abs(self.joints() - q).max())
            if error <= ARRIVAL_TOLERANCE_RAD:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(f'The camera arm stopped {np.degrees(error):.1f} deg from its target pose. '
                                   'Check its path and status, then reset again.')
            time.sleep(.05)
        # Match MDPO: let the camera stop swaying before images are used.
        time.sleep(1)
        return self.joints()

    def close(self):
        if self.piper is not None:
            piper, self.piper = self.piper, None
            # Disconnect only; do not issue an unrequested park/home move.
            piper.DisconnectPort()


def reset_teaching_mode(can='can_piper'):
    """Return the camera arm from teaching mode to CAN joint control, holding its pose.

    piper_sdk requires this reset after teaching (or MIT) mode. Hold the arm while
    it runs: the joints can briefly lose torque.
    """
    controller = AgileXPositionController(can)
    try:
        controller._connect()
        controller.piper.MotionCtrl_1(0x02, 0, 0)
        time.sleep(1)
        deadline = time.monotonic() + controller.enable_timeout
        while not controller.piper.EnablePiper():
            if time.monotonic() >= deadline:
                raise TimeoutError(f'AgileX arm did not re-enable on {can}')
            time.sleep(.01)
        # Target the measured pose, so switching modes cannot replay an old command.
        here = [int(u) for u in np.rint(controller.joints() * _AGILEX_FACTOR)]
        controller.piper.MotionCtrl_2(CAN_CONTROL, 0x01, 10, 0x00)
        controller.piper.JointCtrl(*here)
        time.sleep(.5)
        mode = controller.piper.GetArmStatus().arm_status.ctrl_mode
        print(f'Camera arm control mode: {mode}; joints [rad]: {np.round(controller.joints(), 3)}')
        if int(mode) != CAN_CONTROL:
            raise RuntimeError('The camera arm did not switch to CAN control; power-cycle it and retry.')
    finally:
        controller.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='AgileX camera-arm maintenance')
    parser.add_argument('command', choices=['reset'])
    parser.add_argument('--can', default='can_piper')
    reset_teaching_mode(parser.parse_args().can)
