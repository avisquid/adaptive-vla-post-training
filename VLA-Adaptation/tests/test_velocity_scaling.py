d"""Velocity scaling at a MuJoCo FR3 state; no robot is opened."""
from unittest.mock import Mock, patch

import numpy as np
import pytest

native = pytest.importorskip("aiofranka.native")
try:
    native.load_native()
except ImportError as error:
    pytest.skip(f"aiofranka's native loop is not built: {error}", allow_module_level=True)

from aiofranka import NativeFrankaController, RobotInterface

from real.joint_limits import joint_limits, joint_velocity_limits
from real.velocity_scaling import control_jacobian, joint_velocity, scale_to_velocity_limit, twist_between

LOWER, UPPER = np.asarray(joint_limits()).T
QPOS = (LOWER + UPPER) / 2
QPOS[3] = -2.0
LIMIT = np.asarray(joint_velocity_limits())
PERIOD = 1 / 50


@pytest.fixture(scope="module")
def measured():
    """A cycle's state and its measured control-frame pose, with a 10 cm TCP."""
    with patch("mujoco.viewer.launch_passive", return_value=Mock()):
        robot = RobotInterface(None)
    robot.data.qpos[:7] = QPOS
    controller = NativeFrankaController(robot)
    controller.switch("osc")
    controller.set_tcp([0, 0, .1])
    controller.step()
    state, transform = dict(controller.state), np.array(controller.control_transform)
    return state, state["ee"] @ transform, transform


def moved(start, offset, angle):
    goal = start.copy()
    goal[:3, 3] += offset
    c, s = np.cos(angle), np.sin(angle)
    goal[:3, :3] = goal[:3, :3] @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return goal


def test_joint_velocity_produces_the_commanded_twist(measured):
    state, start, transform = measured
    rate = twist_between(start, moved(start, [.003, -.002, .001], .01)) / PERIOD
    dq = joint_velocity(state, transform, rate)
    np.testing.assert_allclose(control_jacobian(state, transform) @ dq, rate, atol=1e-9)


def test_a_slow_move_is_unchanged(measured):
    state, start, transform = measured
    goal = moved(start, [.002, 0, 0], .005)
    scaled, s, ratio = scale_to_velocity_limit(state, goal, transform, PERIOD, LIMIT)
    assert s == 1. and 0 < ratio < 1
    assert scaled is goal


def test_a_fast_move_puts_the_fastest_joint_at_its_limit_along_the_same_direction(measured):
    state, start, transform = measured
    goal = moved(start, [.08, -.06, .04], .3)
    scaled, s, ratio = scale_to_velocity_limit(state, goal, transform, PERIOD, LIMIT)
    assert ratio > 1 and s == pytest.approx(1 / ratio)
    np.testing.assert_allclose(twist_between(start, scaled), s * twist_between(start, goal), atol=1e-9)
    dq = joint_velocity(state, transform, twist_between(start, scaled) / PERIOD)
    assert np.max(np.abs(dq) / LIMIT) == pytest.approx(1.)
