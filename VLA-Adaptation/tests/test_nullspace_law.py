"""The nullspace law against aiofranka's native OSC, in MuJoCo; no robot is opened."""
from unittest.mock import Mock, patch

import numpy as np
import pytest

pytest.importorskip("numba")
native = pytest.importorskip("aiofranka.native")
try:
    native.load_native()
except ImportError as error:
    pytest.skip(f"aiofranka's native loop is not built: {error}", allow_module_level=True)

from aiofranka import NativeFrankaController, RobotInterface

from real.joint_limits import joint_limits
from real.nullspace_law import AvoidanceSettings, limit_bias, limit_torque, robust_inverse, use_nullspace_osc

LOWER, UPPER = np.asarray(joint_limits()).T
CENTER = (LOWER + UPPER) / 2
SETTINGS = vars(AvoidanceSettings())


def simulated_controller(qpos):
    with patch("mujoco.viewer.launch_passive", return_value=Mock()):
        robot = RobotInterface(None)
    robot.data.qpos[:7] = qpos
    robot.data.qvel[:7] = np.linspace(-.2, .2, 7)
    controller = NativeFrankaController(robot)
    controller.ee_kp = [300, 300, 300, 1000, 1000, 1000]
    controller.ee_kd = 10.
    # The first cycle starts from zero torque; the loop's rate limit would
    # saturate both laws alike, so compare the laws' own torques.
    controller.clip = False
    return controller


def first_torque(qpos, law):
    """One control cycle from qpos toward a goal 3 cm and 0.1 rad away, at a TCP."""
    controller = simulated_controller(qpos)
    controller.switch("osc")
    controller.set_tcp([0, 0, .1])
    if law:
        use_nullspace_osc(controller, SETTINGS)
        controller.set_tcp([0, 0, .1])
    goal = controller.ee_desired.copy()
    goal[:3, 3] += [.03, -.02, .01]
    c, s = np.cos(.1), np.sin(.1)
    goal[:3, :3] = goal[:3, :3] @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    controller.ee_desired = goal
    controller.step()
    return np.asarray(controller.last_command).copy()


def test_law_matches_native_osc_away_from_limits():
    qpos = CENTER.copy()
    qpos[3] = -2.0  # A typical elbow; every joint is far from its limits.
    assert np.all(np.minimum(qpos - LOWER, UPPER - qpos) > SETTINGS["activation_distance"])
    np.testing.assert_allclose(first_torque(qpos, law=True), first_torque(qpos, law=False), atol=1e-9)


def test_law_adds_a_push_away_from_a_nearby_limit():
    qpos = CENTER.copy()
    qpos[3] = -2.0
    qpos[0] = UPPER[0] - .1
    difference = first_torque(qpos, law=True) - first_torque(qpos, law=False)
    assert np.abs(difference).max() > 1e-3
    assert np.abs(difference).max() <= SETTINGS["max_projected_torque"] + 1e-9


def test_bias_is_zero_outside_band_and_full_strength_near_limits():
    args = (LOWER, UPPER, SETTINGS["activation_distance"], SETTINGS["full_strength_distance"], SETTINGS["max_bias_torque"])
    np.testing.assert_array_equal(limit_bias(CENTER, *args), np.zeros(7))
    np.testing.assert_allclose(limit_bias(LOWER + .01, *args), np.full(7, SETTINGS["max_bias_torque"]))
    np.testing.assert_allclose(limit_bias(UPPER - .01, *args), np.full(7, -SETTINGS["max_bias_torque"]))


def test_projected_torque_is_bounded_and_produces_no_task_space_force():
    rng = np.random.default_rng(0)
    torque_limit = np.array([87, 87, 87, 87, 12, 12, 12.])
    for _ in range(50):
        a = rng.normal(size=(7, 7))
        mm, jac = a @ a.T + .5 * np.eye(7), rng.normal(size=(6, 7)) * .4
        minv = robust_inverse(mm)
        mx = robust_inverse(jac @ minv @ jac.T)
        jbar = minv @ jac.T @ mx
        projector = np.eye(7) - jac.T @ jbar.T
        nominal = rng.normal(size=7) * 5
        extra = limit_torque(LOWER + .02, projector, nominal, torque_limit, LOWER, UPPER,
                             *SETTINGS.values())
        assert np.abs(extra).max() <= SETTINGS["max_projected_torque"] + 1e-9
        assert np.all(np.abs(nominal + extra) <= np.maximum(torque_limit, np.abs(nominal)) + 1e-9)
        np.testing.assert_allclose(jbar.T @ extra, np.zeros(6), atol=1e-8)


@pytest.mark.parametrize("settings", [
    {"activation_distance": .04}, {"full_strength_distance": .7}, {"max_bias_torque": 0.}, {"max_projected_torque": np.nan}])
def test_invalid_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        AvoidanceSettings(**settings)
