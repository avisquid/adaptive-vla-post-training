"""real.robot.FR3 against a fake controller; no robot, CAN or camera is opened."""
import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest
from omegaconf import OmegaConf

from real.joint_limits import joint_limits
from real.robot import FR3, Ticker
from real.rotation_utils import homogeneous_to_pose_10d

LOWER, UPPER = np.asarray(joint_limits()).T
CENTER = (LOWER + UPPER) / 2


class FakeController:
    def __init__(self):
        self.error = None
        self.running = True
        self.state = {"qpos": CENTER.copy(), "qvel": np.zeros(7), "ee": np.eye(4)}
        self.ee_desired = None
        self.move = Mock(side_effect=self._arrive)
        self.switch = Mock()
        self.cycles = 0

    def loop_stats(self):
        self.cycles += 1  # The loop advances between reads.
        return {"count": self.cycles}

    def _arrive(self, target):
        self.state["qpos"] = np.asarray(target, dtype=float).copy()


def robot(**overrides):
    cfg = OmegaConf.create({"robot": {
        "ip": "unused", "freq": 1000, "ee_kp": [300] * 3 + [1000] * 3, "ee_kd": [10] * 6,
        "home_qpos": CENTER.tolist(), "reset_tolerance": .03, "realtime_cpu": None, "read_tool": False,
        "nullspace_avoidance": {"enabled": False}, **overrides}})
    fr3 = FR3(cfg)
    fr3.controller = FakeController()
    fr3.robot = SimpleNamespace(state=fr3.controller.state)  # aiofranka.Robot.state
    return fr3


def target_10d(x=.02):
    pose = np.eye(4)
    pose[:3, 3] = [x, -.01, .01]
    return pose, homogeneous_to_pose_10d(pose, .03)


def test_step_commands_the_target_unchanged():
    fr3 = robot()
    pose, action = target_10d()
    applied = fr3.step(action)
    np.testing.assert_allclose(fr3.controller.ee_desired, pose, atol=1e-7)
    np.testing.assert_allclose(applied, action, atol=1e-7)


def test_reset_validates_before_moving_then_holds_in_osc():
    fr3 = robot()
    for bad in [[0] * 7, [100] * 7, [1, 2], [float("nan")] * 7, (LOWER + .01).tolist()]:
        with pytest.raises(ValueError):
            fr3.reset(bad)
    fr3.controller.move.assert_not_called()
    target = (CENTER + .1).tolist()
    fr3.reset(target)
    np.testing.assert_allclose(fr3.controller.move.call_args.args[0], target)
    fr3.controller.switch.assert_called_once_with("osc")
    fr3.reset()  # Home when omitted.
    np.testing.assert_allclose(fr3.controller.move.call_args.args[0], CENTER)


def test_reset_reports_a_target_it_did_not_reach():
    fr3 = robot()
    fr3.controller.move.side_effect = None  # The arm stays where it is.
    with pytest.raises(RuntimeError, match="from its target"):
        fr3.reset((CENTER + .1).tolist())


def test_a_stopped_loop_refuses_every_command():
    fr3 = robot()
    fr3.controller.error = "cartesian_reflex"
    for call in [fr3.get_qpos, fr3.hold, lambda: fr3.step(target_10d()[1]),
                 lambda: fr3.reset(CENTER.tolist())]:
        with pytest.raises(RuntimeError, match="cartesian_reflex"):
            call()
    fr3.controller.move.assert_not_called()
    assert fr3.controller.ee_desired is None
    fr3.controller.error, fr3.controller.running = None, False
    with pytest.raises(RuntimeError, match="not running"):
        fr3.get_ee_pose()


def test_loop_error_is_kept_instead_of_exiting():
    # FR3.check() relies on aiofranka.Controller keeping the error; NativeFrankaController's
    # own watcher would sys.exit(1) here. Recheck it when upgrading aiofranka.
    pytest.importorskip("aiofranka.native").load_native()
    from aiofranka import Controller, Robot
    with patch("mujoco.viewer.launch_passive", return_value=Mock()):
        controller = Controller(Robot(None))
    engine = controller._engine
    engine.robot.stop = Mock()
    with patch.object(engine, "_watch", AsyncMock(return_value="cartesian_reflex")), \
            patch.object(engine, "_report") as report:
        asyncio.run(engine._run())
    assert controller.error == "cartesian_reflex"
    report.assert_called_once_with("cartesian_reflex")
    engine.robot.stop.assert_called_once()


def test_ticker_paces_without_drift_and_skips_a_missed_period():
    ticker = Ticker(100)
    start = time.perf_counter()
    for _ in range(5):
        ticker.wait()
    paced = time.perf_counter() - start
    time.sleep(.05)  # Five periods late.
    start = time.perf_counter()
    ticker.wait()
    after_stall, deadline = time.perf_counter() - start, ticker.deadline
    assert .045 <= paced < .07
    assert after_stall < .005  # Re-anchored: no sleep and no burst of catch-up ticks.
    assert deadline >= time.perf_counter() - .01



def test_velocity_scaling_shortens_the_move_from_the_measured_pose():
    fr3 = robot(freq=50, velocity_scaling={"enabled": True, "limit_fraction": .5})
    c = fr3.controller
    c.state.update(jac=np.vstack([np.eye(3, 7), np.eye(3, 7, 3)]), mm=np.eye(7))
    c.control_transform = np.eye(4)
    c.ee_desired = np.eye(4) * 7  # The previous target does not matter.
    # 0.0524 m in one 20 ms tick is 2.62 m/s on joint 1: twice half its limit.
    applied = fr3.step(target_10d(x=.0524)[1])
    assert fr3.velocity_ratio == pytest.approx(2.)
    assert fr3.velocity_scale == pytest.approx(.5)
    np.testing.assert_allclose(c.ee_desired[:3, 3], [.0262, -.005, .005], atol=1e-9)
    np.testing.assert_allclose(applied[:3], [.0262, -.005, .005], atol=1e-7)


def blas_threads():
    from threadpoolctl import threadpool_info
    return {info["num_threads"] for info in threadpool_info() if info["user_api"] == "blas"}


class Starting(FakeController):
    """A controller for FR3.start(), recording the order of its calls."""
    events = None

    def __init__(self, _):
        super().__init__()
        self.realtime_cpu = None
        self.register_law = Mock(side_effect=lambda law: self.events.append("compile"))
        self.switch = Mock(side_effect=lambda law: self.events.append("switch"))
        self.start = Mock(side_effect=lambda: self.events.append(("start", blas_threads())))


def start_with_fakes(fr3, events, robot_close=None):
    robot = SimpleNamespace(state=None, close=Mock(side_effect=lambda: events.append("close_robot")))
    gripper = Mock()
    gripper.close.side_effect = lambda: events.append("close_gripper")
    def make(r):
        controller = Starting(r)
        robot.state = controller.state
        return controller
    Starting.events = events
    with patch("real.robot.check_native_installation"), patch("real.robot.Robot", return_value=robot), \
            patch("real.robot.Controller", make), \
            patch("real.robot.AgileXGripper", side_effect=lambda *a: events.append("gripper") or gripper):
        fr3.start()
    return robot, gripper


def test_start_compiles_the_law_and_limits_blas_before_taking_torque():
    events = []
    fr3 = robot(nullspace_avoidance={"enabled": True})
    fr3.gripper_cfg = OmegaConf.create({"CAN": "unused", "torque": 2})
    robot_, gripper = start_with_fakes(fr3, events)
    assert events[0] == "compile" and events[1] == ("start", {1})
    assert events.index("switch") < events.index("gripper")
    assert fr3.controller.null_kp.tolist() == fr3.controller.null_kd.tolist() == [1] * 7
    np.testing.assert_array_equal(fr3.controller.control_transform, np.eye(4))
    fr3.close()
    fr3.close()
    assert events[-2:] == ["close_robot", "close_gripper"] and fr3.blas_limits is None
    robot_.close.assert_called_once()
    gripper.close.assert_called_once()


def test_a_failed_start_releases_the_robot():
    events = []
    fr3 = robot()
    fr3.gripper_cfg = OmegaConf.create({"CAN": "unused", "torque": 2})
    original = blas_threads()
    with patch("real.robot.AgileXGripper", side_effect=RuntimeError("no CAN")), \
            pytest.raises(RuntimeError, match="no CAN"):
        robot_ = SimpleNamespace(state=None, close=Mock())
        def make(r):
            controller = Starting(r)
            robot_.state = controller.state
            return controller
        Starting.events = events
        with patch("real.robot.check_native_installation"), patch("real.robot.Robot", return_value=robot_), \
                patch("real.robot.Controller", make):
            fr3.start()
    robot_.close.assert_called_once()
    assert fr3.robot is None and blas_threads() == original


def test_stale_state_is_refused():
    fr3 = robot()
    fr3.controller.loop_stats = lambda: {"count": 7}  # The loop stopped advancing.
    fr3.get_qpos()
    fr3.last_cycle_seen -= .2
    with pytest.raises(RuntimeError, match="stale"):
        fr3.get_qpos()


def test_one_process_owns_the_arm():
    from real.robot import acquire_control
    with patch("real.robot.check_native_installation"):
        lease = acquire_control("test-arm-lease")
        try:
            with pytest.raises(RuntimeError, match="owns the robot"):
                acquire_control("test-arm-lease")
        finally:
            lease.close()
        acquire_control("test-arm-lease").close()  # Released with its file.


def test_native_loop_runs_the_law_in_mujoco():
    pytest.importorskip("aiofranka.native").load_native()
    from aiofranka import Robot
    original = blas_threads()
    fr3 = robot(nullspace_avoidance={"enabled": True}, freq=50)
    fr3.controller = fr3.robot = None
    with patch("real.robot.Robot", lambda *a, **k: Robot(None, read_tool=False)), \
            patch("mujoco.viewer.launch_passive", return_value=Mock()):
        fr3.start()
        try:
            assert fr3.controller.running and blas_threads() == {1}
            for _ in range(10):
                fr3.step(homogeneous_to_pose_10d(fr3.get_ee_pose(), .03))
            fr3.check()
            assert np.isfinite(fr3.get_qpos()).all()
        finally:
            fr3.close()
    assert blas_threads() == original
