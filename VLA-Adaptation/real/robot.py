"""The FR3 arm and its AgileX gripper, driven by aiofranka's native 1 kHz loop.

aiofranka 0.9 runs the torque loop in C++ on its own real-time thread, so this
process's threads and GIL cannot delay a torque command. A new target is a plain
attribute write that the loop takes at its next cycle. aiofranka.Controller runs
its event loop in a thread of its own, so its calls here are plain blocking calls.
"""
import os
import time

import numpy as np
from aiofranka import Controller, Robot

from real.agilex import GRIPPER_OPEN_WIDTH, AgileXGripper
from real.joint_limits import joint_velocity_limits, reset_target
from real.nullspace_law import compile_nullspace_osc, use_nullspace_osc
from real.rotation_utils import homogeneous_to_pose_10d, pose_10d_to_homogeneous
from real.velocity_scaling import scale_to_velocity_limit

NATIVE_PINS = {"aiofranka": "0.9.0", "pylibfranka": "0.21.3", "mujoco": "3.12.0"}


def check_native_installation():
    """Check the pinned native runtime without opening the robot."""
    from importlib.metadata import version
    for package, expected in NATIVE_PINS.items():
        if version(package) != expected:
            raise RuntimeError(f"The native control loop requires {package}=={expected}; "
                               "recreate the env with python -m real.scripts.setup_adaptive")
    from aiofranka.native import load_native
    load_native()


def acquire_control(ip):
    """Reserve the arm at ip for this process; no hardware is opened. Close the result to release it."""
    import fcntl
    import hashlib
    from pathlib import Path
    check_native_installation()
    identity = hashlib.sha256(str(ip).encode()).hexdigest()[:16]
    lease = (Path("/tmp") / f"rsm-native-{os.getuid()}-{identity}.lock").open("a")
    try:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lease.close()
        raise RuntimeError(f"Another collector or deployment owns the robot at {ip}; stop it first") from None
    except BaseException:
        lease.close()
        raise
    return lease


class Ticker:
    """Fixed-rate waits without drift. After falling a full period behind it
    re-anchors, so one slow tick does not cause a burst of catch-up commands."""

    def __init__(self, freq):
        self.period = 1 / float(freq)
        self.deadline = None

    def restart(self):
        self.deadline = None

    def wait(self):
        if self.deadline is None:
            self.deadline = time.perf_counter()
        self.deadline += self.period
        remaining = self.deadline - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
        elif remaining < -self.period:
            self.deadline = time.perf_counter()


class FR3:
    """One arm in OSC with an optional gripper.

    Config: robot.{ip, freq, ee_kp, ee_kd, home_qpos, reset_tolerance, realtime_cpu,
    read_tool, nullspace_avoidance, velocity_scaling} and gripper.{CAN, torque}.
    """

    def __init__(self, cfg):
        self.cfg = cfg.robot
        self.gripper_cfg = cfg.get("gripper")
        self.ticker = Ticker(self.cfg.freq)
        self.robot = None
        self.controller = None
        self.gripper = None
        self.blas_limits = None
        self.last_cycle_count = self.last_cycle_seen = None
        scaling = self.cfg.get("velocity_scaling") or {}
        self.velocity_scaling = bool(scaling.get("enabled", False))
        fraction = float(scaling.get("limit_fraction", 1.))
        if not 0 < fraction <= 1:
            raise ValueError("robot.velocity_scaling.limit_fraction must be in (0, 1]")
        self.velocity_limit = fraction * np.asarray(joint_velocity_limits())
        self.velocity_scale = 1.  # Factor applied to the last step's move.
        self.velocity_ratio = 0.  # Its unscaled max |joint velocity| / limit.

    def start(self):
        check_native_installation()
        try:
            # Threads started after this call (cameras, SpaceMouse, UI, staging) inherit the
            # calling thread's scheduling. Keep it if libfranka raises it to SCHED_FIFO: they
            # would preempt the robot NIC's interrupt threads and delay its packets. The
            # 1 kHz loop sets its own priority and CPU.
            policy, param = os.sched_getscheduler(0), os.sched_getparam(0)
            try:
                self.robot = Robot(str(self.cfg.ip), read_tool=bool(self.cfg.read_tool))
            finally:
                os.sched_setscheduler(0, policy, param)
            # Keeps a loop error (e.g. a reflex) for check() instead of exiting the process.
            self.controller = Controller(self.robot)
            self.controller.realtime_cpu = self.cfg.realtime_cpu
            if (self.cfg.get("nullspace_avoidance") or {}).get("enabled", False):
                compile_nullspace_osc(self.controller)  # Before taking torque control.
            # The law's 6x6/7x7 matrices must not wait on BLAS worker threads, which share
            # the other CPUs with cameras and encoding. Limits this process only; the
            # inference and dataset-writer processes keep their own pools.
            from threadpoolctl import threadpool_limits
            self.blas_limits = threadpool_limits(limits=1, user_api="blas")
            self.controller.start()
            self._use_osc()  # Hold the pose start() measured, in our law.
            deadline = time.monotonic() + 1.
            while int(self.controller.loop_stats()["count"]) < 1:
                self.check()
                if time.monotonic() >= deadline:
                    raise RuntimeError("The control loop's state did not start advancing")
                time.sleep(.001)
            self.state  # noqa: B018 - Validates the first state.
            if self.gripper_cfg is not None:
                self.gripper = AgileXGripper(str(self.gripper_cfg.CAN), self.gripper_cfg.torque)
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup_error:
                error.add_note(f"Robot cleanup also failed: {cleanup_error}")
            raise

    def close(self):
        """Stop the loop, give back torque control and disconnect."""
        robot, gripper, limits = self.robot, self.gripper, self.blas_limits
        self.robot = self.gripper = self.blas_limits = None
        self.last_cycle_count = self.last_cycle_seen = None
        try:
            if robot is not None:
                robot.close()
        finally:
            try:
                if gripper is not None:
                    gripper.close()
            finally:
                if limits is not None:
                    limits.restore_original_limits()

    def _use_osc(self):
        """Hold the measured pose in OSC: switching resets the targets to it."""
        c = self.controller
        c.ee_kp = np.asarray(self.cfg.ee_kp, dtype=np.float64)
        c.ee_kd = np.asarray(self.cfg.ee_kd, dtype=np.float64)
        # aiofranka's defaults, set explicitly so a reset never inherits other values.
        c.null_kp = c.null_kd = np.ones(7)
        c.control_transform = np.eye(4)
        avoidance = dict(self.cfg.get("nullspace_avoidance") or {})
        if avoidance.pop("enabled", False):
            use_nullspace_osc(c, avoidance)
        else:
            c.switch("osc")

    def check(self):
        """Raise if the control loop has stopped."""
        if self.controller is None or self.robot is None:
            raise RuntimeError("The robot is not started; restart the collector.")
        if self.controller.error:
            raise RuntimeError(f"Robot control loop stopped: {self.controller.error}. "
                               "Recover the robot in Desk, run `aiofranka unlock`, then restart the collector.")
        if not self.controller.running:
            raise RuntimeError("Robot control loop is not running; restart the collector.")

    @property
    def state(self):
        """What the loop read at its last cycle: qpos, qvel, ee, jac, mm, last_torque.

        Raises if the loop's cycle count has not changed for 100 ms: its state is stale.
        """
        self.check()
        count, now = int(self.controller.loop_stats()["count"]), time.monotonic()
        # The count restarts with each stats window, so any change is progress.
        if count != self.last_cycle_count:
            self.last_cycle_count, self.last_cycle_seen = count, now
        elif now - self.last_cycle_seen >= .1:
            raise RuntimeError("Robot state is stale: the control loop stopped advancing. "
                               "Recover the robot, then restart the collector.")
        return self.robot.state

    def reset(self, qpos=None, *, open_gripper=True):
        """Move to qpos (home when omitted) in joint impedance, then hold it in OSC."""
        target = np.asarray(reset_target(list(self.cfg.home_qpos if qpos is None else qpos)))
        if open_gripper and self.gripper is not None:
            self.gripper.step(GRIPPER_OPEN_WIDTH)
        self.check()
        self.controller.move(target)
        self._use_osc()
        self.ticker.restart()
        error = float(np.abs(self.get_qpos() - target).max())
        if error > float(self.cfg.reset_tolerance):
            raise RuntimeError(f"Reset stopped {error:.3f} rad from its target; clear the arm's path and reset again.")

    def step(self, action_10d):
        """Command one absolute EE pose and gripper width, then wait for the next tick.

        Returns the accepted action. Joint position limits are left to the arm's
        firmware. With robot.velocity_scaling, a move from the measured pose whose joint velocity over one tick would
        exceed the limits is shortened by one factor (self.velocity_scale); see
        real/velocity_scaling.py.
        """
        pose, width = pose_10d_to_homogeneous(action_10d)
        if not np.isfinite(pose).all():
            raise ValueError("Pose target is not finite")
        if self.gripper is not None:
            self.gripper.step(width)
        state = self.state
        self.velocity_scale, self.velocity_ratio = 1., 0.
        if self.velocity_scaling:
            transform = np.array(self.controller.control_transform, dtype=np.float64)
            pose, self.velocity_scale, self.velocity_ratio = scale_to_velocity_limit(
                state, pose, transform, self.ticker.period, self.velocity_limit)
        self.controller.ee_desired = pose
        self.ticker.wait()
        return homogeneous_to_pose_10d(pose, width)

    def hold(self):
        """Hold the measured pose, without waiting for a tick."""
        pose = self.get_ee_pose()
        self.controller.ee_desired = pose
        return pose

    def get_ee_pose(self):
        return self.state["ee"].copy()

    def get_qpos(self):
        return self.state["qpos"].copy()

    def get_gripper_width(self):
        return float(self.gripper.get_state())

    def get_state_10d(self):
        return homogeneous_to_pose_10d(self.get_ee_pose(), self.get_gripper_width())
