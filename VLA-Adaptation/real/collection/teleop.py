"""One SpaceMouse, one arm, and one gripper target."""
import time

import numpy as np

from real.agilex import GRIPPER_OPEN_WIDTH
from real.rotation_utils import homogeneous_to_pose_10d, pose_10d_to_homogeneous


def taken(previous_measured_state, current_measured_state):
    """Preserve the original collector's measured-motion predicate and rtol."""
    return not np.allclose(previous_measured_state, current_measured_state, atol=1e-4)


class GripperToggle:
    def __init__(self, open_width, closed_width, initial_width):
        self.open_width = open_width
        self.closed_width = closed_width
        self.target = initial_width
        self.armed = True

    def update(self, buttons):
        pressed = any(buttons)
        if pressed and self.armed:
            midpoint = (self.open_width + self.closed_width) / 2
            self.target = self.closed_width if self.target >= midpoint else self.open_width
            self.armed = False
        elif not pressed:
            self.armed = True
        return self.target


class SimulatedEnv:
    """Explicit hardware-free preview, never imported by a hardware driver.

    It is its own robot, with the same interface as FR3RealEnv and real.robot.FR3.
    """
    def __init__(self, freq, camera_names):
        self.freq = freq
        self.pose = np.eye(4)
        self.pose[:3, 3] = [.45, 0, .3]
        self.width = GRIPPER_OPEN_WIDTH
        self.qpos = np.array([.1, -.4, -.07, -2.6, -.03, 2.2, -.7])
        self.camera_names = list(camera_names)
        self.robot = self

    def start(self):
        pass

    def close(self):
        pass

    def hold(self):
        return self.get_ee_pose()

    def get_ee_pose(self):
        return self.pose.copy()

    def get_gripper_width(self):
        return self.width

    def get_qpos(self):
        return self.qpos.copy()

    def get_state_10d(self):
        return homogeneous_to_pose_10d(self.pose, self.width)

    def _get_obs(self):
        # Obviously synthetic preview images, with no claims of a live camera feed.
        images = {name: np.full((240, 320, 3), (85, 110, 90), dtype=np.uint8) for name in self.camera_names}
        return {"state": self.get_state_10d(), "images": images, "task": "simulation", "capture_monotonic_s": time.monotonic()}

    def reset(self, options=None):
        if options and options.get("qpos") and any(options["qpos"]):
            self.qpos = np.array(options["qpos"])
        if options and options.get("open_gripper"):
            self.width = GRIPPER_OPEN_WIDTH
        return self._get_obs(), {}

    def step(self, action):
        self.pose, self.width = pose_10d_to_homogeneous(action)
        time.sleep(1 / self.freq)
        return self._get_obs(), 0., False, False, {"applied_action": np.asarray(action).copy()}
