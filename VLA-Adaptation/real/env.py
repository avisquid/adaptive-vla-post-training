"""Gym-style FR3 environment: the arm (real.robot.FR3) plus RealSense cameras.

Observation (dict):
    images : {camera name: (H, W, 3) uint8 RGB}
    state  : (10,) float32 — [xyz(3), rot6d(6), gripper_width(1)]
    task   : str           — natural-language task description

Action: (10,) in the state layout, an absolute desired EE pose and gripper width.
``env.step(action)`` commands it and waits for the next control tick, so
callers do not add their own sleep. Image preprocessing is left to the model.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from real.perception.rs_device import MultiRSDevice
from real.robot import FR3


class FR3RealEnv:
    def __init__(self, cfg, task: str):
        self.cfg = cfg
        self.task = task
        self.robot = FR3(cfg)
        self.cameras = None

    def start(self) -> None:
        try:
            self.robot.start()
            self.cameras = MultiRSDevice(self.cfg.task.perception)
            self.cameras.start()
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        try:
            self.robot.close()
        finally:
            if self.cameras is not None:
                cameras, self.cameras = self.cameras, None
                cameras.stop()

    def reset(self, *, options: dict | None = None) -> tuple[dict, dict]:
        """Reset to options['qpos'] (home when omitted), then read an observation."""
        options = options or {}
        self.robot.reset(options.get("qpos"), open_gripper=options.get("open_gripper", "qpos" not in options))
        return self._get_obs(), {}

    def step(self, action_10d: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        """Returns the gym 5-tuple; episodes are managed by the caller."""
        applied_action = self.robot.step(action_10d)
        return self._get_obs(), 0.0, False, False, {
            "applied_action": applied_action, "velocity_scale": self.robot.velocity_scale}

    def _get_obs(self) -> dict[str, Any]:
        frames = self.cameras.get_frames()
        images: dict[str, np.ndarray] = {}
        for name, frame in frames.items():
            if frame is None:
                raise RuntimeError(f"No frame from camera '{name}'. Check that it is connected and warmed up.")
            images[name] = np.ascontiguousarray(frame.color, dtype=np.uint8)
        return {"images": images, "state": self.robot.get_state_10d(), "task": self.task,
                "capture_monotonic_s": time.monotonic(),
                "camera_timestamps": {name: frame.timestamp for name, frame in frames.items()}}
