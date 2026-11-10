"""Threaded SpaceMouse reader producing task-space delta actions."""

from __future__ import annotations

import threading
import time

import numpy as np
import pyspacemouse
from scipy.spatial.transform import Rotation


class SpaceMouseController:
    """Threaded 3Dconnexion SpaceMouse reader.

    Samples the device at ``freq`` Hz in a daemon thread and exposes the
    latest per-tick delta as a 4x4 homogeneous transform. The translation and
    rotation components are clipped to conservative magnitudes so a single
    tick can be applied directly to a current EE pose every control cycle.

    """

    def __init__(self, freq: float = 100.0, path: str | None = None):
        # ``path`` pins a specific HID device; None opens the first SpaceMouse.
        # pyspacemouse 2.x returns a device from open(); 1.x returns a bool and
        # exposes read()/close() at module level.
        self._device = pyspacemouse.open(path=path)
        if not self._device:
            raise RuntimeError(f"Failed to open SpaceMouse device (path={path!r})")
        if self._device is True:
            self._device = pyspacemouse  # 1.x has one module-owned device.
        self._started = True
        self._freq = freq
        self._lock = threading.Lock()

        # Default state: identity transform (no delta).
        self._state = np.eye(4)
        self._buttons: tuple[int, ...] = tuple()
        self._sample_time = 0.0
        self._error: Exception | None = None

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while self._started:
            try:
                state = self._device.read()
                if state is None:
                    time.sleep(1.0 / self._freq)
                    continue
            except Exception as exc:
                with self._lock:
                    self._error = exc
                    self._state = np.eye(4)
                    self._buttons = ()
                return
            translation = np.clip(
                np.array([state.x, state.y, -state.z]) * -0.005,
                -0.005,
                0.005,
            )
            rot = np.clip(
                np.array([0.4 * state.pitch, -0.4 * state.roll, -state.yaw]) * 1.5,
                -1,
                1,
            )
            rot_delta = Rotation.from_euler("XYZ", rot, degrees=True).as_matrix()

            homogeneous_delta = np.eye(4)
            homogeneous_delta[:3, :3] = rot_delta
            homogeneous_delta[:3, 3] = translation

            buttons = tuple(state.buttons) if hasattr(state, "buttons") else tuple()

            with self._lock:
                self._state = homogeneous_delta
                self._buttons = buttons
                self._sample_time = time.monotonic()

            time.sleep(1.0 / self._freq)

    def snapshot(self, max_age: float = 0.5):
        """One coherent motion/button sample, rejecting a stopped HID reader."""
        with self._lock:
            if self._error is not None:
                raise RuntimeError(f"SpaceMouse reader failed: {self._error}")
            if time.monotonic() - self._sample_time > max_age:
                raise RuntimeError("SpaceMouse input is stale")
            return self._state.copy(), self._buttons

    def close(self) -> None:
        self._started = False
        self._thread.join(timeout=1.0)
        self._device.close()

