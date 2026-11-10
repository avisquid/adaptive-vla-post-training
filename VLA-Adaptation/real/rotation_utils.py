"""Single-arm 10D poses: xyz, first two rotation columns, gripper width.

The column convention matches the original cached module and recorded data.
"""
from __future__ import annotations

import numpy as np


def matrix_to_rot6d(rotation):
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError(f"Expected (3, 3) rotation, got {rotation.shape}")
    return np.concatenate([rotation[:, 0], rotation[:, 1]])


def rot6d_to_matrix(value):
    value = np.asarray(value, dtype=np.float64).reshape(-1)
    if value.shape != (6,) or not np.isfinite(value).all():
        raise ValueError("Expected six finite rotation components")
    first, second = value[:3], value[3:]
    length = np.linalg.norm(first)
    if length < 1e-10:
        raise ValueError("Degenerate first rotation column")
    first = first / length
    second = second - np.dot(first, second) * first
    length = np.linalg.norm(second)
    if length < 1e-10:
        raise ValueError("Collinear rotation columns")
    second = second / length
    return np.stack([first, second, np.cross(first, second)], axis=1)


def homogeneous_to_pose_10d(transform, gripper_width):
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("Expected a finite (4, 4) transform")
    return np.concatenate([transform[:3, 3], matrix_to_rot6d(transform[:3, :3]), [float(gripper_width)]]).astype(np.float32)


def pose_10d_to_homogeneous(action):
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    if action.shape != (10,) or not np.isfinite(action).all():
        raise ValueError("Expected a finite (10,) single-arm action")
    transform = np.eye(4)
    transform[:3, :3] = rot6d_to_matrix(action[3:9])
    transform[:3, 3] = action[:3]
    return transform, float(action[9])


def orthonormalize_rot6d_action(action):
    action = np.asarray(action)
    transform, width = pose_10d_to_homogeneous(action)
    return homogeneous_to_pose_10d(transform, width).astype(action.dtype)
