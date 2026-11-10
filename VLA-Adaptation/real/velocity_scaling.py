"""Scale an OSC target step so its commanded joint velocity stays within the limits.

Before FR3.step writes a target, ``scale_to_velocity_limit`` takes the move from the
measured pose of the controlled frame as a twist over one control period. It maps that twist to
joint velocities with the OSC's dynamically consistent inverse, M^-1 J^T (J M^-1 J^T)^-1
(aiofranka 0.9.0's native ``osc``, as ported in ``real/nullspace_law.py``). Joint
velocity is linear in the twist, so dividing the whole move (translation and rotation
together) by the largest |dq_i| / limit_i puts the fastest joint exactly at its limit
and keeps the move's direction. Measured from the arm, the move includes its tracking
lag, so the target never leads the arm by more than the limit allows in one period.
This bounds the commanded pose error, not the measured motion: the OSC tracks it as a
spring, and the joint-limit stop and the loop's own rate limit and clip are unchanged.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def robust_inverse(a):
    return np.linalg.inv(a) if abs(np.linalg.det(a)) > 1e-2 else np.linalg.pinv(a)


def control_jacobian(state, control_transform):
    """The Jacobian of the OSC's controlled frame, whose origin moves with v + w x r."""
    flange = state["ee"]
    jac = np.array(state["jac"], dtype=np.float64)
    r = (flange @ control_transform)[:3, 3] - flange[:3, 3]
    jac[:3] += np.cross(jac[3:].T, r).T
    return jac


def twist_between(start, goal):
    """Translation and rotation vector of goal relative to start, in the base frame."""
    return np.concatenate([goal[:3, 3] - start[:3, 3],
                           Rotation.from_matrix(goal[:3, :3] @ start[:3, :3].T).as_rotvec()])


def along(start, twist, s):
    """The pose s of the way along twist from start."""
    pose = np.eye(4)
    pose[:3, 3] = start[:3, 3] + s * twist[:3]
    pose[:3, :3] = Rotation.from_rotvec(s * twist[3:]).as_matrix() @ start[:3, :3]
    return pose


def joint_velocity(state, control_transform, twist_rate):
    """Joint velocities the OSC uses for a task-space velocity at the measured state."""
    jac = control_jacobian(state, control_transform)
    minv = robust_inverse(np.asarray(state["mm"]))
    return minv @ jac.T @ robust_inverse(jac @ minv @ jac.T) @ twist_rate


def scale_to_velocity_limit(state, goal, control_transform, period, limit):
    """Return (goal, s, ratio): the move from the measured pose to goal, scaled by s.

    goal is an OSC target in the control frame (ee @ control_transform), as
    controller.ee_desired takes it. ratio is max |dq_i| / limit_i of the unscaled
    move; s = min(1, 1/ratio).
    """
    start = state["ee"] @ control_transform
    twist = twist_between(start, goal)
    dq = joint_velocity(state, control_transform, twist / period)
    ratio = float(np.max(np.abs(dq) / limit))
    if ratio <= 1.:
        return goal, 1., ratio
    s = 1. / ratio
    return along(start, twist, s), s, ratio
