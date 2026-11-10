"""Stock aiofranka OSC plus one bounded joint-limit torque in its nullspace.

``nullspace_osc`` ports the native OSC law of aiofranka 0.9.0 (``src/native/laws.h``)
to a Numba control law, so it runs inside the native 1 kHz loop. The only addition
is ``limit_torque``: a smooth push away from nearby joint limits, projected into the
OSC nullspace so the end-effector target is unaffected. The native loop applies its
torque-rate limit and clip afterwards, as for the stock law.
"""
from dataclasses import dataclass

import numba
import numpy as np
from aiofranka import control_law

from real.joint_limits import joint_limits


@dataclass(frozen=True)
class AvoidanceSettings:
    activation_distance: float = .60  # rad from a physical joint limit
    full_strength_distance: float = .05
    max_bias_torque: float = 2.0      # Nm before projection
    max_projected_torque: float = 1.5 # Nm per joint after uniform scaling

    def __post_init__(self):
        if any(not np.isfinite(v) or v <= 0 for v in vars(self).values()):
            raise ValueError('Nullspace avoidance settings must be finite and positive')
        if not .05 <= self.full_strength_distance < self.activation_distance < 1.:
            raise ValueError('Require .05 <= full_strength_distance < activation_distance < 1 rad')


@numba.njit
def robust_inverse(a):
    return np.linalg.inv(a) if abs(np.linalg.det(a)) > 1e-2 else np.linalg.pinv(a)


@numba.njit
def rotation_error(goal, current):
    """Rotation vector of goal @ current.T, as scipy's Rotation.as_rotvec computes it."""
    m = goal @ current.T
    q = np.zeros(4)  # x, y, z, w
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    i = int(np.argmax(np.array([m[0, 0], m[1, 1], m[2, 2], trace])))
    if i == 3:
        q[0], q[1], q[2], q[3] = m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1], 1 + trace
    else:
        j, k = (i + 1) % 3, (i + 2) % 3
        q[i] = 1 - trace + 2 * m[i, i]
        q[j] = m[j, i] + m[i, j]
        q[k] = m[k, i] + m[i, k]
        q[3] = m[k, j] - m[j, k]
    q /= np.linalg.norm(q)
    if q[3] < 0:
        q = -q
    angle = 2 * np.arctan2(np.linalg.norm(q[:3]), q[3])
    scale = 2 + angle**2 / 12 + 7 * angle**4 / 2880 if angle <= 1e-3 else angle / np.sin(angle / 2)
    return scale * q[:3]


@numba.njit
def smoothstep(x):
    x = min(max(x, 0.), 1.)
    return x * x * (3. - 2. * x)


@numba.njit
def limit_bias(q, lower, upper, activation, full_strength, max_bias):
    """Negative potential gradient, zero outside the activation band."""
    width = activation - full_strength
    bias = np.zeros(7)
    for i in range(7):
        bias[i] = max_bias * (smoothstep((activation - (q[i] - lower[i])) / width)
                              - smoothstep((activation - (upper[i] - q[i])) / width))
    return bias


@numba.njit
def limit_torque(q, projector, nominal, torque_limit, lower, upper,
                 activation, full_strength, max_bias, max_projected):
    extra = projector @ limit_bias(q, lower, upper, activation, full_strength, max_bias)
    # Scale the whole vector, not single joints: that keeps it in the nullspace.
    scale = min(1., max_projected / max(np.max(np.abs(extra)), 1e-12))
    active = False
    for i in range(7):
        if abs(extra[i]) > 1e-12:
            # Leave the nominal torque room under its magnitude bound.
            room = (torque_limit[i] - np.sign(extra[i]) * nominal[i]) / abs(extra[i])
            scale = min(scale, room)
            active = True
    if active:
        scale = max(0., scale)
    return scale * extra


@control_law(params={"limit_lower": 7, "limit_upper": 7, "activation_distance": 1,
                     "full_strength_distance": 1, "max_bias_torque": 1, "max_projected_torque": 1})
def nullspace_osc(s, p, m, tau):
    flange = s.ee.copy()
    ee = flange @ p.control_transform
    # Jacobian of the control frame, whose origin moves with v + w x r.
    r = ee[:3, 3] - flange[:3, 3]
    jac = s.jac.copy()
    for c in range(7):
        wx, wy, wz = jac[3, c], jac[4, c], jac[5, c]
        jac[0, c] += wy * r[2] - wz * r[1]
        jac[1, c] += wz * r[0] - wx * r[2]
        jac[2, c] += wx * r[1] - wy * r[0]

    goal = p.ee_desired.copy()
    twist = np.zeros(6)
    twist[:3] = goal[:3, 3] - ee[:3, 3]
    twist[3:] = rotation_error(goal[:3, :3].copy(), ee[:3, :3].copy())
    dq = s.qvel.copy()

    minv = robust_inverse(s.mm.copy())
    mx = robust_inverse(jac @ minv @ jac.T)
    feedback = jac.T @ mx @ (p.ee_kp * twist - p.ee_kd * (jac @ dq))
    posture = p.null_kp * (p.initial_qpos - s.qpos) - p.null_kd * dq
    projector = np.eye(7) - jac.T @ (minv @ jac.T @ mx).T
    nominal = feedback + projector @ posture

    tau[:] = nominal + limit_torque(
        s.qpos.copy(), projector, nominal, p.torque_limit.copy(), p.limit_lower.copy(), p.limit_upper.copy(),
        p.activation_distance[0], p.full_strength_distance[0], p.max_bias_torque[0], p.max_projected_torque[0])
    return 0 if np.isfinite(tau).all() else 1


@numba.njit
def warm_linalg():
    """Use once each BLAS/LAPACK routine the law can call.

    Numba looks up a routine's address on its first use while holding the GIL. In the
    1 kHz loop that first use waits for whichever Python thread holds it, and the robot
    aborts with communication_constraints_violation. robust_inverse's two branches
    matter: one may first run only at some later pose.
    """
    a, v = np.eye(7), np.ones(7)
    robust_inverse(a)         # det, inv
    robust_inverse(1e-3 * a)  # pinv
    rotation_error(np.eye(3), np.eye(3))
    return (a @ a.T)[0, 0] + (a @ v)[0]


def compile_nullspace_osc(controller):
    """Compile the law and resolve its routines here, in a Python thread, not in the loop.
    Call it before the loop takes torque control, so compiling never competes with it."""
    warm_linalg()
    controller.register_law(nullspace_osc)


def use_nullspace_osc(controller, settings):
    """Register the law, write its parameters, then switch to it."""
    settings = AvoidanceSettings(**settings)
    compile_nullspace_osc(controller)
    controller.limit_lower, controller.limit_upper = np.asarray(joint_limits()).T
    for name, value in vars(settings).items():
        setattr(controller, name, value)
    controller.switch(nullspace_osc)
