"""Task-level reset randomization, shared without connecting to hardware."""
import numpy as np

from real.joint_limits import RESET_MARGIN_RAD, joint_limits, reset_target


def noise_degrees(value=2.0):
    raw = np.asarray(value)
    if raw.dtype.kind not in 'ifu' or raw.shape not in [(), (7,)]:
        raise ValueError('qpos_noise_deg must be a nonnegative number or seven numbers')
    result = np.broadcast_to(raw.astype(float), (7,)).copy()
    if not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError('qpos_noise_deg must be finite and nonnegative')
    return result


def sample_reset_qpos(nominal, degrees=2.0, rng=None):
    """Uniform per-joint offsets, conditioned on the existing reset limits.

    Away from limits: nominal + radians(degrees) * uniform(-1, 1, size=7).
    Near limits, sample uniformly from the intersection rather than clipping,
    which would concentrate probability at the boundary.
    """
    q = np.asarray(reset_target(nominal), dtype=float)
    amplitude = np.deg2rad(noise_degrees(degrees))
    lower, upper = np.asarray(joint_limits()).T
    # Tiny inset avoids floating-point subtraction falling below the buffer.
    low = np.maximum(-amplitude, lower + RESET_MARGIN_RAD + 1e-12 - q)
    high = np.minimum(amplitude, upper - RESET_MARGIN_RAD - 1e-12 - q)
    fixed = amplitude == 0
    low[fixed] = high[fixed] = 0
    if np.any(low > high):
        raise ValueError('No valid reset perturbation within the configured range')
    offset = (rng if rng is not None else np.random.default_rng()).uniform(low, high)
    target = reset_target((q + offset).tolist())
    return {'nominal': q.tolist(), 'target': target,
            'qpos_noise_deg': np.rad2deg(amplitude).tolist(),
            'noise_rad': (np.asarray(target) - q).tolist(),
            'noise_distribution': 'independent uniform offsets conditioned on reset joint limits'}
