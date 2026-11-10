"""Validate measured reset targets against the checked-in FR3 joint limits."""
from functools import lru_cache
import math
from numbers import Real
from pathlib import Path
import xml.etree.ElementTree as ET


def validate_qpos(value: object) -> list[float]:
    """Validate seven measured joint angles, without inventing missing data."""
    if not isinstance(value, list) or len(value) != 7:
        raise ValueError("qpos must contain exactly seven joint angles in radians")
    if any(isinstance(x, bool) or not isinstance(x, Real) for x in value):
        raise ValueError("qpos must contain numbers, not strings or booleans")
    result = [float(x) for x in value]
    if not all(math.isfinite(x) for x in result):
        raise ValueError("qpos contains a non-finite joint angle")
    return result


def is_recorded(value: object) -> bool:
    if value is None:
        return False
    return any(x != 0.0 for x in validate_qpos(value))


# Project policy for repeatable reset poses, not a replacement for robot limits.
# Positions right at a limit can fault under even gentle hand-guided motion.
RESET_MARGIN_RAD = 0.05


@lru_cache(maxsize=1)
def joint_limits():
    root = ET.parse(Path(__file__).with_name("fr3_calibration.urdf")).getroot()
    return [(float(node.attrib["lower"]), float(node.attrib["upper"])) for i in range(1, 8)
            for node in [root.find(f"joint[@name='fr3_joint{i}']/limit")]]


@lru_cache(maxsize=1)
def joint_velocity_limits():
    root = ET.parse(Path(__file__).with_name("fr3_calibration.urdf")).getroot()
    return [float(root.find(f"joint[@name='fr3_joint{i}']/limit").attrib["velocity"]) for i in range(1, 8)]


def reset_target(value):
    result = validate_qpos(value)
    if not is_recorded(result):
        raise ValueError("Reset qpos is an unrecorded all-zero placeholder")
    for i, (angle, (low, high)) in enumerate(zip(result, joint_limits())):
        if not low <= angle <= high:
            raise ValueError(f"Joint {i + 1}: {angle} rad is outside [{low}, {high}]")
        margin = min(angle - low, high - angle)
        if margin < RESET_MARGIN_RAD:
            side = "lower" if angle - low < high - angle else "upper"
            raise ValueError(
                f"Joint {i + 1} is only {margin:.6f} rad from its {side} limit. "
                f"Recapture with at least {RESET_MARGIN_RAD:.2f} rad of clearance; "
                "this pose can trigger joint_velocity_violation during hand-guiding."
            )
    return result
