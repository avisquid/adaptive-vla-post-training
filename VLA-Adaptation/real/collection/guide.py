"""Task definition loading for the collector. This module never controls or records a robot."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import yaml

from real.joint_limits import reset_target
from real.reset_sampling import noise_degrees
from real.agilex import validate_agilex_qpos


class Conflict(ValueError):
    """An operator action no longer applies to the displayed revision."""


def load_task(path: Path, data: dict | None = None) -> dict:
    path = path.expanduser().resolve()
    data = yaml.safe_load(path.read_text()) if data is None else data
    if not isinstance(data, dict) or not isinstance(data.get("states"), list) or not data["states"]:
        raise ValueError("Task config must contain a nonempty states list")
    states = []
    for i, item in enumerate(data["states"]):
        if not isinstance(item, dict) or not isinstance(item.get("label"), str) or not item["label"].strip():
            raise ValueError(f"State {i} needs a label")
        reference = item.get("image")
        if reference is not None and not isinstance(reference, str):
            raise ValueError(f"State {i} image must be a path or null")
        image_path = (path.parent / reference).resolve() if reference else None
        supported = image_path is not None and image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".gif"}
        image_ready = bool(supported and image_path.is_file())
        image_version = hashlib.sha256(
            f"{image_path}:{image_path.stat().st_mtime_ns}:{image_path.stat().st_size}".encode()
        ).hexdigest()[:16] if image_ready else None
        qpos = item.get("qpos")
        pose_valid = isinstance(qpos, list) and len(qpos) == 7 and all(
            type(x) in (int, float) and math.isfinite(x) for x in qpos
        )
        pose_error = None
        try:
            reset_target(qpos)
        except ValueError as exc:
            pose_error = str(exc)
        states.append({
            "id": i, "label": item["label"], "image": reference,
            "image_url": f"/api/reference/{i}?v={image_version}" if image_ready else None,
            "image_path": str(image_path) if image_ready else None,
            "image_status": "ready" if image_ready else "missing" if not reference or not image_path.is_file() else "unsupported",
            "qpos": qpos if pose_valid else None,
            "qpos_ready": pose_error is None,
            "qpos_error": pose_error,
            "terminal": item.get("terminal") is True,
            "outcome": item.get("outcome", "unspecified"),
        })
    combinations = data.get("edge_combinations")
    if not isinstance(combinations, list) or not combinations:
        raise ValueError("Task config must contain edge_combinations")
    routes, seen = [], set()
    for i, pair in enumerate(combinations):
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"Sequence {i} must be [state IDs, quota]")
        ids, quota = pair
        if not isinstance(ids, list) or len(ids) < 2 or any(type(s) is not int or not 0 <= s < len(states) for s in ids):
            raise ValueError(f"Sequence {i} needs at least two valid state IDs")
        if type(quota) is not int or quota < 1:
            raise ValueError(f"Sequence {i} quota must be a positive integer")
        if any(a == b for a, b in zip(ids, ids[1:])) or tuple(ids) in seen:
            raise ValueError(f"Sequence {i} is duplicate or has adjacent repeated states")
        if any(states[s]["terminal"] for s in ids[:-1]):
            raise ValueError(f"Sequence {i} continues beyond a terminal state")
        seen.add(tuple(ids))
        routes.append({"id": i, "states": ids, "quota": quota})
    reward_states = data.get("reward_states", [])
    if (not isinstance(reward_states, list)
            or any(type(s) is not int or not 0 <= s < len(states) for s in reward_states)
            or len(set(reward_states)) != len(reward_states)):
        raise ValueError("reward_states must be a list of unique valid state IDs")
    reset_noise = noise_degrees(data.get('qpos_noise_deg', 2.0)).tolist()
    agilex_qpos = data.get('agilex_qpos')
    if agilex_qpos is not None:
        agilex_qpos = validate_agilex_qpos(agilex_qpos).tolist()
    agilex_can = data.get('agilex_can', 'can_piper')
    if not isinstance(agilex_can, str) or not agilex_can:
        raise ValueError('agilex_can must be a nonempty string')
    # Includes reward rules, order and quotas: never reinterpret existing data.
    fingerprint = hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return {"name": path.stem.replace("_", " ").title(), "config": str(path),
            "fingerprint": fingerprint, "states": states, "routes": routes,
            "reward_states": reward_states.copy(), "qpos_noise_deg": reset_noise,
            "agilex_qpos": agilex_qpos, "agilex_can": agilex_can}
