"""The model worker's configuration: one rollout preset, as a plain dict.

Presets live in real/config/rollout/. The dashboard passes ``cfg.rollout``; offline
tools load one with :func:`load_rollout`. Everything is validated here, once, before
a worker starts.
"""
from __future__ import annotations

from pathlib import Path
import re

from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROLLOUT_DIR = PROJECT_ROOT / "real" / "config" / "rollout"
KINDS = ("mock", "openpi")


def artifact_path(value) -> Path:
    """Resolve relative model assets against the repository, independently of cwd."""
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def load_rollout(name, **overrides):
    """A rollout preset by name (e.g. "pi05_kitchen_sft_v5"), with top-level fields overridden.

    Presets may extend others through a Hydra ``defaults`` list (with ``_self_`` last).
    """
    preset = OmegaConf.load(ROLLOUT_DIR / f"{name}.yaml")
    bases = [load_rollout(base) for base in preset.pop("defaults", []) if base != "_self_"]
    return OmegaConf.merge(*bases, preset, overrides)


def worker_config(rollout, *, simulate, wait_forever=False) -> dict:
    """Validate a rollout preset and return the dict the worker loads its policy from.

    wait_forever (dry runs, for debugging) disables the load, warmup and reset timeouts.
    """
    if OmegaConf.is_config(rollout):
        rollout = OmegaConf.to_container(rollout, resolve=True)
    kind = rollout.get("kind", "openpi")
    if kind not in KINDS:
        raise ValueError(f"rollout.kind must be one of {KINDS}, not {kind!r}")
    if kind == "mock" and not simulate:
        raise ValueError("rollout.kind=mock (the tests' synthetic policy) requires collect.simulate=true")
    checkpoint = None
    if kind != "mock":
        checkpoint = artifact_path(rollout["checkpoint"])
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"rollout.checkpoint is not a checkpoint directory: {checkpoint}")
    if kind == "openpi" and not rollout.get("config_name"):
        raise ValueError("OpenPI rollouts require rollout.config_name from the trained cooking config")
    cameras = list(rollout["cameras"])
    if not cameras or len(set(cameras)) != len(cameras):
        raise ValueError("rollout.cameras must be a nonempty list of distinct camera names")
    if len(cameras) != 2:
        raise ValueError("pi0.5 and the mock policy take exactly two cameras: global view, then wrist view")
    execute_k = rollout.get("execute_k")
    if execute_k is not None and (isinstance(execute_k, bool) or not isinstance(execute_k, int) or execute_k < 1):
        raise ValueError("rollout.execute_k must be null or a positive integer")
    inference = rollout.get("inference") or {}
    environment = dict(inference.get("environment") or {})
    if any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(key)) or not isinstance(value, str)
           for key, value in environment.items()):
        raise ValueError("rollout.inference.environment must map environment variable names to strings")
    return {
        "kind": kind, "simulate": bool(simulate),
        "checkpoint": None if checkpoint is None else str(checkpoint),
        "config_name": rollout.get("config_name"), "prompt": str(rollout["prompt"]),
        "cameras": cameras, "device": str(rollout.get("device", "cuda:0")),
        "batch_size": int(rollout.get("batch_size", 1)), "candidate_horizon": rollout.get("candidate_horizon"),
        "execute_k": execute_k,
        "inference": {"python": inference.get("python"), "environment": environment,
                      "timeout_s": None if wait_forever else float(inference.get("timeout_s", 120))},
    }
