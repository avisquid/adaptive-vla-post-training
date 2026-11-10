"""Policy inference adapters. This module never connects to robot hardware.

The real dependencies are deliberately lazy: mock evaluation and config checks run in a
CPU environment. Missing trained-model code is an installation error, never a reason to
substitute a mock model. The worker owns the loaded policy.
"""
from __future__ import annotations

import importlib.metadata
import importlib.util
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

from bc.proposer import ProposerActor

# The trained adapters see these role names, in this order, whatever the task calls the cameras.
CAMERA_ROLES = ("global", "wrist")


class ModelDependencyError(RuntimeError):
    """The selected worker environment cannot reconstruct a trained checkpoint."""


def _available(module):
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError, AttributeError):
        return False


def dependency_report(proposer_kind):
    """Report missing *compatible source*, not just pip package names."""
    if proposer_kind == "mock":
        return {"python": sys.executable, "missing": [], "mock": True}
    missing = []
    if proposer_kind == "openpi" and not _available("openpi.policies.policy_config"):
        missing.append("the pinned Openpi revision with cooking training configs and matching input/output transforms")
    return {"python": sys.executable, "missing": missing, "mock": False}


def _cpu_info():
    model = platform.processor() or platform.machine()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    return {"type": "cpu", "model": model, "logical_cores": os.cpu_count()}


def _hardware_info(proposer, device, mock):
    if mock:
        return {"proposal": _cpu_info()}
    kind = getattr(proposer, "kind", None)
    if kind == "openpi":
        d = proposer.jax_device
        return {"proposal": {"backend": "jax", "type": d.platform,
                             "device_index": int(d.id), "model": d.device_kind}}
    import torch
    d = torch.device(device)
    info = _cpu_info()
    if d.type == "cuda":
        index = d.index if d.index is not None else torch.cuda.current_device()
        p = torch.cuda.get_device_properties(index)
        info = {"type": "cuda", "device_index": index, "model": p.name,
                "compute_capability": [p.major, p.minor], "total_memory_bytes": p.total_memory,
                "cuda_runtime": torch.version.cuda}
    info["backend"] = "torch"
    return {"proposal": info}


def _source_roots(mock):
    """Only actual loaded model package roots; never the whole research checkout."""
    if mock:
        return {}
    roots = {}
    for name in ("openpi", "bc"):
        module = sys.modules.get(name)
        origin = getattr(module, "__file__", None)
        if origin:
            roots[name] = str(Path(origin).resolve().parent)
        elif module is not None:
            # Fingerprint the actual imported source directories of namespace packages.
            paths = list(getattr(module, "__path__", []))
            for index, path in enumerate(paths):
                roots[f"{name}_{index}"] = str(Path(path).resolve())
    return roots


def _canonical_obs(raw, cameras):
    """Select the trained views; recorder-only cameras never reach a model."""
    if len(cameras) != 2 or len(set(cameras)) != 2:
        raise ValueError("these trained adapters require two distinct ordered cameras: global view, wrist view")

    def frame(value):
        state = np.asarray(value["state"], dtype=np.float32)
        if state.shape != (10,) or not np.isfinite(state).all():
            raise ValueError("observation state must be finite float32[10]")
        images = {}
        for role, camera in zip(CAMERA_ROLES, cameras):
            image = np.asarray(value["images"][camera])
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"camera {camera} must supply RGB uint8[H,W,3]")
            images[role] = np.ascontiguousarray(image)
        return {"images": images, "state": state, "history": [], "t": int(value.get("t", 0))}

    observation = frame(raw)
    history = raw.get("history") or []
    if isinstance(history, dict):
        history = [history]
    observation["history"] = [frame(previous) for previous in history]
    return observation


class _MockProposal(ProposerActor):
    kind = "mock"
    To = 1

    def __init__(self, horizon):
        self.H = horizon
        self.rng = np.random.default_rng(0)

    def reset(self):
        pass

    def candidates(self, obs, n, horizon):
        candidates = np.broadcast_to(obs["state"], (n, horizon, 10)).copy()
        # Preserve the measured orientation and gripper. Only tiny absolute position
        # offsets vary; these are synthetic plumbing fixtures, never trained models.
        candidates[..., :3] += self.rng.normal(0, 0.001, (n, horizon, 3)).astype(np.float32)
        return candidates


class PolicyPlanner:
    """Draw one action chunk from the policy and execute it unchanged."""

    def __init__(self, proposer, *, horizon, cameras, metadata):
        self.proposer = proposer
        self.horizon = int(horizon)
        self.cameras = list(cameras)
        self._metadata = metadata
        self.forward_passes = 0

    def metadata(self):
        return dict(self._metadata)

    def reset(self):
        self.proposer.reset()

    def plan(self, obs, seed):
        started = time.perf_counter()
        proposal_seed = int(np.random.SeedSequence(int(seed)).generate_state(1)[0])
        self.proposer.rng = np.random.default_rng(proposal_seed)
        kind = getattr(self.proposer, "kind", None)
        if kind == "openpi" and hasattr(self.proposer.policy, "_rng"):
            import jax
            self.proposer.policy._rng = jax.device_put(jax.random.PRNGKey(proposal_seed), self.proposer.jax_device)
        proposal_obs = _canonical_obs(obs, self.cameras)
        prompt = obs.get("task") if isinstance(obs.get("task"), str) else None
        if kind == "openpi" and prompt is not None:
            self.proposer.prompt = prompt
        used = getattr(self.proposer, "prompt", prompt)
        self.forward_passes += 1
        print(f"[Worker] forward pass {self.forward_passes} prompt: {used!r}", flush=True)
        # The proposer API samples n chunks; a rollout executes exactly one.
        actions = np.asarray(self.proposer.candidates(proposal_obs, 1, self.horizon), dtype=np.float32)
        expected = (1, self.horizon, 10)
        if actions.shape != expected or not np.isfinite(actions).all():
            raise ValueError(f"policy must return finite absolute actions with shape {expected}; got {actions.shape}")
        # Echo the instruction this plan was conditioned on, so the driver can verify it.
        return {"chosen_actions": actions[0], "proposal_seed": proposal_seed,
                "prompt": used,
                "timings": {"planning_s": time.perf_counter() - started}}


def build_planner(config):
    """Load the policy from real.deployment.config.worker_config's dict."""
    kind, cameras, device = config["kind"], config["cameras"], config["device"]
    report = dependency_report(kind)
    if report["missing"]:
        raise ModelDependencyError(f"Inference environment {report['python']} is missing:\n- " + "\n- ".join(report["missing"]) +
                                   "\nSet rollout.inference.python to the environment holding the pinned training dependencies.")
    assets = {}
    if kind == "mock":
        requested_horizon = config.get("candidate_horizon")
        proposer = _MockProposal(8 if requested_horizon is None else int(requested_horizon))
    else:
        checkpoint = config["checkpoint"]
        assets["proposal_policy"] = checkpoint
        from real.policies import openpi
        config_name = config["config_name"]
        try:
            proposer = openpi.OpenpiPolicy(config_name, checkpoint, seed=0, batch=int(config["batch_size"]),
                                           device=device, prompt=config.get("prompt", openpi.PROMPT))
        except (KeyError, ValueError) as exc:
            raise ModelDependencyError(f"Openpi config {config_name!r} must be provided by the pinned cooking training tree: {exc}") from exc
    requested_horizon = config.get("candidate_horizon")
    horizon = int(proposer.H) if requested_horizon is None else int(requested_horizon)
    if horizon > proposer.H:
        raise ValueError(f"requested candidate horizon {horizon} exceeds trained action horizon {proposer.H}")
    if horizon < 1:
        raise ValueError("candidate_horizon must be positive")
    execute_k = config.get("execute_k")
    if execute_k is not None and execute_k > horizon:
        raise ValueError(f"rollout.execute_k={execute_k} exceeds the policy's {horizon}-action chunk")
    versions = {}
    for package in ("numpy", "torch", "torchvision", "jax", "diffusers", "openpi", "lerobot"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    metadata = {
        "mock": kind == "mock", "proposer_kind": kind,
        "device": "cpu" if kind == "mock" else device, "python": sys.executable,
        "candidate_horizon": horizon, "trained_action_horizon": int(proposer.H),
        "observation_horizon": int(proposer.To), "action_dim": 10,
        "action_layout": ["x", "y", "z", "r00", "r10", "r20", "r01", "r11", "r21", "gripper_width"],
        "action_units": "absolute metres / rot6d columns / gripper width metres",
        "policy_cameras": list(cameras), "artifacts": assets,
        "dependency_versions": versions,
        "inference_hardware": _hardware_info(proposer, device, kind == "mock"),
        "source_roots": _source_roots(kind == "mock"),
    }
    if hasattr(proposer, "provenance"):
        metadata["policy_transform"] = proposer.provenance
    return PolicyPlanner(proposer, horizon=horizon, cameras=cameras, metadata=metadata)
