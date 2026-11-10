"""pi0.5 through OpenPI: absolute action chunks from the pinned cooking checkpoints.

``OpenpiPolicy.candidates(obs, n, horizon)`` returns float32[n, horizon, 10] in absolute
units (metres / rot6d / gripper width in metres), exactly what ``FR3RealEnv.step`` takes.
OpenPI carries its own normalization statistics inside the checkpoint directory.
"""
from __future__ import annotations

import numpy as np

from bc.proposer import ProposerActor

PROMPT = "Follow the displayed cooking state sequence"

# Action-expert samples per batch. Set this to N for one parallel candidate batch.
# Prefix encoding is always batch 1; 0 retains the upstream serial reference path.
DEFAULT_OPENPI_BATCH = 16


class OpenpiPolicy(ProposerActor):
    """pi0.5 through openpi.

    The vision/language prefix is encoded once per ``candidates`` call. Its KV
    cache is reused by every action-expert batch, with independent candidate
    noise. ``batch >= n`` generates the entire pool in parallel. Smaller batches
    bound action-expert memory without repeating prefix work. Caches never
    survive into the next observation. ``batch=0`` selects upstream serial
    sampling as a reference. Failures raise rather than silently falling back.
    """

    kind = "openpi"

    def __init__(self, config_name: str, ckpt: str, seed: int = 0, prompt: str = PROMPT,
                 batch: int = DEFAULT_OPENPI_BATCH, device: str | None = None):
        import jax
        from openpi.policies import policy_config as pc
        from openpi.training import config as openpi_config

        cfg = openpi_config.get_config(config_name)
        if device is None:
            self.jax_device = jax.devices()[0]
        else:
            backend = "cpu" if str(device) == "cpu" else "gpu"
            index = int(str(device).split(":", 1)[1]) if ":" in str(device) else 0
            available = jax.devices(backend)
            if not 0 <= index < len(available):
                raise ValueError(f"device {device!r} is unavailable to JAX: {available}")
            self.jax_device = available[index]
        with jax.default_device(self.jax_device):
            self.policy = pc.create_trained_policy(cfg, ckpt)
        self.rng = np.random.default_rng(seed)
        self.H = int(cfg.model.action_horizon)
        self.D = int(cfg.model.action_dim)
        self.To = 1                      # pi0.5 is single-frame
        self.prompt = prompt
        self.ckpt = ckpt
        if int(batch) < 0:
            raise ValueError("openpi batch size must be nonnegative")
        self.batch = int(batch)
        self._batched_ok = self.batch > 0 and not getattr(self.policy, "_is_pytorch_model", False)
        if not getattr(self.policy, "_is_pytorch_model", False):
            from openpi.shared import nnx_utils
            from .openpi_cache import SharedPrefixSampler

            sampler = SharedPrefixSampler(self.policy._model)
            self._encode_prefix = nnx_utils.module_jit(sampler.encode)
            self._sample_cached = nnx_utils.module_jit(sampler.sample)

    def reset(self):
        pass

    def _obs_dict(self, obs):
        print(f"prompt: {self.prompt}")
        return {
            "observation/image": obs["images"]["global"],
            "observation/wrist_image": obs["images"]["wrist"],
            "observation/state": obs["state"].astype(np.float32),
            "prompt": self.prompt,
        }

    def candidates(self, obs, n, horizon, noise=None):
        if hasattr(self, "jax_device"):
            import jax
            with jax.default_device(self.jax_device):
                return self._candidates_impl(obs, n, horizon, noise)
        return self._candidates_impl(obs, n, horizon, noise)

    def _candidates_impl(self, obs, n, horizon, noise=None):
        if n < 1 or horizon < 1 or horizon > self.H:
            raise ValueError(f"require N >= 1 and 1 <= horizon <= trained horizon {self.H}")
        self.last_sampling_stats = {"sampler_batch_calls": 0, "sampler_rows_returned": 0,
                                    "prefix_encode_calls": 0, "prefix_encode_rows": 0}
        if noise is None:
            noise = self.rng.standard_normal((n, self.H, self.D)).astype(np.float32)
        noise = np.asarray(noise, dtype=np.float32).reshape(n, self.H, self.D)
        if self._batched_ok:
            return self._candidates_batched(obs, n, horizon, noise)
        if self.batch > 0:
            raise RuntimeError("this Openpi backend cannot batch candidates; explicitly set batch_size=0 for serial inference")
        return self._candidates_serial(obs, n, horizon, noise)

    def _candidates_serial(self, obs, n, horizon, noise):
        ob = self._obs_dict(obs)
        out = []
        for i in range(n):
            self.last_sampling_stats["sampler_batch_calls"] += 1
            out.append(np.asarray(self.policy.infer(ob, noise=noise[i])["actions"])[:horizon, :10])
            self.last_sampling_stats["sampler_rows_returned"] += 1
            self.last_sampling_stats["prefix_encode_calls"] += 1
            self.last_sampling_stats["prefix_encode_rows"] += 1
        return np.stack(out).astype(np.float32)

    def _candidates_batched(self, obs, n, horizon, noise):
        import jax
        import jax.numpy as jnp
        from openpi.models import model as _model

        p = self.policy
        inputs = p._input_transform(jax.tree.map(lambda x: x, self._obs_dict(obs)))
        single = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs)
        context = self._encode_prefix(_model.Observation.from_dict(single))
        self.last_sampling_stats["prefix_encode_calls"] += 1
        self.last_sampling_stats["prefix_encode_rows"] += 1
        out = []
        for lo in range(0, n, self.batch):
            b = min(self.batch, n - lo)
            p._rng, _ = jax.random.split(p._rng)
            self.last_sampling_stats["sampler_batch_calls"] += 1
            actions = self._sample_cached(
                context, **{**p._sample_kwargs, "noise": jnp.asarray(noise[lo:lo + b])},
            )
            actions = np.asarray(actions)
            self.last_sampling_stats["sampler_rows_returned"] += int(actions.shape[0])
            for i in range(b):
                a = p._output_transform({"state": inputs["state"], "actions": actions[i]})["actions"]
                out.append(np.asarray(a)[:horizon, :10])
        return np.stack(out).astype(np.float32)
