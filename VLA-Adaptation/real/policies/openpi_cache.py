"""Shared observation-prefix sampling for the pinned OpenPI Pi0/Pi0.5 model.

This adapter splits ``Pi0.sample_actions`` at its existing prefix KV cache. It
uses the loaded model's encoders, attention, and action projections unchanged;
no installed OpenPI code or methods are patched. Keep the Euler update, masks,
and stopping condition aligned with the pinned upstream sampler.
"""
from __future__ import annotations

from typing import NamedTuple

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.models import model as model_lib
from openpi.models.pi0 import make_attn_mask


class PrefixContext(NamedTuple):
    # Gemma scans over layers, so each K/V array is [layers, 1, tokens, heads, dim].
    kv_cache: tuple[jax.Array, jax.Array]
    mask: jax.Array
    state: jax.Array


class SharedPrefixSampler(nnx.Module):
    """Encode one observation, then sample independent action batches from it."""

    def __init__(self, model):
        self.model = model

    def encode(self, observation):
        if observation.state.shape[0] != 1:
            raise ValueError("shared-prefix encoding requires exactly one observation")
        observation = model_lib.preprocess_observation(None, observation, train=False)
        tokens, mask, ar_mask = self.model.embed_prefix(observation)
        positions = jnp.cumsum(mask, axis=1) - 1
        _, kv_cache = self.model.PaliGemma.llm(
            [tokens, None], mask=make_attn_mask(mask, ar_mask), positions=positions,
        )
        return PrefixContext(kv_cache, mask, observation.state)

    def sample(self, context: PrefixContext, *, noise, num_steps=10):
        if noise.ndim != 3 or noise.shape[1:] != (self.model.action_horizon, self.model.action_dim):
            raise ValueError("noise must have shape [candidates, action_horizon, action_dim]")
        b = noise.shape[0]
        if context.mask.shape[0] != 1 or context.state.shape[0] != 1:
            raise ValueError("prefix context must belong to one observation")
        # Broadcast inside the compiled action sampler. No repeated images or
        # language tokens enter this function, and prefix encoding cannot rerun.
        kv_cache = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (x.shape[0], b, *x.shape[2:])), context.kv_cache,
        )
        prefix_mask = jnp.broadcast_to(context.mask, (b, context.mask.shape[1]))
        observation = model_lib.Observation(
            images={}, image_masks={},
            state=jnp.broadcast_to(context.state, (b, context.state.shape[1])),
        )
        dt = -1.0 / num_steps

        def step(carry):
            x_t, time = carry
            suffix, suffix_mask, suffix_ar_mask, adarms_cond = self.model.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, b),
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = jnp.broadcast_to(
                prefix_mask[:, None, :], (b, suffix.shape[1], prefix_mask.shape[1]),
            )
            full_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            (_, suffix_out), _ = self.model.PaliGemma.llm(
                [None, suffix], mask=full_mask, positions=positions,
                kv_cache=kv_cache, adarms_cond=[None, adarms_cond],
            )
            velocity = self.model.action_out_proj(suffix_out[:, -self.model.action_horizon:])
            return x_t + dt * velocity, time + dt

        actions, _ = jax.lax.while_loop(
            lambda carry: carry[1] >= -dt / 2, step, (noise, 1.0),
        )
        return actions
