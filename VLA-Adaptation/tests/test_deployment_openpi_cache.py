"""Shared-prefix sampling agrees with upstream integration and refreshes per observation.

Run on CPU (JAX_PLATFORMS=cpu); these tests do not load weights or devices.
"""
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
nnx = pytest.importorskip("flax.nnx")
pytest.importorskip("openpi")

from openpi.models import model as model_lib
from openpi.models.pi0 import Pi0
from openpi.shared import nnx_utils
from real.policies.openpi_cache import SharedPrefixSampler
from real.policies.openpi import OpenpiPolicy


@pytest.fixture(autouse=True)
def cpu_only():
    with jax.default_device(jax.devices("cpu")[0]):
        yield


class TinyLLM(nnx.Module):
    def __call__(self, xs, *, mask, positions, kv_cache=None, adarms_cond=None):
        prefix, suffix = xs
        if prefix is not None:
            k = jnp.stack([prefix[..., None, :], prefix[..., None, :] + .1])
            return (prefix, None), (k, k * .5)
        k, v = kv_cache
        assert k.shape[1] == suffix.shape[0]
        # Depend on the per-layer cache, attention mask, and token positions so
        # missing masking or broadcasting along the wrong axis changes actions.
        allowed = mask[:, 0, :k.shape[2]]
        features = ((k + v).mean(axis=(0, 3, 4)) * allowed).sum(axis=1)
        result = .4 * suffix + .03 * features[:, None, None] + .01 * positions[..., None]
        return (None, result), kv_cache


class TinyModel(nnx.Module):
    action_horizon = 3
    action_dim = 10

    def __init__(self):
        self.PaliGemma = nnx.Dict(llm=TinyLLM())

    def embed_prefix(self, obs):
        rgb = jnp.stack([x.mean(axis=(1, 2, 3)) for x in obs.images.values()], axis=1)
        tokens = jnp.broadcast_to(rgb[..., None], (*rgb.shape, self.action_dim))
        mask = jnp.stack(list(obs.image_masks.values()), axis=1)
        return tokens, mask, jnp.zeros(rgb.shape[1], dtype=bool)

    def embed_suffix(self, obs, noise, time):
        tokens = noise + .03 * obs.state[:, None, :] + .1 * time[:, None, None]
        return (tokens, jnp.ones(tokens.shape[:2], dtype=bool),
                jnp.array([True, False, False]), None)

    def action_out_proj(self, x):
        return x

    def sample_actions(self, rng, observation, **kwargs):
        # The reference loop comes from installed OpenPI, not this adapter.
        return Pi0.sample_actions(self, rng, observation, **kwargs)


def inputs(value=.2):
    return {
        "image": {name: np.full((224, 224, 3), value + i, np.float32)
                  for i, name in enumerate(model_lib.IMAGE_KEYS)},
        "image_mask": {name: np.array(i != 1) for i, name in enumerate(model_lib.IMAGE_KEYS)},
        "state": np.arange(10, dtype=np.float32) * .01,
    }


@pytest.mark.parametrize("batch,steps", [(1, 1), (5, 3), (128, 10)])
def test_cached_sampler_matches_upstream_for_identical_noise(batch, steps):
    model = TinyModel()
    adapter = SharedPrefixSampler(model)
    encode = nnx_utils.module_jit(adapter.encode)
    sample = nnx_utils.module_jit(adapter.sample)
    reference = nnx_utils.module_jit(model.sample_actions)
    single = model_lib.Observation.from_dict(jax.tree.map(lambda x: jnp.asarray(x)[None], inputs()))
    repeated = jax.tree.map(lambda x: jnp.broadcast_to(x, (batch, *x.shape[1:])), single)
    noise = jnp.asarray(np.random.default_rng(0).normal(size=(batch, 3, 10)).astype(np.float32))
    context = encode(single)
    expected = reference(jax.random.key(0), repeated, noise=noise, num_steps=steps)
    actual = sample(context, noise=noise, num_steps=steps)
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
    if batch > 1:
        assert not np.array_equal(actual[0], actual[1])


def make_policy(batch):
    model = TinyModel()
    adapter = SharedPrefixSampler(model)
    policy = OpenpiPolicy.__new__(OpenpiPolicy)
    policy.H, policy.D, policy.batch, policy._batched_ok = 3, 10, batch, True
    policy.rng = np.random.default_rng(0)
    policy.policy = SimpleNamespace(
        _input_transform=lambda x: x, _sample_kwargs={"num_steps": 3},
        _output_transform=lambda x: {"actions": x["actions"] * 2 + x["state"]},
        _rng=jax.random.key(0),
    )
    policy._obs_dict = lambda x: x
    policy._encode_prefix = nnx_utils.module_jit(adapter.encode)
    policy._sample_cached = nnx_utils.module_jit(adapter.sample)
    return policy


def test_candidate_batches_share_one_encoding_and_new_observation_refreshes_it():
    policy = make_policy(2)
    encoded, decoded = [], []
    encode, sample = policy._encode_prefix, policy._sample_cached

    def record_encode(obs):
        context = encode(obs)
        encoded.append(context)
        return context

    def record_sample(context, **kwargs):
        decoded.append(context)
        return sample(context, **kwargs)

    policy._encode_prefix, policy._sample_cached = record_encode, record_sample
    noise = np.random.default_rng(7).normal(size=(5, 3, 10)).astype(np.float32)
    first = policy.candidates(inputs(), 5, 2, noise=noise)
    assert first.shape == (5, 2, 10)  # Truncate after sampling the trained horizon.
    assert len(encoded) == 1 and len(decoded) == 3  # Includes the last partial batch.
    assert all(context is encoded[0] for context in decoded)
    assert policy.last_sampling_stats == {
        "prefix_encode_calls": 1, "prefix_encode_rows": 1,
        "sampler_batch_calls": 3, "sampler_rows_returned": 5,
    }
    policy.batch = 128
    together = policy.candidates(inputs(), 5, 2, noise=noise)
    np.testing.assert_allclose(together, first, atol=1e-6, rtol=1e-6)
    assert policy.last_sampling_stats["sampler_batch_calls"] == 1
    changed = policy.candidates(inputs(.8), 5, 2, noise=noise)
    assert len(encoded) == 3
    assert decoded[-1] is encoded[-1]
    assert not np.allclose(changed, first)
    # Output transforms are still applied once, after flow sampling.
    reference = np.asarray(sample(encoded[0], noise=jnp.asarray(noise), num_steps=3))
    np.testing.assert_allclose(first, (reference * 2 + inputs()["state"])[:, :2], atol=1e-6)
