"""The synthetic policy that stands in for a trained model in the deployment and rollout tests.

worker_config accepts kind "mock" only with simulate=True.
"""
from omegaconf import OmegaConf

MOCK_ROLLOUT = {
    "kind": "mock", "checkpoint": None, "config_name": None,
    "prompt": "Follow the displayed cooking state sequence", "cameras": ["global", "wrist"],
    "device": "cpu", "execute_k": 4, "batch_size": 1, "candidate_horizon": 8, "plan_timeout_s": 5.0,
    "inference": {"python": None, "environment": {}, "timeout_s": 60, "warmup_steps": 1},
}


def mock_rollout(**overrides):
    return OmegaConf.merge(OmegaConf.create(MOCK_ROLLOUT), overrides)
