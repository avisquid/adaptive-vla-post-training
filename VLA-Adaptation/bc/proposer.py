"""Small hardware- and framework-independent proposer contract."""
from abc import ABC, abstractmethod


class ProposerActor(ABC):
    """Propose absolute xyz/rotation6d/gripper chunks for any ranking algorithm.

    ``To`` is the trained observation history, ``H`` the available action horizon.
    The planner supplies an isolated NumPy ``rng`` before each proposal request.
    ``candidates`` returns float32[N, horizon, 10] in physical units, without
    clipping, smoothing, scoring, or executing any actions.
    """
    To: int
    H: int

    @abstractmethod
    def reset(self):
        """Reset episode-local policy state."""

    @abstractmethod
    def candidates(self, obs, n, horizon):
        """Generate candidates from the current observation and control-step history."""
