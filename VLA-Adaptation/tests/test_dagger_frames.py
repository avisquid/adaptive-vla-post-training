"""dagger.frames on a small rollout dataset written like the collector's; no model is loaded."""
import json

import pandas as pd
import pytest

from dagger.frames import correction_indices


def write_dataset(root, episodes, intervention=True):
    """episodes: [(per-frame intervention labels, handoffs)], stored as LeRobot does."""
    features = {"action": {}, **({"intervention": {}} if intervention else {})}
    (root / "meta/collection/episodes").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "meta/info.json").write_text(json.dumps({"features": features}))
    start = 0
    for episode, (labels, handoffs) in enumerate(episodes):
        pd.DataFrame({"index": range(start, start + len(labels)), "episode_index": episode,
                      "frame_index": range(len(labels)), "intervention": labels}).to_parquet(
            root / f"data/chunk-000/episode_{episode:06d}.parquet")
        (root / f"meta/collection/episodes/{episode}.json").write_text(
            json.dumps({"episode_index": episode, "handoffs": handoffs}))
        start += len(labels)
    return root


def test_only_whole_chunks_of_corrections_are_selected(tmp_path):
    # Policy 0-9, SpaceMouse 10-19, paused 20-24 (also labelled 1), SpaceMouse 25-29.
    rollout = ([0] * 10 + [1] * 20, [{"controller": "paused", "frame": 10}, {"controller": "human", "frame": 10},
                                     {"controller": "paused", "frame": 20}, {"controller": "human", "frame": 25}])
    teleop = ([1] * 5, None)  # A pure SpaceMouse episode, metadata without handoffs.
    dataset = write_dataset(tmp_path / "rollouts", [rollout, teleop])
    assert correction_indices(dataset, horizon=4) == [*range(10, 17), 25, 26, 30, 31]
    assert correction_indices(dataset, horizon=6) == [*range(10, 15)]  # Too short: 25-29 and the teleop episode.


def test_a_teleop_dataset_is_refused(tmp_path):
    dataset = write_dataset(tmp_path / "teleop", [([0] * 5, None)], intervention=False)
    with pytest.raises(ValueError, match="rollout dataset"):
        correction_indices(dataset, horizon=4)
