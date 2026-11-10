"""The correction frames of one rollout dataset: where the operator drove with the SpaceMouse.

The collector labels a frame intervention=1 when the SpaceMouse drove it or the rollout
was paused; the episode's handoffs say when it was paused. A training sample is a frame
whose whole action chunk is corrections from one episode, so no policy action or paused
hold is ever a training target.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def paused_ranges(handoffs, length):
    """The [start, end) frame ranges in which a rollout was paused."""
    handoffs = handoffs or []  # Absent in episodes recorded before handoffs were logged.
    ranges = []
    for i, handoff in enumerate(handoffs):
        if handoff["controller"] == "paused":
            end = handoffs[i + 1]["frame"] if i + 1 < len(handoffs) else length
            ranges.append((handoff["frame"], end))
    return ranges


def correction_indices(dataset, horizon):
    """Dataset indices of the frames that start an action chunk of `horizon` corrections."""
    dataset = Path(dataset)
    info = json.loads((dataset / "meta/info.json").read_text())
    if "intervention" not in info["features"]:
        raise ValueError(f"{dataset} has no intervention label: DAgger needs a rollout dataset")
    handoffs = {}
    for path in (dataset / "meta/collection/episodes").glob("*.json"):
        record = json.loads(path.read_text())
        handoffs[record["episode_index"]] = record.get("handoffs")
    frames = pd.concat([pd.read_parquet(path, columns=["index", "episode_index", "frame_index", "intervention"])
                        for path in sorted((dataset / "data").glob("*/*.parquet"))])
    selected = []
    for episode, rows in frames.groupby("episode_index"):
        rows = rows.sort_values("frame_index")
        correction = rows["intervention"].to_numpy() == 1
        for start, end in paused_ranges(handoffs.get(episode), len(rows)):
            correction[start:end] = False
        if len(correction) < horizon:
            continue
        # The chunk starting at frame i is frames i .. i + horizon - 1.
        whole_chunk = np.lib.stride_tricks.sliding_window_view(correction, horizon).all(axis=1)
        selected.extend(rows["index"].to_numpy()[:len(whole_chunk)][whole_chunk].tolist())
    return sorted(selected)
