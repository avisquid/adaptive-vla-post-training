"""Push a locally collected LeRobot dataset to the Hugging Face Hub.

Usage:
    python -m real.scripts.push_dataset_to_hub \
        --root /home/ankile/lrds/teleop_smoketest \
        --repo-id your-hf-username/teleop_smoketest

Requires ``huggingface-cli login`` (or HF_TOKEN in the environment) to have
write access to the target repo. Stop collection after saving at least one
episode. LeRobot 0.3.3 commits via save_episode(), without a finalize() method.
"""

# 
# huggingface-cli upload <user>/meta-slim-up /home/ankile/lrds/meta-green-slim-up

from __future__ import annotations

import argparse
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Local dataset directory (the one containing meta/info.json).",
    )
    p.add_argument(
        "--repo-id",
        required=True,
        help="Target Hub repo id, e.g. 'your-username/my_dataset'.",
    )
    p.add_argument(
        "--private",
        action="store_true",
        help="Create the Hub repo as private.",
    )
    p.add_argument(
        "--no-videos",
        action="store_true",
        help="Skip uploading the videos/ shards (metadata + parquet only).",
    )
    p.add_argument(
        "--branch",
        default=None,
        help="Push to a specific branch instead of the default.",
    )
    p.add_argument(
        "--tags",
        nargs="*",
        default=None,
        help="Optional list of dataset card tags.",
    )
    p.add_argument(
        "--license",
        default="apache-2.0",
        help="Dataset license (default: apache-2.0).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not (args.root / "meta" / "info.json").exists():
        raise SystemExit(
            f"No LeRobot dataset found at {args.root} "
            f"(missing meta/info.json)."
        )

    pending = args.root.parent / f".{args.root.name}.collection-staging" / "pending_commit.json"
    if pending.exists():
        raise SystemExit(f"Inspect the interrupted commit before publishing: {pending}")
    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.root, download_videos=False, video_backend="pyav")
    print(
        f"Loaded dataset at {args.root} "
        f"({dataset.meta.total_episodes} episodes, "
        f"{dataset.meta.total_frames} frames)."
    )
    print(f"Pushing to https://huggingface.co/datasets/{args.repo_id} ...")

    dataset.push_to_hub(
        branch=args.branch,
        tags=args.tags,
        license=args.license,
        push_videos=not args.no_videos,
        private=args.private,
    )
    print("Done.")


if __name__ == "__main__":
    main()
