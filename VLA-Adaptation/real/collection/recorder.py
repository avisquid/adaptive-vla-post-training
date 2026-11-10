"""Bounded local staging and the pinned LeRobot 0.3.3 writer adapter."""
from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import queue
import shutil
import threading
import time

import numpy as np

from real.collection.dataset_card import dataset_card


AXES = ["x", "y", "z", "r6d_0", "r6d_1", "r6d_2", "r6d_3", "r6d_4", "r6d_5", "gripper"]


def check_writer_dependencies():
    if version("lerobot") != "0.3.3" or version("datasets") != "3.6.0":
        raise RuntimeError("Use conda env vla with lerobot==0.3.3 and datasets==3.6.0; see requirements-collection.txt")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def features(camera_shapes, videos, intervention=False):
    result = {key: {"dtype": "float32", "shape": (10,), "names": AXES}
              for key in ("observation.state", "action")}
    for name, shape in camera_shapes.items():
        result[f"observation.images.{name}"] = {"dtype": "video" if videos else "image", "shape": tuple(shape), "names": ["height", "width", "channel"]}
    for name in ("route_index", "segment_index", "edge_from", "edge_to", "goal_reached", "reached_state_id"):
        result[f"annotation.{name}"] = {"dtype": "int64", "shape": (1,), "names": None}
    result["timing.control_tick_index"] = {"dtype": "int64", "shape": (1,), "names": None}
    result["timing.capture_elapsed_s"] = {"dtype": "float64", "shape": (1,), "names": None}
    result["next.reward"] = {"dtype": "float32", "shape": (1,), "names": None}
    if intervention:
        # Rollout datasets: 0 = policy action, 1 = human (SpaceMouse) action.
        result["intervention"] = {"dtype": "int64", "shape": (1,), "names": None}
    return result


class StagedEpisode:
    def __init__(self, root, identity, queue_size=16):
        self.path = Path(root) / identity
        self.path.mkdir(parents=True, exist_ok=False)
        self.count = 0
        self.error = None
        self.queue = queue.Queue(maxsize=queue_size)
        self.thread = threading.Thread(target=self._write, name="episode-staging", daemon=True)
        self.thread.start()
        self.closed = False

    def _write(self):
        while True:
            item = self.queue.get()
            try:
                if item is None:
                    return
                index, frame = item
                if self.error is None:
                    path = self.path / f"{index:07d}.npz"
                    with path.with_suffix(".tmp").open("wb") as stream:
                        np.savez(stream, **frame)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(path.with_suffix(".tmp"), path)
            except Exception as exc:
                self.error = exc
            finally:
                self.queue.task_done()

    def add(self, frame):
        if self.closed or self.error:
            raise RuntimeError(f"Staging unavailable: {self.error}")
        try:
            self.queue.put_nowait((self.count, {k: np.asarray(v).copy() for k, v in frame.items()}))
        except queue.Full as exc:
            raise RuntimeError("Staging writer fell behind; attempt interrupted") from exc
        self.count += 1

    def checkpoint(self, metadata):
        atomic_json(self.path / "episode.json", metadata)

    def flush(self):
        self.queue.join()
        if self.error:
            raise RuntimeError(f"Staging failed: {self.error}")

    def close(self):
        if not self.closed:
            self.queue.put(None)
            self.thread.join()
            self.closed = True
        if self.error:
            raise RuntimeError(f"Staging failed: {self.error}")


class Recorder:
    def __init__(self, root, repo_id, task, fps, videos=True, threads=4, simulation=False, encoding=None,
                 intervention=False):
        from real.dataset.locking import DatasetLock
        self._dataset_lock = None
        # Checked before locking too, so an interrupted commit is reported even
        # while another process holds the dataset.
        pending = Path(root).parent / f".{Path(root).name}.collection-staging/pending_commit.json"
        if pending.exists():
            raise RuntimeError(f"An earlier commit was interrupted. Preserve and inspect {pending.parent}; automatic append is disabled.")
        self._dataset_lock = DatasetLock(root)
        self._dataset_lock.__enter__()
        try:
            self._initialize(root, repo_id, task, fps, videos, threads, simulation, encoding, intervention)
        except BaseException:
            self._dataset_lock.close()
            raise

    def _initialize(self, root, repo_id, task, fps, videos, threads, simulation, encoding, intervention=False):
        self.root = Path(root)
        self.repo_id = repo_id
        self.task = task
        self.fps, self.videos, self.threads = fps, videos, threads
        self.simulation = simulation
        self.intervention = bool(intervention)
        self.encoding = {'preset': 'veryfast', 'crf': 30, 'gop': 2, 'threads': 2, **(encoding or {})}
        if self.encoding['preset'] not in {'medium', 'fast', 'veryfast', 'superfast', 'ultrafast'}:
            raise ValueError('Unsupported video encoding preset')
        for key, low, high in [('crf', 0, 51), ('gop', 1, 250), ('threads', 1, 8)]:
            value = self.encoding[key]
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f'video_encoding.{key} must be an integer in [{low}, {high}]')
        self.staging = self.root.parent / f".{self.root.name}.collection-staging"
        self.staging.mkdir(parents=True, exist_ok=True)
        self.meta = self.root / "meta/collection"
        self.dataset = None
        self.counts = [0] * len(task["routes"])
        if (self.staging / "pending_commit.json").exists():
            raise RuntimeError(f"An earlier commit was interrupted. Preserve and inspect {self.staging}; automatic append is disabled to avoid duplicate episodes.")
        if (self.root / "meta/info.json").exists() and not (self.meta / "task.json").exists():
            raise ValueError("Existing dataset lacks route metadata. Choose a new repo_id.")
        if (self.root / "meta/info.json").exists():
            has_intervention = "intervention" in json.loads((self.root / "meta/info.json").read_text())["features"]
            if has_intervention != self.intervention:
                raise ValueError(
                    "Rollout collection needs a dataset with the per-frame intervention label; choose a new "
                    "repo_id for rollouts." if self.intervention else
                    "This is a rollout dataset (it has an intervention label); launch with a rollout preset "
                    "or choose another repo_id for teleop demonstrations.")
        if (self.meta / "task.json").exists():
            saved = json.loads((self.meta / "task.json").read_text())
            if saved["fingerprint"] != task["fingerprint"] or saved["simulation"] != simulation:
                raise ValueError("Task definition or simulation mode differs from the dataset")
            records = list((self.meta / "episodes").glob("*.json"))
            for path in records:
                record = json.loads(path.read_text())
                self.counts[record["route_index"]] += 1
            info = json.loads((self.root / "meta/info.json").read_text())
            if len(records) != info["total_episodes"]:
                raise RuntimeError("Dataset and route commits disagree; inspect before appending")

    def start(self, identity):
        return StagedEpisode(self.staging, identity)

    def _open(self, camera_shapes):
        check_writer_dependencies()
        from real.collection.video_writer import CollectionVideoDataset as LeRobotDataset
        expected = features(camera_shapes, self.videos, self.intervention)
        if (self.root / "meta/info.json").exists():
            info = json.loads((self.root / "meta/info.json").read_text())
            if info["fps"] != self.fps or info["total_episodes"] < 1:
                raise ValueError("Cannot resume an empty/incompatible dataset; inspect the pending commit")
            for key, spec in expected.items():
                found = info["features"].get(key, {})
                if found.get("dtype") != spec["dtype"] or tuple(found.get("shape", ())) != spec["shape"]:
                    raise ValueError(f"Dataset feature mismatch: {key}")
            dataset = LeRobotDataset(self.repo_id, root=self.root, download_videos=False, video_backend="pyav")
            dataset.start_image_writer(num_processes=0, num_threads=self.threads)
        else:
            dataset = LeRobotDataset.create(repo_id=self.repo_id, root=self.root, fps=self.fps,
                        features=expected, robot_type="fr3_agilex", use_videos=self.videos,
                        image_writer_threads=self.threads, video_backend="pyav")
            snapshot = json.loads(json.dumps(self.task))
            snapshot["simulation"] = self.simulation
            snapshot["recording_contract"] = "quasi-static taken transitions; nominal timestamp=retained index/fps; rotation uses first two columns"
            snapshot["reward_states"] = self.task.get("reward_states", [])
            if self.intervention:
                snapshot["intervention_contract"] = ("intervention=0 for policy actions and 1 for human SpaceMouse actions "
                                                     "or a paused hold; a rollout may pause and resume any number of times "
                                                     "(handoffs in episode metadata)")
            snapshot["reward_contract"] = "next.reward=1 on the last retained transition before a goal marker reaching a reward_states state ID; 0 on every other row"
            for state in snapshot["states"]:
                source = Path(state.pop("image_path"))
                relative = f"states/{state['id']}{source.suffix.lower()}"
                dest = self.meta / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, dest)
                state["image"] = relative
                state.pop("image_url", None)
                state["image_sha256"] = hashlib.sha256(dest.read_bytes()).hexdigest()
            atomic_json(self.meta / "task.json", snapshot)
            (self.root / "README.md").write_text(dataset_card(
                self.repo_id, snapshot, self.fps, camera_shapes, self.videos, self.simulation))
        self.dataset = dataset

    def commit(self, episode, metadata, camera_shapes, prompt):
        started = time.perf_counter()
        episode.flush()
        atomic_json(self.staging / "pending_commit.json", {"episode_uuid": metadata["attempt_id"], "staging": str(episode.path), "expected_episode_index": sum(self.counts)})
        if self.dataset is None:
            self._open(camera_shapes)
        dataset = self.dataset
        if self.videos:
            dataset.prepare_staged_video(episode, self.encoding)
        opened = time.perf_counter()
        episode_index = int(dataset.meta.total_episodes)
        boundaries = metadata["goals"]
        segment = 0
        route = self.task["routes"][metadata["route_index"]]["states"]
        for i in range(episode.count):
            while i >= boundaries[segment]["boundary_index"]:
                segment += 1
            with np.load(episode.path / f"{i:07d}.npz", allow_pickle=False) as stored:
                frame = {key: stored[key] for key in stored.files}
            reached = i == boundaries[segment]["boundary_index"] - 1
            values = {"route_index": metadata["route_index"], "segment_index": segment,
                      "edge_from": route[segment], "edge_to": route[segment+1],
                      "goal_reached": int(reached), "reached_state_id": route[segment+1] if reached else -1}
            frame.update({f"annotation.{key}": np.array([value], dtype=np.int64) for key, value in values.items()})
            frame["next.reward"] = np.array(
                [float(reached and route[segment + 1] in self.task.get("reward_states", []))], dtype=np.float32)
            # Rollout frames carry their own prompt; teleop frames use the session prompt.
            task = str(frame.pop("task")) if "task" in frame else prompt
            dataset.add_frame(frame, task=task)
        replayed = time.perf_counter()
        try:
            dataset.save_episode()
        finally:
            dataset._staged_video = None
        metadata = {**metadata, "episode_index": episode_index, "accepted": True, "route_completed": True,
                    "final_state_id": route[-1], "task_outcome": self.task["states"][route[-1]]["outcome"],
                    "encoding": ({**self.encoding, "codec": "h264", "pixel_format": "yuv420p",
                                  "source": "staged_npz"} if self.videos else None),
                    "save_timing_s": {"open": opened - started, "replay": replayed - opened,
                                      "finalize": time.perf_counter() - replayed,
                                      "video_encode": getattr(dataset, 'encoding_seconds', 0.)}}
        arrival_dir = self.meta / "arrivals" / metadata["attempt_id"]
        arrival_dir.mkdir(parents=True, exist_ok=True)
        for source in episode.path.glob("arrival-*.npz"):
            shutil.copyfile(source, arrival_dir / source.name)
        atomic_json(self.meta / "episodes" / f"{metadata['attempt_id']}.json", metadata)
        episode.checkpoint(metadata)
        (self.staging / "pending_commit.json").unlink()
        self.counts[metadata["route_index"]] += 1
        return metadata

    def close(self):
        try:
            if self.dataset is not None:
                self.dataset.stop_image_writer()
        finally:
            if self._dataset_lock is not None:
                self._dataset_lock.close()
