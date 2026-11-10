"""Render a rollout dataset's episodes showing who drives each frame and what it commands.

Each output frame has the recorded cameras side by side, framed in the controller's
colour (policy blue, SpaceMouse orange, paused grey), the commanded setpoint motion and
gripper, and a timeline of the episode with a green band over the DAgger samples: the
frames whose whole action chunk is corrections (dagger.frames), which training starts from.
The SpaceMouse itself is not recorded: the command is the change of the recorded absolute
action (the setpoint) over the last 0.1 s, in the robot base frame.

Run from VLA-Adaptation, in rsm or adaptive:

    python -m dagger.overlay ~/lrds/DAgger_part1                 # every episode
    python -m dagger.overlay ~/lrds/DAgger_part1 --episodes 0 7  # some of them

Videos go to <dataset>/overlays/episode_<index>.mp4. LeRobot and OpenPI read only the paths
in meta/info.json, so the folder does not change the dataset.
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd

from dagger.frames import correction_indices
from real.agilex import GRIPPER_OPEN_WIDTH
from real.rotation_utils import rot6d_to_matrix

COLORS = {"policy": (66, 135, 245), "human": (255, 140, 0), "paused": (140, 140, 140)}  # RGB
LABELS = {"policy": "POLICY", "human": "SPACEMOUSE", "paused": "PAUSED"}
TRAINED = (60, 220, 90)
HEADER, BORDER, TIMELINE, INSET = 92, 8, 30, 76
FONT = cv2.FONT_HERSHEY_SIMPLEX


def controllers(intervention, handoffs):
    """Who drove each frame: the handoffs, and the intervention label before the first one."""
    names = np.where(intervention == 1, "human", "policy").astype(object)
    handoffs = handoffs or []  # Pure SpaceMouse episodes, or ones recorded before handoffs were logged.
    for i, handoff in enumerate(handoffs):
        end = handoffs[i + 1]["frame"] if i + 1 < len(handoffs) else len(names)
        names[handoff["frame"]:end] = handoff["controller"]
    return names


def commands(action, fps, window_s=0.1):
    """Per frame: setpoint velocity (m/s, base frame) and rotation rate (rad/s), averaged over the
    preceding window so the readout does not flicker at the recording rate."""
    velocity = np.zeros((len(action), 3))
    velocity[1:] = np.diff(action[:, :3], axis=0) * fps
    rotations = [rot6d_to_matrix(a[3:9]) for a in action]
    rate = np.zeros(len(action))
    for t in range(1, len(action)):
        cos = (np.trace(rotations[t] @ rotations[t - 1].T) - 1) / 2
        rate[t] = np.arccos(np.clip(cos, -1, 1)) * fps
    window = max(1, round(window_s * fps))
    trailing = lambda x: pd.DataFrame(x).rolling(window, min_periods=1).mean().to_numpy()
    return trailing(velocity), trailing(rate)[:, 0]


def timeline(names, trained, width):
    """The episode as a strip: controller colour per frame, a green band over trained frames."""
    strip = np.zeros((TIMELINE, width, 3), np.uint8)
    columns = (np.arange(width) * len(names) / width).astype(int)
    for x, t in enumerate(columns):
        strip[8:, x] = COLORS[names[t]]
        if trained[t]:
            strip[:6, x] = TRAINED
    return strip


def text(image, string, origin, scale, color, thickness=1):
    cv2.putText(image, string, origin, FONT, scale, color, thickness, cv2.LINE_AA)


def draw_inset(canvas, x0, velocity, full_scale_mps=0.25):
    """Top-down arrow of the commanded xy velocity (+x up, +y left) and a bar for z. Lengths go
    with the square root of speed, so slow SpaceMouse motion stays visible next to the policy's."""
    y0 = (HEADER - INSET) // 2
    cv2.rectangle(canvas, (x0, y0), (x0 + INSET, y0 + INSET), (70, 70, 70), 1)
    centre = np.array([x0 + INSET // 2 - 6, y0 + INSET // 2])
    radius = INSET // 2 - 10
    scaled = lambda v: np.sign(v) * np.sqrt(np.minimum(np.abs(v) / full_scale_mps, 1))
    xy = np.array([-velocity[1], -velocity[0]])
    speed = np.linalg.norm(xy)
    if speed > 2e-3:
        tip = centre + xy / speed * scaled(speed) * radius
        cv2.arrowedLine(canvas, tuple(centre), tuple(tip.astype(int)), (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.3)
    text(canvas, "+x", (centre[0] - 7, y0 + 11), 0.35, (160, 160, 160))
    bar_x, mid = x0 + INSET - 8, y0 + INSET // 2
    height = int(scaled(velocity[2]) * radius)
    cv2.line(canvas, (bar_x, mid - radius), (bar_x, mid + radius), (70, 70, 70), 1)
    if height:
        cv2.rectangle(canvas, (bar_x - 3, mid), (bar_x + 3, mid - height), (255, 255, 255), -1)
    text(canvas, "z", (bar_x - 3, y0 + INSET - 3), 0.35, (160, 160, 160))


def render(dataset, episode, record, trained_frames, cameras, tile_width, out):
    dataset = Path(dataset)
    info = json.loads((dataset / "meta/info.json").read_text())
    fps = info["fps"]
    chunk = episode // info["chunks_size"]
    frames = pd.read_parquet(dataset / info["data_path"].format(episode_chunk=chunk, episode_index=episode))
    action = np.stack(frames["action"].to_numpy()).astype(np.float64)
    names = controllers(frames["intervention"].to_numpy(), record.get("handoffs"))
    trained = np.zeros(len(frames), bool)
    trained[trained_frames] = True
    velocity, rate = commands(action, fps)

    tile_height = round(tile_width * 3 / 4 / 2) * 2
    width = len(cameras) * tile_width + 2 * BORDER
    height = HEADER + tile_height + 2 * BORDER + TIMELINE
    strip = timeline(names, trained, width)
    sources = [av.open(str(dataset / info["video_path"].format(
        episode_chunk=chunk, video_key=f"observation.images.{camera}", episode_index=episode))) for camera in cameras]
    for source in sources:
        source.streams.video[0].thread_type = "AUTO"
    output = av.open(str(out), "w")
    stream = output.add_stream("libx264", rate=fps)
    stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
    stream.options = {"crf": "23", "preset": "veryfast"}
    try:
        decoded = zip(*(source.decode(video=0) for source in sources))
        for t, images in enumerate(decoded):
            if t >= len(frames):
                break
            controller = names[t]
            canvas = np.full((height, width, 3), 24, np.uint8)
            canvas[HEADER:HEADER + tile_height + 2 * BORDER] = COLORS[controller]
            for i, image in enumerate(images):
                x = BORDER + i * tile_width
                canvas[HEADER + BORDER:HEADER + BORDER + tile_height, x:x + tile_width] = cv2.resize(
                    image.to_ndarray(format="rgb24"), (tile_width, tile_height), interpolation=cv2.INTER_AREA)
                text(canvas, cameras[i], (x + 6, HEADER + BORDER + 18), 0.5, (255, 255, 255))

            text(canvas, LABELS[controller], (12, 38), 1.1, COLORS[controller], 3)
            text(canvas, f"episode {episode}  frame {t}/{len(frames) - 1}  {t / fps:6.2f} s", (12, 66), 0.5,
                 (210, 210, 210))
            v = velocity[t] * 1000
            gripper = action[t, 9]
            state = "OPEN" if gripper > GRIPPER_OPEN_WIDTH / 2 else "CLOSED"
            x = 380
            text(canvas, f"command  v [{v[0]:+6.0f} {v[1]:+6.0f} {v[2]:+6.0f}] mm/s   |v| {np.linalg.norm(v):4.0f}",
                 (x, 30), 0.55, (255, 255, 255))
            text(canvas, f"rotation {np.degrees(rate[t]):5.1f} deg/s   gripper {gripper * 1000:4.1f} mm {state}",
                 (x, 56), 0.55, (255, 255, 255))
            if trained[t]:
                text(canvas, "DAgger sample", (x, 82), 0.5, TRAINED, 2)
            draw_inset(canvas, width - INSET - 12, velocity[t])

            canvas[-TIMELINE:] = strip
            cursor = int(t * width / len(frames))
            cv2.line(canvas, (cursor, height - TIMELINE), (cursor, height - 1), (255, 255, 255), 2)
            for packet in stream.encode(av.VideoFrame.from_ndarray(canvas, format="rgb24")):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    finally:
        output.close()
        for source in sources:
            source.close()
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path, help="Rollout dataset directory, <collect.dataset_root>/<repo_id>")
    parser.add_argument("--episodes", type=int, nargs="*", help="Episode indices (default: all)")
    parser.add_argument("--cameras", nargs="+", default=["global", "wrist", "observer"])
    parser.add_argument("--tile-width", type=int, default=480, help="Width of each camera tile, pixels")
    parser.add_argument("--horizon", type=int, default=50,
                        help="Action chunk length DAgger trains on, for the marked frames (the checkpoint's)")
    parser.add_argument("--out", type=Path, help="Output directory (default: <dataset>/overlays)")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()

    dataset = args.dataset.expanduser().resolve()
    out = (args.out or dataset / "overlays").expanduser()
    out.mkdir(parents=True, exist_ok=True)
    records = {}
    for path in (dataset / "meta/collection/episodes").glob("*.json"):
        record = json.loads(path.read_text())
        records[record["episode_index"]] = record
    index = pd.concat([pd.read_parquet(path, columns=["index", "episode_index", "frame_index"])
                       for path in sorted((dataset / "data").glob("*/*.parquet"))]).set_index("index")
    trained = index.loc[correction_indices(dataset, args.horizon)]
    episodes = args.episodes if args.episodes else sorted(index["episode_index"].unique())

    with ProcessPoolExecutor(args.workers) as pool:
        jobs = [pool.submit(render, dataset, episode, records.get(episode, {}),
                            trained.loc[trained["episode_index"] == episode, "frame_index"].to_numpy(),
                            args.cameras, args.tile_width, out / f"episode_{episode:06d}.mp4")
                for episode in episodes]
        for job in jobs:
            print(job.result(), flush=True)


if __name__ == "__main__":
    main()
