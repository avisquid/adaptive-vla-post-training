"""Pinned LeRobot 0.3.3 writer with direct staged-frame H.264 encoding.

Keep stock tables, statistics, metadata and indexing. Only materialize the PNGs
that LeRobot samples for image statistics; encode all video frames from NPZ.
"""
import shutil
import time

import av
import numpy as np
from lerobot.datasets.compute_stats import sample_indices
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import write_info


class CollectionVideoDataset(LeRobotDataset):
    def prepare_staged_video(self, episode, settings):
        self._staged_video = (episode.path, episode.count)
        self._stats_indices = set(sample_indices(episode.count))
        self._encoding = settings
        self.encoding_seconds = 0.

    def _save_image(self, image, fpath):
        if getattr(self, '_staged_video', None) is not None:
            index = self.episode_buffer['size']
            is_video = any(fpath == self._get_image_file_path(
                self.episode_buffer['episode_index'], key, index) for key in self.meta.video_keys)
            if is_video and index not in self._stats_indices:
                return
        super()._save_image(image, fpath)

    def encode_episode_videos(self, episode_index):
        if getattr(self, '_staged_video', None) is None:
            return super().encode_episode_videos(episode_index)
        source, count = self._staged_video
        started = time.perf_counter()
        old_level = av.logging.get_level()
        av.logging.set_level(av.logging.ERROR)
        try:
            for key in self.meta.video_keys:
                path = self.root / self.meta.get_video_file_path(episode_index, key)
                if path.exists():
                    raise FileExistsError(f'Refusing to overwrite episode video: {path}')
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix('.encoding.mp4')
                h, w, _ = self.features[key]['shape']
                options = {'preset': self._encoding['preset'], 'crf': str(self._encoding['crf']),
                           'g': str(self._encoding['gop'])}
                with av.open(str(temporary), 'w') as container:
                    stream = container.add_stream('libx264', rate=self.fps, options=options)
                    stream.width, stream.height, stream.pix_fmt = w, h, 'yuv420p'
                    stream.codec_context.thread_count = self._encoding['threads']
                    for index in range(count):
                        with np.load(source / f'{index:07d}.npz', allow_pickle=False) as row:
                            rgb = row[key]
                        if rgb.shape != (h, w, 3) or rgb.dtype != np.uint8:
                            raise ValueError(f'Invalid staged video frame {index}: {key}')
                        for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format='rgb24')):
                            container.mux(packet)
                    for packet in stream.encode():
                        container.mux(packet)
                temporary.replace(path)
                # The raw NPZ source remains available after successful save.
                img_dir = self._get_image_file_path(episode_index, key, 0).parent
                if img_dir.exists():
                    shutil.rmtree(img_dir)
            if self.meta.video_keys and episode_index == 0:
                self.meta.update_video_info()
                write_info(self.meta.info, self.meta.root)
        finally:
            self.encoding_seconds = time.perf_counter() - started
            av.logging.set_level(old_level)
