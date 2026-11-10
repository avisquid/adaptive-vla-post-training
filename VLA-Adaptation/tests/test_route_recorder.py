"""Actual pinned LeRobot round trips; run in conda env vla."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import uuid
from unittest.mock import patch

import numpy as np

from real.collection.guide import load_task
from real.collection.recorder import Recorder
from real.collection.runtime import Collection
from real.collection.teleop import SimulatedEnv
from omegaconf import OmegaConf
import yaml


def recorder_test_task():
    """Keep the two-edge test fixture independent of the live collection order."""
    path = Path(__file__).resolve().parents[1] / 'real/config/task/tabletop.yaml'
    data = yaml.safe_load(path.read_text())
    data['edge_combinations'] = [[[0, 1, 3], 50]] + [
        route for route in data['edge_combinations'] if route[0] != [0, 1, 3]]
    return load_task(path, data=data)


@unittest.skipUnless(importlib.util.find_spec('lerobot'), 'Run in vla to exercise LeRobot')
class RecorderRoundTripTests(unittest.TestCase):
    def test_runtime_commits_full_route_with_paused_arrivals_and_resumes_counts(self):
        task = recorder_test_task()
        cfg = OmegaConf.create({'collect': {'simulate':True, 'dry_run':False, 'gripper_open_width':.069,
            'gripper_closed_width':.004, 'task_prompt':'Follow the tabletop route', 'ui':{'heartbeat_timeout':30}}})
        with tempfile.TemporaryDirectory() as tmp:
            writer = Recorder(Path(tmp)/'dataset','test/tabletop',task,50,videos=False,threads=1,simulation=True)
            runtime = Collection(cfg,task,SimulatedEnv(50,['global']),recorder=writer)
            def action(name):
                runtime.command({'action':name,'client_id':'test','event_id':str(uuid.uuid4()),'revision':runtime.session.revision})
                runtime.process_command()
                if name == 'accept':
                    runtime.finish_save(wait=True)
                self.assertIsNone(runtime.session.error)
            try:
                with patch('time.sleep'):
                    action('reset'); action('start')
                    identity = runtime.session.attempt_id
                    for _ in range(2):
                        for _ in range(4): runtime.step()
                        before = runtime.session.frames
                        action('simulate_motion')
                        for _ in range(12): runtime.step()
                        self.assertEqual(runtime.session.frames-before,10)
                        for _ in range(4): runtime.step()
                        self.assertEqual(runtime.session.frames-before,10)
                        action('goal')
                        runtime.staged.flush()
                action('accept')
                self.assertEqual(runtime.session.counts[0],1)
                self.assertEqual(runtime.session.mode,'setup')
                meta=json.loads((writer.meta/'episodes'/f'{identity}.json').read_text())
                self.assertEqual(meta['segments'],[{'edge':[0,1],'start':0,'end':10},{'edge':[1,3],'start':10,'end':20}])
                self.assertEqual(len(list((writer.meta/'arrivals'/identity).glob('*.npz'))),2)
                self.assertEqual(meta['reset']['state_id'],0)
            finally:
                runtime.close()
                if runtime.staged: runtime.staged.close()
                writer.close()
            reopened=Recorder(Path(tmp)/'dataset','test/tabletop',task,50,videos=False,simulation=True)
            self.assertEqual(reopened.counts[0],1)

    def test_create_save_reopen_append_and_portable_annotations(self):
        for videos in [False, True]:
            with self.subTest(videos=videos), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / 'dataset'
                task = recorder_test_task()
                for attempt in range(2):
                    writer = Recorder(root, 'test/tabletop', task, 50, videos=videos, threads=1, simulation=True)
                    self.assertEqual(writer.counts[0], attempt)
                    identity = str(uuid.uuid4())
                    staged = writer.start(identity)
                    for i in range(8):
                        staged.add({'observation.state':np.array([.4+i*.001,0,.3,1,0,0,0,1,0,.069],dtype=np.float32),
                                    'action':np.array([.4+(i+1)*.001,0,.3,1,0,0,0,1,0,.069],dtype=np.float32),
                                    'observation.images.global':np.full((32,32,3),i*15,dtype=np.uint8),
                                    'timing.capture_elapsed_s':np.array([i*.1],dtype=np.float64),
                                    'timing.control_tick_index':np.array([i*5],dtype=np.int64)})
                    meta={'attempt_id':identity,'route_index':0,'goals':[{'boundary_index':3},{'boundary_index':8}],
                          'events':[], 'segments':[{'edge':[0,1],'start':0,'end':3},{'edge':[1,3],'start':3,'end':8}]}
                    try:
                        writer.commit(staged,meta,{'global':(32,32,3)},'Follow the tabletop route')
                    finally:
                        staged.close()
                        writer.close()
                from lerobot.datasets.lerobot_dataset import LeRobotDataset
                dataset=LeRobotDataset('test/tabletop',root=root,download_videos=False,video_backend='pyav')
                self.assertEqual(dataset.meta.total_episodes,2)
                self.assertEqual(dataset.meta.total_frames,16)
                first=dataset[0]
                self.assertEqual(tuple(first['observation.state'].shape),(10,))
                self.assertEqual(first['annotation.edge_from'].item(),0)
                self.assertEqual(dataset[2]['annotation.goal_reached'].item(),1)
                self.assertEqual(dataset[3]['annotation.edge_from'].item(),1)
                self.assertEqual(dataset[7]['annotation.reached_state_id'].item(),3)
                snapshot=json.loads((root/'meta/collection/task.json').read_text())
                self.assertEqual(len(snapshot['states']),7)
                self.assertTrue(all((root/'meta/collection'/s['image']).is_file() for s in snapshot['states']))
                self.assertEqual(len(list((root/'meta/collection/episodes').glob('*.json'))),2)
                self.assertFalse((root.parent/'.dataset.collection-staging/pending_commit.json').exists())

    def test_rollout_frames_store_their_own_prompt_as_the_lerobot_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'dataset'
            task = recorder_test_task()
            writer = Recorder(root, 'test/tabletop', task, 50, videos=False, threads=1, simulation=True, intervention=True)
            identity = str(uuid.uuid4())
            staged = writer.start(identity)
            prompts = ['Place the tomato in the pan.'] * 3 + ['Place the pan on the stove.'] * 5
            for i, prompt in enumerate(prompts):
                staged.add({'observation.state':np.zeros(10,dtype=np.float32), 'action':np.zeros(10,dtype=np.float32),
                            'observation.images.global':np.zeros((32,32,3),dtype=np.uint8),
                            'timing.capture_elapsed_s':np.array([i*.1]), 'timing.control_tick_index':np.array([i]),
                            'intervention':np.array([int(i >= 6)]), 'task':np.array(prompt)})
            meta = {'attempt_id':identity,'route_index':0,'goals':[{'boundary_index':3},{'boundary_index':8}],
                    'events':[], 'segments':[{'edge':[0,1],'start':0,'end':3},{'edge':[1,3],'start':3,'end':8}]}
            try:
                writer.commit(staged, meta, {'global':(32,32,3)}, 'Session prompt, unused for rollout frames')
            finally:
                staged.close()
                writer.close()
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            dataset = LeRobotDataset('test/tabletop', root=root, download_videos=False)
            self.assertEqual([dataset[i]['task'] for i in range(8)], prompts)
            self.assertEqual(set(dataset.meta.tasks.values()), set(prompts))

    def test_pending_commit_blocks_append(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = recorder_test_task()
            writer=Recorder(Path(tmp)/'dataset','test/tabletop',task,50)
            (writer.staging/'pending_commit.json').write_text('{}')
            with self.assertRaisesRegex(RuntimeError,'commit was interrupted'):
                Recorder(Path(tmp)/'dataset','test/tabletop',task,50)

    def test_direct_video_keeps_every_frame_and_original_sampled_statistics(self):
        import av
        from lerobot.datasets.compute_stats import get_feature_stats, sample_indices
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        count = 128  # Exceeds LeRobot's minimum statistics sample of 100 frames.
        sampled = sample_indices(count)
        self.assertLess(len(sampled), count)
        shapes = {'wrist': (32, 32, 3), 'global': (24, 40, 3)}
        images = {name: np.stack([np.full(shape, i * 2, dtype=np.uint8)
                                 for i in range(count)]) for name, shape in shapes.items()}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'dataset'
            task = recorder_test_task()
            writer = Recorder(root, 'test/tabletop', task, 50, threads=1, simulation=True)
            identity = str(uuid.uuid4())
            staged = writer.start(identity)
            try:
                for i in range(count):
                    staged.add({'observation.state': np.zeros(10, dtype=np.float32),
                                'action': np.zeros(10, dtype=np.float32),
                                'timing.capture_elapsed_s': np.array([i / 50], dtype=np.float64),
                                'timing.control_tick_index': np.array([i], dtype=np.int64),
                                **{f'observation.images.{name}': frames[i]
                                   for name, frames in images.items()}})
                    if i % 8 == 7:
                        staged.flush()
                meta = {'attempt_id': identity, 'route_index': 0,
                        'goals': [{'boundary_index': 64}, {'boundary_index': count}]}
                writer._open(shapes)
                encode = writer.dataset.encode_episode_videos
                def verify_sampled_pngs(episode_index):
                    self.assertEqual(len(list(root.glob('images/**/*.png'))), len(sampled) * len(shapes))
                    return encode(episode_index)
                with patch.object(writer.dataset, 'encode_episode_videos', side_effect=verify_sampled_pngs):
                    record = writer.commit(staged, meta, shapes, 'Test every video frame')
                self.assertEqual(record['encoding']['preset'], 'veryfast')
                self.assertGreater(record['save_timing_s']['video_encode'], 0)
            finally:
                staged.close()
                writer.close()
            dataset = LeRobotDataset('test/tabletop', root=root, download_videos=False, video_backend='pyav')
            self.assertEqual(len(dataset), count)
            unsampled = next(i for i in range(count) if i not in sampled)
            for name, frames in images.items():
                key = f'observation.images.{name}'
                stats = get_feature_stats(frames[sampled].transpose(0, 3, 1, 2), axis=(0, 2, 3), keepdims=True)
                for stat, value in stats.items():
                    expected = value if stat == 'count' else (value / 255).squeeze(0)
                    np.testing.assert_allclose(dataset.meta.stats[key][stat], expected, atol=1e-12, rtol=1e-12)
                with av.open(str(root / dataset.meta.get_video_file_path(0, key))) as video:
                    decoded = [frame.to_ndarray(format='rgb24') for frame in video.decode(video=0)]
                self.assertEqual(len(decoded), count)
                # The frames omitted from PNG statistics still appear, in order, in the video.
                np.testing.assert_allclose(np.mean(decoded, axis=(1, 2, 3)), np.arange(count) * 2, atol=4)
                self.assertAlmostEqual(dataset[unsampled][key].mean().item(), unsampled * 2 / 255, delta=4/255)
