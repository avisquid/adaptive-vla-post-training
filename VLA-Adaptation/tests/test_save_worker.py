"""The dataset writer process behind the collector; actual pinned LeRobot, run in adaptive."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from real.collection.runtime import Collection
from real.collection.save_worker import ProcessRecorder
from real.collection.teleop import SimulatedEnv
from test_route_recorder import recorder_test_task

pytestmark = pytest.mark.skipif(not importlib.util.find_spec('lerobot'), reason='Run in adaptive to exercise LeRobot')

CFG = OmegaConf.create({'collect': {'simulate': True, 'dry_run': False, 'gripper_open_width': .069,
    'gripper_closed_width': .004, 'task_prompt': 'Follow the tabletop route', 'ui': {'heartbeat_timeout': 30}}})


def writer(root, task, **kwargs):
    return ProcessRecorder(root, 'test/tabletop', task, 50, videos=False, threads=1, simulation=True, **kwargs)


def record_route(runtime):
    """Record the two-edge route up to review, as the operator would."""
    def action(name):
        runtime.command({'action': name, 'client_id': 'test', 'event_id': str(uuid.uuid4()),
                         'revision': runtime.session.revision})
        runtime.process_command()
        assert runtime.session.error is None
    with patch('time.sleep'):
        action('reset')
        action('start')
        for _ in range(2):
            action('simulate_motion')
            for _ in range(12):
                runtime.step()
            runtime.staged.flush()  # Unpaced steps would outrun the staging queue.
            action('goal')
    return action


def test_commits_run_in_the_writer_process_and_resume(tmp_path):
    task, root = recorder_test_task(), tmp_path / 'dataset'
    recorder = writer(root, task, realtime_cpu=31)
    runtime = Collection(CFG, task, SimulatedEnv(50, ['global']), recorder=recorder)
    try:
        assert 31 not in recorder.worker['cpus'] and recorder.worker['pid'] != os.getpid()
        for i in range(2):
            action = record_route(runtime)
            identity = runtime.session.attempt_id
            action('accept')
            runtime.finish_save(wait=True)
            assert runtime.session.counts[0] == recorder.counts[0] == i + 1
            meta = json.loads((root / 'meta/collection/episodes' / f'{identity}.json').read_text())
            assert meta['episode_index'] == i
            assert meta['save_worker']['pid'] == recorder.process.pid
            assert meta['save_worker']['cpu_threads'] == 1
    finally:
        runtime.close()
        recorder.close()
    assert recorder.process.poll() == 0
    resumed = writer(root, task)  # The child released the dataset lock.
    try:
        assert resumed.counts[0] == 2
    finally:
        resumed.close()


@pytest.mark.parametrize('fault', ['missing_frame', 'dead_worker'])
def test_a_failed_save_preserves_the_draft(tmp_path, fault):
    task, root = recorder_test_task(), tmp_path / 'dataset'
    recorder = writer(root, task)
    runtime = Collection(CFG, task, SimulatedEnv(50, ['global']), recorder=recorder)
    try:
        action = record_route(runtime)
        staged = runtime.staged.path
        if fault == 'missing_frame':
            (staged / '0000000.npz').unlink()
        else:
            recorder.process.kill()
            recorder.process.wait()
        action('accept')
        runtime.finish_save(wait=True)
        assert runtime.fatal and 'Dataset save failed' in runtime.session.error
        assert recorder.counts[0] == 0
        assert (staged / 'episode.json').is_file() and (staged / '0000001.npz').is_file()
    finally:
        runtime.close()
        recorder.close()


def test_the_collector_process_never_imports_the_dataset_stack(tmp_path):
    script = '''
import sys
from real.collection.guide import load_task
from real.collection.save_worker import ProcessRecorder
from pathlib import Path
task = load_task(Path('real/config/task/tabletop.yaml'))
writer = ProcessRecorder(sys.argv[1], 'test/tabletop', task, 50, videos=False, simulation=True)
try:
    for module in ('torch', 'datasets', 'lerobot'):
        assert module not in sys.modules, module
finally:
    writer.close()
'''
    project = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, '-c', script, str(tmp_path / 'dataset')], check=True, timeout=120, cwd=project)
