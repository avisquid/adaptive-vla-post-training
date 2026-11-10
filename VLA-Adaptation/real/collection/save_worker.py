"""Persistent dataset writer process. Only sealed paths and metadata cross the pipes.

LeRobot's replay, Parquet writes and video encoding run in this child, started before
the robot, so they never share the collector's GIL, BLAS pools or scheduling with the
native torque loop. The child also stays off robot.realtime_cpu.
"""
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


class ProcessRecorder:
    """Recorder facade; the child owns LeRobot and the dataset lock until close."""

    def __init__(self, root, repo_id, task, fps, videos=True, threads=2, simulation=False,
                 encoding=None, intervention=False, *, cpu_threads=1, startup_timeout_s=120,
                 timeout_s=300, realtime_cpu=None):
        if type(cpu_threads) is not int or cpu_threads < 1:
            raise ValueError('collect.save.cpu_threads must be a positive integer')
        if not 0 < startup_timeout_s < float('inf') or not 0 < timeout_s < float('inf'):
            raise ValueError('Save worker timeouts must be finite and positive')
        self.root = Path(root).expanduser().resolve()
        self.staging = self.root.parent / f'.{self.root.name}.collection-staging'
        self.counts = []
        self.timeout_s = float(timeout_s)
        self.closed = self.broken = False
        self.process = self._request_pipe = self._response_pipe = None
        self._io = ThreadPoolExecutor(max_workers=1, thread_name_prefix='save-pipe')
        request_read, request_write = os.pipe()
        response_read, response_write = os.pipe()
        try:
            project = Path(__file__).resolve().parents[2]
            env = dict(os.environ)
            for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                        'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
                env[key] = str(cpu_threads)
            env['PYTHONPATH'] = str(project) + os.pathsep + env.get('PYTHONPATH', '')
            self._request_pipe = os.fdopen(request_write, 'w')
            self._response_pipe = os.fdopen(response_read, 'r')
            self.process = subprocess.Popen(
                [sys.executable, '-m', 'real.collection.save_worker',
                 '--request-fd', str(request_read), '--response-fd', str(response_write)],
                pass_fds=(request_read, response_write), cwd=project, env=env,
                start_new_session=True)
            os.close(request_read)
            request_read = None
            os.close(response_write)
            response_write = None
            self.worker = self._request({'kind': 'open', 'root': str(self.root),
                'repo_id': repo_id, 'task': task, 'fps': fps, 'videos': videos, 'threads': threads,
                'simulation': simulation, 'encoding': encoding, 'intervention': intervention,
                'cpu_threads': cpu_threads, 'realtime_cpu': realtime_cpu}, startup_timeout_s)
            self.counts = self.worker['counts']
        except BaseException:
            self.broken = True
            self.close()
            raise
        finally:
            for fd in (request_read, response_write):
                if fd is not None:
                    os.close(fd)

    def _exchange(self, message):
        self._request_pipe.write(json.dumps(message, allow_nan=False) + '\n')
        self._request_pipe.flush()
        line = self._response_pipe.readline(4 * 1024 * 1024)
        if not line or not line.endswith('\n'):
            raise RuntimeError(f'Save worker exited or returned an incomplete reply (pid {self.process.pid})')
        reply = json.loads(line)
        if not reply['ok']:
            raise RuntimeError('Save worker: ' + reply['error'] + '\n' + reply.get('traceback', ''))
        return reply['result']

    def _request(self, message, timeout=None):
        if self.closed or self.broken:
            raise RuntimeError('Save worker is unavailable; restart collection after checking the staged demo')
        timeout = self.timeout_s if timeout is None else float(timeout)
        try:
            return self._io.submit(self._exchange, message).result(timeout=timeout)
        except BaseException as error:
            self.broken = True
            self._stop_process()
            if isinstance(error, TimeoutError):
                raise RuntimeError(f'Save worker timed out after {timeout:g}s; staged demo preserved') from error
            raise

    def start(self, identity):
        if self.broken or self.closed or self.process.poll() is not None:
            raise RuntimeError('Save worker stopped; restart collection before recording')
        from real.collection.recorder import StagedEpisode
        return StagedEpisode(self.staging, identity)

    def commit(self, episode, metadata, camera_shapes, prompt):
        """Same contract as Recorder.commit; the parent has already sealed the episode."""
        episode.flush()
        if not episode.closed or episode.error:
            raise ValueError('Seal the staged episode before sending it to the save worker')
        if episode.path.resolve() != self.staging / metadata['attempt_id']:
            raise ValueError('Staged episode identity does not match its metadata')
        result = self._request({'kind': 'commit', 'metadata': metadata, 'frames': episode.count,
                                'camera_shapes': {name: list(shape) for name, shape in camera_shapes.items()},
                                'prompt': prompt})
        self.counts[:] = result['counts']
        return result['metadata']

    def _stop_process(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()

    def close(self):
        if self.closed:
            return
        try:
            if not self.broken and self.process is not None and self.process.poll() is None:
                self._request({'kind': 'close'}, timeout=10)
                self.process.wait(timeout=5)
        except Exception:
            pass  # Save errors are reported by commit; cleanup must still release the child.
        finally:
            self.closed = True
            self._stop_process()
            self._io.shutdown(wait=True, cancel_futures=True)
            for pipe in (self._request_pipe, self._response_pipe):
                if pipe is not None:
                    try:
                        pipe.close()
                    except BrokenPipeError:
                        pass


class _SealedEpisode:
    def __init__(self, path, count):
        self.path, self.count = path, count

    def flush(self):
        pass  # The parent flushed and joined the staging thread before this request.

    def checkpoint(self, metadata):
        from real.collection.recorder import atomic_json
        atomic_json(self.path / 'episode.json', metadata)


def serve(request, response):
    recorder = limits = worker = None
    try:
        for line in request:
            message = json.loads(line)
            try:
                if message['kind'] == 'open':
                    if recorder is not None:
                        raise ValueError('Save worker already initialized')
                    cpus = os.sched_getaffinity(0)
                    cpus.discard(message['realtime_cpu'])
                    if not cpus:
                        raise ValueError('No non-control CPU available for saving')
                    os.sched_setaffinity(0, cpus)
                    from real.collection.recorder import Recorder
                    recorder = Recorder(message['root'], message['repo_id'], message['task'], message['fps'],
                        videos=message['videos'], threads=message['threads'], simulation=message['simulation'],
                        encoding=message['encoding'], intervention=message['intervention'])
                    import torch
                    import pyarrow
                    from threadpoolctl import threadpool_limits
                    threads = message['cpu_threads']
                    torch.set_num_threads(threads)
                    torch.set_num_interop_threads(threads)
                    pyarrow.set_cpu_count(threads)
                    pyarrow.set_io_thread_count(threads)
                    limits = threadpool_limits(limits=threads)
                    worker = {'pid': os.getpid(), 'cpu_threads': threads, 'cpus': sorted(cpus)}
                    result = {**worker, 'counts': recorder.counts}
                elif message['kind'] == 'commit':
                    metadata = message['metadata']
                    identity = metadata['attempt_id']
                    if not identity or Path(identity).name != identity:
                        raise ValueError('Invalid staged episode identity')
                    if message['frames'] <= 0:
                        raise ValueError('Staged episode has no frames')
                    episode = _SealedEpisode(recorder.staging / identity, message['frames'])
                    camera_shapes = {name: tuple(shape) for name, shape in message['camera_shapes'].items()}
                    result = {'metadata': recorder.commit(episode, {**metadata, 'save_worker': worker},
                        camera_shapes, message['prompt']), 'counts': recorder.counts}
                    print(f"Saved episode {result['metadata']['episode_index']} "
                          f"({message['frames']} frames) in writer process {os.getpid()}", flush=True)
                elif message['kind'] == 'close':
                    recorder.close()
                    recorder = None
                    result = None
                else:
                    raise ValueError('Unknown save worker command')
                response.write(json.dumps({'ok': True, 'result': result}, allow_nan=False) + '\n')
                response.flush()
                if message['kind'] == 'close':
                    break
            except Exception as error:
                response.write(json.dumps({'ok': False, 'error': str(error), 'traceback': traceback.format_exc()}) + '\n')
                response.flush()
                break  # Never retry a possibly partial commit in this writer instance.
    finally:
        if recorder is not None:
            recorder.close()
        if limits is not None:
            limits.restore_original_limits()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request-fd', type=int, required=True)
    parser.add_argument('--response-fd', type=int, required=True)
    args = parser.parse_args()
    with os.fdopen(args.request_fd) as request, os.fdopen(args.response_fd, 'w') as response:
        serve(request, response)


if __name__ == '__main__':
    main()
