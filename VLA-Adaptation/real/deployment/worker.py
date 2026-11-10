"""Local inference process with dedicated pipes, isolated from robot drivers.

Only the parent and its child possess these file descriptors. Pickle is used for
trusted local messages, never for network or user-supplied payloads. Arrays use
the NPY format instead of NumPy's version-dependent pickle module names. Model
stdout remains ordinary console output and cannot corrupt the protocol.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import os
from pathlib import Path
import pickle
import signal
import struct
import subprocess
import sys
import time
import traceback


_ARRAY_TAG = "__rsm_numpy_npy_v1__"


def _encode_arrays(value):
    import numpy as np

    if isinstance(value, (np.ndarray, np.generic)):
        buffer = io.BytesIO()
        np.save(buffer, np.asarray(value), allow_pickle=False)
        return (_ARRAY_TAG, isinstance(value, np.generic), buffer.getvalue())
    if isinstance(value, dict):
        return {key: _encode_arrays(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_encode_arrays(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_encode_arrays(item) for item in value)
    return value


def _decode_arrays(value):
    if isinstance(value, tuple) and len(value) == 3 and value[0] == _ARRAY_TAG:
        import numpy as np

        array = np.load(io.BytesIO(value[2]), allow_pickle=False)
        return array[()] if value[1] else array
    if isinstance(value, dict):
        return {key: _decode_arrays(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_arrays(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_decode_arrays(item) for item in value)
    return value


def _read_exact(stream, size):
    chunks = []
    while size:
        part = stream.read(size)
        if not part:
            raise EOFError("Inference worker closed its response pipe")
        chunks.append(part)
        size -= len(part)
    return b"".join(chunks)


def _receive(stream):
    size = struct.unpack("!Q", _read_exact(stream, 8))[0]
    if size > 512 * 1024 * 1024:
        raise ValueError("Inference message exceeds 512 MiB")
    return _decode_arrays(pickle.loads(_read_exact(stream, size)))


def _send(stream, value):
    payload = pickle.dumps(_encode_arrays(value), protocol=5)
    stream.write(struct.pack("!Q", len(payload)))
    stream.write(payload)
    stream.flush()


class ModelWorker:
    """One persistent planner, at most one request in flight."""

    def __init__(self, config):
        """config: the dict from real.deployment.config.worker_config."""
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference-pipe")
        self.pending = None
        self.process = None
        self.closed = False
        self._request = self._response = None
        # None (dry runs) waits indefinitely, e.g. while the worker sits at a breakpoint.
        self.timeout_s = config["inference"]["timeout_s"]
        request_read, request_write = os.pipe()
        response_read, response_write = os.pipe()
        try:
            inference = config["inference"]
            python = inference["python"] or sys.executable
            root = Path(__file__).resolve().parents[2]
            child_env = dict(os.environ)
            child_env.update(inference["environment"])
            child_env["PYTHONPATH"] = str(root) + os.pathsep + child_env.get("PYTHONPATH", "")
            self.process = subprocess.Popen(
                [str(python), "-m", "real.deployment.worker", "--request-fd", str(request_read),
                 "--response-fd", str(response_write)],
                pass_fds=(request_read, response_write), cwd=root, env=child_env,
            )
        except BaseException:
            os.close(request_write)
            os.close(response_read)
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.closed = True
            raise
        finally:
            os.close(request_read)
            os.close(response_write)
        self._request = os.fdopen(request_write, "wb")
        self._response = os.fdopen(response_read, "rb")
        try:
            self.metadata = self._submit({"kind": "load", "config": config}).result(inference["timeout_s"])
        except BaseException:
            self.close()
            raise

    def _exchange(self, message):
        started = time.perf_counter()
        try:
            _send(self._request, message)
            reply = _receive(self._response)
        except (EOFError, BrokenPipeError) as error:
            raise self._pipe_failure(message["kind"]) from error
        if not reply["ok"]:
            raise RuntimeError("Inference worker: " + reply["error"] + "\n" + reply.get("traceback", ""))
        result = reply["result"]
        if message["kind"] == "plan":
            result["timings"]["worker_planning_s"] = result["timings"]["planning_s"]
            result["timings"]["planning_s"] = time.perf_counter() - started
        return result

    def _pipe_failure(self, operation):
        if self.closed:
            return RuntimeError(f"Inference worker was closed during {operation}")
        # A native abort closes the pipe just before Popen observes its exit.
        # Briefly reap it so the diagnostic includes the actual signal/status.
        status = self.process.poll()
        if status is None:
            try:
                status = self.process.wait(timeout=.2)
            except subprocess.TimeoutExpired:
                pass
        if self.closed:
            return RuntimeError(f"Inference worker was closed during {operation}")
        if status is None:
            detail = "communication pipe closed while its subprocess is still running"
        elif status < 0:
            try:
                name = signal.Signals(-status).name
            except ValueError:
                name = f"signal {-status}"
            detail = f"subprocess terminated by {name} (exit status {status})"
        else:
            detail = f"subprocess exited with status {status}"
        return RuntimeError(
            f"Inference worker failed during {operation}: {detail}. "
            "Native CUDA/XLA fatal diagnostics, if any, appear above in the subprocess stderr."
        )

    def _submit(self, message):
        if self.closed:
            raise RuntimeError("Inference worker is closed")
        if self.pending is not None and not self.pending.done():
            raise RuntimeError("Only one planning request may be in flight")
        self.pending = self.pool.submit(self._exchange, message)
        return self.pending

    def submit_plan(self, observation, seed):
        return self._submit({"kind": "plan", "observation": observation, "seed": int(seed)})

    def submit_reset(self):
        """Reset policy state asynchronously, without blocking the collector loop."""
        return self._submit({"kind": "reset"})

    def reset(self):
        return self.submit_reset().result(timeout=None if self.timeout_s is None else 30)

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.process is not None and self.process.poll() is None:
            # Also interrupts an in-flight GPU request when deployment exits.
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.pool.shutdown(wait=True, cancel_futures=True)
        for stream in (self._request, self._response):
            if stream is not None:
                try:
                    stream.close()
                except BrokenPipeError:
                    # A failed buffered send is flushed again by close(). The
                    # worker's original failure must survive teardown, and the
                    # response pipe still needs to be closed.
                    pass


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-fd", type=int, required=True)
    parser.add_argument("--response-fd", type=int, required=True)
    args = parser.parse_args(argv)
    with os.fdopen(args.request_fd, "rb") as requests, os.fdopen(args.response_fd, "wb") as replies:
        planner = None
        while True:
            try:
                message = _receive(requests)
            except EOFError:
                return
            except Exception as error:
                _send(replies, {"ok": False, "error": f"{type(error).__name__}: {error}",
                                "traceback": traceback.format_exc()})
                return
            try:
                if message["kind"] == "load":
                    from real.deployment.models import build_planner
                    planner = build_planner(message["config"])
                    metadata = getattr(planner, "metadata", {})
                    result = metadata() if callable(metadata) else metadata
                elif planner is None:
                    raise RuntimeError("Load a planner before inference")
                elif message["kind"] == "reset":
                    planner.reset()
                    result = None
                elif message["kind"] == "plan":
                    result = planner.plan(message["observation"], message["seed"])
                else:
                    raise ValueError("Unknown inference operation")
                reply = {"ok": True, "result": result}
            except Exception as error:
                reply = {"ok": False, "error": f"{type(error).__name__}: {error}",
                         "traceback": traceback.format_exc()}
            _send(replies, reply)


if __name__ == "__main__":
    main()
