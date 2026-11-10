"""Native worker exits retain useful diagnostics without GPU or hardware access."""
import io
import os
import signal
import subprocess
import sys
from unittest.mock import Mock, patch

import pytest

from real.deployment.worker import ModelWorker


def worker_for(process, *, closed=False):
    worker = object.__new__(ModelWorker)
    worker.closed = closed
    worker.process = process
    worker._request = io.BytesIO()
    worker._response = io.BytesIO()
    return worker


@pytest.mark.parametrize("operation", ["load", "plan", "reset"])
def test_closed_pipe_includes_operation_status_and_original_exception(operation):
    worker = worker_for(Mock(poll=Mock(return_value=23)))
    with pytest.raises(RuntimeError) as failure:
        worker._exchange({"kind": operation})
    message = str(failure.value)
    assert f"during {operation}" in message
    assert "exited with status 23" in message
    assert "CUDA/XLA" in message and "stderr" in message
    assert isinstance(failure.value.__cause__, EOFError)
    worker.process.wait.assert_not_called()


def test_broken_request_pipe_reports_signal_and_preserves_cause():
    worker = worker_for(Mock(poll=Mock(return_value=-signal.SIGABRT)))
    error = BrokenPipeError("child exited")
    with patch("real.deployment.worker._send", side_effect=error), \
            pytest.raises(RuntimeError, match="SIGABRT") as failure:
        worker._exchange({"kind": "load"})
    assert failure.value.__cause__ is error


def test_child_exit_race_gets_a_short_bounded_wait():
    process = Mock(poll=Mock(return_value=None), wait=Mock(return_value=-signal.SIGSEGV))
    worker = worker_for(process)
    with pytest.raises(RuntimeError, match="SIGSEGV"):
        worker._exchange({"kind": "plan"})
    process.wait.assert_called_once_with(timeout=.2)


def test_pipe_closure_does_not_wait_forever_for_a_live_child():
    process = Mock(poll=Mock(return_value=None))
    process.wait.side_effect = subprocess.TimeoutExpired("worker", .2)
    worker = worker_for(process)
    with pytest.raises(RuntimeError, match="subprocess is still running"):
        worker._exchange({"kind": "plan"})
    process.wait.assert_called_once_with(timeout=.2)


def test_intentional_close_is_not_reported_as_a_native_crash():
    process = Mock()
    worker = worker_for(process, closed=True)
    with pytest.raises(RuntimeError, match="was closed during plan") as failure:
        worker._exchange({"kind": "plan"})
    assert "CUDA" not in str(failure.value)
    assert isinstance(failure.value.__cause__, EOFError)
    process.poll.assert_not_called()
    process.wait.assert_not_called()


@pytest.mark.parametrize("exit_statement,diagnostic", [
    ("os._exit(23)", "exited with status 23"),
    ("os.abort()", "SIGABRT"),
])
def test_actual_native_child_exit_is_reported(exit_statement, diagnostic):
    script = (
        "import os, resource, sys; "
        "resource.setrlimit(resource.RLIMIT_CORE, (0, 0)); "
        "sys.stdin.buffer.read(8); " + exit_statement
    )
    with subprocess.Popen([sys.executable, "-c", script], stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        worker = worker_for(process)
        worker._request, worker._response = process.stdin, process.stdout
        try:
            with pytest.raises(RuntimeError, match=diagnostic) as failure:
                worker._exchange({"kind": "plan"})
            assert isinstance(failure.value.__cause__, (EOFError, BrokenPipeError))
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)


def test_configured_environment_reaches_child_without_changing_parent(monkeypatch):
    from mock_rollout import mock_rollout
    from real.deployment.config import worker_config

    cfg = worker_config(mock_rollout(), simulate=True)
    cfg["inference"]["environment"] = {"RSM_TEST_WORKER_SETTING": "configured"}
    monkeypatch.setenv("RSM_TEST_WORKER_SETTING", "inherited")
    script = """
import os, sys
from real.deployment.worker import _receive, _send
with os.fdopen(int(sys.argv[1]), 'rb') as requests, os.fdopen(int(sys.argv[2]), 'wb') as replies:
    _receive(requests)
    _send(replies, {'ok': True, 'result': {'setting': os.environ['RSM_TEST_WORKER_SETTING']}})
"""
    popen = subprocess.Popen

    def launch_probe(args, **kwargs):
        request_fd, response_fd = kwargs["pass_fds"]
        return popen([args[0], "-c", script, str(request_fd), str(response_fd)], **kwargs)

    monkeypatch.setattr("real.deployment.worker.subprocess.Popen", launch_probe)
    worker = ModelWorker(cfg)
    try:
        assert worker.metadata["setting"] == "configured"
        assert os.environ["RSM_TEST_WORKER_SETTING"] == "inherited"
    finally:
        worker.close()


def test_failed_constructor_closes_both_pipes_without_masking_worker_exit(monkeypatch):
    from mock_rollout import mock_rollout
    from real.deployment.config import worker_config

    cfg = worker_config(mock_rollout(), simulate=True)
    popen, fdopen = subprocess.Popen, os.fdopen
    streams = []

    def launch_failed_child(args, **kwargs):
        process = popen([args[0], "-c", "import os; os._exit(23)"], **kwargs)
        process.wait(timeout=5)
        return process

    def track_stream(*args, **kwargs):
        stream = fdopen(*args, **kwargs)
        streams.append(stream)
        return stream

    monkeypatch.setattr("real.deployment.worker.subprocess.Popen", launch_failed_child)
    monkeypatch.setattr("real.deployment.worker.os.fdopen", track_stream)
    try:
        with pytest.raises(RuntimeError, match="during load: subprocess exited with status 23") as failure:
            ModelWorker(cfg)
        assert isinstance(failure.value.__cause__, BrokenPipeError)
        assert len(streams) == 2
        assert all(stream.closed for stream in streams)
    finally:
        for stream in streams:
            try:
                stream.close()
            except BrokenPipeError:
                pass
