# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Verify subprocess probing, streaming line capture, early exit, and timeout handling."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import TYPE_CHECKING, Any

import pytest

from reach.runtime._subprocess import (
    _ProcessGroupController,
    _StderrDrainer,
    run_subprocess_probe,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def test_run_subprocess_probe_real_execution(tmp_path: Path) -> None:
    """Verify live subprocess probe captures full stdout and reaps children closing stdout early."""
    cmd = (
        ["sh", "-c", "printf 'hello\\nworld\\n'"]
        if os.name != "nt"
        else [sys.executable, "-c", "print('hello'); print('world')"]
    )
    completed, err = run_subprocess_probe(cmd, tmp_path)

    assert err is None
    assert completed is not None
    assert completed.returncode == 0
    assert completed.stdout == "hello\nworld\n"

    if os.name != "nt":
        early_close_cmd = ["sh", "-c", "printf 'done\\n'; exec >&-; sleep 0.03"]
        res, close_err = run_subprocess_probe(early_close_cmd, tmp_path, timeout_s=2.0)
        assert close_err is None
        assert res is not None
        assert "done" in res.stdout


def test_run_subprocess_probe_real_early_exit(tmp_path: Path) -> None:
    """Verify on_line predicate early-terminates process when target line appears."""
    if os.name != "nt":
        cmd = [
            "sh",
            "-c",
            "printf 'line1\\nSTOP_HERE\\n'; sleep 0.2 >/dev/null 2>&1; printf 'line3\\n'",
        ]
    else:
        script = (
            "import sys, time\n"
            "print('line1', flush=True)\n"
            "print('STOP_HERE', flush=True)\n"
            "time.sleep(2)\n"
            "print('line3', flush=True)\n"
        )
        cmd = [sys.executable, "-c", script]

    def _predicate(line: str) -> bool:
        return "STOP_HERE" in line

    completed, err = run_subprocess_probe(cmd, tmp_path, timeout_s=5.0, on_line=_predicate)

    assert err is None
    assert completed is not None
    assert "line1" in completed.stdout
    assert "STOP_HERE" in completed.stdout
    assert "line3" not in completed.stdout


def test_run_subprocess_probe_large_stderr_does_not_deadlock(tmp_path: Path) -> None:
    """Verify large stderr volume drains concurrently without pipe deadlock."""
    payload_size = 256 * 1024
    if os.name != "nt":
        cmd = [
            "sh",
            "-c",
            f"head -c {payload_size} /dev/zero >&2; printf 'probe_completed\\n'",
        ]
    else:
        script = (
            "import sys\n"
            f"sys.stderr.write('E' * {payload_size})\n"
            "sys.stderr.flush()\n"
            "sys.stdout.write('probe_completed\\n')\n"
            "sys.stdout.flush()\n"
        )
        cmd = [sys.executable, "-c", script]
    completed, err = run_subprocess_probe(cmd, tmp_path, timeout_s=4.0)

    assert err is None
    assert completed is not None
    assert "probe_completed" in completed.stdout
    assert len(completed.stderr) == payload_size


def _sleep_cmd(seconds: int = 10, *, emit_ping: bool = False) -> list[str]:
    """Return a cross-platform command that sleeps for the given duration."""
    if os.name != "nt":
        return (
            ["sh", "-c", f"printf 'ping\\n'; sleep {seconds}"]
            if emit_ping
            else ["sleep", str(seconds)]
        )
    prefix = "print('ping', flush=True); " if emit_ping else ""
    return [sys.executable, "-c", f"import time; {prefix}time.sleep({seconds})"]


@pytest.mark.parametrize(
    ("emit_ping", "timeout_s"),
    [
        pytest.param(False, 0.015, id="silent-hang"),
        pytest.param(True, 0.001, id="streaming-hang"),
    ],
)
def test_run_subprocess_probe_watchdog_terminates_hung_process(
    tmp_path: Path,
    emit_ping: bool,
    timeout_s: float,
) -> None:
    """Verify watchdog terminates both silent and streaming hung subprocesses on timeout."""
    start = time.monotonic()
    completed, err = run_subprocess_probe(
        _sleep_cmd(10, emit_ping=emit_ping),
        tmp_path,
        timeout_s=timeout_s,
    )
    elapsed = time.monotonic() - start

    assert completed is None
    assert err == "timeout"
    assert elapsed < 2.0


def test_run_subprocess_probe_real_spawn_failure(tmp_path: Path) -> None:
    """Verify invalid executable path gracefully returns spawn failure error."""
    cmd = ["/path/to/nonexistent/reach_executable_binary_probe"]
    completed, err = run_subprocess_probe(cmd, tmp_path)

    assert completed is None
    assert err is not None
    assert "failed to spawn process" in err


def test_run_subprocess_probe_mock_mode_happy_path(
    mock_subprocess: Callable[..., Any],
    tmp_path: Path,
) -> None:
    """Verify monkeypatched subprocess.run is handled cleanly via mock path."""
    mock_subprocess(stdout="mock line 1\nmock line 2\n")
    completed, err = run_subprocess_probe(["dummy"], tmp_path)

    assert err is None
    assert completed is not None
    assert completed.stdout == "mock line 1\nmock line 2\n"


def test_run_subprocess_probe_mock_mode_early_exit(
    mock_subprocess: Callable[..., Any],
    tmp_path: Path,
) -> None:
    """Verify monkeypatched subprocess.run applies on_line filtering and truncation."""
    mock_subprocess(stdout="line 1\nMATCH_LINE\nline 3\n")

    def _predicate(line: str) -> bool:
        return "MATCH_LINE" in line

    completed, err = run_subprocess_probe(["dummy"], tmp_path, on_line=_predicate)

    assert err is None
    assert completed is not None
    assert completed.stdout == "line 1\nMATCH_LINE\n"


def test_run_subprocess_probe_mock_mode_timeout(
    mock_subprocess: Callable[..., Any],
    tmp_path: Path,
) -> None:
    """Verify monkeypatched subprocess.run timeout returns timeout error."""
    mock_subprocess(side_effect=subprocess.TimeoutExpired(cmd="dummy", timeout=1))
    completed, err = run_subprocess_probe(["dummy"], tmp_path)

    assert completed is None
    assert err == "timeout"


def test_run_subprocess_probe_mock_mode_spawn_error(
    mock_subprocess: Callable[..., Any],
    tmp_path: Path,
) -> None:
    """Verify monkeypatched subprocess.run OSError returns spawn error."""
    mock_subprocess(side_effect=FileNotFoundError("missing executable"))
    completed, err = run_subprocess_probe(["dummy"], tmp_path)

    assert completed is None
    assert err is not None
    assert "failed to spawn process: missing executable" in err


def test_process_group_controller_lifecycle(tmp_path: Path) -> None:
    """Verify ProcessGroupController gracefully terminates and kills spawned processes."""
    proc = subprocess.Popen(  # noqa: S603
        _sleep_cmd(10),
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        text=True,
        start_new_session=True,
    )
    try:
        controller = _ProcessGroupController(proc)

        # Terminate gracefully
        controller.terminate_gracefully(wait_timeout=0.5)
        assert proc.poll() is not None

        # Redundant signal calls on reaped process should not raise
        controller.send_signal(15)
        controller.kill()
    finally:
        if proc.stdout and not proc.stdout.closed:
            proc.stdout.close()
        if proc.stderr and not proc.stderr.closed:
            proc.stderr.close()


def test_stderr_drainer_accumulates_lines() -> None:
    """Verify StderrDrainer reads and joins lines from an input stream."""
    stream = iter(["line1\n", "line2\n", "line3"])
    drainer = _StderrDrainer(stream)
    result = drainer.join(timeout=1.0)
    assert result == "line1\nline2\nline3"


def test_stderr_drainer_handles_none_stream() -> None:
    """Verify StderrDrainer handles None stream without error."""
    drainer = _StderrDrainer(None)
    result = drainer.join(timeout=1.0)
    assert result == ""


def test_stderr_drainer_thread_safe_concurrent_reads() -> None:
    """Verify StderrDrainer safely joins while background thread appends chunks."""

    def slow_stream() -> Any:
        for i in range(15):
            time.sleep(0.0002)
            yield f"line {i}\n"

    drainer = _StderrDrainer(slow_stream())
    _ = [drainer.join(timeout=0.001) for _ in range(3)]
    full = drainer.join(timeout=2.0)
    assert "line 0\n" in full
    assert "line 14\n" in full
