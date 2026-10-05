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

"""Encapsulate low-level process probing, stream handling, and failure formatting."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from reach.runtime._env import sanitize_subprocess_env

#: Stored reference to standard library subprocess.run to detect test monkeypatching.
_ORIGINAL_SUBPROCESS_RUN = subprocess.run

#: Default timeout in seconds when waiting for a process to terminate gracefully.
_DEFAULT_TERMINATE_WAIT_TIMEOUT: float = 1.0

#: Default timeout in seconds when joining the stderr drain background thread.
_DEFAULT_STDERR_JOIN_TIMEOUT: float = 1.0


class _ProcessGroupController:
    """Manage process lifetime, signals, and process group termination."""

    def __init__(self, proc: subprocess.Popen[str]) -> None:
        self.proc = proc

    def send_signal(self, sig: int) -> None:
        """Send a signal to the process group, falling back to the single process."""
        try:
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(self.proc.pid), sig)
            else:
                self.proc.send_signal(sig)
        except (ProcessLookupError, OSError):
            pass

    def terminate_gracefully(self, wait_timeout: float = _DEFAULT_TERMINATE_WAIT_TIMEOUT) -> None:
        """Send SIGTERM to process group, escalating to SIGKILL on timeout."""
        self.send_signal(signal.SIGTERM)
        try:
            self.proc.wait(timeout=wait_timeout)
        except subprocess.TimeoutExpired:
            self.send_signal(signal.SIGKILL)
            with contextlib.suppress(OSError, subprocess.SubprocessError):
                self.proc.wait(timeout=wait_timeout)

    def kill(self) -> None:
        """Kill the process group immediately using SIGKILL."""
        self.send_signal(signal.SIGKILL)


def _run_mock_probe(
    cmd: list[str],
    workdir: Path,
    timeout_s: float | None = None,
    env: Mapping[str, str] | None = None,
    on_line: Callable[[str], bool] | None = None,
) -> tuple[subprocess.CompletedProcess[str] | None, str | None]:
    """Execute monkeypatched subprocess.run probe for test suites."""
    try:
        completed = subprocess.run(
            cmd,
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            stdin=subprocess.DEVNULL,
            env=dict(env) if env is not None else sanitize_subprocess_env(dict(os.environ)),
        )
    except subprocess.TimeoutExpired:
        return None, "timeout"
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"failed to spawn process: {exc}"

    if on_line is not None and completed.stdout:
        matched_lines: list[str] = []
        for line in completed.stdout.splitlines(keepends=True):
            matched_lines.append(line)
            if on_line(line):
                break
        return subprocess.CompletedProcess(
            args=completed.args,
            returncode=completed.returncode,
            stdout="".join(matched_lines),
            stderr=completed.stderr,
        ), None
    return completed, None


def _spawn_probe_process(
    cmd: list[str],
    workdir: Path,
    env: Mapping[str, str] | None,
) -> tuple[subprocess.Popen[str] | None, str | None]:
    """Spawn a child process in a new session with piped standard streams."""
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=workdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            start_new_session=True,
            env=dict(env) if env is not None else sanitize_subprocess_env(dict(os.environ)),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"failed to spawn process: {exc}"
    return proc, None


class _StderrDrainer:
    """Asynchronously drain and accumulate lines from a subprocess stderr stream."""

    def __init__(self, stream: Iterable[str] | None) -> None:
        """Start a background daemon thread to drain lines from stream."""
        self._lines: list[str] = []
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._drain,
            args=(stream,),
            daemon=True,
        )
        self._thread.start()

    def _drain(self, stream: Iterable[str] | None) -> None:
        """Drain lines from stream into internal buffer until EOF."""
        if stream is None:
            return
        with contextlib.suppress(OSError, ValueError):
            for chunk in stream:
                with self._lock:
                    self._lines.append(chunk)

    def join(self, timeout: float = _DEFAULT_STDERR_JOIN_TIMEOUT) -> str:
        """Wait for the drainer thread to complete and return accumulated text."""
        self._thread.join(timeout=timeout)
        with self._lock:
            return "".join(self._lines)


def _stream_process_output(
    proc: subprocess.Popen[str],
    controller: _ProcessGroupController,
    start_time: float,
    timeout_s: float | None,
    on_line: Callable[[str], bool] | None,
) -> tuple[list[str], bool, str | None]:
    """Stream process stdout lines, checking timeouts and early-exit predicates."""
    if proc.stdout is None:
        return [], False, "subprocess stdout unavailable"

    stdout_lines: list[str] = []
    early_stopped = False

    try:
        for line in proc.stdout:
            stdout_lines.append(line)
            if timeout_s is not None and time.monotonic() - start_time > timeout_s:
                controller.terminate_gracefully(wait_timeout=_DEFAULT_TERMINATE_WAIT_TIMEOUT)
                return stdout_lines, False, "timeout"
            if on_line is not None and on_line(line):
                early_stopped = True
                break
    except Exception as exc:  # noqa: BLE001
        controller.kill()
        return stdout_lines, False, f"failed during process execution: {exc}"
    finally:
        if early_stopped and proc.poll() is None:
            controller.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=_DEFAULT_TERMINATE_WAIT_TIMEOUT)
            except subprocess.TimeoutExpired:
                controller.kill()
                with contextlib.suppress(OSError, subprocess.SubprocessError):
                    proc.wait(timeout=_DEFAULT_TERMINATE_WAIT_TIMEOUT)

            if proc.stdout and not proc.stdout.closed:
                with contextlib.suppress(OSError, ValueError):
                    extra_out = proc.stdout.read()
                    if extra_out:
                        stdout_lines.extend(extra_out.splitlines(keepends=True))

    if timeout_s is not None and time.monotonic() - start_time > timeout_s:
        controller.terminate_gracefully(wait_timeout=_DEFAULT_TERMINATE_WAIT_TIMEOUT)
        return stdout_lines, False, "timeout"

    return stdout_lines, early_stopped, None


def _drain_and_reap_process(
    proc: subprocess.Popen[str],
    controller: _ProcessGroupController,
    drainer: _StderrDrainer,
    start_time: float,
    timeout_s: float | None,
) -> tuple[list[str], str, str | None]:
    """Drain remaining process streams and wait for termination."""
    remaining_lines: list[str] = []
    if timeout_s is not None and time.monotonic() - start_time > timeout_s:
        controller.terminate_gracefully(wait_timeout=_DEFAULT_TERMINATE_WAIT_TIMEOUT)
        return remaining_lines, drainer.join(), "timeout"

    if proc.poll() is None:
        remaining_time = (
            max(0.1, timeout_s - (time.monotonic() - start_time)) if timeout_s is not None else None
        )
        try:
            proc.wait(timeout=remaining_time)
        except subprocess.TimeoutExpired:
            controller.terminate_gracefully(wait_timeout=_DEFAULT_TERMINATE_WAIT_TIMEOUT)
            return remaining_lines, drainer.join(), "timeout"

    if proc.stdout and not proc.stdout.closed:
        with contextlib.suppress(OSError, ValueError):
            stdout_rem = proc.stdout.read()
            if stdout_rem:
                remaining_lines.extend(stdout_rem.splitlines(keepends=True))
            proc.stdout.close()

    stderr_rem = drainer.join()
    if proc.stderr and not proc.stderr.closed:
        with contextlib.suppress(OSError):
            proc.stderr.close()

    return remaining_lines, stderr_rem, None


def run_subprocess_probe(
    cmd: list[str],
    workdir: Path,
    timeout_s: float | None = None,
    env: Mapping[str, str] | None = None,
    on_line: Callable[[str], bool] | None = None,
) -> tuple[subprocess.CompletedProcess[str] | None, str | None]:
    """Execute a CLI probe subprocess capturing output, streaming lines, and handling early exit."""
    if subprocess.run is not _ORIGINAL_SUBPROCESS_RUN:
        return _run_mock_probe(cmd, workdir, timeout_s, env, on_line)

    start_time = time.monotonic()
    proc, spawn_err = _spawn_probe_process(cmd, workdir, env)
    if proc is None:
        return None, spawn_err

    drainer = _StderrDrainer(proc.stderr)

    controller = _ProcessGroupController(proc)
    watchdog: threading.Timer | None = None
    if timeout_s is not None:
        watchdog = threading.Timer(timeout_s, controller.terminate_gracefully)
        watchdog.daemon = True
        watchdog.start()

    try:
        stdout_lines, _early_stopped, stream_err = _stream_process_output(
            proc, controller, start_time, timeout_s, on_line
        )
        if stream_err is not None:
            controller.terminate_gracefully(wait_timeout=_DEFAULT_TERMINATE_WAIT_TIMEOUT)
            drainer.join()
            return None, stream_err

        rem_lines, stderr_rem, reap_err = _drain_and_reap_process(
            proc, controller, drainer, start_time, timeout_s
        )
        if reap_err is not None:
            return None, reap_err

        stdout_lines.extend(rem_lines)
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=proc.returncode if proc.returncode is not None else 0,
            stdout="".join(stdout_lines),
            stderr=stderr_rem,
        ), None
    finally:
        if watchdog is not None:
            watchdog.cancel()
        if proc.stdout and not proc.stdout.closed:
            with contextlib.suppress(OSError):
                proc.stdout.close()
        if proc.stderr and not proc.stderr.closed:
            with contextlib.suppress(OSError):
                proc.stderr.close()


def process_failure_reason(completed: subprocess.CompletedProcess[str]) -> str:
    """Extract trailing stderr line or returncode description from a completed process."""
    detail = (completed.stderr or "").strip().splitlines()
    return detail[-1] if detail else f"exit {completed.returncode}"


def format_subprocess_error(
    agent_name: str,
    err: str | None,
    timeout_s: float | None = None,
) -> str:
    """Format standardized probe error message for timeout or process spawn failure."""
    if err == "timeout":
        timeout_str = f" after {timeout_s}s" if timeout_s is not None else ""
        return f"{agent_name} process timed out{timeout_str}"
    return f"failed to spawn {agent_name}: {err}" if err else f"{agent_name} subprocess failed"


def iter_json_lines(lines: Iterable[str]) -> Iterable[dict[str, Any]]:
    """Parse and yield JSON objects line-by-line from streaming output."""
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            event = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            yield event


def extract_content_reasoning(content: Iterable[Any]) -> list[str]:
    """Extract thought, thinking, or text strings from assistant message content items."""
    reasoning: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        raw: Any = None
        if item_type in ("thought", "thinking"):
            raw = item.get("thinking") or item.get("thought") or item.get("text")
        elif item_type == "text":
            raw = item.get("text")

        if raw and (cleaned := str(raw).strip()):
            reasoning.append(cleaned)
    return reasoning


def check_tool_leak(
    observed_tools: Iterable[str],
    allowed_tools: Iterable[str] | None,
) -> str | None:
    """Detect whether observed tools contain forbidden tools outside allowed set."""
    if allowed_tools is None:
        return None
    leaked = tuple(sorted(set(observed_tools) - frozenset(allowed_tools)))
    return f"tool leak: {', '.join(leaked)}" if leaked else None
