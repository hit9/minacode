"""Background shell jobs tracked by a session: a process handle plus its status/tail surface.

`Session.jobs` maps job ids to these; the tool layer (`BashTool`, `Job`) starts them and this
class owns the process lifecycle and the merged stdout+stderr tail.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import ClassVar


@dataclass
class BackgroundJob:
    """A non-blocking shell process tracked by the session. Output is either redirected to a log
    file on disk (jobs started via `Job(start)`) or accumulated in an in-memory tail buffer by a
    drainer thread (jobs promoted from a running BashTool call after bash_wait_timeout). Both
    variants expose the same tail/status/wait/kill surface."""

    id: str
    command: str
    process: subprocess.Popen[bytes]
    log_path: str
    started_at: float
    status: str = "running"
    exit_code: int | None = None
    # Memory-backed tail, populated by BashTool.promote_to_job's drainer thread. When set, tail()
    # reads from here instead of log_path. Bounded at BUFFER_LIMIT chars by the drainer.
    stream_buffer: list[str] | None = None
    stream_lock: threading.Lock | None = None

    BUFFER_LIMIT: ClassVar[int] = 32 * 1024  # per-stream tail cap in chars

    def update_status(self) -> None:
        if self.status != "running":
            return
        code = self.process.poll()
        if code is not None:
            self.status = "done"
            self.exit_code = code

    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    def kill(self, grace: float = 3.0) -> None:
        """SIGTERM, wait grace seconds, then SIGKILL if still running. Removes the log file."""
        if self.status == "running":
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except OSError:
                self.process.terminate()
            try:
                self.process.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except OSError:
                    self.process.kill()
                self.process.wait()
            self.update_status()
            if self.status == "running":
                self.status = "killed"
                self.exit_code = -1
        if self.log_path:
            with contextlib.suppress(OSError):
                os.unlink(self.log_path)

    def tail(self, limit: int) -> str:
        """Return the last `limit` chars from the merged stdout+stderr log."""
        limit = max(0, limit)
        if self.stream_buffer is not None:
            with self.stream_lock or contextlib.nullcontext():
                text = "".join(self.stream_buffer)
        else:
            try:
                with open(self.log_path, "rb") as file:
                    file.seek(0, 2)
                    size = file.tell()
                    # UTF-8 is up to 4 bytes/char; read a little extra so decoding produces at least `limit` chars.
                    file.seek(max(0, size - limit * 4), 0)
                    text = file.read().decode("utf-8", errors="replace")
            except OSError:
                return ""
        if len(text) <= limit:
            return text
        if limit <= 3:
            return "." * limit
        return "..." + text[-(limit - 3) :]
