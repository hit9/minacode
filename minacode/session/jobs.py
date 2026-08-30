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
    file on disk (jobs started via `Job(start)`) or accumulated in an in-memory tail buffer before
    and after a BashTool call is promoted at bash_wait_timeout. Both
    variants expose the same tail/status/wait/kill surface."""

    id: str
    command: str
    process: subprocess.Popen[bytes]
    log_path: str
    started_at: float
    status: str = "running"
    exit_code: int | None = None
    # Memory-backed tail populated across BashTool promotion. When set, tail() reads from here
    # instead of log_path. Bounded at BUFFER_LIMIT chars by append_stream().
    stream_buffer: list[str] | None = None
    stream_lock: threading.Lock | None = None
    stream_truncated: bool = False

    BUFFER_LIMIT: ClassVar[int] = 32 * 1024  # promoted-job tail cap in chars

    @staticmethod
    def normalize_id(value: object) -> str:
        """Canonicalize the bare numeric shorthand accepted at every job lookup boundary."""
        job_id = str(value or "").strip()
        return f"job.{job_id}" if job_id and not job_id.startswith("job.") and job_id.isdigit() else job_id

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

    def append_stream(self, text: str) -> None:
        """Append promoted-Bash output while keeping its in-memory tail bounded."""
        if not text or self.stream_buffer is None:
            return
        with self.stream_lock or contextlib.nullcontext():
            self.stream_buffer.append(text)
            total = sum(len(part) for part in self.stream_buffer)
            while total > self.BUFFER_LIMIT and len(self.stream_buffer) > 1:
                total -= len(self.stream_buffer.pop(0))
                self.stream_truncated = True
            if total > self.BUFFER_LIMIT:
                self.stream_buffer[0] = self.stream_buffer[0][-self.BUFFER_LIMIT :]
                self.stream_truncated = True

    def log_snapshot(self, max_bytes: int) -> tuple[str, bool] | None:
        """Return a bounded log snapshot, or None when disk-backed output is unavailable.

        The storage choice stays private to the job. Promoted Bash output is already a bounded
        tail; a disk-backed Job log keeps both ends without loading an unbounded file into the UI.
        """
        max_bytes = max(1, max_bytes)
        if self.stream_buffer is not None:
            with self.stream_lock or contextlib.nullcontext():
                text = "".join(self.stream_buffer)
                truncated = self.stream_truncated
            if truncated:
                text = "... earlier job output omitted ...\n" + text
            return text, truncated
        try:
            with open(self.log_path, "rb") as file:
                file.seek(0, os.SEEK_END)
                size = file.tell()
                if size <= max_bytes:
                    file.seek(0)
                    return file.read().decode("utf-8", errors="replace"), False
                head_size = max_bytes // 2
                tail_size = max_bytes - head_size
                file.seek(0)
                head = file.read(head_size).decode("utf-8", errors="replace")
                file.seek(-tail_size, os.SEEK_END)
                tail = file.read(tail_size).decode("utf-8", errors="replace")
                return head + "\n... middle of job log omitted ...\n" + tail, True
        except OSError:
            return None
