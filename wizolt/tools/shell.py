"""Shell tools: foreground commands and background jobs."""

from __future__ import annotations

import codecs
import contextlib
import os
import re
import selectors
import shlex
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from typing import Any, ClassVar, cast

from wizolt.base import Json, ToolArgs, ToolError
from wizolt.session import BackgroundJob, Session
from wizolt.tools.base import Tool


class BashTool(Tool):
    """Run one bash invocation in the workspace, streaming its output as it arrives.

    Confirmation is the default, and the read-only allowlist is the narrow exception — it exists
    because this tool replaced the dedicated listing and search tools, and prompting for every `ls`
    would be unusable. Auto-approval must hold for the whole command, not its first word: every stage
    of a pipeline or `&&` chain must independently be read-only, redirection to a real path or command
    substitution disqualifies it, and wrappers that can hide execution are never approved. This is a
    prompting heuristic, not a sandbox; confirmation remains the real boundary.

    The process gets its own session, so cancelling kills the whole group instead of orphaning
    children of an already-exited shell. Output is decoded incrementally per stream, so a multibyte
    character split across reads survives. If it remains active past the foreground wait timeout,
    the same process is registered as a background job and its bounded output tail remains available
    through `Job`.
    """

    NAME = "Bash"
    _DEV_NULL_REDIRECT_RE: ClassVar[re.Pattern] = re.compile(r"(?:\d*>>?|&>|<)\s*/dev/null(?![\w./])")
    _BACKGROUND_AMP_RE: ClassVar[re.Pattern] = re.compile(r"(?<!&)&(?!&)")
    _CONTROL_OPERATOR_RE: ClassVar[re.Pattern] = re.compile(r"&&|\|\||[|;\n]")
    LOG_LEXER = "bash"
    DESCRIPTION = (
        "Run one bash shell invocation starting in the workspace; returns exit_code/stdout/stderr and shows live output. Compose several steps into one "
        "invocation with `&&`, `||`, `|`, and `;` rather than issuing separate calls. Avoid unbounded output; "
        "limit noisy commands with head/tail/sed/rg filters or command-specific limits, and inspect large outputs in chunks. "
        "Never use it to read or print secrets (private keys, credentials, tokens, `.env`)."
    )
    MUTATES = True
    live_output: Callable[[str, str], None] | None = None

    def __init__(self, session: Session, args: ToolArgs):
        super().__init__(session, args)
        self._process_lock = threading.Lock()
        self._process: subprocess.Popen[bytes] | None = None

    def cancel(self) -> None:
        with self._process_lock:
            proc = self._process
        if proc is not None and proc.poll() is None:
            self.kill_process_group(proc)

    # Read-only executables that only inspect the filesystem/repo. A command built solely from these
    # (and safe git subcommands) auto-runs without a confirmation prompt in non-yolo mode, replacing
    # the dedicated List/Find/LineCount/read-only-Git tools that were removed in favour of Bash.
    # fmt: off
    SAFE_COMMANDS: ClassVar[frozenset[str]] = frozenset(
        {
            # Common read-only inspection commands. The obvious file-writing forms (`sort -o`,
            # `uniq IN OUT`, `sed -i`, `tree -o`) are guarded below; we do not chase exotic paths
            # like sed's `w` command — common sense over exhaustive safety.
            "ls", "cat", "head", "tail", "wc", "find", "grep", "egrep", "fgrep", "rg", "sort", "uniq",
            "sed", "tree", "cut", "tr", "nl", "comm", "column", "fold", "paste", "join", "echo", "printf", "pwd",
            "stat", "file", "basename", "dirname", "realpath", "readlink", "which", "type",
            "diff", "cmp", "date", "printenv", "du", "df", "jq", "true", "test", "uname", "hostname",
            # Benign builtin the model routinely prefixes (cd changes the subshell dir only).
            "cd",
        }
    )
    SAFE_GIT_SUBCOMMANDS: ClassVar[frozenset[str]] = frozenset(
        {"status", "diff", "log", "show", "rev-parse", "ls-files", "grep", "blame", "describe",
         "shortlog", "cat-file", "ls-tree", "rev-list", "for-each-ref", "diff-tree"}
    )
    # fmt: on

    def needs_confirmation(self) -> bool:
        try:
            return not self.is_readonly(self.command())
        except ToolError:
            return True

    @classmethod
    def is_readonly(cls, command: str) -> bool:
        """Conservatively classify a command as safe to auto-run. Bias hard toward False: a false
        'safe' would run a mutating command without consent, while a false 'unsafe' only costs a
        confirmation prompt. Rejects anything that can write, execute arbitrary code, or background."""
        command = command.strip()
        if not command:
            return False
        # Normalize away the ubiquitous harmless redirections — discarding output to /dev/null and
        # merging stderr/stdout — so the common `cmd 2>/dev/null` / `cmd >/dev/null 2>&1` forms are
        # not treated as file writes.
        scan = cls._DEV_NULL_REDIRECT_RE.sub(" ", command)
        scan = scan.replace("2>&1", " ").replace(">&2", " ")
        # Anything still redirecting to/from a real path, or substituting a command, can write or
        # run arbitrary code.
        if any(ch in scan for ch in (">", "<", "`")) or "$(" in scan:
            return False
        # Reject a lone background & (detaches a process); && and || are allowed sequence operators.
        if cls._BACKGROUND_AMP_RE.search(scan):
            return False
        # Split on every control operator (&& || | ; newline) and require EVERY stage to be a safe
        # read-only command — so `git log && rm x` is not auto-approved on the strength of `git log`.
        return all(cls._safe_segment(part) for part in cls._CONTROL_OPERATOR_RE.split(scan) if part.strip())

    @classmethod
    def _safe_segment(cls, segment: str) -> bool:
        try:
            tokens = shlex.split(segment)
        except ValueError:
            return False
        if not tokens:
            return False
        cmd = tokens[0]
        # Env assignments and wrapper commands can hide arbitrary execution — never auto-approve.
        # fmt: off
        if "=" in cmd or cmd in {"env", "sudo", "eval", "exec", "command", "xargs", "nohup", "time",
                                 "watch", "bash", "sh", "zsh", "tee", "awk", "python", "python3"}:
            return False
        # fmt: on
        if cmd == "git":
            return cls._safe_git(tokens)
        if cmd not in cls.SAFE_COMMANDS:
            return False
        # Flags/args that turn a read-only command into a writer.
        if cmd == "find" and any(t in {"-delete", "-exec", "-execdir", "-ok", "-okdir", "-fprint", "-fprint0", "-fprintf", "-fls"} for t in tokens):
            return False
        if cmd == "sed" and any(t.startswith(("-i", "--in-place")) for t in tokens):
            return False
        if cmd == "tree" and any(t.startswith(("-o", "--output")) for t in tokens):
            return False  # `tree -o FILE` writes the listing to a file
        if cmd == "sort" and any(t.startswith(("-o", "--output")) for t in tokens):
            return False  # `sort -o FILE` / `--output=FILE` writes to a file
        # `uniq INPUT OUTPUT` writes the second file operand.
        return not (cmd == "uniq" and cls._uniq_writes(tokens))

    @staticmethod
    def _uniq_writes(tokens: list[str]) -> bool:
        # uniq writes only in the two-operand form `uniq [OPTS] INPUT OUTPUT`. Count positional
        # operands, skipping the numeric argument that follows a value-taking short flag.
        value_flags = {"-f", "-s", "-w", "--skip-fields", "--skip-chars", "--check-chars"}
        operands = 0
        skip_next = False
        for token in tokens[1:]:
            if skip_next:
                skip_next = False
            elif token in value_flags:
                skip_next = True
            elif not token.startswith("-"):
                operands += 1
        return operands >= 2

    @classmethod
    def _safe_git(cls, tokens: list[str]) -> bool:
        index = 1
        while index < len(tokens) and tokens[index] == "--no-pager":
            index += 1
        if index >= len(tokens):
            return False
        sub = tokens[index]
        if sub not in cls.SAFE_GIT_SUBCOMMANDS:
            return False
        args = tokens[index + 1 :]
        if any(t == "--output" or t.startswith("--output=") for t in args):
            return False
        return not (sub == "grep" and any(t.startswith(("-O", "--open-files-in-pager")) for t in args))

    @classmethod
    def params_schema(cls) -> Json:
        # fmt: off
        return cls.object_schema({
            "command": {"type": "string", "minLength": 1, "pattern": "^.*\\S.*$", "description": "Bash command to run starting in the workspace; filter noisy output with head/tail/rg"},
        }, ["command"])
        # fmt: on

    @classmethod
    def payload_args(cls, payload: Json) -> ToolArgs:
        command = str(payload.get("command") or "")
        if not command.strip():
            raise ToolError("Bash command must be non-empty")
        return [command]

    def command(self) -> str:
        command = self.strings(min_count=1, max_count=1)[0]
        if not command.strip():
            raise ToolError("Bash command must be non-empty")
        return command

    def short_args(self) -> list[str]:
        return [self.command()]

    def call(self) -> str:
        command = self.command()
        bash = shutil.which("bash") or "bash"
        proc = None
        try:
            proc = subprocess.Popen(
                [bash, "-lc", command], cwd=self.session.cwd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True
            )
            with self._process_lock:
                self._process = proc
            assert proc.stdout is not None and proc.stderr is not None
            return self.stream_process(proc)
        except KeyboardInterrupt:
            self.kill_and_collect(proc)
            raise
        finally:
            with self._process_lock:
                if self._process is proc:
                    self._process = None
            if self.live_output is not None:
                self.live_output("", "")

    def stream_process(self, proc: subprocess.Popen[bytes]) -> str:
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        # Per-stream incremental decoders so a multibyte UTF-8 character split across two 4096-byte
        # reads is decoded once it is complete, instead of being mangled into replacement chars.
        self._decoders = {"stdout": codecs.getincrementaldecoder("utf-8")("replace"), "stderr": codecs.getincrementaldecoder("utf-8")("replace")}
        selector = selectors.DefaultSelector()
        stdout, stderr = proc.stdout, proc.stderr
        assert stdout is not None and stderr is not None
        selector.register(stdout, selectors.EVENT_READ, "stdout")
        selector.register(stderr, selectors.EVENT_READ, "stderr")
        timed_out = False
        started = time.monotonic()
        shell_deadline = started + self.session.settings.shell_timeout
        wait_budget = self.session.settings.bash_wait_timeout
        # Auto-promotion: if the command hasn't exited within bash_wait_timeout, hand the still-
        # running proc to the background jobs registry and return control to the model with a
        # partial-output payload. Disabled when the setting is 0 or the wait budget is already
        # >= shell_timeout (in which case we would kill on the same deadline anyway).
        promote_deadline = started + wait_budget if wait_budget and wait_budget < self.session.settings.shell_timeout else None
        try:
            while selector.get_map() or proc.poll() is None:
                now = time.monotonic()
                if promote_deadline is not None and now >= promote_deadline and proc.poll() is None:
                    # Don't drain here: drain_selector does BLOCKING os.reads, which would wait
                    # until bash produced more output (or exited) — defeating the whole point of
                    # promotion. Whatever data the streaming loop already read is the partial
                    # payload; anything still in-flight becomes the drainer thread's first read.
                    return self.promote_to_job(proc, selector, stdout_parts, stderr_parts)
                remaining = shell_deadline - now
                if remaining <= 0:
                    timed_out = True
                    self.kill_process_group(proc)
                    proc.wait()
                    self.drain_selector(selector, stdout_parts, stderr_parts)
                    break
                wait = min(0.2, remaining, promote_deadline - now if promote_deadline is not None else remaining)
                if selector.get_map():
                    for key, _ in selector.select(max(0.0, wait)):
                        self.read_stream_chunk(selector, key, stdout_parts, stderr_parts)
                else:
                    time.sleep(max(0.0, wait))
            if proc.returncode is None:
                proc.wait()
        finally:
            selector.close()
        stdout, stderr = "".join(stdout_parts), "".join(stderr_parts)
        if timed_out:
            stderr += ("\n" if stderr else "") + "timeout"
            return self.process_result("BashToolResult", -1, stdout, stderr)
        return self.process_result("BashToolResult", proc.returncode or 0, stdout, stderr)

    def promote_to_job(
        self,
        proc: subprocess.Popen[bytes],
        selector: selectors.BaseSelector,
        stdout_parts: list[str],
        stderr_parts: list[str],
    ) -> str:
        """Hand off a still-running Bash proc to the background job registry. Closes the streaming
        selector, starts a drainer thread that keeps reading proc.stdout/stderr into an in-memory
        tail buffer (bounded), and returns a partial-output payload for the model."""
        # Take pipe handles before closing the selector so the drainer can keep reading them.
        stdout_pipe, stderr_pipe = proc.stdout, proc.stderr
        with contextlib.suppress(OSError):
            selector.close()
        self.session.job_counter += 1
        job_id = f"job.{self.session.job_counter}"
        partial_stdout = "".join(stdout_parts)
        partial_stderr = "".join(stderr_parts)
        buffer: list[str] = []
        buffer_lock = threading.Lock()
        job = BackgroundJob(
            id=job_id,
            command=self.command(),
            process=proc,
            log_path="",
            started_at=time.monotonic() - self.session.settings.bash_wait_timeout,
            stream_buffer=buffer,
            stream_lock=buffer_lock,
        )
        self.session.jobs[job_id] = job
        # Output already consumed by the foreground selector belongs to the same job history as
        # bytes drained after promotion. Seed it before the drainer threads start so Ctrl-O and
        # later Job calls do not begin halfway through the command.
        job.append_stream(partial_stdout)
        job.append_stream(partial_stderr)

        def drain_pipe(pipe: Any) -> None:
            if pipe is None:
                return
            try:
                # read1 returns whatever is immediately available (line-buffered producers ship one
                # line per call), so a slow trickle of output lands in the tail buffer promptly
                # instead of blocking until a full 4KB is buffered.
                for chunk in iter(lambda: pipe.read1(4096), b""):
                    text = chunk.decode("utf-8", errors="replace")
                    job.append_stream(text)
            except (OSError, ValueError):
                return

        threading.Thread(target=drain_pipe, args=(stdout_pipe,), daemon=True).start()
        threading.Thread(target=drain_pipe, args=(stderr_pipe,), daemon=True).start()
        # Leads with status, not wait: this note is read right after backgrounding handed control
        # back, and waiting is what gives it away again. Getting on with other work is the point.
        note = (
            f"backgrounded after {self.session.settings.bash_wait_timeout}s; still running as {job_id}. "
            f'Keep working; check it later with Job(action="status"|"wait"|"kill", job="{job_id}").'
        )
        partial_stderr = partial_stderr + ("\n" if partial_stderr else "") + note
        return self.process_result("BashToolResult", -1, partial_stdout, partial_stderr)

    def drain_selector(self, selector: selectors.BaseSelector, stdout_parts: list[str], stderr_parts: list[str]) -> None:
        for key in list(selector.get_map().values()):
            while self.read_stream_chunk(selector, key, stdout_parts, stderr_parts):
                pass

    def read_stream_chunk(
        self,
        selector: selectors.BaseSelector,
        key: selectors.SelectorKey,
        stdout_parts: list[str],
        stderr_parts: list[str],
    ) -> bool:
        try:
            data = os.read(cast(Any, key.fileobj).fileno(), 4096)
        except OSError:
            data = b""
        eof = not data
        if eof:
            with contextlib.suppress(Exception):
                selector.unregister(key.fileobj)
            with contextlib.suppress(Exception):
                cast(Any, key.fileobj).close()
        # final=True on EOF flushes any bytes still buffered in the decoder (e.g. a truncated
        # trailing character) so they are not silently dropped.
        text = self._decoders[key.data].decode(data, final=eof)
        if text:
            (stdout_parts if key.data == "stdout" else stderr_parts).append(text)
            if self.live_output is not None:
                self.live_output(str(key.data), text)
        return not eof

    @staticmethod
    def kill_process_group(proc: subprocess.Popen[bytes]) -> None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            with contextlib.suppress(OSError):
                proc.kill()

    @classmethod
    def kill_and_collect(cls, proc: subprocess.Popen[bytes] | None) -> tuple[str, str]:
        if proc is None:
            return "", ""
        cls.kill_process_group(proc)
        stdout, stderr = proc.communicate()

        def decode(value: bytes | str | None) -> str:
            return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value or ""

        return decode(stdout), decode(stderr)


class JobTool(Tool):
    NAME = "Job"
    DESCRIPTION = "Start, monitor, wait for, list, and kill background shell jobs. Processes run in their own process group and do not block the agent."
    MUTATES = True
    ACTIONS: ClassVar[tuple[str, ...]] = ("start", "status", "wait", "list", "kill")
    MAX_JOBS: ClassVar[int] = 8
    DEFAULT_LIMIT: ClassVar[int] = 4096
    # How long one wait may hold the agent. Backgrounding hands control back and waiting is the one
    # way to give it away again, so a wait always ends: at the model's timeout, at MAX_WAIT, or at
    # Ctrl-C. Keep every individual wait short; a still-running job remains addressable and can be
    # checked again later without parking the agent for minutes at a time.
    DEFAULT_WAIT: ClassVar[int] = 20
    MAX_WAIT: ClassVar[int] = 20
    POLL_INTERVAL: ClassVar[float] = 0.1
    # How often the job's log tail is streamed into the live preview while a wait polls. Aligned
    # with the preview's own TICK so the two repaint together; the poll slices stay finer because
    # only the interrupt response needs POLL_INTERVAL resolution.
    LIVE_INTERVAL: ClassVar[float] = 0.3
    live_output: Callable[[str, str], None] | None = None

    def __init__(self, session: Session, args: ToolArgs):
        super().__init__(session, args)
        self._interrupted = threading.Event()

    def cancel(self) -> None:
        """Ctrl-C during a wait. This abandons the wait, not the job: the command keeps running and
        stays addressable through `Job`, which is the whole point of having backgrounded it."""
        self._interrupted.set()

    @classmethod
    def params_schema(cls) -> Json:
        # fmt: off
        return cls.object_schema({
            "action": {"type": "string", "enum": list(cls.ACTIONS), "description": "Operation to perform"},
            "command": {"type": "string", "minLength": 1, "description": "Shell command to run for action=start"},
            "job": {"type": "string", "description": "Job id for action=status, wait, or kill"},
            "timeout": {"type": "integer", "minimum": 0, "description": f"Seconds to wait for action=wait; omit or 0 waits {cls.DEFAULT_WAIT}s, and every wait is capped at {cls.MAX_WAIT}s. A wait that ends with the job still running says so; check it again later"},
            "limit": {"type": "integer", "minimum": 1, "description": "Max characters of stdout/stderr to return; default 4096"},
        }, ["action"])
        # fmt: on

    def payload(self) -> Json:
        return self.single_dict_arg("Job requires a single object argument")

    def resolved_action(self, payload: Json) -> str:
        action = str(payload.get("action") or "").strip()
        if action not in self.ACTIONS:
            raise ToolError(f"unknown action: {action!r}")
        return action

    def blocks_agent(self) -> bool:
        """A `wait`, or a `status` with a positive timeout: the two actions that hold the agent in
        _await_process. A malformed call blocks nothing -- it is rejected before it runs."""
        try:
            payload = self.payload()
            action = self.resolved_action(payload)
            timeout = self.requested_timeout(payload)
        except ToolError:
            return False
        return action == "wait" or (action == "status" and timeout > 0)

    def needs_confirmation(self) -> bool:
        return self.resolved_action(self.payload()) in {"start", "kill", "wait"}

    def short_args(self) -> list[str]:
        payload = self.payload()
        action = self.resolved_action(payload)
        if action == "start":
            return [str(payload.get("command") or "")]
        if action == "list":
            return ["list"]
        return [action, str(payload.get("job") or "")]

    @classmethod
    def log_lexer(cls, args: ToolArgs) -> str:
        payload = args[0] if len(args) == 1 and isinstance(args[0], dict) else {}
        return "bash" if payload.get("action") == "start" else cls.LOG_LEXER

    def call(self) -> str:
        payload = self.payload()
        action = self.resolved_action(payload)
        if action == "start":
            return self._start(payload)
        if action == "status":
            return self._status(payload)
        if action == "wait":
            return self._wait(payload)
        if action == "list":
            return self._list()
        if action == "kill":
            return self._kill(payload)
        raise ToolError(f"unhandled action: {action!r}")

    def _start(self, payload: Json) -> str:
        command = str(payload.get("command") or "").strip()
        if not command:
            raise ToolError("start requires a non-empty command")
        active = len(self.session.running_jobs())
        if active >= self.MAX_JOBS:
            raise ToolError(f"too many active jobs ({active}/{self.MAX_JOBS}); kill or wait for one first")
        self.session.job_counter += 1
        job_id = f"job.{self.session.job_counter}"
        # Log to disk (stdout+stderr merged) so we don't need a threaded drainer to keep the
        # subprocess's OS-level pipe buffers from filling. The command is wrapped in a `{ ...; }`
        # group so the redirection captures every stage of a compound command, not just the last
        # (`a; b && c` would otherwise leak its earlier stages to the inherited stdout).
        # `start_new_session` makes this shell its own process-group leader and the command inherits
        # that group, so killpg(pid) reaches the command and its children; running it directly (no
        # `exec`) keeps builtins like `cd` working.
        fd, log_path = tempfile.mkstemp(prefix=f"nc-{job_id}-", suffix=".log")
        os.close(fd)
        proc = subprocess.Popen(
            ["bash", "-lc", f"{{ {command}; }} > {shlex.quote(log_path)} 2>&1"],
            cwd=self.session.cwd,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        self.session.jobs[job_id] = BackgroundJob(id=job_id, command=command, process=proc, log_path=log_path, started_at=time.monotonic())
        return f"Started {job_id}: {command}"

    @staticmethod
    def requested_timeout(payload: Json) -> int:
        """The `timeout` the model asked for, 0 when it asked for none. A `status` waits only when
        this is positive; a `wait` always waits, falling back to DEFAULT_WAIT. Non-numeric text is
        rejected by name rather than raised as a bare int() ValueError the model has to decode."""
        raw = payload.get("timeout")
        try:
            return max(0, int(raw or 0))
        except (TypeError, ValueError):
            raise ToolError(f"timeout must be a whole number of seconds, got {raw!r}") from None

    def wait_budget(self, payload: Json) -> int:
        """The seconds one wait may hold the agent: what was asked for, clamped to MAX_WAIT, or
        DEFAULT_WAIT when nothing was asked for."""
        timeout = self.requested_timeout(payload)
        return self.DEFAULT_WAIT if timeout <= 0 else min(timeout, self.MAX_WAIT)

    def _await_process(self, job: BackgroundJob, payload: Json) -> bool:
        """Wait for the job, in slices, so Ctrl-C lands within POLL_INTERVAL instead of after the
        whole budget. Returns whether the wait was interrupted. A single blocking process.wait()
        would be simpler but unreachable from the cancelling thread. While polling, the job's log
        tail is streamed into the live preview on LIVE_INTERVAL slices; the throttle keeps the
        disk re-read in `tail` from fighting the poll loop."""
        deadline = time.monotonic() + self.wait_budget(payload)
        baseline = ""
        last_stream = 0.0
        while job.process.poll() is None and time.monotonic() < deadline:
            if self._interrupted.wait(self.POLL_INTERVAL):
                break
            if self.live_output is None or time.monotonic() - last_stream < self.LIVE_INTERVAL:
                continue
            last_stream = time.monotonic()
            # The tail is the diff source for the preview; 8000 matches BashLivePreview.MAX_CHARS.
            text = job.tail(8000)
            if text.startswith(baseline):
                if len(text) > len(baseline):
                    self.live_output("output", text[len(baseline) :])
                    baseline = text
            elif text:
                # The prefix relation broke: the log outgrew the tail window, so `...` shifted the
                # whole frame and the delta is not recoverable by prefix matching. Push the entire
                # visible tail -- the preview keeps only its own last MAX_CHARS, so the overlap
                # with what was already pushed falls off and the rolling window stays correct.
                self.live_output("output", text)
                baseline = text
        job.update_status()
        return self._interrupted.is_set()

    def _status(self, payload: Json) -> str:
        try:
            job = self._resolve_job(payload)
            interrupted = self._await_process(job, payload) if self.requested_timeout(payload) else False
            return self._format(job, payload, interrupted=interrupted)
        finally:
            # Close the live region on every exit path (done, budget exhausted, Ctrl-C, a ToolError
            # from _resolve_job): the runner has already started it, and leaving it open would leak
            # a preview that keeps ticking. A non-blocking status never opened it, so this is a no-op.
            if self.live_output is not None:
                self.live_output("", "")

    def _wait(self, payload: Json) -> str:
        try:
            job = self._resolve_job(payload)
            interrupted = self._await_process(job, payload)
            return self._format(job, payload, interrupted=interrupted)
        finally:
            if self.live_output is not None:
                self.live_output("", "")

    def _list(self) -> str:
        if not self.session.jobs:
            return "No jobs."
        self.session.running_jobs()
        rows = []
        for job in self.session.jobs.values():
            exit_code = job.exit_code if job.status != "running" else "-"
            rows.append(f"| {job.id} | {job.status} | {exit_code} | {job.command[:60]} |")
        return "Jobs:\n| id | status | exit | command |\n|---|---|---|---|\n" + "\n".join(rows)

    def _kill(self, payload: Json) -> str:
        job = self._resolve_job(payload)
        job.kill()
        return f"Killed {job.id} (status={job.status}, exit_code={job.exit_code})"

    def _resolve_job(self, payload: Json) -> BackgroundJob:
        job_id = BackgroundJob.normalize_id(payload.get("job"))
        if not job_id:
            raise ToolError("job id required")
        job = self.session.jobs.get(job_id)
        if job is None:
            raise ToolError(f"unknown job: {job_id!r}")
        job.update_status()
        return job

    def _format(self, job: BackgroundJob, payload: Json, *, interrupted: bool = False) -> str:
        limit = max(1, int(payload.get("limit") or self.DEFAULT_LIMIT))
        output = job.tail(limit)
        lines = [
            f"Job: {job.id}",
            f"Status: {job.status}",
            f"Command: {job.command}",
            f"Elapsed: {job.elapsed():.1f}s",
        ]
        if job.exit_code is not None:
            lines.append(f"Exit code: {job.exit_code}")
        if job.status == "running":
            # A wait that comes back while the job runs returns the same shape as one that comes
            # back because it finished. Without this the output below reads as the final result.
            reason = "the user interrupted the wait" if interrupted else f"the wait ended; one wait lasts at most {self.MAX_WAIT}s"
            lines.append(f"Still running ({reason}); any output above is partial. Do other work and check back, or wait again.")
        if output:
            lines.extend(["--- output ---", output])
        return "\n".join(lines)
