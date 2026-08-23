import shlex
import subprocess
import sys
import threading
import time

import pytest

import minacode.render as render_module
from minacode.base import LogBlock, LogEdge, LogLine, LogRole, ToolCall, ToolError
from minacode.cli import CommandLoop
from minacode.cli.commands import ps_command
from minacode.context import ContextManager
from minacode.engine import Agent
from minacode.render import BashLivePreview, LiveSpark, UiPrinter
from minacode.runner import ToolDisplay, ToolRunner
from minacode.session import Session
from minacode.tools import BashTool, JobTool, Tool


def session(tmp_path):
    return Session(cwd=str(tmp_path))


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({}, "unknown action"),
        ({"action": "unknown"}, "unknown action"),
        ({"action": "start", "command": "  "}, "non-empty command"),
        ({"action": "status"}, "job id required"),
        ({"action": "status", "job": "job.99"}, "unknown job"),
    ],
)
def test_job_validation_errors_are_actionable(tmp_path, payload, message):
    with pytest.raises(ToolError, match=message):
        JobTool(session(tmp_path), [payload]).call()


def test_job_wait_and_list_report_completed_output(tmp_path):
    s = session(tmp_path)
    assert JobTool(s, [{"action": "list"}]).call() == "No jobs."
    JobTool(s, [{"action": "start", "command": "printf completed"}]).call()

    waited = JobTool(s, [{"action": "wait", "job": "job.1", "timeout": 2}]).call()
    listed = JobTool(s, [{"action": "list"}]).call()

    assert "Status: done" in waited
    assert "Exit code: 0" in waited
    assert "--- output ---\ncompleted" in waited
    assert "| job.1 | done | 0 | printf completed |" in listed


def test_job_wait_is_bounded_and_says_the_job_is_still_running(tmp_path, monkeypatch):
    """Backgrounding hands control back to the agent; waiting must not take it away for good.
    A wait ends at the model's timeout or at MAX_WAIT, whichever is shorter, and a job that
    outlives it is reported as still running rather than looking like it finished."""
    s = session(tmp_path)
    # Sub-second budgets keep the waits real but fast: each wait parks for 0.2s (the poll
    # interval is 0.1s), and the ceiling clamps even an absurd requested timeout.
    monkeypatch.setattr(JobTool, "DEFAULT_WAIT", 0.2)
    monkeypatch.setattr(JobTool, "MAX_WAIT", 0.2)
    JobTool(s, [{"action": "start", "command": "sleep 30"}]).call()

    for payload in ({}, {"timeout": 0}, {"timeout": 3600}):
        started = time.monotonic()
        waited = JobTool(s, [{"action": "wait", "job": "job.1", **payload}]).call()
        elapsed = time.monotonic() - started

        assert elapsed < 2, f"wait with {payload} blocked for {elapsed:.1f}s"
        assert elapsed >= 0.15  # the budget was actually spent, not skipped
        assert "Status: running" in waited
        assert "Still running (the wait ended" in waited
        assert "Exit code:" not in waited

    # status with a timeout goes through the same budget.
    started = time.monotonic()
    assert "Still running" in JobTool(s, [{"action": "status", "job": "job.1", "timeout": 3600}]).call()
    assert time.monotonic() - started < 2

    JobTool(s, [{"action": "kill", "job": "job.1"}]).call()


def test_job_wait_honours_a_longer_model_timeout_up_to_the_ceiling(tmp_path, monkeypatch):
    s = session(tmp_path)
    # The job outlives the default budget by a wide margin, so only a longer timeout sees it
    # through; the elapsed-time range pins that the wait really parked.
    monkeypatch.setattr(JobTool, "DEFAULT_WAIT", 0.1)
    monkeypatch.setattr(JobTool, "MAX_WAIT", 900)
    JobTool(s, [{"action": "start", "command": "sleep 0.5; printf slow-done"}]).call()

    # The default would have given up at 0.1s; asking for 30 sees the job through to the end.
    started = time.monotonic()
    waited = JobTool(s, [{"action": "wait", "job": "job.1", "timeout": 30}]).call()

    assert 0.3 < time.monotonic() - started < 5
    assert "Status: done" in waited
    assert "--- output ---\nslow-done" in waited
    assert JobTool(s, [{"action": "wait", "job": "job.1"}]).wait_budget({}) == 0.1  # DEFAULT_WAIT
    assert JobTool(s, [{"action": "wait", "job": "job.1"}]).wait_budget({"timeout": 3600}) == 900  # MAX_WAIT
    # A non-numeric timeout is named in the error rather than surfacing as a bare int() ValueError.
    with pytest.raises(ToolError, match="whole number of seconds"):
        JobTool(s, [{"action": "wait", "job": "job.1", "timeout": "1m"}]).call()
    # The same call reports itself as non-blocking, so the runner's pre-block never raises on it.
    assert JobTool(s, [{"action": "wait", "job": "job.1", "timeout": "1m"}]).blocks_agent() is False


def test_job_wait_budget_is_always_capped_at_twenty_seconds(tmp_path):
    tool = JobTool(session(tmp_path), [{"action": "wait", "job": "job.1"}])

    assert tool.wait_budget({}) == 20
    assert tool.wait_budget({"timeout": 0}) == 20
    assert tool.wait_budget({"timeout": 19}) == 19
    assert tool.wait_budget({"timeout": 20}) == 20
    assert tool.wait_budget({"timeout": 21}) == 20
    assert tool.wait_budget({"timeout": 3600}) == 20
    assert "capped at 20s" in JobTool.params_schema()["properties"]["timeout"]["description"]


def test_job_wait_is_interruptible_and_leaves_the_job_running(tmp_path, monkeypatch):
    """Ctrl-C during a wait reaches JobTool through the runner and abandons the wait only: the
    command keeps running, so the agent gets control back without losing the job."""
    s = session(tmp_path)
    monkeypatch.setattr(JobTool, "MAX_WAIT", 900)
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)
    JobTool(s, [{"action": "start", "command": "sleep 60"}]).call()
    tool = JobTool(s, [{"action": "wait", "job": "job.1", "timeout": 900}])
    result = []

    thread = threading.Thread(target=lambda: result.append(runner.call_tool(tool)))
    thread.start()
    deadline = time.monotonic() + 2
    while runner._active_job.value is not tool and time.monotonic() < deadline:
        time.sleep(0.01)
    started = time.monotonic()
    runner.cancel()
    thread.join(timeout=5)

    assert not thread.is_alive(), "cancel did not interrupt the wait"
    assert time.monotonic() - started < 3
    assert "Still running (the user interrupted the wait)" in result[0]
    assert s.jobs["job.1"].process.poll() is None  # the job itself survives the interrupt
    JobTool(s, [{"action": "kill", "job": "job.1"}]).call()


def test_job_wait_streams_log_tail_to_live_output(tmp_path, monkeypatch):
    """A wait streams the job's log tail into the live preview in increments, and closes the
    region when the wait ends."""
    s = session(tmp_path)
    monkeypatch.setattr(JobTool, "POLL_INTERVAL", 0.01)
    JobTool(s, [{"action": "start", "command": "printf 'line one\\nline two\\n'; sleep 0.1"}]).call()
    events = []
    tool = JobTool(s, [{"action": "wait", "job": "job.1", "timeout": 1}])
    tool.live_output = lambda stream, text: events.append((stream, text))

    tool.call()

    assert events[-1] == ("", "")
    deltas = [text for stream, text in events if stream == "output"]
    assert deltas, "no output was streamed into the live preview"
    assert "".join(deltas) == "line one\nline two\n"  # 增量拼接后恰好是全部输出
    JobTool(s, [{"action": "kill", "job": "job.1"}]).call()


def test_job_wait_streams_short_log_incrementally_without_duplicates(tmp_path, monkeypatch):
    """A short log appended in two phases spaced past the live interval streams each increment
    exactly once: the second tail is a continuation of the first, never a repeat of the whole
    frame."""
    s = session(tmp_path)
    monkeypatch.setattr(JobTool, "POLL_INTERVAL", 0.01)
    monkeypatch.setattr(JobTool, "LIVE_INTERVAL", 0.01)
    JobTool(s, [{"action": "start", "command": "printf 'one\\n'; sleep 0.1; printf 'two\\n'; sleep 0.1"}]).call()
    events = []
    tool = JobTool(s, [{"action": "wait", "job": "job.1", "timeout": 1}])
    tool.live_output = lambda stream, text: events.append((stream, text))

    tool.call()

    deltas = [text for stream, text in events if stream == "output"]
    assert "".join(deltas) == "one\ntwo\n"
    JobTool(s, [{"action": "kill", "job": "job.1"}]).call()


def test_job_wait_keeps_streaming_after_log_outgrows_tail_window(tmp_path, monkeypatch):
    """Once the log passes the 8000-char tail window the `...` prefix breaks suffix matching;
    the wait then pushes the whole visible tail so the preview keeps rolling instead of freezing."""
    s = session(tmp_path)
    monkeypatch.setattr(JobTool, "POLL_INTERVAL", 0.01)
    monkeypatch.setattr(JobTool, "LIVE_INTERVAL", 0.01)
    command = "printf 'a%.0s' {1..6000}; sleep 0.2; printf 'b%.0s' {1..4000}; sleep 0.1"
    JobTool(s, [{"action": "start", "command": command}]).call()
    events = []
    tool = JobTool(s, [{"action": "wait", "job": "job.1", "timeout": 1}])
    tool.live_output = lambda stream, text: events.append((stream, text))

    tool.call()

    deltas = [text for stream, text in events if stream == "output"]
    assert events[-1] == ("", "")
    assert deltas[0] == "a" * 6000  # 窗口内时按增量推送
    # 日志超过窗口后，推送的是完整的可见尾部：带 `...` 前缀且以最新输出结尾
    assert deltas[-1].startswith("...") and deltas[-1].endswith("b" * 100)
    assert len(deltas[-1]) == 8000
    JobTool(s, [{"action": "kill", "job": "job.1"}]).call()


def test_job_wait_stream_clears_when_budget_is_exhausted(tmp_path, monkeypatch):
    """The live region is closed even when the wait ends by running out of budget."""
    s = session(tmp_path)
    monkeypatch.setattr(JobTool, "DEFAULT_WAIT", 0.2)
    monkeypatch.setattr(JobTool, "MAX_WAIT", 0.2)
    monkeypatch.setattr(JobTool, "POLL_INTERVAL", 0.01)
    JobTool(s, [{"action": "start", "command": "sleep 30"}]).call()
    events = []
    tool = JobTool(s, [{"action": "wait", "job": "job.1"}])
    tool.live_output = lambda stream, text: events.append((stream, text))

    tool.call()

    assert "Still running" in tool._format(s.jobs["job.1"], {"action": "wait", "job": "job.1"})
    assert events[-1] == ("", "")
    JobTool(s, [{"action": "kill", "job": "job.1"}]).call()


def test_job_wait_stream_clears_on_cancel(tmp_path, monkeypatch):
    """Ctrl-C abandons the wait and still closes the live region; the job keeps running."""
    s = session(tmp_path)
    monkeypatch.setattr(JobTool, "MAX_WAIT", 900)
    monkeypatch.setattr(JobTool, "POLL_INTERVAL", 0.01)
    # One line of output so the first poll pushes an event and the cancel fires right away,
    # instead of the loop below waiting out its whole deadline on a silent job.
    JobTool(s, [{"action": "start", "command": "printf 'x\\n'; sleep 30"}]).call()
    tool = JobTool(s, [{"action": "wait", "job": "job.1", "timeout": 900}])
    events = []
    tool.live_output = lambda stream, text: events.append((stream, text))
    result = []
    thread = threading.Thread(target=lambda: result.append(tool.call()))
    thread.start()
    deadline = time.monotonic() + 2
    while not events and time.monotonic() < deadline:
        time.sleep(0.01)
    tool.cancel()
    thread.join(timeout=5)

    assert not thread.is_alive(), "cancel did not interrupt the wait"
    assert ("output", "x\n") in events  # 流式输出在 cancel 前已到达
    assert s.jobs["job.1"].process.poll() is None  # 中断只放弃 wait,不杀 job
    assert events[-1] == ("", "")
    JobTool(s, [{"action": "kill", "job": "job.1"}]).call()


def test_job_wait_stream_clears_when_job_resolution_fails(tmp_path):
    """A wait on an unknown job raises ToolError and still closes the live region the runner
    already opened."""
    s = session(tmp_path)
    events = []
    tool = JobTool(s, [{"action": "wait", "job": "job.99"}])
    tool.live_output = lambda stream, text: events.append((stream, text))

    with pytest.raises(ToolError, match="unknown job"):
        tool.call()

    assert events == [("", "")]


def _job_call_lines(blocks) -> list[str]:
    """The `Job ...` root lines among the emitted blocks, in order."""
    return [line[0].text for block in blocks if isinstance(block, LogBlock) for line in block.walk() if line[0].label == "Job"]


def test_job_wait_prints_call_line_before_blocking(tmp_path, monkeypatch):
    """A Job wait blocks the agent with no live stream, so under yolo the runner prints the call
    line as soon as the wait starts -- before the result lands -- so the user can see the agent is
    waiting instead of a blank screen. The finish block then hangs its children under that root."""
    monkeypatch.setattr(JobTool, "DEFAULT_WAIT", 30)
    monkeypatch.setattr(JobTool, "POLL_INTERVAL", 0.01)
    s = session(tmp_path)
    s.settings.yolo = True  # no approval block, so the pre-block is the only thing drawing the root
    blocks: list[LogBlock | str] = []
    runner = ToolRunner(s, ContextManager(s), input_fn=lambda _prompt: "y", output_fn=blocks.append)
    JobTool(s, [{"action": "start", "command": "sleep 0.3; printf done"}]).call()

    runner.run([ToolCall("call_1", "Job", [{"action": "wait", "job": "job.1", "timeout": 30}])])

    # The first output is the call line printed before the block: a leaf LogBlock whose root is a
    # LogLine carrying the tool name and args, with no children yet.
    first = blocks[0]
    assert isinstance(first, LogBlock)
    root = next(first.walk())[0]
    assert root.label == "Job"
    assert "wait" in root.text and "job.1" in root.text
    # The finish block comes after, with no root of its own (the pre-block already drew it) and a
    # stored/done child hanging underneath.
    finish_blocks = [
        block for block in blocks[1:] if isinstance(block, LogBlock) and not any(isinstance(item, LogLine) and item.label == "Job" for item in block.items)
    ]
    assert finish_blocks, "no rootless finish block after the pre-block call line"
    assert any(line[0].label in {"stored", "done"} for block in finish_blocks for line in block.walk())
    # A non-blocking action (list) prints no pre-block call line: only the finish block appears.
    blocks.clear()
    runner.run([ToolCall("call_2", "Job", [{"action": "list"}])])
    assert len(blocks) == 1


def test_job_wait_call_line_is_not_repeated_after_an_approval(tmp_path, monkeypatch):
    """A Job wait needs confirmation, and the approval block already drew the call line before the
    block starts. The pre-block must stand down there, or the same line lands twice in a row."""
    monkeypatch.setattr(JobTool, "DEFAULT_WAIT", 30)
    monkeypatch.setattr(JobTool, "POLL_INTERVAL", 0.01)
    s = session(tmp_path)
    assert s.settings.yolo is False  # the approval path is the default one
    blocks: list[LogBlock | str] = []
    runner = ToolRunner(s, ContextManager(s), input_fn=lambda _prompt: "y", output_fn=blocks.append)
    JobTool(s, [{"action": "start", "command": "sleep 0.3; printf done"}]).call()

    runner.run([ToolCall("call_1", "Job", [{"action": "wait", "job": "job.1", "timeout": 30}])])

    call_lines = _job_call_lines(blocks)
    assert len(call_lines) == 1, f"the call line was drawn {len(call_lines)} times: {call_lines}"
    assert "wait" in call_lines[0] and "job.1" in call_lines[0]


def test_bash_behaviors(tmp_path):
    s = session(tmp_path)
    bash = BashTool(s, ["printf out; printf err >&2; exit 3"]).call()
    assert "* exit_code: 3" in bash
    assert "<stdout>\nout\n</stdout>" in bash
    assert "<stderr>\nerr\n</stderr>" in bash

    # Multibyte UTF-8 output large enough to span 4096-byte read boundaries must decode cleanly
    # (regression: per-chunk decoding mangled split characters into replacement chars).
    wide = BashTool(s, ['python3 -c "print(chr(0x4e2d)*3000)"']).call()
    assert "�" not in wide
    assert wide.count(chr(0x4E2D)) == 3000


def test_bash_starts_in_workspace_but_can_create_external_directory_after_approval(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    s = session(workspace)
    prompts = []
    runner = ToolRunner(s, ContextManager(s), input_fn=lambda prompt: prompts.append(prompt) or "y", output_fn=lambda text: None)

    runner.run([ToolCall("mkdir", "Bash", ["mkdir ../external"])])

    assert (tmp_path / "external").is_dir()
    assert len(prompts) == 1


def test_bash_cancel_kills_active_process(tmp_path):
    tool = BashTool(session(tmp_path), ["sleep 30"])
    finished = threading.Event()

    def run():
        tool.call()
        finished.set()

    thread = threading.Thread(target=run)
    thread.start()
    deadline = time.monotonic() + 1
    while tool._process is None and time.monotonic() < deadline:
        time.sleep(0.01)

    tool.cancel()

    assert finished.wait(timeout=1)
    thread.join(timeout=1)


def test_bash_fast_command_does_not_promote(tmp_path):
    s = session(tmp_path)
    s.settings.bash_wait_timeout = 5
    s.settings.shell_timeout = 30

    output = BashTool(s, ["printf hi"]).call()

    assert "* exit_code: 0" in output
    assert "hi" in output
    assert "backgrounded" not in output
    assert not s.jobs


def test_bash_live_preview_skips_unchanged_redraws(monkeypatch):
    printed = []
    now = [100.4]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    monkeypatch.setattr(render_module, "print_formatted_text", lambda ft, **kw: printed.append("".join(t for _, t in ft)))

    class FakeOut:
        def write_raw(self, s=""):
            pass

        def erase_end_of_line(self):
            pass

        def flush(self):
            pass

    p = BashLivePreview()
    p.output = FakeOut()
    p.active = True
    p.started_at = 100.0

    p.render()
    first = len(printed)
    p.render()
    assert len(printed) == first

    now[0] = 101.1
    p.render()
    assert len(printed) > first
    # BashLivePreview uses sub-second precision (`1.1s`) so the ticker feels live.
    assert any(LiveSpark.GLYPH + "running… 1.1s" in line for line in printed[first:])


def test_bash_promoted_job_is_killable(tmp_path):
    s = session(tmp_path)
    s.settings.bash_wait_timeout = 0.2
    s.settings.shell_timeout = 5

    BashTool(s, ["sleep 60"]).call()
    assert "job.1" in s.jobs
    job = s.jobs["job.1"]
    job.kill()
    assert job.status in {"done", "killed"}
    assert job.process.poll() is not None


def test_bash_promotion_disabled_when_wait_timeout_zero(tmp_path):
    s = session(tmp_path)
    s.settings.bash_wait_timeout = 0
    s.settings.shell_timeout = 0.2

    output = BashTool(s, ["sleep 5"]).call()

    assert "* exit_code: -1" in output
    assert "timeout" in output
    assert "backgrounded" not in output
    assert not s.jobs


def test_bash_readonly_auto_approval_classification(tmp_path):
    s = session(tmp_path)

    def readonly(command):
        return not BashTool(s, [command]).needs_confirmation()

    # Safe read-only commands auto-run (no confirmation prompt in non-yolo mode).
    assert readonly("ls -la")
    assert readonly("cat file.txt")
    assert readonly("wc -l minacode.py")
    assert readonly("find . -name '*.py'")
    assert readonly("rg needle src")
    assert readonly("git status --short")
    assert readonly("git --no-pager status --short")
    assert readonly("git diff HEAD~1")
    assert readonly("cat a | grep foo | wc -l")  # pipeline of safe commands
    assert readonly("ls && cat README.md")  # sequence of safe commands
    assert readonly("cd /Users/x/proj && git log --oneline -10")  # cd prefix is a benign builtin
    assert readonly("cd a; ls")
    assert readonly("ls -la && find . -maxdepth 2 -type f | grep -v .git | sort | head -80")
    assert readonly("cat f | sort -u | uniq -c")  # sort/uniq are read-only in pipelines
    assert readonly("grep foo f 2>/dev/null")  # discarding stderr is not a file write
    assert readonly("ls -la >/dev/null 2>&1")  # /dev/null + stderr-merge
    assert readonly("cat f | sed -n '1,20p'")  # sed for read-only filtering
    assert readonly("tree -L 2 src")

    # Anything that writes, executes code, mutates git, or hides execution still asks.
    assert not readonly("rm -rf build")
    # Every stage of a chain is validated — a safe first command must not whitelist a mutating one.
    assert not readonly("git log && rm -rf x")
    assert not readonly("ls ; rm x")
    assert not readonly("cat f && python3 evil.py")
    assert not readonly("git log & rm x")  # backgrounding
    assert not readonly("git commit -m x")
    assert not readonly("git checkout main")
    assert not readonly("echo hi > out.txt")  # redirection
    assert not readonly("cat >/dev/nullx")  # /dev/null is only a prefix; writes real file /dev/nullx
    assert not readonly("echo x >/dev/null.bak")  # /dev/null prefix of a real file
    assert not readonly("cat 2>/dev/nullish")  # /dev/null prefix on a stderr redirect
    assert not readonly("cat >/dev/null2>&1")  # writes /dev/null2, not the null device
    assert not readonly("cat $(cmd)")  # command substitution
    assert not readonly("python3 script.py")  # arbitrary code
    assert not readonly("find . -delete")  # destructive flag
    assert not readonly("find . -name x -fprint0 out")  # file-writing flag
    assert not readonly("cat f > g")  # redirection to a real file
    assert not readonly("sed -i s/a/b/ f")  # in-place edit
    assert not readonly("sort -o out.txt f")  # sort output file
    assert not readonly("tree -o out.txt")  # tree output file
    assert not readonly("sed -i s/a/b/ f")  # in-place edit
    assert not readonly("git diff --output=patch.txt")  # file-writing git option
    assert not readonly("git grep -O needle")  # opens files via pager/editor
    assert not readonly("git --paginate log")  # can invoke configured pager
    assert not readonly("ls & rm x")  # backgrounding
    assert not readonly("ls; rm x")  # unsafe stage in a sequence
    assert not readonly("FOO=1 env")  # env assignment / wrapper


def test_bash_slow_command_promotes_to_job(tmp_path):
    s = session(tmp_path)
    s.settings.bash_wait_timeout = 0.2
    s.settings.shell_timeout = 5

    output = BashTool(s, ["printf early; sleep 0.5; printf late"]).call()

    assert "* exit_code: -1" in output
    assert "early" in output
    assert "backgrounded after 0.2s" in output
    assert "job.1" in output
    assert "job.1" in s.jobs
    job = s.jobs["job.1"]
    assert job.stream_buffer is not None
    # `early` was consumed by the foreground streaming loop before promotion, so it lives in the
    # Bash result payload (asserted above). `late` was produced after the drainer took over, so it
    # lives in the promoted job's tail buffer.
    job.process.wait(timeout=5)
    for _ in range(50):
        if "late" in job.tail(4096):
            break
        import time as _t

        _t.sleep(0.05)
    assert "late" in job.tail(4096)
    job.update_status()
    assert job.status == "done"
    assert job.exit_code == 0


def test_bash_timeout_and_live_output(tmp_path):
    s = session(tmp_path)
    s.settings.shell_timeout = 0.2
    events = []
    tool = BashTool(s, ["printf live; sleep 5"])
    tool.live_output = lambda stream, text: events.append((stream, text))

    output = tool.call()

    assert "* exit_code: -1" in output
    assert "live" in output
    assert "timeout" in output
    assert ("stdout", "live") in events
    assert events[-1] == ("", "")


def test_bash_timeout_applies_after_output_streams_close(tmp_path):
    s = session(tmp_path)
    s.settings.shell_timeout = 0.05

    output = BashTool(s, ["exec 1>&- 2>&-; sleep 1"]).call()

    assert "* exit_code: -1" in output
    assert "timeout" in output


def test_job_captures_large_output_via_log_file(tmp_path):
    s = session(tmp_path)
    code = 'import sys; sys.stdout.write("x" * 1000000)'
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"
    JobTool(s, [{"action": "start", "command": command}]).call()
    job = s.jobs["job.1"]

    try:
        job.process.wait(timeout=2)
    finally:
        if job.process.poll() is None:
            job.kill(grace=0.1)

    job.update_status()
    assert job.status == "done"
    assert job.exit_code == 0
    assert job.tail(100) == "..." + "x" * 97


def test_job_start_captures_every_stage_of_a_compound_command(tmp_path):
    """The whole command is grouped before redirection, so output from early stages (not just the
    last) lands in the job log instead of leaking to the inherited stdout."""
    s = session(tmp_path)
    JobTool(s, [{"action": "start", "command": "printf first; printf second && printf third"}]).call()
    job = s.jobs["job.1"]

    try:
        job.process.wait(timeout=2)
    finally:
        if job.process.poll() is None:
            job.kill(grace=0.1)

    job.update_status()
    assert job.status == "done"
    log = job.tail(1000)
    assert "first" in log and "second" in log and "third" in log


def test_job_start_reclaims_finished_capacity(tmp_path, monkeypatch):
    s = session(tmp_path)
    monkeypatch.setattr(JobTool, "MAX_JOBS", 1)
    JobTool(s, [{"action": "start", "command": "true"}]).call()
    s.jobs["job.1"].process.wait(timeout=2)

    result = JobTool(s, [{"action": "start", "command": "true"}]).call()

    assert result.startswith("Started job.2")
    s.jobs["job.2"].process.wait(timeout=2)


def test_job_start_runs_shell_builtins_and_compound_commands(tmp_path):
    """`Job(start)` must run commands through the shell rather than `exec` the first word, or
    builtins like `cd` and compound commands like `cd dir && cmd` fail with `exec: cd: not found`."""
    s = session(tmp_path)
    sub = tmp_path / "sub"
    sub.mkdir()
    JobTool(s, [{"action": "start", "command": f"cd {shlex.quote(str(sub))} && printf marker"}]).call()
    job = s.jobs["job.1"]

    try:
        job.process.wait(timeout=2)
    finally:
        if job.process.poll() is None:
            job.kill(grace=0.1)

    job.update_status()
    assert job.status == "done"
    assert job.exit_code == 0
    assert "marker" in job.tail(100)


def test_job_start_uses_bash_highlighting(tmp_path):
    s = session(tmp_path)
    runner = ToolRunner(s, ContextManager(s))
    start = ToolCall("j1", "Job", [{"action": "start", "command": "pytest -q"}])
    wait = ToolCall("j2", "Job", [{"action": "wait", "job": "job.1"}])

    start_line = runner.log_root(runner.short_call(start), call=start)
    wait_line = runner.log_root(runner.short_call(wait), call=wait)

    assert start_line.syntax == "bash"
    assert wait_line.syntax == "tool-args"
    wait_segments = UiPrinter(output_fn=lambda text: None).log_segments(LogBlock([wait_line]))
    assert ("fg:#d2a8ff", "job.1") in wait_segments


def test_job_status_accepts_bare_numeric_id(tmp_path):
    s = session(tmp_path)
    JobTool(s, [{"action": "start", "command": "true"}]).call()
    s.jobs["job.1"].process.wait(timeout=2)

    result = JobTool(s, [{"action": "status", "job": "1"}]).call()

    assert "Status: done" in result
    assert "Exit code: 0" in result


def test_job_tail_respects_limits_smaller_than_ellipsis(tmp_path):
    s = session(tmp_path)
    JobTool(s, [{"action": "start", "command": "printf abcdef"}]).call()
    job = s.jobs["job.1"]
    job.process.wait(timeout=2)

    assert job.tail(1) == "."
    assert job.tail(2) == ".."
    assert job.tail(3) == "..."


def test_kill_finished_job_does_not_signal_stale_process(tmp_path):
    s = session(tmp_path)
    JobTool(s, [{"action": "start", "command": "true"}]).call()
    s.jobs["job.1"].process.wait(timeout=2)

    result = JobTool(s, [{"action": "kill", "job": "job.1"}]).call()

    assert "status=done" in result
    assert "exit_code=0" in result


def test_ps_hides_jobs_that_finished_without_polling(tmp_path):
    s = session(tmp_path)
    JobTool(s, [{"action": "start", "command": "true"}]).call()
    s.jobs["job.1"].process.wait(timeout=2)
    command_loop = CommandLoop(Agent(s), input_fn=lambda prompt="": "", output_fn=lambda text: None)

    assert ps_command(command_loop, "") == "No active jobs (1 total)."


def test_tool_runner_approved_live_bash_does_not_repeat_command(tmp_path):
    s = session(tmp_path)
    events = []
    runner = ToolRunner(s, ContextManager(s), input_fn=lambda prompt: "", output_fn=lambda text: events.append(("display", str(text))))
    runner.live_start = lambda: events.append(("start", ""))
    runner.live_output = lambda stream, text: events.append((stream, text))

    runner.run([ToolCall("bash", "Bash", ["bash -lc 'printf approved'"])])

    display = [text for kind, text in events if kind == "display"]
    assert display[0].startswith("  Bash  ")
    assert "approval required" not in display[0]
    assert display[-1].startswith("    ├ output")
    assert "Ctrl-O for more" in display[-1]
    assert "    └ stored tr." in display[-1]
    assert display[-1].endswith("[approved]")
    assert sum(text.startswith("  Bash  ") for text in display) == 1
    assert sum("printf approved" in text for text in display) == 1


def test_tool_runner_bash_preview_keeps_literal_closing_tags(tmp_path):
    s = session(tmp_path)
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)
    output = Tool.process_result("BashToolResult", 0, "before </stdout> after", "before </stderr> after")

    preview = runner.bash_result_preview(output, runner.BASH_TRANSCRIPT_PREVIEW_LINES)

    assert "before </stdout> after" in preview
    assert "before </stderr> after" in preview


def test_tool_runner_bash_preview_omits_past_limit(tmp_path):
    s = session(tmp_path)
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)
    limit = 24
    lines = [f"line {index}" for index in range(limit + 1)]

    preview = runner.preview_lines("\n".join(lines), limit)

    assert len(preview) == limit + 1
    assert preview[0] == "line 0"
    assert preview[limit // 2] == "... 1 line omitted ..."
    assert preview[-1] == lines[-1]


def test_tool_runner_compact_bash_result_keeps_bounded_output_without_live_frame(tmp_path):
    s = session(tmp_path)
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)
    output = Tool.process_result("BashToolResult", 0, "visible output", "")

    display = str(
        runner.finish_display(
            ToolCall("bash", "Bash", ["printf visible"]),
            "tr.1",
            output,
            failed=False,
            d=ToolDisplay(nested_display=True),
        )
    )

    assert display.startswith("    ├ output Ctrl-O for more")
    assert "visible output" in display


def test_tool_runner_failed_live_bash_does_not_repeat_command(tmp_path, monkeypatch):
    s = session(tmp_path)
    output = []
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: output.append(str(text)))
    runner.live_start = lambda: None
    runner.live_output = lambda _stream, _text: None
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("spawn failed")))

    runner.run([ToolCall("bash", "Bash", ["printf duplicate"])])

    assert output[0] == "  Bash  printf duplicate"
    assert output[1].startswith("    └ error ")
    assert "printf duplicate" not in output[1]
    assert "spawn failed" in output[1]


def test_tool_runner_finish_display_bounds_bash_output(tmp_path):
    s = session(tmp_path)
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)
    stdout = "\n".join(f"out {index}" for index in range(20))
    output = Tool.process_result("BashToolResult", 0, stdout, "err")

    display = str(runner.finish_display(ToolCall("bash", "Bash", ["printf lots"]), "tr.1", output, failed=False))

    assert display.startswith("  Bash  printf lots\n")
    assert "    ├ output Ctrl-O for more" in display
    assert "out 0" in display
    assert "... 17 lines omitted ..." in display
    assert "out 18" in display and "out 19" in display
    assert "err" in display
    assert display.endswith("    └ stored tr.1")


def test_tool_runner_finish_display_keeps_bounded_bash_output_after_live_preview(tmp_path):
    s = session(tmp_path)
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)
    output = Tool.process_result("BashToolResult", 0, "live output", "")

    display = str(runner.finish_display(ToolCall("bash", "Bash", ["printf live"]), "tr.1", output, failed=False))

    assert "    ├ output Ctrl-O for more" in display
    assert "live output" in display
    assert display.endswith("    └ stored tr.1")


def test_tool_runner_prints_bash_header_before_live_output(tmp_path):
    s = session(tmp_path)
    events = []
    runner = ToolRunner(
        s,
        ContextManager(s),
        input_fn=lambda prompt: (_ for _ in ()).throw(AssertionError("unexpected prompt")),
        output_fn=lambda text: events.append(("display", str(text))),
    )
    runner.live_start = lambda: events.append(("start", ""))
    runner.live_output = lambda stream, text: events.append((stream, text))

    runner.run([ToolCall("bash", "Bash", ["printf live"])])

    assert events[0] == ("display", "  Bash  printf live")
    assert events[1] == ("start", "")
    assert ("stdout", "live") in events
    assert events[-1][0] == "display"
    assert "    ├ output" in events[-1][1]
    assert "Ctrl-O for more" in events[-1][1]
    assert "live" in events[-1][1]
    assert "    └ stored tr." in events[-1][1]
    assert sum("printf live" in text for kind, text in events if kind == "display") == 1
    assert sum("Bash" in text for kind, text in events if kind == "display") == 1
    assert "live" in s.tool_records[-1].output


def test_tool_runner_starts_bash_live_preview_before_output(tmp_path):
    s = session(tmp_path)
    s.settings.yolo = True
    events = []
    runner = ToolRunner(s, ContextManager(s), input_fn=lambda prompt: (_ for _ in ()).throw(AssertionError("unexpected prompt")), output_fn=lambda text: None)
    runner.live_start = lambda: events.append(("start", ""))
    runner.live_output = lambda stream, text: events.append((stream, text))

    runner.run([ToolCall("bash", "Bash", ["printf live"])])

    assert events[0] == ("start", "")
    assert ("stdout", "live") in events
    assert events[-1] == ("", "")


def test_tool_runner_job_wait_starts_live_preview_with_budget(tmp_path, monkeypatch):
    """A blocking Job wait opens the same live preview as Bash, handing it the wait budget for
    the countdown; a non-blocking status opens nothing."""
    monkeypatch.setattr(JobTool, "POLL_INTERVAL", 0.01)
    s = session(tmp_path)
    s.settings.yolo = True
    events = []
    runner = ToolRunner(
        s,
        ContextManager(s),
        input_fn=lambda prompt: (_ for _ in ()).throw(AssertionError("unexpected prompt")),
        output_fn=lambda text: None,
    )
    runner.live_start = lambda budget=None: events.append(("start", budget))
    runner.live_output = lambda stream, text: events.append((stream, text))
    JobTool(s, [{"action": "start", "command": "sleep 0.2; printf done"}]).call()

    runner.run([ToolCall("call_1", "Job", [{"action": "wait", "job": "job.1", "timeout": 5}])])

    assert events[0] == ("start", 5)
    assert events[-1] == ("", "")

    # A status without a timeout holds nothing and opens no live region.
    events.clear()
    runner.run([ToolCall("call_2", "Job", [{"action": "status", "job": "job.1"}])])
    assert not any(kind == "start" for kind, _text in events)


def test_uiprinter_renders_bash_preview_like_live_output():
    ui = UiPrinter(output_fn=lambda text: None)
    block = LogBlock.hierarchy(
        LogLine("Bash", "cmd", LogRole.TOOL),
        [
            LogLine("", "stderr:", LogRole.OUTPUT, LogEdge.CONTINUE),
            LogLine("", "  Traceback", LogRole.OUTPUT, LogEdge.CONTINUE),
            LogLine("", "    File x", LogRole.OUTPUT, LogEdge.CONTINUE),
            LogLine("", "  AttributeError", LogRole.OUTPUT, LogEdge.CONTINUE),
        ],
    )
    segs = ui.log_segments(block)

    assert ("ansibrightblack", "stderr:") in segs
    assert ("ansibrightblack", "  Traceback") in segs
    assert ("ansibrightblack", "    File x") in segs
    assert ("ansibrightblack", "  AttributeError") in segs


def test_uiprinter_syntax_highlights_bash_arguments(tmp_path):
    s = session(tmp_path)
    line = ToolRunner(s, ContextManager(s)).log_root("Bash cd /tmp && printf '%s\\n' value")

    assert line.syntax == "bash"
    segments = UiPrinter(output_fn=lambda text: None).log_segments(LogBlock([line]))
    assert ("fg:#79c0ff", "cd") in segments
    assert ("fg:#79c0ff", "printf") in segments
    assert ("fg:#a5d6ff", "'%s\\n'") in segments
    assert not any("bg:" in style for style, _text in segments)
