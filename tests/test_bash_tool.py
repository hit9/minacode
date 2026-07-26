import shlex
import sys
import threading
import time

import pytest

import minacode as n


def session(tmp_path):
    return n.Session(cwd=str(tmp_path))


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
    with pytest.raises(n.ToolError, match=message):
        n.JobTool(session(tmp_path), [payload]).call()


def test_job_wait_and_list_report_completed_output(tmp_path):
    s = session(tmp_path)
    assert n.JobTool(s, [{"action": "list"}]).call() == "No jobs."
    n.JobTool(s, [{"action": "start", "command": "printf completed"}]).call()

    waited = n.JobTool(s, [{"action": "wait", "job": "job.1", "timeout": 2}]).call()
    listed = n.JobTool(s, [{"action": "list"}]).call()

    assert "Status: done" in waited
    assert "Exit code: 0" in waited
    assert "--- output ---\ncompleted" in waited
    assert "| job.1 | done | 0 | printf completed |" in listed


def test_bash_behaviors(tmp_path):
    s = session(tmp_path)
    bash = n.BashTool(s, ["printf out; printf err >&2; exit 3"]).call()
    assert "* exit_code: 3" in bash
    assert "<stdout>\nout\n</stdout>" in bash
    assert "<stderr>\nerr\n</stderr>" in bash

    # Multibyte UTF-8 output large enough to span 4096-byte read boundaries must decode cleanly
    # (regression: per-chunk decoding mangled split characters into replacement chars).
    wide = n.BashTool(s, ['python3 -c "print(chr(0x4e2d)*3000)"']).call()
    assert "�" not in wide
    assert wide.count(chr(0x4E2D)) == 3000


def test_bash_cancel_kills_active_process(tmp_path):
    tool = n.BashTool(session(tmp_path), ["sleep 30"])
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

    output = n.BashTool(s, ["printf hi"]).call()

    assert "* exit_code: 0" in output
    assert "hi" in output
    assert "backgrounded" not in output
    assert not s.jobs


def test_bash_live_preview_skips_unchanged_redraws(monkeypatch):
    printed = []
    now = [100.4]
    monkeypatch.setattr(n.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(n.render, "print_formatted_text", lambda ft, **kw: printed.append("".join(t for _, t in ft)))

    class FakeOut:
        def write_raw(self, s=""):
            pass

        def erase_end_of_line(self):
            pass

        def flush(self):
            pass

    p = n.BashLivePreview()
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
    assert any("  ├ running… 1.1s" in line for line in printed[first:])


def test_bash_promoted_job_is_killable(tmp_path):
    s = session(tmp_path)
    s.settings.bash_wait_timeout = 0.2
    s.settings.shell_timeout = 5

    n.BashTool(s, ["sleep 60"]).call()
    assert "job.1" in s.jobs
    job = s.jobs["job.1"]
    job.kill()
    assert job.status in {"done", "killed"}
    assert job.process.poll() is not None


def test_bash_promotion_disabled_when_wait_timeout_zero(tmp_path):
    s = session(tmp_path)
    s.settings.bash_wait_timeout = 0
    s.settings.shell_timeout = 0.2

    output = n.BashTool(s, ["sleep 5"]).call()

    assert "* exit_code: -1" in output
    assert "timeout" in output
    assert "backgrounded" not in output
    assert not s.jobs


def test_bash_readonly_auto_approval_classification(tmp_path):
    s = session(tmp_path)

    def readonly(command):
        return not n.BashTool(s, [command]).needs_confirmation()

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

    output = n.BashTool(s, ["printf early; sleep 0.5; printf late"]).call()

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
    tool = n.BashTool(s, ["printf live; sleep 5"])
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

    output = n.BashTool(s, ["exec 1>&- 2>&-; sleep 1"]).call()

    assert "* exit_code: -1" in output
    assert "timeout" in output


def test_job_captures_large_output_via_log_file(tmp_path):
    s = session(tmp_path)
    code = 'import sys; sys.stdout.write("x" * 1000000)'
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"
    n.JobTool(s, [{"action": "start", "command": command}]).call()
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
    n.JobTool(s, [{"action": "start", "command": "printf first; printf second && printf third"}]).call()
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
    monkeypatch.setattr(n.JobTool, "MAX_JOBS", 1)
    n.JobTool(s, [{"action": "start", "command": "true"}]).call()
    s.jobs["job.1"].process.wait(timeout=2)

    result = n.JobTool(s, [{"action": "start", "command": "true"}]).call()

    assert result.startswith("Started job.2")
    s.jobs["job.2"].process.wait(timeout=2)


def test_job_start_runs_shell_builtins_and_compound_commands(tmp_path):
    """`Job(start)` must run commands through the shell rather than `exec` the first word, or
    builtins like `cd` and compound commands like `cd dir && cmd` fail with `exec: cd: not found`."""
    s = session(tmp_path)
    sub = tmp_path / "sub"
    sub.mkdir()
    n.JobTool(s, [{"action": "start", "command": f"cd {shlex.quote(str(sub))} && printf marker"}]).call()
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
    runner = n.ToolRunner(s, n.ContextManager(s))
    start = n.ToolCall("j1", "Job", [{"action": "start", "command": "pytest -q"}])
    wait = n.ToolCall("j2", "Job", [{"action": "wait", "job": "job.1"}])

    start_line = runner.log_root(runner.short_call(start), call=start)
    wait_line = runner.log_root(runner.short_call(wait), call=wait)

    assert start_line.syntax == "bash"
    assert wait_line.syntax == "tool-args"
    wait_segments = n.UiPrinter(output_fn=lambda text: None).log_segments(n.LogBlock([wait_line]))
    assert ("fg:#d2a8ff", "job.1") in wait_segments


def test_job_status_accepts_bare_numeric_id(tmp_path):
    s = session(tmp_path)
    n.JobTool(s, [{"action": "start", "command": "true"}]).call()
    s.jobs["job.1"].process.wait(timeout=2)

    result = n.JobTool(s, [{"action": "status", "job": "1"}]).call()

    assert "Status: done" in result
    assert "Exit code: 0" in result


def test_job_tail_respects_limits_smaller_than_ellipsis(tmp_path):
    s = session(tmp_path)
    n.JobTool(s, [{"action": "start", "command": "printf abcdef"}]).call()
    job = s.jobs["job.1"]
    job.process.wait(timeout=2)

    assert job.tail(1) == "."
    assert job.tail(2) == ".."
    assert job.tail(3) == "..."


def test_kill_finished_job_does_not_signal_stale_process(tmp_path):
    s = session(tmp_path)
    n.JobTool(s, [{"action": "start", "command": "true"}]).call()
    s.jobs["job.1"].process.wait(timeout=2)

    result = n.JobTool(s, [{"action": "kill", "job": "job.1"}]).call()

    assert "status=done" in result
    assert "exit_code=0" in result


def test_ps_hides_jobs_that_finished_without_polling(tmp_path):
    s = session(tmp_path)
    n.JobTool(s, [{"action": "start", "command": "true"}]).call()
    s.jobs["job.1"].process.wait(timeout=2)
    command_loop = n.CommandLoop(n.Agent(s), input_fn=lambda prompt="": "", output_fn=lambda text: None)

    assert command_loop.ps_command("") == "No active jobs (1 total)."


def test_tool_runner_approved_live_bash_does_not_repeat_command(tmp_path):
    s = session(tmp_path)
    events = []
    runner = n.ToolRunner(s, n.ContextManager(s), input_fn=lambda prompt: "", output_fn=lambda text: events.append(("display", str(text))))
    runner.live_start = lambda: events.append(("start", ""))
    runner.live_output = lambda stream, text: events.append((stream, text))

    runner.run([n.ToolCall("bash", "Bash", ["bash -lc 'printf approved'"])])

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
    runner = n.ToolRunner(s, n.ContextManager(s), output_fn=lambda text: None)
    output = n.Tool.process_result("BashToolResult", 0, "before </stdout> after", "before </stderr> after")

    preview = runner.bash_result_preview(output)

    assert "before </stdout> after" in preview
    assert "before </stderr> after" in preview


def test_tool_runner_bash_preview_omits_past_limit(tmp_path):
    s = session(tmp_path)
    runner = n.ToolRunner(s, n.ContextManager(s), output_fn=lambda text: None)
    lines = [f"line {index}" for index in range(n.ToolRunner.BASH_PREVIEW_LINES + 1)]

    preview = runner.preview_lines("\n".join(lines))

    assert len(preview) == n.ToolRunner.BASH_PREVIEW_LINES + 1
    assert preview[0] == "line 0"
    assert preview[n.ToolRunner.BASH_PREVIEW_LINES // 2] == "... 1 line omitted ..."
    assert preview[-1] == lines[-1]


def test_tool_runner_compact_bash_result_keeps_bounded_output_without_live_frame(tmp_path):
    s = session(tmp_path)
    runner = n.ToolRunner(s, n.ContextManager(s), output_fn=lambda text: None)
    output = n.Tool.process_result("BashToolResult", 0, "visible output", "")

    display = str(
        runner.finish_display(
            n.ToolCall("bash", "Bash", ["printf visible"]),
            "tr.1",
            output,
            failed=False,
            d=n.ToolDisplay(nested_display=True),
        )
    )

    assert display.startswith("    ├ output Ctrl-O for more")
    assert "visible output" in display


def test_tool_runner_failed_live_bash_does_not_repeat_command(tmp_path, monkeypatch):
    s = session(tmp_path)
    output = []
    runner = n.ToolRunner(s, n.ContextManager(s), output_fn=lambda text: output.append(str(text)))
    runner.live_start = lambda: None
    runner.live_output = lambda _stream, _text: None
    monkeypatch.setattr(n.subprocess, "Popen", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("spawn failed")))

    runner.run([n.ToolCall("bash", "Bash", ["printf duplicate"])])

    assert output[0] == "  Bash  printf duplicate"
    assert output[1].startswith("    └ error ")
    assert "printf duplicate" not in output[1]
    assert "spawn failed" in output[1]


def test_tool_runner_finish_display_bounds_bash_output(tmp_path):
    s = session(tmp_path)
    runner = n.ToolRunner(s, n.ContextManager(s), output_fn=lambda text: None)
    stdout = "\n".join(f"out {index}" for index in range(20))
    output = n.Tool.process_result("BashToolResult", 0, stdout, "err")

    display = str(runner.finish_display(n.ToolCall("bash", "Bash", ["printf lots"]), "tr.1", output, failed=False))

    assert display.startswith("  Bash  printf lots\n")
    assert "    ├ output Ctrl-O for more" in display
    assert "out 0" in display
    assert "... 17 lines omitted ..." in display
    assert "out 18" in display and "out 19" in display
    assert "err" in display
    assert display.endswith("    └ stored tr.1")


def test_tool_runner_finish_display_keeps_bounded_bash_output_after_live_preview(tmp_path):
    s = session(tmp_path)
    runner = n.ToolRunner(s, n.ContextManager(s), output_fn=lambda text: None)
    output = n.Tool.process_result("BashToolResult", 0, "live output", "")

    display = str(runner.finish_display(n.ToolCall("bash", "Bash", ["printf live"]), "tr.1", output, failed=False))

    assert "    ├ output Ctrl-O for more" in display
    assert "live output" in display
    assert display.endswith("    └ stored tr.1")


def test_tool_runner_prints_bash_header_before_live_output(tmp_path):
    s = session(tmp_path)
    events = []
    runner = n.ToolRunner(
        s,
        n.ContextManager(s),
        input_fn=lambda prompt: (_ for _ in ()).throw(AssertionError("unexpected prompt")),
        output_fn=lambda text: events.append(("display", str(text))),
    )
    runner.live_start = lambda: events.append(("start", ""))
    runner.live_output = lambda stream, text: events.append((stream, text))

    runner.run([n.ToolCall("bash", "Bash", ["printf live"])])

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
    runner = n.ToolRunner(
        s, n.ContextManager(s), input_fn=lambda prompt: (_ for _ in ()).throw(AssertionError("unexpected prompt")), output_fn=lambda text: None
    )
    runner.live_start = lambda: events.append(("start", ""))
    runner.live_output = lambda stream, text: events.append((stream, text))

    runner.run([n.ToolCall("bash", "Bash", ["printf live"])])

    assert events[0] == ("start", "")
    assert ("stdout", "live") in events
    assert events[-1] == ("", "")


def test_uiprinter_renders_bash_preview_like_live_output():
    ui = n.UiPrinter(output_fn=lambda text: None)
    block = n.LogBlock.hierarchy(
        n.LogLine("Bash", "cmd", n.LogRole.TOOL),
        [
            n.LogLine("", "stderr:", n.LogRole.OUTPUT, n.LogEdge.CONTINUE),
            n.LogLine("", "  Traceback", n.LogRole.OUTPUT, n.LogEdge.CONTINUE),
            n.LogLine("", "    File x", n.LogRole.OUTPUT, n.LogEdge.CONTINUE),
            n.LogLine("", "  AttributeError", n.LogRole.OUTPUT, n.LogEdge.CONTINUE),
        ],
    )
    segs = ui.log_segments(block)

    assert ("ansibrightblack", "stderr:") in segs
    assert ("ansibrightblack", "  Traceback") in segs
    assert ("ansibrightblack", "    File x") in segs
    assert ("ansibrightblack", "  AttributeError") in segs


def test_uiprinter_syntax_highlights_bash_arguments(tmp_path):
    s = session(tmp_path)
    line = n.ToolRunner(s, n.ContextManager(s)).log_root("Bash cd /tmp && printf '%s\\n' value")

    assert line.syntax == "bash"
    segments = n.UiPrinter(output_fn=lambda text: None).log_segments(n.LogBlock([line]))
    assert ("fg:#79c0ff", "cd") in segments
    assert ("fg:#79c0ff", "printf") in segments
    assert ("fg:#a5d6ff", "'%s\\n'") in segments
    assert not any("bg:" in style for style, _text in segments)
