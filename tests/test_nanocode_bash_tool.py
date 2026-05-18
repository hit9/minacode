import os
import signal
import time

from nanocode import BashTool, RuntimeSettings, Session


def test_bash_tool_runs_command_and_returns_output(tmp_path):
    session = Session(cwd=str(tmp_path))

    tool = BashTool.make(session, ["printf hello"])
    result = tool.call()

    assert tool.requires_confirmation(session) is True
    assert result == "\n".join(
        [
            "<BashToolResult>",
            "* exit_code: 0",
            "<stdout>",
            "hello",
            "</stdout>",
            "</BashToolResult>",
        ]
    )


def test_bash_tool_returns_nonzero_exit_and_stderr(tmp_path):
    session = Session(cwd=str(tmp_path))

    result = BashTool.make(session, ["printf nope >&2; exit 7"]).call()

    assert "<BashToolResult>" in result
    assert "* exit_code: 7" in result
    assert "<stderr>\nnope\n</stderr>" in result


def test_bash_tool_times_out_and_reports_timeout(tmp_path):
    session = Session(cwd=str(tmp_path), settings=RuntimeSettings(shell_timeout=0))

    result = BashTool.make(session, ["sleep 5 & wait"]).call()

    assert "* exit_code: -1" in result
    assert "timeout" in result


def test_bash_tool_kills_process_group_on_interrupt(tmp_path, monkeypatch):
    session = Session(cwd=str(tmp_path), settings=RuntimeSettings(shell_timeout=30))
    pid_file = tmp_path / "pid"
    tool = BashTool.make(session, [f"echo $$ > {pid_file}; printf started; sleep 30"])
    original_read_chunk = BashTool._read_stream_chunk

    def interrupt_on_output(selector, key, stdout_parts, stderr_parts):
        result = original_read_chunk(selector, key, stdout_parts, stderr_parts)
        if "started" in "".join(stdout_parts):
            raise KeyboardInterrupt()
        return result

    monkeypatch.setattr(BashTool, "_read_stream_chunk", staticmethod(interrupt_on_output))

    try:
        result = tool.call()
        assert "* exit_code: -1" in result
        assert "* interrupted: true" in result
        assert "* reason: user_ctrl_c" in result
        pid = int(pid_file.read_text(encoding="utf-8").strip())
        for _ in range(20):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            raise AssertionError("bash process was not killed")
    finally:
        if pid_file.exists():
            pid = int(pid_file.read_text(encoding="utf-8").strip())
            try:
                os.killpg(pid, signal.SIGKILL)
            except OSError:
                pass
