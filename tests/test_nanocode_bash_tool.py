from nanocode import BashTool, Session


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
    session = Session(cwd=str(tmp_path), shell_timeout=0)

    result = BashTool.make(session, ["sleep 5 & wait"]).call()

    assert "* exit_code: -1" in result
    assert "timeout" in result
