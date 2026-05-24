import shutil
import subprocess

import pytest

from nanocode import GitTool, Session, ToolCallError


def test_git_tool_runs_readonly_git_command(tmp_path):
    if not shutil.which("git"):
        pytest.skip("git not installed")
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True, text=True)
    (tmp_path / "sample.txt").write_text("hello\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))

    tool = GitTool.make(session, ["status", "--short"])
    result = tool.call()

    assert tool.requires_confirmation(session) is False
    assert "<GitToolResult>" in result
    assert "* exit_code: 0" in result
    assert "?? sample.txt" in result


def test_git_tool_marks_mutating_commands_for_confirmation(tmp_path):
    if not shutil.which("git"):
        pytest.skip("git not installed")
    session = Session(cwd=str(tmp_path))

    tool = GitTool.make(session, ["add", "sample.txt"])

    assert tool.requires_confirmation(session) is True


def test_git_tool_rejects_empty_args(tmp_path):
    session = Session(cwd=str(tmp_path))

    with pytest.raises(ToolCallError, match="requires at least one git arg"):
        GitTool.make(session, [])


def test_git_tool_rejects_cwd_outside_session_cwd(tmp_path):
    if not shutil.which("git"):
        pytest.skip("git not installed")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = Session(cwd=str(repo))

    with pytest.raises(ToolCallError, match="path outside cwd"):
        GitTool.make(session, ["cwd=..", "status"])
