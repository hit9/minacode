import pytest

from nanocode import CreateFileTool, Agent, Session, ToolCallError


def test_create_file_tool_creates_missing_file(tmp_path):
    path = tmp_path / "created.txt"
    session = Session(cwd=str(tmp_path))

    tool = CreateFileTool.make(session, ["created.txt", "alpha\n"])
    display = tool.preview()
    result = tool.call()

    assert tool.requires_confirmation(session) is True
    assert "+alpha\n" in display
    assert path.read_text(encoding="utf-8") == "alpha\n"
    assert result == "\n".join(
        [
            "<CreateFileToolResult>",
            "* path: created.txt",
            "* created: true",
            "</CreateFileToolResult>",
        ]
    )


def test_create_file_tool_rejects_existing_file(tmp_path):
    path = tmp_path / "created.txt"
    path.write_text("existing\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))

    tool = CreateFileTool.make(session, ["created.txt", "alpha\n"])

    assert "file already exists" in tool.preview()
    with pytest.raises(ToolCallError, match="file already exists"):
        tool.call()
    assert path.read_text(encoding="utf-8") == "existing\n"


def test_create_file_tool_creates_missing_parent_inside_cwd(tmp_path):
    path = tmp_path / "nested" / "created.txt"
    session = Session(cwd=str(tmp_path))

    tool = CreateFileTool.make(session, ["nested/created.txt", "alpha\n"])
    result = tool.call()

    assert path.read_text(encoding="utf-8") == "alpha\n"
    assert "* path: nested/created.txt" in result


def test_create_file_tool_rejects_missing_parent_outside_cwd(tmp_path):
    outside = tmp_path.parent / (tmp_path.name + "-outside") / "created.txt"
    session = Session(cwd=str(tmp_path))

    tool = CreateFileTool.make(session, [str(outside), "alpha\n"])

    with pytest.raises(ToolCallError, match="No such file or directory"):
        tool.call()
    assert not outside.exists()


def test_main_agent_can_execute_create_file_tool(tmp_path):
    path = tmp_path / "created.txt"
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)

    latest = agent.execute_tool_calls(
        [{"name": "CreateFile", "intention": "create sample file", "args": ["created.txt", "alpha\n"]}],
        confirm=lambda call, tool: True,
    )

    assert path.read_text(encoding="utf-8") == "alpha\n"
    assert "<CreateFileToolResult>" in latest
    assert agent.blackboard.checks_required is True
