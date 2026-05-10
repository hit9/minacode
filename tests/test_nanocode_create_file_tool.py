import pytest

from nanocode import CreateFileTool, MainAgent, Session, ToolCallError


def test_create_file_tool_creates_missing_file(tmp_path):
    path = tmp_path / "created.txt"
    session = Session(cwd=str(tmp_path))

    tool = CreateFileTool.make(session, ["created.txt", "alpha\n"])
    display = tool.display()
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

    assert "file already exists" in tool.display()
    with pytest.raises(ToolCallError, match="file already exists"):
        tool.call()
    assert path.read_text(encoding="utf-8") == "existing\n"


def test_main_agent_can_execute_create_file_tool(tmp_path):
    path = tmp_path / "created.txt"
    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)

    latest = agent.execute_tool_calls(
        [{"name": "CreateFile", "intention": "create sample file", "args": ["created.txt", "alpha\n"]}],
        confirm=lambda call, tool: True,
    )

    assert path.read_text(encoding="utf-8") == "alpha\n"
    assert "<CreateFileToolResult>" in latest
    assert agent.blackboard.verification_required is True
