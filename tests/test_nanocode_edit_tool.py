import pytest

from nanocode import EditTool, Session, ToolCallError


def test_edit_tool_replaces_first_exact_match(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\nbeta\nbeta\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))

    tool = EditTool.make(session, ["sample.txt", "beta", "BETA"])
    display = tool.preview()
    result = tool.call()

    assert tool.requires_confirmation(session) is True
    assert "-beta\n" in display
    assert "+BETA\n" in display
    assert path.read_text(encoding="utf-8") == "alpha\nBETA\nbeta\n"
    assert result == "\n".join(
        [
            "<EditToolResult>",
            "* path: sample.txt",
            "* replacements: 1",
            "</EditToolResult>",
        ]
    )


def test_edit_tool_raises_when_find_text_is_missing(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))

    tool = EditTool.make(session, ["sample.txt", "missing", "replacement"])

    with pytest.raises(ToolCallError, match="target `find` text not found"):
        tool.call()


def test_edit_tool_creates_missing_file_with_empty_find(tmp_path):
    path = tmp_path / "created.txt"
    session = Session(cwd=str(tmp_path))

    tool = EditTool.make(session, ["created.txt", "", "alpha\n"])
    display = tool.preview()
    result = tool.call()

    assert "+alpha\n" in display
    assert path.read_text(encoding="utf-8") == "alpha\n"
    assert result == "\n".join(
        [
            "<EditToolResult>",
            "* path: created.txt",
            "* created: true",
            "</EditToolResult>",
        ]
    )


def test_edit_tool_rejects_wrong_arg_count_with_actionable_error(tmp_path):
    session = Session(cwd=str(tmp_path))

    with pytest.raises(ToolCallError, match=r'Edit args error: got 0 args; expected \["filepath", "find", "replace"\]'):
        EditTool.make(session, [])


def test_edit_tool_rejects_empty_find_text_for_existing_file(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))

    tool = EditTool.make(session, ["sample.txt", "", "replacement"])

    assert "empty find creates missing files only" in tool.preview()
    with pytest.raises(ToolCallError, match="empty find creates missing files only"):
        tool.call()
    assert path.read_text(encoding="utf-8") == "alpha\n"


def test_edit_tool_display_falls_back_when_find_text_is_missing(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))

    tool = EditTool.make(session, ["sample.txt", "missing", "replacement"])

    assert tool.preview() == f'Edit({path}, find="missing")'
