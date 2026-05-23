import pytest

from nanocode import ListTool, Session, ToolCallError


def test_list_dir_tool_lists_filtered_entries_relative_to_cwd(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "app.py").write_text("print('hi')\n", encoding="utf-8")
    (src / "notes.md").write_text("notes\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))

    tool = ListTool.make(session, ["src", "*.py"])

    assert tool.requires_confirmation(session) is False
    assert tool.call() == "\n".join(
        [
            "<ListToolResult>",
            "* (file): src/app.py",
            "</ListToolResult>",
        ]
    )


def test_list_dir_tool_sorts_dirs_before_files_then_by_name(tmp_path):
    (tmp_path / "b.txt").write_text("b\n", encoding="utf-8")
    (tmp_path / "a.txt").write_text("a\n", encoding="utf-8")
    (tmp_path / "z_dir").mkdir()
    (tmp_path / "a_dir").mkdir()
    session = Session(cwd=str(tmp_path))

    result = ListTool.make(session, ["."]).call()

    assert result == "\n".join(
        [
            "<ListToolResult>",
            "* (dir): a_dir",
            "* (dir): z_dir",
            "* (file): a.txt",
            "* (file): b.txt",
            "</ListToolResult>",
        ]
    )


def test_list_dir_tool_defaults_to_cwd(tmp_path):
    (tmp_path / "sample.txt").write_text("alpha\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))

    result = ListTool.make(session, []).call()

    assert result == "\n".join(
        [
            "<ListToolResult>",
            "* (file): sample.txt",
            "</ListToolResult>",
        ]
    )


def test_list_dir_tool_rejects_non_directory(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))

    tool = ListTool.make(session, ["sample.txt"])

    with pytest.raises(ToolCallError, match="not a directory"):
        tool.call()
