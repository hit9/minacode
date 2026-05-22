import re

import pytest

from nanocode import Agent, EditFileTool, ReadTool, Session, ToolCallError


def _anchors(read_result: str) -> list[str]:
    return re.findall(r"^(\d+:[0-9a-f]{6})\|", read_result, re.MULTILINE)


def _read_anchors(session: Session, filepath: str, range_token: str = "0,0") -> list[str]:
    args = [filepath] if range_token == "0,0" else [filepath, range_token]
    return _anchors(ReadTool.make(session, args).call())


def test_edit_file_replaces_range_from_read_anchors(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    anchors = _read_anchors(session, "sample.txt")

    tool = EditFileTool.make(session, ["sample.txt", [{"op": "replace", "start": anchors[1], "end": anchors[1], "content": "BETA\n"}]])
    display = tool.preview()
    result = tool.call()

    assert tool.requires_confirmation(session) is True
    assert "-beta\n" in display
    assert "+BETA\n" in display
    assert path.read_text(encoding="utf-8") == "alpha\nBETA\ngamma\n"
    assert result == "\n".join(
        [
            "<EditFileToolResult>",
            "* path: sample.txt",
            "* edits: 1",
            "* range[1]: 1:2",
            "</EditFileToolResult>",
        ]
    )


def test_edit_file_accepts_full_hashline_anchor(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\nbeta\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    read_result = ReadTool.make(session, ["sample.txt"]).call()
    full_hashline = next(line for line in read_result.splitlines() if line.endswith("|beta"))

    EditFileTool.make(session, ["sample.txt", [{"op": "replace", "start": full_hashline, "end": full_hashline, "content": "BETA\n"}]]).call()

    assert path.read_text(encoding="utf-8") == "alpha\nBETA\n"


def test_edit_file_inserts_and_deletes_atomically(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\nbeta\ngamma\ndelta\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    anchors = _read_anchors(session, "sample.txt")

    result = EditFileTool.make(
        session,
        [
            "sample.txt",
            [
                {"op": "insert_after", "start": anchors[0], "content": "inserted\n"},
                {"op": "delete", "start": anchors[2], "end": anchors[2], "content": ""},
                {"op": "replace", "start": anchors[3], "end": anchors[3], "content": "DELTA\n"},
            ],
        ],
    ).call()

    assert "* edits: 3" in result
    assert path.read_text(encoding="utf-8") == "alpha\ninserted\nbeta\nDELTA\n"


def test_edit_file_rejects_stale_anchor_without_writing(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\nbeta\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    anchors = _read_anchors(session, "sample.txt")
    path.write_text("alpha\nchanged\n", encoding="utf-8")

    tool = EditFileTool.make(session, ["sample.txt", [{"op": "replace", "start": anchors[1], "end": anchors[1], "content": "BETA\n"}]])

    assert "stale anchor" in tool.preview()
    with pytest.raises(ToolCallError, match="stale anchor"):
        tool.call()
    assert path.read_text(encoding="utf-8") == "alpha\nchanged\n"


def test_edit_file_rejects_overlapping_edits_without_writing(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    anchors = _read_anchors(session, "sample.txt")

    tool = EditFileTool.make(
        session,
        [
            "sample.txt",
            [
                {"op": "replace", "start": anchors[0], "end": anchors[1], "content": "AB\n"},
                {"op": "replace", "start": anchors[1], "end": anchors[2], "content": "BG\n"},
            ],
        ],
    )

    with pytest.raises(ToolCallError, match="overlap"):
        tool.call()
    assert path.read_text(encoding="utf-8") == "alpha\nbeta\ngamma\n"


def test_edit_file_rejects_missing_files(tmp_path):
    session = Session(cwd=str(tmp_path))
    tool = EditFileTool.make(session, ["missing.txt", [{"op": "insert_after", "start": "0:abcdef", "content": "alpha\n"}]])

    assert "use CreateFile" in tool.preview()
    with pytest.raises(ToolCallError, match="use CreateFile"):
        tool.call()


def test_edit_file_rejects_wrong_arg_shape(tmp_path):
    session = Session(cwd=str(tmp_path))

    with pytest.raises(ToolCallError, match="requires args: filepath, edits"):
        EditFileTool.make(session, [])
    with pytest.raises(ToolCallError, match="edits cannot be empty"):
        EditFileTool.make(session, ["sample.txt", []])
    with pytest.raises(ToolCallError, match="edit op must be"):
        EditFileTool.make(session, ["sample.txt", [{"op": "move", "start": "0:abcdef"}]])


def test_agent_executes_edit_file_with_structured_args(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\nbeta\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    anchors = _read_anchors(session, "sample.txt")
    agent = Agent(session)

    latest = agent.execute_tool_calls(
        [
            {
                "name": "EditFile",
                "intention": "replace beta",
                "args": ["sample.txt", [{"op": "replace", "start": anchors[1], "end": anchors[1], "content": "BETA\n"}]],
            }
        ],
        confirm=lambda call, tool: True,
    )

    assert path.read_text(encoding="utf-8") == "alpha\nBETA\n"
    assert "<EditFileToolResult>" in latest
    assert agent.blackboard.verification_required is True
