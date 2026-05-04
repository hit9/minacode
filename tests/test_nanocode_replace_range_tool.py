import pytest

from nanocode import ReadTool, ReplaceRangeTool, Session, ToolCallError


def _fingerprint(read_result: str) -> str:
    return read_result.split("<fingerprint>", 1)[1].split("</fingerprint>", 1)[0]


def test_replace_range_tool_replaces_range_when_fingerprint_matches(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    fingerprint = _fingerprint(ReadTool.make(session, ["sample.txt", "1", "2"]).call())

    tool = ReplaceRangeTool.make(session, ["sample.txt", "1", "2", fingerprint, "BETA\n"])
    display = tool.display()
    result = tool.call()

    assert ReplaceRangeTool.name() == "ReplaceRange"
    assert tool.requires_confirmation(session) is True
    assert display.startswith("--- ")
    assert "-beta\n" in display
    assert "+BETA\n" in display
    assert path.read_text(encoding="utf-8") == "alpha\nBETA\ngamma\n"
    assert result == "\n".join(
        [
            "<ReplaceRangeToolResult>",
            "* path: sample.txt",
            "* range: 1:2",
            f"* fingerprint: {fingerprint}",
            "</ReplaceRangeToolResult>",
        ]
    )


def test_replace_range_tool_rejects_fingerprint_mismatch(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\nbeta\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))

    tool = ReplaceRangeTool.make(session, ["sample.txt", "1", "2", "bad", "BETA\n"])

    display = tool.display()

    assert display.startswith("ReplaceRange(")
    assert "# preview unavailable: fingerprint mismatch" in display
    assert "current " in display
    with pytest.raises(ToolCallError, match="fingerprint mismatch"):
        tool.call()
    assert path.read_text(encoding="utf-8") == "alpha\nbeta\n"


def test_replace_range_tool_replaces_to_eof_when_end_is_zero(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    fingerprint = _fingerprint(ReadTool.make(session, ["sample.txt", "1", "0"]).call())

    tool = ReplaceRangeTool.make(session, ["sample.txt", "1", "0", fingerprint, "tail\n"])
    result = tool.call()

    assert path.read_text(encoding="utf-8") == "alpha\ntail\n"
    assert "* range: 1:3" in result


def test_replace_range_tool_inserts_when_start_equals_end(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\ngamma\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    fingerprint = _fingerprint(ReadTool.make(session, ["sample.txt", "1", "1"]).call())

    ReplaceRangeTool.make(session, ["sample.txt", "1", "1", fingerprint, "beta\n"]).call()

    assert path.read_text(encoding="utf-8") == "alpha\nbeta\ngamma\n"


def test_replace_range_tool_rejects_no_change(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\nbeta\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    fingerprint = _fingerprint(ReadTool.make(session, ["sample.txt", "1", "2"]).call())

    tool = ReplaceRangeTool.make(session, ["sample.txt", "1", "2", fingerprint, "beta\n"])

    with pytest.raises(ToolCallError, match="range replacement produced no changes"):
        tool.call()
    assert path.read_text(encoding="utf-8") == "alpha\nbeta\n"
