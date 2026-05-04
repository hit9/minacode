import json

import pytest

from nanocode import Agent, BatchReplaceRangesTool, RangeFingerprintStore, ReadTool, ReplaceRangeTool, Session, ToolCallError


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


def test_replace_range_tool_adds_line_break_before_following_content(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    fingerprint = _fingerprint(ReadTool.make(session, ["sample.txt", "1", "2"]).call())

    ReplaceRangeTool.make(session, ["sample.txt", "1", "2", fingerprint, "BETA"]).call()

    assert path.read_text(encoding="utf-8") == "alpha\nBETA\ngamma\n"


def test_replace_range_tool_relocates_cached_fingerprint_after_line_shift(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    fingerprint = _fingerprint(ReadTool.make(session, ["sample.txt", "2", "3"]).call())
    path.write_text("zero\nalpha\nbeta\ngamma\n", encoding="utf-8")

    result = ReplaceRangeTool.make(session, ["sample.txt", "2", "3", fingerprint, "GAMMA\n"]).call()

    assert path.read_text(encoding="utf-8") == "zero\nalpha\nbeta\nGAMMA\n"
    assert "* range: 3:4" in result
    assert "* relocated_from: 2:3" in result


def test_replace_range_tool_rejects_ambiguous_cached_relocation(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    fingerprint = _fingerprint(ReadTool.make(session, ["sample.txt", "1", "2"]).call())
    path.write_text("zero\nalpha\nbeta\nbeta\ngamma\n", encoding="utf-8")

    tool = ReplaceRangeTool.make(session, ["sample.txt", "1", "2", fingerprint, "BETA\n"])

    with pytest.raises(ToolCallError, match="cached range matched multiple locations"):
        tool.call()
    assert path.read_text(encoding="utf-8") == "zero\nalpha\nbeta\nbeta\ngamma\n"


def test_replace_range_tool_rejects_full_file_fingerprint_for_partial_range(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    fingerprint = _fingerprint(ReadTool.make(session, ["sample.txt"]).call())

    tool = ReplaceRangeTool.make(session, ["sample.txt", "1", "2", fingerprint, "BETA\n"])
    display = tool.display()

    assert display.startswith("ReplaceRange(")
    assert "# preview unavailable: fingerprint mismatch" in display
    assert "call Read(filepath, 1, 2)" in display
    assert "--- " not in display
    with pytest.raises(ToolCallError, match=r"call Read\(filepath, 1, 2\)"):
        tool.call()
    assert path.read_text(encoding="utf-8") == "alpha\nbeta\ngamma\n"


def test_replace_range_tool_rejects_fingerprint_mismatch(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\nbeta\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))

    tool = ReplaceRangeTool.make(session, ["sample.txt", "1", "2", "bad", "BETA\n"])

    display = tool.display()

    assert display.startswith("ReplaceRange(")
    assert "# preview unavailable: fingerprint mismatch" in display
    assert "current " in display
    assert "call Read(filepath, 1, 2)" in display
    with pytest.raises(ToolCallError, match=r"call Read\(filepath, 1, 2\)"):
        tool.call()
    assert path.read_text(encoding="utf-8") == "alpha\nbeta\n"


def test_replace_range_cache_is_bounded(tmp_path):
    session = Session(cwd=str(tmp_path))
    store = session.range_fingerprints

    for index in range(RangeFingerprintStore.MAX_ENTRIES + 5):
        store.remember(filepath=str(tmp_path / "sample.txt"), start=index, end=index + 1, content="line " + str(index))

    assert len(store) == RangeFingerprintStore.MAX_ENTRIES


def test_replace_range_cache_clears_when_goal_changes(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    _fingerprint(ReadTool.make(session, ["sample.txt", "1", "2"]).call())

    Agent(session).apply_response({"goal_update": "new goal"})

    assert len(session.range_fingerprints) == 0


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


def test_replace_ranges_tool_applies_multiple_ranges_against_one_snapshot(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\nbeta\ngamma\ndelta\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    beta = _fingerprint(ReadTool.make(session, ["sample.txt", "1", "2"]).call())
    delta = _fingerprint(ReadTool.make(session, ["sample.txt", "3", "4"]).call())
    edits = json.dumps(
        [
            {"start": 1, "end": 2, "fingerprint": beta, "content": "BETA\nextra\n"},
            {"start": 3, "end": 4, "fingerprint": delta, "content": "DELTA\n"},
        ]
    )

    tool = BatchReplaceRangesTool.make(session, ["sample.txt", edits])
    display = tool.display()
    result = tool.call()

    assert BatchReplaceRangesTool.name() == "BatchReplaceRanges"
    assert tool.requires_confirmation(session) is True
    assert "-beta\n" in display
    assert "+BETA\n" in display
    assert path.read_text(encoding="utf-8") == "alpha\nBETA\nextra\ngamma\nDELTA\n"
    assert "* edits: 2" in result
    assert "* range 1: 1:2" in result
    assert "* range 2: 3:4" in result


def test_replace_ranges_tool_adds_line_break_before_following_content(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    beta = _fingerprint(ReadTool.make(session, ["sample.txt", "1", "2"]).call())
    edits = json.dumps([{"start": 1, "end": 2, "fingerprint": beta, "content": "BETA"}])

    BatchReplaceRangesTool.make(session, ["sample.txt", edits]).call()

    assert path.read_text(encoding="utf-8") == "alpha\nBETA\ngamma\n"


def test_replace_ranges_tool_rejects_overlapping_ranges(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    first = _fingerprint(ReadTool.make(session, ["sample.txt", "0", "2"]).call())
    second = _fingerprint(ReadTool.make(session, ["sample.txt", "1", "3"]).call())
    edits = json.dumps(
        [
            {"start": 0, "end": 2, "fingerprint": first, "content": "one\n"},
            {"start": 1, "end": 3, "fingerprint": second, "content": "two\n"},
        ]
    )

    tool = BatchReplaceRangesTool.make(session, ["sample.txt", edits])

    with pytest.raises(ToolCallError, match="resolved ranges overlap"):
        tool.call()
    assert path.read_text(encoding="utf-8") == "alpha\nbeta\ngamma\n"


def test_replace_ranges_tool_rejects_full_file_fingerprint_for_partial_range(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    fingerprint = _fingerprint(ReadTool.make(session, ["sample.txt"]).call())
    edits = json.dumps([{"start": 1, "end": 2, "fingerprint": fingerprint, "content": "BETA\n"}])

    tool = BatchReplaceRangesTool.make(session, ["sample.txt", edits])
    display = tool.display()

    assert display.startswith("BatchReplaceRanges(")
    assert "# preview unavailable: fingerprint mismatch" in display
    assert "call Read(filepath, 1, 2)" in display
    assert "--- " not in display
    with pytest.raises(ToolCallError, match=r"call Read\(filepath, 1, 2\)"):
        tool.call()
    assert path.read_text(encoding="utf-8") == "alpha\nbeta\ngamma\n"
