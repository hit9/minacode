import pytest

from nanocode import MainAgent, RangeFingerprintStore, ReadTool, ReplaceRangeTool, Session, ToolCallArgError, ToolCallError


def _fingerprint(read_result: str) -> str:
    return read_result.split("<fingerprint>", 1)[1].split("</fingerprint>", 1)[0]


def test_replace_range_tool_replaces_range_when_fingerprint_matches(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    fingerprint = _fingerprint(ReadTool.make(session, ["sample.txt", "1", "2"]).call())

    tool = ReplaceRangeTool.make(session, ["sample.txt", "1", "2", fingerprint, "BETA\n"])
    display = tool.preview()
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


def test_replace_range_tool_creates_missing_file_with_empty_zero_range(tmp_path):
    path = tmp_path / "created.txt"
    session = Session(cwd=str(tmp_path))

    tool = ReplaceRangeTool.make(session, ["created.txt", "0", "0", "", "alpha\n"])
    display = tool.preview()
    result = tool.call()

    assert "+alpha\n" in display
    assert path.read_text(encoding="utf-8") == "alpha\n"
    assert result == "\n".join(
        [
            "<ReplaceRangeToolResult>",
            "* path: created.txt",
            "* range: 0:0",
            f"* fingerprint: {RangeFingerprintStore().remember(filepath=str(path), start=0, end=0, content='')}",
            "* created: true",
            "</ReplaceRangeToolResult>",
        ]
    )


def test_replace_range_tool_warns_for_broad_preview_ranges(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("".join("line " + str(index) + "\n" for index in range(25)), encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    fingerprint = _fingerprint(ReadTool.make(session, ["sample.txt", "0,25"]).call())

    display = ReplaceRangeTool.make(session, ["sample.txt", "0", "25", fingerprint, "replacement\n"]).preview()

    assert display.startswith("# warning: broad range replacement; prefer smaller semantic ranges\n--- ")


def test_replace_range_tool_rejects_public_multi_range_args(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\nbeta\ngamma\ndelta\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    beta_fingerprint = _fingerprint(ReadTool.make(session, ["sample.txt", "1", "2"]).call())
    delta_fingerprint = _fingerprint(ReadTool.make(session, ["sample.txt", "3", "4"]).call())

    with pytest.raises(ToolCallArgError, match="requires exactly 5 args"):
        ReplaceRangeTool.make(
            session,
            ["sample.txt", "1", "2", beta_fingerprint, "BETA\n", "3", "4", delta_fingerprint, "DELTA\n"],
        )

    assert path.read_text(encoding="utf-8") == "alpha\nbeta\ngamma\ndelta\n"


def test_agent_merges_consecutive_same_file_replace_range_calls(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\nbeta\ngamma\ndelta\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    beta_fingerprint = _fingerprint(ReadTool.make(session, ["sample.txt", "1", "2"]).call())
    delta_fingerprint = _fingerprint(ReadTool.make(session, ["sample.txt", "3", "4"]).call())
    agent = MainAgent(session)
    confirmations = []

    latest = agent.execute_tool_calls(
        [
            {"name": "ReplaceRange", "intention": "replace beta", "args": ["sample.txt", "1", "2", beta_fingerprint, "BETA\n"]},
            {"name": "ReplaceRange", "intention": "replace delta", "args": ["sample.txt", "3", "4", delta_fingerprint, "DELTA\n"]},
        ],
        confirm=lambda call, tool: confirmations.append(call.executed) or True,
    )

    assert len(agent.tool_runner.latest_executions) == 1
    assert confirmations[0].startswith('ReplaceRange("sample.txt", "1", "2"')
    assert "replace beta; replace delta" in session.tool_result_store["tr.1"].description
    assert "* replacements: 2" in latest
    assert path.read_text(encoding="utf-8") == "alpha\nBETA\ngamma\nDELTA\n"


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


def test_replace_range_tool_accepts_full_file_fingerprint_for_partial_range(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    fingerprint = _fingerprint(ReadTool.make(session, ["sample.txt"]).call())

    tool = ReplaceRangeTool.make(session, ["sample.txt", "1", "2", fingerprint, "BETA\n"])
    display = tool.preview()
    result = tool.call()

    assert display.startswith("--- ")
    assert "# preview unavailable" not in display
    assert "-beta\n" in display
    assert "+BETA\n" in display
    assert "* range: 1:2" in result
    assert path.read_text(encoding="utf-8") == "alpha\nBETA\ngamma\n"


def test_replace_range_tool_reports_fingerprint_cached_range(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    fingerprint = _fingerprint(ReadTool.make(session, ["sample.txt", "0", "3"]).call())
    path.write_text("alpha\nBETA\ngamma\n", encoding="utf-8")

    tool = ReplaceRangeTool.make(session, ["sample.txt", "1", "2", fingerprint, "BETA\n"])

    display = tool.preview()
    assert "this fingerprint was cached for range(s): 0:3" in display
    with pytest.raises(ToolCallError, match=r"cached for range\(s\): 0:3"):
        tool.call()


def test_replace_range_tool_rejects_fingerprint_mismatch(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\nbeta\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))

    tool = ReplaceRangeTool.make(session, ["sample.txt", "1", "2", "bad", "BETA\n"])

    display = tool.preview()

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


def test_replace_range_cache_survives_goal_rewording(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    fingerprint = _fingerprint(ReadTool.make(session, ["sample.txt", "1", "2"]).call())

    MainAgent(session).apply_response({"actions": [{"type": "goal", "text": "new goal"}]})

    ReplaceRangeTool.make(session, ["sample.txt", "1", "2", fingerprint, "BETA\n"]).call()

    assert path.read_text(encoding="utf-8") == "alpha\nBETA\ngamma\n"


def test_replace_range_cache_clears_when_main_goal_finishes(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    _fingerprint(ReadTool.make(session, ["sample.txt", "1", "2"]).call())
    agent = MainAgent(session)

    agent.cancel_current_goal()

    assert len(session.range_fingerprints) == 0


def test_replace_range_cache_clears_when_new_main_run_starts(tmp_path):
    class FakeModelClient:
        def request(self, system_prompt, user_prompt, *, activity="main"):
            return {"actions": [{"type": "chat", "text": "done"}]}

    path = tmp_path / "sample.txt"
    path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    _fingerprint(ReadTool.make(session, ["sample.txt", "1", "2"]).call())
    agent = MainAgent(session)
    agent.model_client = FakeModelClient()

    agent.run("new task")

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


def test_replace_range_tool_rejects_wide_fingerprint_for_empty_insert_range(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    fingerprint = _fingerprint(ReadTool.make(session, ["sample.txt"]).call())
    path.write_text("zero\nalpha\nbeta\ngamma\n", encoding="utf-8")

    tool = ReplaceRangeTool.make(session, ["sample.txt", "1", "1", fingerprint, "INSERT\n"])

    assert "# preview unavailable: fingerprint mismatch" in tool.preview()
    with pytest.raises(ToolCallError, match=r"call Read\(filepath, 1, 1\)"):
        tool.call()
    assert path.read_text(encoding="utf-8") == "zero\nalpha\nbeta\ngamma\n"


def test_replace_range_tool_rejects_no_change(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\nbeta\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    fingerprint = _fingerprint(ReadTool.make(session, ["sample.txt", "1", "2"]).call())

    tool = ReplaceRangeTool.make(session, ["sample.txt", "1", "2", fingerprint, "beta\n"])

    with pytest.raises(ToolCallError, match="range replacement produced no changes"):
        tool.call()
    assert path.read_text(encoding="utf-8") == "alpha\nbeta\n"
