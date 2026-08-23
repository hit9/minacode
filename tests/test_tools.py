import json
import os
import shutil

import code_symbol_index as csi
import pytest

from minacode.base import (
    LogBlock,
    LogEdge,
    LogLine,
    LogRole,
    ToolCall,
    ToolError,
)
from minacode.config import (
    Config,
)
from minacode.context import ContextManager
from minacode.model import ModelClient
from minacode.render import UiPrinter
from minacode.runner import ToolRunner
from minacode.session import HistorySegment, Session
from minacode.tools import (
    TOOL_REGISTRY,
    TOOLS,
    BashTool,
    CodeIndex,
    EditTool,
    InspectCodeTool,
    MCPTool,
    NoteTool,
    ReadTool,
    RecallContextTool,
    RecallTool,
    SearchTool,
    SkillTool,
    Tool,
    ViewImageTool,
)


def session(tmp_path):
    return Session(cwd=str(tmp_path))


def _q(*items):
    """Wrap question item dicts into the Ask tool payload args."""
    return [{"questions": list(items)}]


def test_base_tool_helpers_validate_shared_argument_contracts(tmp_path):
    class DemoTool(Tool):
        NAME = "Demo"

    tool = DemoTool(session(tmp_path), ["one", "two"])

    assert tool.strings(min_count=1, max_count=2) == ["one", "two"]
    assert tool.preview() == "Demo(one, two)"
    assert Tool.line_range([1, 3]) == (1, 3)
    assert Tool.compact({"key": "a long value"}, 16) == '{"key":"a lon...'
    assert Tool.compile_regex("needle").search("NEEDLE")
    assert not Tool.compile_regex("needle", case_sensitive=True).search("NEEDLE")

    with pytest.raises(ToolError, match="requires 1 string args"):
        DemoTool(session(tmp_path), []).strings(min_count=1, max_count=1)
    with pytest.raises(ToolError, match="args must be strings"):
        DemoTool(session(tmp_path), [1]).strings()
    with pytest.raises(ToolError, match=r"range must be \[start,end\] integers"):
        Tool.line_range([True, 2])
    with pytest.raises(ToolError, match="range values must be >= 0"):
        Tool.line_range([-1, 2])
    with pytest.raises(ToolError, match="invalid regex"):
        Tool.compile_regex("[")

    assert ViewImageTool in TOOLS
    assert TOOL_REGISTRY["ViewImage"] is ViewImageTool


def test_read_anchor_parsing_accepts_display_and_index_formats():
    short = ReadTool.line_hash("line\n")
    indexed = ReadTool.indexed_line_hash("line\n")

    # The model writes a 1-based line number; parsing returns the 0-based index behind it.
    assert ReadTool.parse_anchor(f"anchor=7:{short} | line") == (6, short)
    assert ReadTool.parse_anchor(f"7:{indexed}") == (6, indexed)
    assert ReadTool.require_anchor(f"7:{short}") == (6, short)
    assert ReadTool.parse_anchor("not-an-anchor") is None
    with pytest.raises(ToolError, match="invalid anchor"):
        ReadTool.require_anchor("not-an-anchor")


def test_strict_schema_handles_optional_enum_union_and_container_without_mutation():
    original = {
        "type": "object",
        "properties": {
            "required": {"type": "integer"},
            "enum": {"type": "string", "enum": ["a"]},
            "union": {"type": ["string", "null"]},
            "multi": {"type": ["string", "number"]},
            "items": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        },
        "required": ["required"],
    }

    strict = Tool._strict_schema(original)

    assert original["properties"]["enum"] == {"type": "string", "enum": ["a"]}
    assert strict["required"] == ["required", "enum", "union", "multi", "items"]
    assert strict["additionalProperties"] is False
    assert strict["properties"]["required"] == {"type": "integer"}
    assert strict["properties"]["enum"] == {"type": ["string", "null"], "enum": ["a", None]}
    assert strict["properties"]["union"] == {"type": ["string", "null"]}
    assert strict["properties"]["multi"] == {"type": ["string", "number", "null"]}
    assert strict["properties"]["items"] == {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}]}


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ([], "at least one query object"),
        (["needle"], "query objects"),
        ([{"pattern": "needle", "extra": True}], "unexpected field"),
        ([{"pattern": ""}], "requires pattern"),
        ([{"pattern": "needle", "context": True}], "context must be"),
        ([{"pattern": "needle", "context": SearchTool.MAX_CONTEXT + 1}], "context must be"),
    ],
)
def test_search_request_validation_is_actionable(tmp_path, args, message):
    with pytest.raises(ToolError, match=message):
        SearchTool(session(tmp_path), args).requests()


def test_skill_tool_without_library_reports_missing_capability(tmp_path):
    s = session(tmp_path)
    s.skills = None

    with pytest.raises(ToolError, match="no skills are installed"):
        SkillTool(s, ["missing"]).call()


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ([], "at least one"),
        (["file.py"], "must be .* objects"),
        ([{"path": "file.py", "extra": True}], "unexpected field"),
        ([{"path": ""}], "non-empty path"),
        ([{"path": "file.py", "ranges": []}], "non-empty ranges"),
    ],
)
def test_read_target_validation_is_actionable(tmp_path, args, message):
    with pytest.raises(ToolError, match=message):
        ReadTool(session(tmp_path), args).targets()


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"unknown": True}, "unexpected field"),
        ({"append_known": "fact"}, "append_known must be an array"),
        ({"replace_known": "fact"}, "replace_known must be an array"),
        ({"action": "update"}, "update requires"),
    ],
)
def test_note_validation_errors_are_actionable(tmp_path, payload, message):
    with pytest.raises(ToolError, match=message):
        NoteTool(session(tmp_path), [payload]).call()


def test_reading_a_materialized_tool_output_needs_no_confirmation(tmp_path):
    """The marker of a truncated result hands the model an absolute path outside the workspace, so
    the out-of-workspace prompt would stop the model from reading a file minacode itself wrote and
    told it to read. Assets are exempt; anything else outside the workspace still asks."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    s = Session(cwd=str(workspace), config=Config(data_dir=str(tmp_path / "data")))
    large = "\n".join(f"line {index}" for index in range(20000))
    key = s.store_tool_result("Bash", ["big"], large)
    ContextManager(s).bound_output(large, key)
    asset = os.path.join(s.images.assets_dir(), key + ".txt")
    outside = tmp_path / "elsewhere.txt"
    outside.write_text("private\n", encoding="utf-8")

    assert ReadTool(s, [{"path": asset}]).needs_confirmation() is False
    assert SearchTool(s, [{"path": asset, "pattern": "line 5"}]).needs_confirmation() is False
    assert ReadTool(s, [{"path": str(outside)}]).needs_confirmation() is True
    assert SearchTool(s, [{"path": str(outside), "pattern": "private"}]).needs_confirmation() is True
    # Reading it really does return the full output the marker promised.
    assert "line 19999" in ReadTool(s, [{"path": asset}]).call()


def test_mcp_tool_handles_missing_manager_and_invalid_arguments(tmp_path):
    s = session(tmp_path)
    s.mcp = None
    tool = MCPTool(s, [{"action": "call", "server": "docs", "tool": "read", "arguments": {}}])

    assert tool.needs_confirmation() is False
    with pytest.raises(ToolError, match="MCP not configured"):
        tool.call()
    with pytest.raises(ToolError, match="arguments must be an object"):
        MCPTool(s, [{"action": "call", "server": "docs", "tool": "read", "arguments": []}]).call()


def test_code_index_failure_helpers_keep_session_state_consistent(tmp_path, monkeypatch):
    s = session(tmp_path)
    index = CodeIndex(s)
    monkeypatch.setattr(csi, "status", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("status failed")))

    assert CodeIndex.status_line("ready") == "index✓ synced"
    assert CodeIndex.status_line("error", "broken") == "index! error: broken"
    assert index.status() == ("error", "status failed")
    assert s.state.code_index_status == "error"
    assert s.state.code_index_error == "status failed"
    assert index.fail(" update failed ") == "update failed"
    assert s.state.code_index_notice == "error"

    index.finish()
    assert s.state.code_index_notice == ""
    assert s.state.code_index_error == ""
    assert s.state.code_index_status == "synced"














































































































def test_read_and_search_success_paths(tmp_path):
    (tmp_path / "sample.py").write_text("alpha\nNeedle\nomega\n", encoding="utf-8")
    (tmp_path / "blob.bin").write_bytes(b"a\0b")
    s = session(tmp_path)

    read = ReadTool(s, [{"path": "sample.py", "ranges": [[1, 2], [3, 0]]}]).call()
    single_range = ReadTool(s, [{"path": "sample.py", "ranges": [1, 2]}]).call()
    full_default = ReadTool(s, [{"path": "sample.py"}]).call()
    alpha_hash = ReadTool.line_hash("alpha\n")
    needle_hash = ReadTool.line_hash("Needle\n")
    omega_hash = ReadTool.line_hash("omega\n")
    assert f"anchor=1:{alpha_hash} | alpha" in read
    assert f"anchor=2:{needle_hash} | Needle" in read
    assert f"anchor=3:{omega_hash} | omega" in read
    assert f"anchor=1:{alpha_hash} | alpha" in single_range
    assert f"anchor=2:{needle_hash} | Needle" in single_range
    assert f"anchor=3:{omega_hash} | omega" in full_default
    assert "<total_lines>3</total_lines>" in full_default  # Read reports the line count (replaces LineCount)

    found = SearchTool(s, [{"pattern": "needle", "path": "."}]).call()
    assert f"sample.py anchor=2:{needle_hash} | Needle" in found

    multiline = SearchTool(s, [{"pattern": "alpha\\nNeedle", "path": "sample.py"}]).call()
    assert "sample.py anchor=1:" in multiline


def test_read_search_and_anchors_report_one_based_line_numbers(tmp_path):
    """Read, Search, and Edit anchors must number lines the way `grep -n`, tracebacks, and diffs
    do, so a line number seen in one place can be used in another without adjustment."""
    s = session(tmp_path)
    lines = ["alpha", "beta", "Needle", "omega"]  # grep -n numbers these 1..4
    (tmp_path / "sample.py").write_text("\n".join(lines) + "\n", encoding="utf-8")
    beta_hash = ReadTool.line_hash("beta\n")
    needle_hash = ReadTool.line_hash("Needle\n")

    # A 1-based inclusive range returns exactly the requested lines, and no others.
    read = ReadTool(s, [{"path": "sample.py", "ranges": [[2, 3]]}]).call()
    assert "<range>2:3</range>" in read
    assert f"anchor=2:{beta_hash} | beta" in read
    assert f"anchor=3:{needle_hash} | Needle" in read
    assert "alpha" not in read and "omega" not in read

    # Search agrees with Read on the same line, through both the ripgrep and the Python backend.
    found = SearchTool(s, [{"pattern": "needle", "path": "."}]).call()
    assert f"sample.py anchor=3:{needle_hash} | Needle" in found
    multiline = SearchTool(s, [{"pattern": "beta\\nNeedle", "path": "sample.py"}]).call()
    assert "sample.py anchor=2:" in multiline

    # An anchor taken from that output edits the line it names, and nothing shifts by one.
    EditTool(s, ["sample.py", [{"op": "replace", "start": f"3:{needle_hash}", "end": f"3:{needle_hash}", "content": "FOUND\n"}]]).call()
    assert (tmp_path / "sample.py").read_text(encoding="utf-8") == "alpha\nbeta\nFOUND\nomega\n"


def test_read_echoes_ranges_the_model_could_have_written(tmp_path):
    """short_args is echoed back to the model in the tool message, not just printed in the
    terminal, so the "read to the end of the file" sentinel must not surface as a literal 0."""
    s = session(tmp_path)
    (tmp_path / "a.py").write_text("one\ntwo\nthree\n", encoding="utf-8")

    def echo(args):
        return ReadTool(s, args).short_args()

    assert echo([{"path": "a.py"}]) == ["a.py"]  # whole file: no range, matching the omitted input
    assert echo([{"path": "a.py", "ranges": [[1, 0]]}]) == ["a.py"]
    assert echo([{"path": "a.py", "ranges": [[2, 0]]}]) == ["a.py 2:"]  # line 2 to the end
    assert echo([{"path": "a.py", "ranges": [[2, 3]]}]) == ["a.py 2:3"]
    assert echo([{"path": "a.py", "ranges": [[1, 1], [3, 0]]}]) == ["a.py 1:1,3:"]

    # Omitting ranges and passing null (what strict-schema providers send) mean the same thing.
    assert ReadTool.payload_args({"path": "a.py"}) == ReadTool.payload_args({"path": "a.py", "ranges": None})


def test_anchor_from_zero_based_session_relocates_instead_of_editing_the_wrong_line(tmp_path):
    """A session started before line numbers became 1-based can replay an anchor that is one line
    low. The content hash makes that recoverable: the anchor is relocated to the line it actually
    describes, never silently applied to the neighbouring line it now points at."""
    s = session(tmp_path)
    (tmp_path / "code.py").write_text("first\nsecond\nthird\n", encoding="utf-8")
    third_hash = ReadTool.line_hash("third\n")
    stale = f"2:{third_hash}"  # "third" was line 2 under the old numbering

    EditTool(s, ["code.py", [{"op": "replace", "start": stale, "end": stale, "content": "THIRD\n"}]]).call()

    assert (tmp_path / "code.py").read_text(encoding="utf-8") == "first\nsecond\nTHIRD\n"

    # When the line's content is duplicated there is no single line to relocate to, so the edit
    # is refused rather than guessed at.
    (tmp_path / "dup.py").write_text("head\nsame\nsame\n", encoding="utf-8")
    same_hash = ReadTool.line_hash("same\n")
    ambiguous = f"1:{same_hash}"  # "same" was line 1 under the old numbering
    with pytest.raises(ToolError, match="stale anchor"):
        EditTool(s, ["dup.py", [{"op": "replace", "start": ambiguous, "end": ambiguous, "content": "x\n"}]]).call()


def test_recall_behaviors(tmp_path):
    s = session(tmp_path)
    first = s.store_tool_result("Read", ["a.txt"], "a0\na1\na2\n")
    second = s.store_tool_result("Search", [{"pattern": "b"}], "b0\nb1\n")

    sliced = RecallTool(s, [{"keys": [first, second], "ranges": [[2, 2]]}]).call()
    assert "a1" in sliced and "a0" not in sliced
    assert "b1" in sliced and "b0" not in sliced

    common_range = RecallTool(s, [{"keys": [first], "ranges": [[0, 1]]}]).call()
    assert "a0" in common_range and "a1" not in common_range

    with pytest.raises(ToolError):
        RecallTool(s, [{"key": first, "ranges": [[2, "bad"]]}]).call()


def test_recall_history_regex_searches_titles_and_text(tmp_path):
    s = session(tmp_path)
    s.history.extend(
        [
            HistorySegment(key="seg.1", title="cache work", text="user:\nStable prefix design"),
            HistorySegment(key="seg.2", title="notes", text="assistant:\nTask Memory placement"),
            HistorySegment(key="seg.3", title="unrelated", text="assistant:\nNothing relevant"),
        ]
    )

    result = RecallContextTool(s, [{"query": "stable prefix|task memory"}]).call()

    assert '<RecallContextSearchResult query="stable prefix|task memory" matches=2>' in result
    assert "seg.1 2" in result
    assert "Stable prefix design" in result
    assert "seg.2 2" in result
    assert "Task Memory placement" in result
    assert "seg.3" not in result


def test_recall_history_regex_supports_key_scope_case_and_limit(tmp_path):
    s = session(tmp_path)
    s.history.extend(
        [
            HistorySegment(key="seg.1", title="one", text="Needle first"),
            HistorySegment(key="seg.2", title="two", text="needle second\nneedle third"),
            HistorySegment(key="seg.3", title="three", text="needle fourth"),
        ]
    )

    result = RecallContextTool(
        s,
        [{"keys": ["seg.1", "seg.2"], "query": "needle", "case_sensitive": True, "limit": 1}],
    ).call()

    assert "matches=1" in result
    assert "seg.1" not in result
    assert "seg.2 1" in result
    assert "needle second" in result
    assert "needle third" not in result
    assert "seg.3" not in result


def test_recall_history_regex_validates_search_arguments(tmp_path):
    s = session(tmp_path)

    for payload in ({"query": "["}, {"query": "x", "limit": 0}, {"keys": ["seg.1"], "case_sensitive": True}):
        with pytest.raises(ToolError):
            RecallContextTool(s, [payload]).call()


def test_recall_history_lists_newest_segments_with_pagination(tmp_path):
    s = session(tmp_path)
    s.history.extend(HistorySegment(key=f"seg.{index}", title=f"task {index}") for index in range(1, 5))

    first = json.loads(RecallContextTool(s, [{"action": "list", "limit": 2}]).call())
    second = json.loads(RecallContextTool(s, [{"action": "list", "limit": 2, "before": first["next_before"]}]).call())

    assert first == {
        "segments": [{"key": "seg.4", "title": "task 4"}, {"key": "seg.3", "title": "task 3"}],
        "total": 4,
        "returned": 2,
        "next_before": "seg.3",
    }
    assert second["segments"] == [{"key": "seg.2", "title": "task 2"}, {"key": "seg.1", "title": "task 1"}]
    assert "next_before" not in second


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"action": "unknown"}, "action must be"),
        ({"action": "list", "keys": ["seg.1"]}, "list accepts only"),
        ({"action": "get", "keys": ["seg.1"], "limit": 1}, "get accepts only"),
        ({"action": "search"}, "search requires query"),
        ({"action": "search", "query": "x", "before": "seg.1"}, "before is only valid"),
        ({"action": "list", "before": "bad"}, "before must look like"),
    ],
)
def test_recall_history_actions_reject_conflicting_fields(tmp_path, payload, message):
    with pytest.raises(ToolError, match=message):
        RecallContextTool(session(tmp_path), [payload]).call()


def test_recall_history_rejects_bad_key_format(tmp_path):
    s = session(tmp_path)

    with pytest.raises(ToolError):
        RecallContextTool(s, [{"keys": ["tr.1"]}]).call()


def test_recall_history_reports_missing_segment(tmp_path):
    s = session(tmp_path)

    assert "seg.9: missing" in RecallContextTool(s, [{"keys": ["seg.9"]}]).call()


def test_recall_history_returns_segment_text(tmp_path):
    s = session(tmp_path)
    s.history.append(HistorySegment(key="seg.1", title="task", text="user:\nfind bug"))

    result = RecallContextTool(s, [{"keys": ["seg.1"]}]).call()

    assert "<RecallContextResult>" in result
    assert 'key="seg.1"' in result
    assert "find bug" in result


def test_reject_collapses_display(tmp_path):
    s = session(tmp_path)
    out = []
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: out.append(str(text)))

    msg = runner.reject(ToolCall("c", "Read", [{"path": "x"}]), "ToolError: Read requires non-empty ranges")

    # display collapses to one quiet line, no full [failed]/error block
    assert any("· rejected: Read requires non-empty ranges" in t for t in out)
    assert not any("[failed]" in t or t.startswith("  error ") for t in out)
    # model still receives the full error
    assert "Read requires non-empty ranges" in msg


def test_search_ignores_hidden_and_gitignored_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    (tmp_path / ".gitignore").write_text("ignored.txt\nignored_dir/\n", encoding="utf-8")
    (tmp_path / "visible.txt").write_text("needle\n", encoding="utf-8")
    (tmp_path / ".hidden.txt").write_text("needle\n", encoding="utf-8")
    (tmp_path / ".hidden_dir").mkdir()
    (tmp_path / ".hidden_dir" / "inside.txt").write_text("needle\n", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("needle\n", encoding="utf-8")
    (tmp_path / "ignored_dir").mkdir()
    (tmp_path / "ignored_dir" / "inside.txt").write_text("needle\n", encoding="utf-8")
    s = session(tmp_path)

    found = SearchTool(s, [{"pattern": "needle", "path": "."}]).call()
    direct_hidden = SearchTool(s, [{"pattern": "needle", "path": ".hidden.txt"}]).call()
    direct_ignored = SearchTool(s, [{"pattern": "needle", "path": "ignored_dir/inside.txt"}]).call()

    assert "visible.txt anchor=1:" in found
    assert ".hidden" not in found
    assert "ignored" not in found
    assert ".hidden.txt anchor=1:" not in direct_hidden
    assert "ignored_dir/inside.txt anchor=1:" not in direct_ignored


def test_single_and_batch_payload_shapes_are_supported():
    assert ModelClient.tool_payload("Read", {"path": "a.py"}) == [{"path": "a.py", "ranges": [[1, 0]]}]
    assert ModelClient.tool_payload("Read", {"path": "a.py", "ranges": [0, 2]}) == [{"path": "a.py", "ranges": [[0, 2]]}]
    assert ModelClient.tool_payload("Read", {"files": [{"path": "a.py", "ranges": [[0, 1]]}]}) == [{"path": "a.py", "ranges": [[0, 1]]}]
    assert ReadTool(Session(cwd="."), [{"path": "minacode.py"}]).targets()[0][1] == [(1, 0)]
    assert ModelClient.tool_payload("Search", {"pattern": "TODO"}) == [{"pattern": "TODO"}]
    assert ModelClient.tool_payload("Search", {"queries": [{"pattern": "TODO"}]}) == [{"pattern": "TODO"}]
    assert ModelClient.tool_payload("Note", {"set_goal": "ship"}) == [{"set_goal": "ship"}]


def test_tool_runner_finish_display_keeps_ask_answer(tmp_path):
    s = session(tmp_path)
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    display = str(runner.finish_display(ToolCall("ask", "Ask", _q({"question": "Which?"})), "tr.1", "typed answer", failed=False))

    assert display.startswith("  Ask  Which? → tr.1\n")
    assert display.endswith("    └ answer typed answer")


def test_tool_runner_reject_records_error_and_returns_failed_message(tmp_path):
    s = Session(cwd=str(tmp_path))
    runner = ToolRunner(s, ContextManager(s))
    call = ToolCall("e1", "Bash", ["bad cmd"])
    out = []
    runner.output_fn = out.append
    result = runner.reject(call, "ToolError: command not found")
    assert len(out) == 1
    assert isinstance(out[0], LogBlock)
    assert "command not found" in str(out[0])
    # Should record the error
    assert len(s.tool_errors) == 1
    assert s.tool_errors[0].name == "Bash"
    assert "command not found" in s.tool_errors[0].error
    # reject returns a plain-text tool-message representation
    assert "failed" in result.lower()
    assert "command not found" in result


def test_tool_runner_short_call_formats_search_and_recall(tmp_path):
    s = session(tmp_path)
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    search = runner.short_call(
        ToolCall(
            "s",
            "Search",
            [
                {"pattern": "done in", "glob": "*.py"},
                {"pattern": "elapsed.*s]", "path": "tests", "context": 2},
            ],
        )
    )
    assert search == 'Search "done in" glob=*.py; "elapsed.*s]" path=tests C=2'

    recall = runner.short_call(ToolCall("r", "Recall", [{"keys": ["tr.4", "tr.5"], "ranges": [[0, 80]]}]))
    assert recall == "Recall tr.4 0:80; tr.5 0:80"

    s.state.known = ["existing"]
    note = runner.short_call(
        ToolCall(
            "m",
            "Note",
            [
                {
                    "set_goal": "ship",
                    "replace_plan": [{"status": "doing", "text": "inspect"}, {"status": "todo", "text": "patch"}],
                    "append_known": ["existing", "new fact"],
                }
            ],
        )
    )
    assert note == "Note goal: ship\nplan:\n  - [~] inspect\n  - [ ] patch\nknown:\n  + new fact"


def test_tool_schemas_are_strict_for_high_risk_tools():
    bash_params = BashTool.schema()["function"]["parameters"]
    assert bash_params["required"] == ["command"]
    assert bash_params["properties"]["command"]["pattern"] == r"^.*\S.*$"

    edit_params = EditTool.schema()["function"]["parameters"]
    assert edit_params["required"] == ["path", "edits"]
    assert set(edit_params["properties"]) == {"path", "edits"}
    assert "the line at end is itself replaced or deleted" in EditTool.schema()["function"]["description"]

    recall_keys = RecallTool.schema()["function"]["parameters"]["properties"]["keys"]
    assert recall_keys["items"]["pattern"] == r"^tr\.\d+$"

    read_params = ReadTool.schema()["function"]["parameters"]
    assert {"path", "ranges", "files"} <= set(read_params["properties"])

    note_params = NoteTool.schema()["function"]["parameters"]
    assert "minItems" not in note_params["properties"]["replace_plan"]
    assert note_params["properties"]["replace_plan"]["items"]["properties"]["status"]["enum"] == ["todo", "doing", "done", "blocked"]
    assert "minItems" not in note_params["properties"]["replace_known"]

    search_params = SearchTool.schema()["function"]["parameters"]
    assert {"pattern", "queries"} <= set(search_params["properties"])
    assert search_params["properties"]["queries"]["items"]["properties"]["context"]["type"] == "integer"

    def walk(value):
        if isinstance(value, dict):
            assert "anyOf" not in value
            assert "prefixItems" not in value
            if isinstance(value.get("pattern"), str):
                assert value["pattern"].startswith("^")
                assert value["pattern"].endswith("$")
            if "items" in value:
                assert isinstance(value["items"], dict)
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    for tool in TOOLS:
        params = tool.schema()["function"]["parameters"]
        assert "args" not in params.get("properties", {})
        walk(tool.schema())


def test_tool_validation_rejects_bad_shapes_without_side_effects(tmp_path):
    s = session(tmp_path)
    (tmp_path / "sample.py").write_text("alpha\n", encoding="utf-8")

    with pytest.raises(ToolError):
        ReadTool(s, [{"path": "sample.py", "ranges": []}]).call()
    with pytest.raises(ToolError):
        EditTool(s, ["a.txt", [{"op": "replace_all", "old": "", "content": "a\n"}]]).call()
    with pytest.raises(ToolError):
        BashTool(s, []).call()
    with pytest.raises(ToolError):
        SearchTool(s, [{"pattern": "["}]).call()
    with pytest.raises(ToolError):
        InspectCodeTool(s, ["inspect", "two words"]).call()

    assert not (tmp_path / "a.txt").exists()
    assert not (tmp_path / "b.txt").exists()


def test_uiprinter_highlights_generic_tool_arguments(tmp_path):
    s = session(tmp_path)
    runner = ToolRunner(s, ContextManager(s))
    line = runner.log_root('Search "done in" glob=*.py C=2')

    assert line.syntax == "tool-args"
    segments = UiPrinter(output_fn=lambda text: None).log_segments(LogBlock([line]))
    assert ("fg:#a5d6ff", '"done in"') in segments
    assert ("fg:#79c0ff", "glob=") in segments
    assert ("fg:#d2a8ff", "2") in segments


def test_uiprinter_renders_note_memory_status_colors():
    ui = UiPrinter(output_fn=lambda text: None)
    segs = ui.segments("goal: ship\ncheck: passed\nplan:\n  - [~] inspect\n  - [x] patch\nknown:\n  + pytest")

    assert ("ansimagenta", "goal: ship") in segs
    assert ("ansimagenta", "check: passed") in segs
    assert ("ansicyan", "plan:") in segs
    assert ("ansiyellow", "  - [~] inspect") in segs
    assert ("ansigreen", "  - [x] patch") in segs
    assert ("ansigreen", "  + pytest") in segs


def test_uiprinter_renders_rejected_line_dim():
    ui = UiPrinter(output_fn=lambda text: None)
    segs = ui.log_segments(LogBlock([LogLine("Read", "· rejected: needs ranges", LogRole.MUTED)]))

    assert any(style == "ansibrightblack" and "rejected" in text for style, text in segs)
    assert not any(style in ("ansired", "ansigreen") for style, text in segs)


def test_uiprinter_renders_stored_result_dim():
    ui = UiPrinter(output_fn=lambda text: None)
    block = LogBlock.hierarchy(None, [LogLine("stored", "tr.50 [approved]", LogRole.META, LogEdge.END)])

    assert ui.log_segments(block) == [
        ("", "    "),
        ("ansibrightblack", "└ "),
        ("ansibrightblack", "stored"),
        ("ansibrightblack", " tr.50 [approved]"),
        ("", "\n"),
    ]


def test_uiprinter_renders_tool_root_without_generic_prefix():
    block = LogBlock([LogLine("Read", "minacode.py 0:100 → tr.6 [auto]", LogRole.TOOL)])
    segments = UiPrinter(output_fn=lambda text: None).log_segments(block)
    text = "".join(value for _style, value in segments)

    assert text == "  Read  minacode.py 0:100 → tr.6 [auto]\n"
    assert any(style == "fg:default" and "minacode.py 0:100 → tr.6 [auto]" in value for style, value in segments)
