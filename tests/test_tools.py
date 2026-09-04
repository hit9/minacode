import json
import os
import shutil

import code_symbol_index as csi
import pytest

from wizolt.base import (
    LogBlock,
    LogEdge,
    LogLine,
    LogRole,
    ToolCall,
    ToolError,
)
from wizolt.config import (
    Config,
)
from wizolt.context import ContextManager
from wizolt.render import UiPrinter
from wizolt.runner import ToolRunner
from wizolt.session import HistorySegment, Session
from wizolt.tools import (
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
    tool_payload,
    toolblocks,
    tooloutput,
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


def test_read_accepts_ranges_in_both_array_forms_and_renders_a_view(tmp_path):
    """The old anchor string formats are gone; Read takes 1-based inclusive ranges either as one
    [start, end] pair or as a list of pairs, and its output carries a `view.N` id once rendered
    with the session's registered drafts."""
    (tmp_path / "sample.py").write_text("a\nb\nc\nd\n", encoding="utf-8")
    s = session(tmp_path)

    out = ReadTool(s, [{"path": "sample.py", "ranges": [[1, 2], [4, 4]]}]).call()
    assert "source=" not in out.retained_text  # retained text carries no view id
    keys = s.register_source_drafts(list(out.drafts))
    rendered = out.render(keys)
    assert 'source="view.1"' in rendered
    assert "1 | a" in rendered and "2 | b" in rendered and "4 | d" in rendered
    assert "3 | c" not in rendered

    # The single [start, end] form is accepted as one range.
    single = ReadTool(s, [{"path": "sample.py", "ranges": [2, 3]}]).call()
    assert "2 | b" in single.retained_text and "3 | c" in single.retained_text


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
    the out-of-workspace prompt would stop the model from reading a file wizolt itself wrote and
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
    assert "line 19999" in ReadTool(s, [{"path": asset}]).call().retained_text


async def test_mcp_tool_handles_missing_manager_and_invalid_arguments(tmp_path):
    s = session(tmp_path)
    s.mcp = None
    tool = MCPTool(s, [{"action": "call", "server": "docs", "tool": "read", "arguments": {}}])

    assert tool.needs_confirmation() is False
    with pytest.raises(ToolError, match="MCP not configured"):
        tool.call()
    with pytest.raises(ToolError, match="arguments must be an object"):
        await MCPTool(s, [{"action": "call", "server": "docs", "tool": "read", "arguments": []}]).call_async()


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
    assert "<Read path=" in read.retained_text
    assert "1 | alpha" in read.retained_text
    assert "2 | Needle" in read.retained_text
    assert "3 | omega" in read.retained_text
    assert "1 | alpha" in single_range.retained_text
    assert "2 | Needle" in single_range.retained_text
    assert "3 | omega" in full_default.retained_text
    assert "total_lines=3" in full_default.retained_text  # Read reports the line count

    found = SearchTool(s, [{"pattern": "needle", "path": "."}]).call()
    assert '<Search pattern="needle" matches=1>' in found.retained_text
    assert '<file path="sample.py"' in found.retained_text
    assert "2 | Needle" in found.retained_text

    multiline = SearchTool(s, [{"pattern": "alpha\\nNeedle", "path": "sample.py"}]).call()
    assert '<file path="sample.py"' in multiline.retained_text
    assert "1 | alpha" in multiline.retained_text


def test_read_search_and_edit_report_one_based_line_numbers(tmp_path):
    """Read, Search, and Edit must number lines the way `grep -n`, tracebacks, and diffs do, so
    a line number seen in one place can be used in another without adjustment."""
    s = session(tmp_path)
    lines = ["alpha", "beta", "Needle", "omega"]  # grep -n numbers these 1..4
    (tmp_path / "sample.py").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # A 1-based inclusive range returns exactly the requested lines, and no others.
    read = ReadTool(s, [{"path": "sample.py", "ranges": [[2, 3]]}]).call()
    assert 'lines="2:3"' in read.retained_text
    assert "2 | beta" in read.retained_text
    assert "3 | Needle" in read.retained_text
    assert "alpha" not in read.retained_text and "omega" not in read.retained_text

    # Search agrees with Read on the same line, through both the ripgrep and the Python backend.
    found = SearchTool(s, [{"pattern": "needle", "path": "."}]).call()
    assert "3 | Needle" in found.retained_text
    multiline = SearchTool(s, [{"pattern": "beta\\nNeedle", "path": "sample.py"}]).call()
    assert "2 | beta" in multiline.retained_text

    # A view taken from that output edits the line it names, and nothing shifts by one.
    key = s.register_source_drafts(list(read.drafts))[0]
    EditTool(s, ["sample.py", key, [{"op": "replace", "start": 3, "end": 3, "content": "FOUND\n"}]]).call()
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


def test_one_based_edit_targets_never_shift_the_wrong_line(tmp_path):
    """View line numbers are 1-based and Edit uses the same 1-based inclusive start/end, so a line
    number seen in one place edits exactly that line -- nothing is one off."""
    s = session(tmp_path)
    (tmp_path / "code.py").write_text("first\nsecond\nthird\n", encoding="utf-8")

    out = ReadTool(s, [{"path": "code.py"}]).call()
    key = s.register_source_drafts(list(out.drafts))[0]

    # 1-based line 3 names "third", not the second line a 0-based scheme would hit.
    EditTool(s, ["code.py", key, [{"op": "replace", "start": 3, "end": 3, "content": "THIRD\n"}]]).call()
    assert (tmp_path / "code.py").read_text(encoding="utf-8") == "first\nsecond\nTHIRD\n"

    # A duplicated line is still targeted by its explicit line number; no guessing is involved.
    (tmp_path / "dup.py").write_text("head\nsame\nsame\n", encoding="utf-8")
    dup = ReadTool(s, [{"path": "dup.py"}]).call()
    dup_key = s.register_source_drafts(list(dup.drafts))[0]
    EditTool(s, ["dup.py", dup_key, [{"op": "replace", "start": 2, "end": 2, "content": "ONE\n"}]]).call()
    assert (tmp_path / "dup.py").read_text(encoding="utf-8") == "head\nONE\nsame\n"


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

    assert "visible.txt" in found.retained_text
    assert ".hidden" not in found.retained_text
    assert "ignored" not in found.retained_text
    assert ".hidden.txt" not in direct_hidden.retained_text
    assert "ignored_dir/inside.txt" not in direct_ignored.retained_text


def test_single_and_batch_payload_shapes_are_supported():
    assert tool_payload("Read", {"path": "a.py"}) == [{"path": "a.py", "ranges": [[1, 0]]}]
    assert tool_payload("Read", {"path": "a.py", "ranges": [0, 2]}) == [{"path": "a.py", "ranges": [[0, 2]]}]
    assert tool_payload("Read", {"files": [{"path": "a.py", "ranges": [[0, 1]]}]}) == [{"path": "a.py", "ranges": [[0, 1]]}]
    assert ReadTool(Session(cwd="."), [{"path": "wizolt.py"}]).targets()[0][1] == [(1, 0)]
    assert tool_payload("Search", {"pattern": "TODO"}) == [{"pattern": "TODO"}]
    assert tool_payload("Search", {"queries": [{"pattern": "TODO"}]}) == [{"pattern": "TODO"}]
    assert tool_payload("Note", {"set_goal": "ship"}) == [{"set_goal": "ship"}]


def test_tool_runner_finish_display_keeps_ask_answer(tmp_path):
    s = session(tmp_path)

    display = str(toolblocks.finish_display(s, ToolCall("ask", "Ask", _q({"question": "Which?"})), "tr.1", "typed answer", failed=False))

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


def test_run_one_rejects_tools_outside_session_whitelist(tmp_path):
    s = session(tmp_path)
    runner = ToolRunner(s, ContextManager(s), input_fn=lambda prompt: "y")

    s.tool_names = ("Read",)
    (message,) = runner.run([ToolCall("c1", "Bash", ["echo hi"])])
    content = str(message["content"])
    assert "failed" in content.lower()
    assert "ToolError: Bash is not available in this session" in content

    # Empty tuple = no filtering (parent behavior): the same call executes.
    s.tool_names = ()
    (message,) = runner.run([ToolCall("c2", "Bash", ["echo hi"])])
    assert "hi" in str(message["content"])


def test_parallel_safe_false_for_tools_outside_whitelist(tmp_path):
    s = session(tmp_path)
    runner = ToolRunner(s, ContextManager(s))

    # Search is normally parallel-safe, but the whitelist excludes it, so it falls back to
    # serial run_one (where the whitelist gate rejects it) instead of execute_readonly.
    s.tool_names = ("Bash",)
    assert not runner.parallel_safe(ToolCall("c1", "Search", [{"pattern": "x"}]))

    # Empty tuple = no filtering: parallel-safety is decided as before.
    s.tool_names = ()
    assert runner.parallel_safe(ToolCall("c2", "Search", [{"pattern": "x"}]))


def test_tool_runner_short_call_formats_search_and_recall(tmp_path):
    s = session(tmp_path)
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    search = tooloutput.short_call(
        runner.session,
        ToolCall(
            "s",
            "Search",
            [
                {"pattern": "done in", "glob": "*.py"},
                {"pattern": "elapsed.*s]", "path": "tests", "context": 2},
            ],
        ),
    )
    assert search == 'Search "done in" glob=*.py; "elapsed.*s]" path=tests C=2'

    recall = tooloutput.short_call(runner.session, ToolCall("r", "Recall", [{"keys": ["tr.4", "tr.5"], "ranges": [[0, 80]]}]))
    assert recall == "Recall tr.4 0:80; tr.5 0:80"

    s.state.known = ["existing"]
    note = tooloutput.short_call(
        runner.session,
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
        ),
    )
    assert note == "Note goal: ship\nplan:\n  - [~] inspect\n  - [ ] patch\nknown:\n  + new fact"


def test_tool_schemas_are_strict_for_high_risk_tools():
    bash_params = BashTool.schema()["function"]["parameters"]
    assert bash_params["required"] == ["command"]
    assert bash_params["properties"]["command"]["pattern"] == r"^.*\S.*$"

    edit_params = EditTool.schema()["function"]["parameters"]
    assert edit_params["required"] == ["path", "edits"]
    assert set(edit_params["properties"]) == {"edits", "path", "source"}
    assert "source=view.N from Read, Search, or InspectCode" in EditTool.schema()["function"]["description"]

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
        EditTool(s, ["a.txt", [{"op": "bogus", "content": "a\n"}]]).call()
    with pytest.raises(ToolError):
        BashTool(s, []).call()
    with pytest.raises(ToolError):
        SearchTool(s, [{"pattern": "["}]).call()
    with pytest.raises(ToolError):
        InspectCodeTool(s, ["inspect", "two words"]).call()

    assert not (tmp_path / "a.txt").exists()
    assert not (tmp_path / "b.txt").exists()


def test_uiprinter_highlights_generic_tool_arguments(tmp_path):
    session(tmp_path)
    line = toolblocks.log_root('Search "done in" glob=*.py C=2')

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
    block = LogBlock([LogLine("Read", "wizolt.py 0:100 → tr.6 [auto]", LogRole.TOOL)])
    segments = UiPrinter(output_fn=lambda text: None).log_segments(block)
    text = "".join(value for _, value in segments)

    assert text == "  Read  wizolt.py 0:100 → tr.6 [auto]\n"
    assert any(style == "fg:default" and "wizolt.py 0:100 → tr.6 [auto]" in value for style, value in segments)


def test_mixed_batch_whitelisted_tool_runs_and_excluded_rejected(tmp_path):
    """A batch mixing an excluded parallel-safe tool with a whitelisted one: the segment router
    never hands the excluded name to execute_readonly, so each call gets its own verdict."""
    s = session(tmp_path)
    s.settings.max_parallel_tools = 4
    s.tool_names = ("Read",)
    (tmp_path / "a.txt").write_text("hello\n", encoding="utf-8")
    runner = ToolRunner(s, ContextManager(s), input_fn=lambda prompt: "y")

    calls = [
        ToolCall("s1", "Search", [{"pattern": "hello"}]),
        ToolCall("r1", "Read", [{"path": "a.txt"}]),
        ToolCall("s2", "Search", [{"pattern": "hello"}]),
    ]
    contents = {message["tool_call_id"]: str(message["content"]) for message in runner.run(calls)}
    assert "hello" in contents["r1"]
    assert "Search is not available in this session" in contents["s1"]
    assert "Search is not available in this session" in contents["s2"]


def test_search_shares_one_view_per_file_across_queries(tmp_path, monkeypatch):
    """A batched Search unions every query's visible rows for one path into one view, so two
    queries hitting the same file cannot hand the model two ids for the same snapshot -- or two
    renderings of it that disagree. The Python backend and ripgrep must agree on that shape."""
    (tmp_path / "a.py").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("alpha\n", encoding="utf-8")

    for rg_available in (True, False):
        s = session(tmp_path)
        if not rg_available:
            monkeypatch.setattr(shutil, "which", lambda name: None)
        out = SearchTool(s, [{"pattern": "alpha", "path": "."}, {"pattern": "gamma", "path": "."}]).call()
        keys = s.register_source_drafts(list(out.drafts))
        text = out.render(keys)

        # Three blocks (a.py twice, b.py once) but two views: a.py's blocks share one id.
        assert len(keys) == 3
        assert len(set(keys)) == 2
        assert len(s.source_views) == 2
        a_view = next(v for v in s.source_views.values() if v.display_path == "a.py")
        assert [line for span in a_view.spans for line in span.lines] == ["alpha\n", "gamma\n"]
        assert text.count(f'source="{a_view.key}"') == 2
        monkeypatch.undo()


def test_search_rows_come_from_the_file_not_the_candidate_finder(tmp_path, monkeypatch):
    """ripgrep only nominates (path, line) candidates. If the file changes before the rows are
    captured, the view must show the file as it is now -- pairing a view id with the text rg
    printed earlier would let an Edit validate against content that no longer exists."""
    path = tmp_path / "a.py"
    path.write_text("needle\nsecond\n", encoding="utf-8")
    s = session(tmp_path)
    tool = SearchTool(s, [{"pattern": "needle", "path": "."}])
    original = SearchTool.find_candidates

    def stale(self, request):
        rows = original(self, request)
        path.write_text("header\nneedle\nsecond\n", encoding="utf-8")  # shifts every candidate line
        return rows

    monkeypatch.setattr(SearchTool, "find_candidates", stale)
    out = tool.call()
    key = s.register_source_drafts(list(out.drafts))[0]
    view_obj = s.get_source_view(key)

    assert view_obj.total_lines == 3
    assert [(span.start, span.lines) for span in view_obj.spans] == [(2, ("needle\n",))]
    assert "2 | needle" in out.retained_text


def test_read_merges_one_view_per_path_across_request_items(tmp_path):
    """A batched Read emits one block and one view per file, not one per requested range. Ranges
    for the same path are unioned, sorted, and merged when they overlap or touch, so the `lines`
    label the model sees is exactly the set of spans the view can validate an edit against."""
    (tmp_path / "a.py").write_text("".join(f"a{index}\n" for index in range(1, 11)), encoding="utf-8")
    (tmp_path / "b.py").write_text("b1\n", encoding="utf-8")
    s = session(tmp_path)

    out = ReadTool(
        s,
        [
            {"path": "a.py", "ranges": [[5, 6]]},
            {"path": "b.py", "ranges": [[1, 1]]},
            {"path": "a.py", "ranges": [[1, 2], [3, 4], [9, 10]]},  # touching 1:2+3:4, disjoint 9:10
        ],
    ).call()
    keys = s.register_source_drafts(list(out.drafts))

    assert len(keys) == 2
    a_view = s.get_source_view(keys[0])
    assert [(span.start, span.end) for span in a_view.spans] == [(1, 6), (9, 10)]
    assert 'lines="1:6,9:10"' in out.retained_text
    assert out.retained_text.count("<Read ") == 2

    # A range inside a span resolves; one crossing the gap between spans is refused as unseen.
    assert a_view.range_lines(5, 6) == ("a5\n", "a6\n")
    with pytest.raises(ToolError, match="source range unseen"):
        a_view.range_lines(6, 9)


def test_search_skips_a_candidate_it_cannot_hydrate(tmp_path):
    """ripgrep nominates a path; the view has to come from reading that path now. When the read
    fails -- the file was removed, or grew past the size limit, between discovery and capture --
    the path is dropped from the result rather than reported with text nothing can vouch for."""
    (tmp_path / "keep.py").write_text("needle here\n", encoding="utf-8")
    (tmp_path / "vanishes.py").write_text("needle here\n", encoding="utf-8")
    s = session(tmp_path)
    tool = SearchTool(s, [{"pattern": "needle", "path": "."}])
    original = SearchTool.find_candidates

    def then_remove(self, request):
        rows = original(self, request)
        os.unlink(tmp_path / "vanishes.py")
        return rows

    SearchTool.find_candidates = then_remove
    try:
        out = tool.call()
    finally:
        SearchTool.find_candidates = original

    keys = s.register_source_drafts(list(out.drafts))
    assert [s.get_source_view(key).display_path for key in keys] == ["keep.py"]
    assert "vanishes.py" not in out.retained_text
    assert '<Search pattern="needle" matches=1>' in out.retained_text


def test_search_stop_hook_kills_and_reaps_its_child(tmp_path):
    """Search spends its time inside ripgrep, so cancelling the turn has to reach that process.

    The child is owned rather than run through `subprocess.run`, which gives no handle to kill:
    a stop kills it, `_run_rg` still reaps it before returning, and no further child is started."""
    import threading
    import time

    from wizolt.tools import SearchTool

    s = session(tmp_path)
    tool = SearchTool(s, [{"pattern": "x"}])
    result: list[object] = []
    thread = threading.Thread(target=lambda: result.append(tool._run_rg(["sleep", "30"])))
    thread.start()
    deadline = time.monotonic() + 2
    while tool._process is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert tool._process is not None, "the child never started"
    child = tool._process

    tool.request_stop()
    thread.join(timeout=5)

    assert not thread.is_alive(), "the killed child was never reaped"
    assert child.poll() is not None  # reaped, not left as a zombie
    assert result and result[0].returncode != 0
    # Once stopped, no further ripgrep invocation is started at all.
    assert tool._run_rg(["sleep", "30"]) is None
