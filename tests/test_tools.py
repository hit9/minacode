import os
import shutil

import code_symbol_index as csi
import pytest

import minacode
from minacode.base import LogBlock, LogEdge, LogLine, LogRole, RuntimeSettings, ToolCall, ToolError
from minacode.context import ContextManager
from minacode.model import ModelClient
from minacode.render import UiPrinter
from minacode.runner import ToolRunner
from minacode.session import HistorySegment, Session, SessionSnapshotCodec
from minacode.tools import (
    TOOL_REGISTRY,
    TOOLS,
    AskTool,
    BashTool,
    CodeIndex,
    EditTool,
    InspectCodeTool,
    MCPTool,
    NextHintsTool,
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

    assert ReadTool.parse_anchor(f"anchor=7:{short} | line") == (7, short)
    assert ReadTool.parse_anchor(f"7:{indexed}") == (7, indexed)
    assert ReadTool.require_anchor(f"7:{short}") == (7, short)
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
        ({}, "requires set_goal"),
    ],
)
def test_note_validation_errors_are_actionable(tmp_path, payload, message):
    with pytest.raises(ToolError, match=message):
        NoteTool(session(tmp_path), [payload]).call()


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


def test_ask_tool_call_basic(tmp_path):
    """call() returns question text when question_fn is None."""
    s = session(tmp_path)
    assert AskTool(s, _q({"question": "Which approach?"})).call() == "Which approach?"


def test_ask_tool_call_callback_passthrough_choices_none(tmp_path):
    """call() passes choices/previews/recommended as None when not provided."""
    s = session(tmp_path)
    calls = []

    def fake_fn(spec, position):
        calls.append((spec, position))
        return "free text answer"

    tool = AskTool(s, _q({"question": "Name?"}))
    tool.question_fn = fake_fn
    assert tool.call() == "free text answer"
    (spec, position) = calls[0]
    assert spec.choices is None
    assert spec.previews is None
    assert spec.recommended is None
    assert position == ""


def test_ask_tool_call_empty_list_raises(tmp_path):
    """call() raises ToolError when questions list is missing or empty."""
    s = session(tmp_path)
    with pytest.raises(ToolError, match="non-empty 'questions' list"):
        AskTool(s, [{"questions": []}]).call()
    with pytest.raises(ToolError, match="non-empty 'questions' list"):
        AskTool(s, [{}]).call()


def test_ask_tool_call_empty_question_raises(tmp_path):
    """call() raises ToolError for empty/missing question text."""
    s = session(tmp_path)
    with pytest.raises(ToolError, match="each question requires a 'question' field"):
        AskTool(s, _q({"question": ""})).call()
    with pytest.raises(ToolError, match="each question requires a 'question' field"):
        AskTool(s, _q({})).call()


def test_ask_tool_call_invalid_args_raises(tmp_path):
    """call() raises ToolError for malformed top-level args."""
    s = session(tmp_path)
    with pytest.raises(ToolError, match="Ask requires named fields"):
        AskTool(s, ["just a string"]).call()
    with pytest.raises(ToolError, match="Ask requires named fields"):
        AskTool(s, []).call()


def test_ask_tool_call_invalid_choices_raises(tmp_path):
    """call() validates choices type."""
    s = session(tmp_path)
    with pytest.raises(ToolError, match="Ask choices must be a list of strings"):
        AskTool(s, _q({"question": "Q", "choices": "not-a-list"})).call()
    with pytest.raises(ToolError, match="Ask choices must be a list of strings"):
        AskTool(s, _q({"question": "Q", "choices": [1, 2, 3]})).call()


def test_ask_tool_call_invalid_previews_raises(tmp_path):
    """call() validates previews type and length."""
    s = session(tmp_path)
    with pytest.raises(ToolError, match="Ask previews must be a list of strings"):
        AskTool(s, _q({"question": "Q", "choices": ["A"], "previews": [1]})).call()
    with pytest.raises(ToolError, match="Ask previews must match choices length"):
        AskTool(s, _q({"question": "Q", "choices": ["A", "B"], "previews": ["only one"]})).call()


def test_ask_tool_call_invalid_recommended_raises(tmp_path):
    """call() validates recommended is an in-range choice index."""
    s = session(tmp_path)
    with pytest.raises(ToolError, match="valid 0-based choice index"):
        AskTool(s, _q({"question": "Q", "choices": ["A", "B"], "recommended": 2})).call()
    with pytest.raises(ToolError, match="valid 0-based choice index"):
        AskTool(s, _q({"question": "Q", "recommended": 0})).call()  # no choices
    with pytest.raises(ToolError, match="valid 0-based choice index"):
        AskTool(s, _q({"question": "Q", "choices": ["A"], "recommended": True})).call()  # bool not int


def test_ask_tool_call_invokes_callback(tmp_path):
    """call() invokes question_fn with question/choices/previews/recommended."""
    s = session(tmp_path)
    calls = []

    def fake_fn(spec, position):
        calls.append((spec, position))
        return "user chose B"

    tool = AskTool(s, _q({"question": "A or B?", "choices": ["A", "B"], "previews": ["PA", "PB"], "recommended": 1}))
    tool.question_fn = fake_fn
    result = tool.call()
    assert result == "user chose B"
    (spec, position) = calls[0]
    assert (spec.question, spec.choices, spec.previews, spec.recommended) == ("A or B?", ["A", "B"], ["PA", "PB"], 1)
    assert position == ""  # a single question carries no position indicator


def test_ask_tool_call_multiple_questions(tmp_path):
    """call() asks each question in sequence and labels the combined answers."""
    s = session(tmp_path)
    asked = []

    def fake_fn(spec, position):
        asked.append((spec.question, position))
        return {"Runtime?": "Node", "Name?": "core"}[spec.question]

    tool = AskTool(
        s,
        _q(
            {"question": "Runtime?", "choices": ["Node", "Deno"]},
            {"question": "Name?"},
        ),
    )
    tool.question_fn = fake_fn
    result = tool.call()
    assert asked == [("Runtime?", "1/2"), ("Name?", "2/2")]  # sequential, with position
    assert result == "Q: Runtime?\nA: Node\n\nQ: Name?\nA: core"


def test_ask_tool_call_no_previews_with_choices(tmp_path):
    """call() allows choices without previews."""
    s = session(tmp_path)
    assert AskTool(s, _q({"question": "Q", "choices": ["A", "B"]})).call() == "Q"


def test_ask_tool_call_with_choices(tmp_path):
    """call() accepts choices and returns fallback question text."""
    s = session(tmp_path)
    assert AskTool(s, _q({"question": "Which?", "choices": ["A", "B"]})).call() == "Which?"


def test_ask_tool_call_with_choices_and_previews(tmp_path):
    """call() accepts choices + previews."""
    s = session(tmp_path)
    tool = AskTool(
        s,
        _q(
            {
                "question": "Which?",
                "choices": ["A", "B"],
                "previews": ["Preview A", "Preview B"],
            }
        ),
    )
    assert tool.call() == "Which?"


def test_ask_tool_registered():
    """AskTool is in TOOLS and TOOL_REGISTRY."""
    assert AskTool.NAME == "Ask"
    assert AskTool in TOOLS
    assert TOOL_REGISTRY["Ask"] is AskTool
    assert "Question" not in TOOL_REGISTRY
    assert not hasattr(minacode, "QuestionTool")


def test_ask_tool_schema():
    """params_schema requires a questions array of question objects, strict."""
    schema = AskTool.params_schema()
    assert schema["type"] == "object"
    assert schema["required"] == ["questions"]
    assert schema["additionalProperties"] is False
    questions = schema["properties"]["questions"]
    assert questions["type"] == "array"
    assert questions["minItems"] == 1
    item = questions["items"]
    assert item["required"] == ["question"]
    assert item["additionalProperties"] is False
    props = item["properties"]
    assert props["question"]["type"] == "string"
    assert props["choices"]["items"]["type"] == "string"
    assert props["previews"]["items"]["type"] == "string"
    assert props["recommended"]["type"] == "integer"


def test_ask_tool_schema_strict(tmp_path):
    """schema() enforces additionalProperties=False at both levels."""
    schema = AskTool.schema()
    params = schema["function"]["parameters"]
    assert params["additionalProperties"] is False
    assert "questions" in params["properties"]
    item = params["properties"]["questions"]["items"]
    assert item["additionalProperties"] is False
    assert "question" in item["properties"]
    assert "choices" in item["properties"]
    assert "previews" in item["properties"]


def test_ask_tool_short_args(tmp_path):
    """short_args() shows the first question and a count of the rest."""
    s = session(tmp_path)
    tool = AskTool(s, _q({"question": "Which approach should I use?"}))
    args = tool.short_args()
    assert len(args) == 1
    assert "Which approach" in args[0]
    assert "more" not in args[0]
    multi = AskTool(s, _q({"question": "First?"}, {"question": "Second?"}))
    assert "(+1 more)" in multi.short_args()[0]
    assert len(AskTool(s, []).short_args()) == 1


def test_ask_tool_validates_batch_before_asking(tmp_path):
    """A malformed later question raises before any question is asked."""
    s = session(tmp_path)
    asked = []

    def fake_fn(spec, position):
        asked.append(spec.question)
        return "x"

    tool = AskTool(
        s,
        _q(
            {"question": "First?", "choices": ["A"]},
            {"question": "Second?", "choices": ["A", "B"], "recommended": 5},  # out of range
        ),
    )
    tool.question_fn = fake_fn
    with pytest.raises(ToolError, match="valid 0-based choice index"):
        tool.call()
    assert asked == []  # validation happens up front, so nothing was asked


def test_ask_tool_wired_in_tool_runner(tmp_path):
    """ToolRunner injects question_fn into AskTool instances."""
    s = session(tmp_path)
    ctx = ContextManager(s)
    captured = []

    def fake_question_fn(spec, position):
        captured.append((spec, position))
        return "test answer"

    runner = ToolRunner(s, ctx, output_fn=lambda text: None)
    runner.question_fn = fake_question_fn
    results = runner.run([ToolCall("q", "Ask", [{"questions": [{"question": "A or B?", "choices": ["A", "B"], "recommended": 0}]}])])
    assert len(results) == 1
    assert results[0]["tool_call_id"] == "q"
    assert results[0]["role"] == "tool"
    assert "test answer" in results[0]["content"]
    (spec, position) = captured[0]
    assert (spec.question, spec.choices, spec.recommended, position) == ("A or B?", ["A", "B"], 0, "")


def test_auto_approved_tool_prints_single_line_with_tag(tmp_path):
    # In yolo mode a confirmation-requiring tool without a preview (Bash) should print only the
    # result line tagged [auto], not a redundant "auto …" pre-line that duplicates the header.
    s = session(tmp_path)
    s.settings.yolo = True
    out = []
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: out.append(str(text)))
    runner.run([ToolCall("b0", "Bash", [":"])])
    assert len(out) == 1
    assert out[0].startswith("  Bash  ")
    assert out[0].rstrip().endswith("[auto]")
    assert sum(line.startswith("  Bash  ") for line in out) == 1


def test_gitignore_cache_cleanup_on_file_delete(tmp_path):
    """Cache entry is removed when .gitignore is deleted."""
    gitignore = tmp_path / ".gitignore"
    gitignore.write_text("delete_me.txt\n", encoding="utf-8")
    s = session(tmp_path)
    tool = SearchTool(s, [{"pattern": "x"}])

    tool.gitignore_patterns(str(tmp_path))
    ws_gitignore = str(gitignore)
    assert ws_gitignore in s._gitignore_cache

    # Delete the .gitignore file
    gitignore.unlink()
    patterns = tool.gitignore_patterns(str(tmp_path))
    assert patterns == []
    assert ws_gitignore not in s._gitignore_cache


def test_gitignore_cache_invalidates_on_file_change(tmp_path):
    """Cache re-reads .gitignore when mtime changes."""
    gitignore = tmp_path / ".gitignore"
    gitignore.write_text("old.txt\n", encoding="utf-8")
    s = session(tmp_path)
    tool = SearchTool(s, [{"pattern": "x"}])

    patterns1 = tool.gitignore_patterns(str(tmp_path))
    assert patterns1 == ["old.txt"]

    ws_gitignore = str(gitignore)
    old_mtime = s._gitignore_cache[ws_gitignore][0]

    # Modify the .gitignore file
    gitignore.write_text("new.txt\n", encoding="utf-8")
    patterns2 = tool.gitignore_patterns(str(tmp_path))
    assert patterns2 == ["new.txt"]

    new_mtime = s._gitignore_cache[ws_gitignore][0]
    assert new_mtime != old_mtime


def test_gitignore_cache_keyed_by_root(tmp_path):
    """Different root directories cache independently."""
    sub = tmp_path / "sub"
    sub.mkdir()
    (tmp_path / ".gitignore").write_text("root_ignored.txt\n", encoding="utf-8")
    (sub / ".gitignore").write_text("sub_ignored.txt\n", encoding="utf-8")

    s = session(tmp_path)
    tool = SearchTool(s, [{"pattern": "x"}])

    # Root patterns include only workspace .gitignore
    root_patterns = tool.gitignore_patterns(str(tmp_path))
    assert "root_ignored.txt" in root_patterns
    assert "sub_ignored.txt" not in root_patterns

    # Sub patterns include workspace + sub .gitignore
    sub_patterns = tool.gitignore_patterns(str(sub))
    assert "root_ignored.txt" in sub_patterns  # workspace always included
    assert "sub_ignored.txt" in sub_patterns

    # Two separate cache entries
    assert len(s._gitignore_cache) == 2


def test_gitignore_cache_noop_when_no_gitignore(tmp_path):
    """When no .gitignore exists, returns empty list and cache stays empty."""
    s = session(tmp_path)
    tool = SearchTool(s, [{"pattern": "x"}])

    patterns = tool.gitignore_patterns(str(tmp_path))
    assert patterns == []
    assert len(s._gitignore_cache) == 0


def test_gitignore_cache_populated_and_reused(tmp_path):
    """Cache stores parsed patterns and reuses them on subsequent calls."""
    (tmp_path / ".gitignore").write_text("ignored.txt\nbuild/\n", encoding="utf-8")
    s = session(tmp_path)
    tool = SearchTool(s, [{"pattern": "x"}])

    # First call populates the cache
    patterns1 = tool.gitignore_patterns(str(tmp_path))
    assert "ignored.txt" in patterns1
    assert "build/" in patterns1

    # Cache should exist for the workspace .gitignore
    ws_gitignore = str(tmp_path / ".gitignore")
    assert ws_gitignore in s._gitignore_cache
    cached_mtime, cached_patterns = s._gitignore_cache[ws_gitignore]
    assert cached_patterns == patterns1

    # Second call reuses cache (mtime unchanged)
    patterns2 = tool.gitignore_patterns(str(tmp_path))
    assert patterns2 == patterns1
    # Cache entry unchanged
    assert s._gitignore_cache[ws_gitignore][0] == cached_mtime


def test_gitignore_cache_preserves_order(tmp_path):
    """After a no-op stat (no change), patterns come from cache unchanged."""
    (tmp_path / ".gitignore").write_text("a.txt\nb.txt\n", encoding="utf-8")
    s = session(tmp_path)
    tool = SearchTool(s, [{"pattern": "x"}])

    p1 = tool.gitignore_patterns(str(tmp_path))
    p2 = tool.gitignore_patterns(str(tmp_path))

    # Same object identity isn't required, but content must match
    assert p1 == p2 == ["a.txt", "b.txt"]


def test_gitignore_cache_shared_across_tools(tmp_path):
    """SearchTool instances share the same gitignore cache via Session."""
    (tmp_path / ".gitignore").write_text("secret.log\n", encoding="utf-8")
    s = session(tmp_path)

    find = SearchTool(s, [{"pattern": "x"}])
    search = SearchTool(s, [{"pattern": "needle", "path": "."}])

    # Find populates the cache
    find_patterns = find.gitignore_patterns(str(tmp_path))
    assert find_patterns == ["secret.log"]

    # Search reuses the same cache entry
    search_patterns = search.gitignore_patterns(str(tmp_path))
    assert search_patterns == find_patterns

    ws_key = str(tmp_path / ".gitignore")
    assert ws_key in s._gitignore_cache
    # Only one cache entry, not duplicated
    assert len(s._gitignore_cache) == 1


def test_gitignore_line_filtering_unchanged(tmp_path):
    """Cache still filters blank lines, comments, and negation patterns."""
    (tmp_path / ".gitignore").write_text("keep.txt\n\n  # comment\n!negated.txt\n  \n", encoding="utf-8")
    s = session(tmp_path)
    tool = SearchTool(s, [{"pattern": "x"}])

    patterns = tool.gitignore_patterns(str(tmp_path))
    assert patterns == ["keep.txt"]

    assert s.tool_errors == []


def test_inspect_code_api_errors_return_failed_result(tmp_path, monkeypatch):
    s = session(tmp_path)
    monkeypatch.setattr(CodeIndex, "available", lambda self: True)
    monkeypatch.setattr(csi, "search", lambda *args, **kwargs: (_ for _ in ()).throw(csi.CodeSymbolIndexError("bad query")))

    result = InspectCodeTool(s, ["find", "Missing"]).call()

    assert "* exit_code: 1" in result
    assert "bad query" in result


def test_inspect_code_modes_call_symbol_index_api(tmp_path, monkeypatch):
    s = session(tmp_path)
    (tmp_path / "sample.py").write_text("class Example:\n    pass\n", encoding="utf-8")
    calls = []

    monkeypatch.setattr(CodeIndex, "available", lambda self: True)
    monkeypatch.setattr(csi, "search", lambda query, **kwargs: calls.append(("search", query, kwargs)) or "search ok")
    monkeypatch.setattr(csi, "inspect", lambda query, **kwargs: calls.append(("inspect", query, kwargs)) or "inspect ok")
    monkeypatch.setattr(csi, "outline", lambda path, **kwargs: calls.append(("outline", path, kwargs)) or "outline ok")
    monkeypatch.setattr(csi, "refs", lambda query, **kwargs: calls.append(("refs", query, kwargs)) or "refs ok")
    monkeypatch.setattr(csi, "impls", lambda query, **kwargs: calls.append(("impls", query, kwargs)) or "impls ok")
    monkeypatch.setattr(csi, "callers", lambda query, **kwargs: calls.append(("callers", query, kwargs)) or "callers ok")
    monkeypatch.setattr(csi, "callees", lambda query, **kwargs: calls.append(("callees", query, kwargs)) or "callees ok")

    assert "search ok" in InspectCodeTool(s, ["find", "Example", {"kind": "class,function", "limit": 10, "exact_only": True}]).call()
    assert "inspect ok" in InspectCodeTool(s, ["inspect", "Example", {"path": "sample.py"}]).call()
    assert "outline ok" in InspectCodeTool(s, ["outline", "sample.py"]).call()
    assert "outline ok" in InspectCodeTool(s, ["outline", "sample.py", {"limit": 300}]).call()
    assert "refs ok" in InspectCodeTool(s, ["refs", "Example", {"all_kinds": True, "offset": 5}]).call()
    assert "impls ok" in InspectCodeTool(s, ["impls", "Example", {"kind": "class"}]).call()
    assert "callers ok" in InspectCodeTool(s, ["callers", "Example", {"depth": 2}]).call()
    assert "callees ok" in InspectCodeTool(s, ["callees", "Example"]).call()

    assert calls[0] == (
        "search",
        "Example",
        {"root": str(tmp_path), "kind": "class,function", "path": None, "exact_only": True, "format": "text", "limit": 10},
    )
    assert calls[1] == (
        "inspect",
        "Example",
        {
            "root": str(tmp_path),
            "kind": None,
            "path": "sample.py",
            "exact_only": False,
            "format": "text",
            "limit": csi.DEFAULT_PAGE_LIMIT,
            "anchors": True,
            "anchor_format": "explicit",
        },
    )
    assert calls[2] == (
        "outline",
        "sample.py",
        {"root": str(tmp_path), "symbol": None, "max_symbols": csi.DEFAULT_MAX_OUTLINE_SYMBOLS, "format": "text"},
    )
    assert calls[3] == (
        "outline",
        "sample.py",
        {"root": str(tmp_path), "symbol": None, "max_symbols": 300, "format": "text"},
    )
    assert calls[4] == (
        "refs",
        "Example",
        {
            "root": str(tmp_path),
            "kind": None,
            "path": None,
            "exact_only": False,
            "format": "text",
            "limit": csi.DEFAULT_MAX_REFERENCES,
            "offset": 5,
            "ref_kinds": "all",
        },
    )
    assert calls[5] == (
        "impls",
        "Example",
        {"root": str(tmp_path), "kind": "class", "path": None, "exact_only": False, "format": "text", "limit": csi.DEFAULT_MAX_IMPLEMENTORS, "offset": 0},
    )
    assert calls[6] == (
        "callers",
        "Example",
        {"root": str(tmp_path), "kind": None, "path": None, "exact_only": False, "format": "text", "limit": csi.DEFAULT_MAX_CALLERS, "depth": 2},
    )
    assert calls[7] == (
        "callees",
        "Example",
        {
            "root": str(tmp_path),
            "kind": None,
            "path": None,
            "exact_only": False,
            "format": "text",
            "limit": csi.DEFAULT_MAX_CALLEES,
            "depth": 3,
            "loose": False,
        },
    )

    assert "refs ok" in InspectCodeTool(s, ["refs", "Example", {"ref_kind": "call,write"}]).call()
    assert calls[8][2]["ref_kinds"] == "call,write"
    assert "callees ok" in InspectCodeTool(s, ["callees", "Example", {"loose": True}]).call()
    assert calls[9][2]["loose"] is True

    with pytest.raises(ToolError):
        InspectCodeTool(s, ["outline", "missing.py"]).call()
    with pytest.raises(ToolError):
        InspectCodeTool(s, ["inspect", "sample.py"]).call()
    with pytest.raises(ToolError):
        InspectCodeTool(s, ["outline", "sample.py", {"limit": 1001}]).call()
    with pytest.raises(ToolError):
        InspectCodeTool(s, ["refs", "sample.py"]).call()
    with pytest.raises(ToolError):
        InspectCodeTool(s, ["callers", "Example", {"depth": 9}]).call()
    with pytest.raises(ToolError):
        InspectCodeTool(s, ["refs", "Example", {"ref_kind": "bogus"}]).call()
    with pytest.raises(ToolError):
        InspectCodeTool(s, ["refs", "Example", {"ref_kind": "call", "all_kinds": True}]).call()


def test_inspect_code_strips_kind_prefix_from_target(tmp_path, monkeypatch):
    s = session(tmp_path)
    calls = []
    monkeypatch.setattr(CodeIndex, "available", lambda self: True)
    monkeypatch.setattr(csi, "search", lambda query, **kwargs: calls.append(query) or "ok")

    # "class Config" with kind "class" -> the redundant leading kind word is dropped.
    InspectCodeTool(s, ["find", "class Config", {"kind": "class"}]).call()
    assert calls[-1] == "Config"

    # Works when the kind option lists several kinds.
    InspectCodeTool(s, ["find", "function handoff", {"kind": "class,function"}]).call()
    assert calls[-1] == "handoff"

    # Only the declared kind is stripped: a bare language keyword is not, and still errors.
    with pytest.raises(ToolError):
        InspectCodeTool(s, ["find", "def foo", {"kind": "function"}]).call()
    # No kind provided -> nothing to key off, still rejected.
    with pytest.raises(ToolError):
        InspectCodeTool(s, ["find", "class Config"]).call()


def test_log_block_aligns_multiline_tool_arguments():
    block = LogBlock.hierarchy(
        LogLine("Bash", 'git commit -m "title\nbody"', LogRole.TOOL, syntax="bash"),
        [LogLine("done", role=LogRole.META, edge=LogEdge.END)],
    )
    expected = '  Bash  git commit -m "title\n        body"\n    └ done'

    assert str(block) == expected
    rendered = "".join(text for _style, text in UiPrinter(output_fn=lambda text: None).log_segments(block))
    assert rendered == expected + "\n"


def test_log_block_wraps_long_tool_arguments_with_hanging_indent(monkeypatch):
    command = 'git commit -m "system prompt: enhance with attitude, updates, review mode, and tooling rules"'
    block = LogBlock([LogLine("Bash", command, LogRole.TOOL, syntax="bash")])

    with monkeypatch.context() as patch:
        patch.setattr(shutil, "get_terminal_size", lambda fallback=(80, 24): os.terminal_size((40, 24)))
        rendered = "".join(text for _style, text in UiPrinter(output_fn=lambda text: None).log_segments(block))

    assert rendered.splitlines() == [
        '  Bash  git commit -m "system prompt:',
        "        enhance with attitude,",
        "        updates, review mode, and",
        '        tooling rules"',
    ]
    assert all(len(line) < 40 for line in rendered.splitlines())


def test_note_tool_replace_known(tmp_path):
    s = session(tmp_path)
    s.state.known = ["old fact"]
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    short = runner.short_call(ToolCall("n", "Note", [{"replace_known": ["new fact a", "new fact b"]}]))
    assert short == "Note known:\n  new fact a\n  new fact b"

    output = []
    runner.output_fn = output.append
    runner.run([ToolCall("n", "Note", [{"replace_known": ["new fact a", "new fact b"]}])])
    assert s.state.known == ["new fact a", "new fact b"]
    assert output == ["known:\n  new fact a\n  new fact b"]

    runner.run([ToolCall("n", "Note", [{"replace_known": []}])])
    assert s.state.known == []


def test_note_tool_set_check(tmp_path):
    s = session(tmp_path)
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    short = runner.short_call(ToolCall("n", "Note", [{"set_check": "pytest -q passed"}]))
    assert short == "Note check: pytest -q passed"

    output = []
    runner.output_fn = output.append
    runner.run([ToolCall("n", "Note", [{"set_check": "pytest -q passed"}])])
    assert s.state.check == "pytest -q passed"
    assert output == ["check: pytest -q passed"]


def test_note_tool_updates_durable_memory_without_result_key(tmp_path):
    s = session(tmp_path)
    s.state.known = ["existing"]
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    output = []
    runner.output_fn = output.append
    runner.run(
        [
            ToolCall(
                "note",
                "Note",
                [
                    {
                        "set_goal": "ship",
                        "replace_plan": [{"status": "doing", "text": "inspect"}, {"status": "todo", "text": "patch"}],
                        "append_known": ["existing", "pytest"],
                    }
                ],
            )
        ]
    )

    assert s.state.goal == "ship"
    assert [vars(item) for item in s.state.plan] == [{"status": "doing", "text": "inspect"}, {"status": "todo", "text": "patch"}]
    assert s.state.known == ["existing", "pytest"]
    assert s.tool_records == []
    assert output == ["goal: ship\nplan:\n  - [~] inspect\n  - [ ] patch\nknown:\n  + pytest"]


def test_note_tool_validates_before_mutating_state(tmp_path):
    s = session(tmp_path)
    s.state.goal = "old goal"
    s.state.plan = ["old plan"]
    s.state.known = ["old fact"]

    with pytest.raises(ToolError) as error:
        NoteTool(s, [{"set_goal": "new goal", "replace_plan": "inspect"}]).call()

    assert str(error.value) == 'Note replace_plan must be an array of plan items, e.g. {"replace_plan":[{"status":"doing","text":"inspect"}]}'
    assert s.state.goal == "old goal"
    assert s.state.plan == ["old plan"]
    assert s.state.known == ["old fact"]

    with pytest.raises(ToolError, match="Note replace_plan status must be one of"):
        NoteTool(s, [{"replace_plan": [{"status": "started", "text": "inspect"}]}]).call()


def test_suggest_tool_sets_transient_quick_hints(tmp_path):
    s = session(tmp_path)
    assert NextHintsTool(s, [{"inputs": ["run the tests", "show the diff"]}]).call() == "Offered 2 quick input(s)"
    assert s.quick_hints == ("run the tests", "show the diff")


def test_suggest_tool_dedupes_and_caps(tmp_path):
    s = session(tmp_path)
    NextHintsTool(s, [{"inputs": ["a", "a", "b", "c", "d", "e"]}]).call()
    assert s.quick_hints == ("a", "b", "c", "d")


def test_suggest_tool_validates_before_writing(tmp_path):
    s = session(tmp_path)
    with pytest.raises(ToolError, match="inputs must be an array"):
        NextHintsTool(s, [{"inputs": "run"}]).call()
    with pytest.raises(ToolError, match="at least one non-empty"):
        NextHintsTool(s, [{"inputs": ["  "]}]).call()
    with pytest.raises(ToolError, match="unexpected field"):
        NextHintsTool(s, [{"inputs": ["a"], "extra": 1}]).call()
    assert s.quick_hints == ()


def test_suggest_tool_does_not_store_result():
    assert NextHintsTool.STORES_RESULT is False


def test_suggest_tool_short_args(tmp_path):
    tool = NextHintsTool(session(tmp_path), [{"inputs": ["run the tests", "show the diff"]}])
    assert tool.short_args() == ["inputs:\n  - run the tests\n  - show the diff"]


def test_quick_hints_are_transient_and_never_serialized(tmp_path):
    s = session(tmp_path)
    s.set_quick_hints(["run the tests", "show the diff"])
    assert s.quick_hints == ("run the tests", "show the diff")
    snapshot = SessionSnapshotCodec.snapshot(s, {})
    assert "quick_hints" not in snapshot
    assert "quick_hints" not in snapshot["state"]
    s.clear_quick_hints()
    assert s.quick_hints == ()


def test_runtime_settings_quick_hints_default_and_override():
    assert RuntimeSettings.from_dict({}).quick_hints is True
    assert RuntimeSettings.from_dict({"runtime": {"quick_hints": False}}).quick_hints is False


def test_resolved_schemas_hide_next_hints_when_disabled(tmp_path):
    s = session(tmp_path)

    def names():
        return {schema["function"]["name"] for schema in Tool.resolved_schemas(s)}

    assert "NextHints" in names()
    s.settings.quick_hints = False
    assert "NextHints" not in names()


def test_read_and_search_success_paths(tmp_path):
    (tmp_path / "sample.py").write_text("alpha\nNeedle\nomega\n", encoding="utf-8")
    (tmp_path / "blob.bin").write_bytes(b"a\0b")
    s = session(tmp_path)

    read = ReadTool(s, [{"path": "sample.py", "ranges": [[0, 2], [2, 0]]}]).call()
    single_range = ReadTool(s, [{"path": "sample.py", "ranges": [0, 2]}]).call()
    full_default = ReadTool(s, [{"path": "sample.py"}]).call()
    alpha_hash = ReadTool.line_hash("alpha\n")
    needle_hash = ReadTool.line_hash("Needle\n")
    omega_hash = ReadTool.line_hash("omega\n")
    assert f"anchor=0:{alpha_hash} | alpha" in read
    assert f"anchor=1:{needle_hash} | Needle" in read
    assert f"anchor=2:{omega_hash} | omega" in read
    assert f"anchor=0:{alpha_hash} | alpha" in single_range
    assert f"anchor=1:{needle_hash} | Needle" in single_range
    assert f"anchor=2:{omega_hash} | omega" in full_default
    assert "<total_lines>3</total_lines>" in full_default  # Read reports the line count (replaces LineCount)

    found = SearchTool(s, [{"pattern": "needle", "path": "."}]).call()
    assert f"sample.py anchor=1:{needle_hash} | Needle" in found

    multiline = SearchTool(s, [{"pattern": "alpha\\nNeedle", "path": "sample.py"}]).call()
    assert "sample.py anchor=0:" in multiline


def test_recall_behaviors(tmp_path):
    s = session(tmp_path)
    first = s.store_tool_result("Read", ["a.txt"], "a0\na1\na2\n")
    second = s.store_tool_result("Search", [{"pattern": "b"}], "b0\nb1\n")

    sliced = RecallTool(s, [{"keys": [first, second], "ranges": [[1, 2]]}]).call()
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

    for payload in ({}, {"query": "["}, {"query": "x", "limit": 0}, {"keys": ["seg.1"], "case_sensitive": True}):
        with pytest.raises(ToolError):
            RecallContextTool(s, [payload]).call()


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

    assert "visible.txt anchor=0:" in found
    assert ".hidden" not in found
    assert "ignored" not in found
    assert ".hidden.txt anchor=0:" not in direct_hidden
    assert "ignored_dir/inside.txt anchor=0:" not in direct_ignored


def test_single_and_batch_payload_shapes_are_supported():
    assert ModelClient.tool_payload("Read", {"path": "a.py"}) == [{"path": "a.py", "ranges": [[0, 0]]}]
    assert ModelClient.tool_payload("Read", {"path": "a.py", "ranges": [0, 2]}) == [{"path": "a.py", "ranges": [[0, 2]]}]
    assert ModelClient.tool_payload("Read", {"files": [{"path": "a.py", "ranges": [[0, 1]]}]}) == [{"path": "a.py", "ranges": [[0, 1]]}]
    assert ReadTool(Session(cwd="."), [{"path": "minacode.py"}]).targets()[0][1] == [(0, 0)]
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
    assert "start/end anchors are inclusive" in EditTool.schema()["function"]["description"]

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
        EditTool(s, ["a.txt", [{"op": "replace_all", "old": "", "new": "a\n"}]]).call()
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
