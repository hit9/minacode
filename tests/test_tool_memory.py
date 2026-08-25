"""tool memory (split from tests/test_tools.py)."""

import json

import pytest
from test_tools import session

from minacode.tools import tooloutput
from minacode.base import (
    ToolCall,
    ToolError,
)
from minacode.config import (
    ConfigFile,
    RuntimeSettings,
)
from minacode.context import ContextManager
from minacode.runner import ToolRunner
from minacode.session import HistorySegment, Session, SessionSnapshotCodec
from minacode.tools import (
    NextHintsTool,
    NoteTool,
    RecallContextTool,
    Tool,
)


def test_note_tool_replace_known(tmp_path):
    s = session(tmp_path)
    s.state.known = ["old fact"]
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    short = tooloutput.short_call(runner.session, ToolCall("n", "Note", [{"replace_known": ["new fact a", "new fact b"]}]))
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

    short = tooloutput.short_call(runner.session, ToolCall("n", "Note", [{"set_check": "pytest -q passed"}]))
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


def test_note_tool_views_selected_state_without_mutating(tmp_path):
    s = session(tmp_path)
    s.state.goal = "ship"
    s.state.plan = [{"status": "doing", "text": "verify"}]

    result = json.loads(NoteTool(s, [{"action": "view", "fields": ["goal", "plan"]}]).call())

    assert result == {"goal": "ship", "plan": [{"status": "doing", "text": "verify"}]}
    assert NoteTool(s, [{"action": "view"}]).needs_confirmation() is False


def test_memory_tools_treat_strict_schema_nulls_as_omitted(tmp_path):
    s = session(tmp_path)
    note_result = json.loads(
        NoteTool(
            s,
            [{"action": "update", "fields": None, "set_goal": "ship", "replace_plan": None, "append_known": None, "replace_known": None, "set_check": None}],
        ).call()
    )
    s.history.append(HistorySegment(key="seg.1", title="cache"))
    list_result = json.loads(
        RecallContextTool(
            s,
            [{"action": "list", "keys": None, "query": None, "case_sensitive": None, "limit": 1, "before": None}],
        ).call()
    )

    assert note_result["changed"] == ["goal"]
    assert list_result["segments"] == [{"key": "seg.1", "title": "cache"}]


def test_note_short_args_treats_strict_schema_nulls_as_omitted(tmp_path):
    payload = {
        "action": None,
        "fields": None,
        "set_goal": None,
        "replace_plan": None,
        "append_known": None,
        "replace_known": None,
        "set_check": None,
    }

    assert NoteTool(session(tmp_path), [payload]).short_args() == ["view all"]


def test_note_empty_goal_and_check_explicitly_clear_state(tmp_path):
    s = session(tmp_path)
    s.state.goal = "ship"
    s.state.check = "tests pass"
    tool = NoteTool(s, [{"set_goal": "", "set_check": ""}])

    assert tool.short_args() == ["goal: (cleared)\ncheck: (cleared)"]
    assert json.loads(tool.call())["changed"] == ["goal", "check"]
    assert s.state.goal == ""
    assert s.state.check == ""


def test_recall_context_distinguishes_a_dropped_segment_from_an_unknown_one(tmp_path):
    """Only the newest segments are retained, so a key below the window is gone for good. Saying
    that, with what is still reachable, is what stops the model from retrying the same key."""
    s = session(tmp_path)
    s.history.extend(HistorySegment(key=f"seg.{number}", title=f"span {number}", text="body") for number in range(7, 10))

    result = RecallContextTool(s, [{"action": "get", "keys": ["seg.3", "seg.99", "seg.8"]}]).call()

    assert "* seg.3: dropped; only the newest 3 segments are kept, from seg.7" in result
    assert "* seg.99: missing" in result
    assert '<Segment key="seg.8" title="span 8">' in result


def test_recall_context_returns_the_summary_as_it_stood_at_that_compaction(tmp_path):
    """The checkpoint carries only the newest summary, and every compaction folds the previous one
    into the next — so the live summary has been through one pass per compaction while this copy
    has been through exactly one. It was already stored; returning it is what makes it reachable.
    """
    s = session(tmp_path)
    s.history.append(HistorySegment(key="seg.1", title="span", text="body", summary="what that span settled"))
    s.history.append(HistorySegment(key="seg.2", title="trimmed", text="body"))  # summarizer failed: no summary

    result = RecallContextTool(s, [{"action": "get", "keys": ["seg.1", "seg.2"]}]).call()

    assert "<SummaryAtCompaction>\nwhat that span settled\n</SummaryAtCompaction>" in result
    assert result.count("<SummaryAtCompaction>") == 1  # a segment with no summary carries no empty block


def test_memory_tools_ignore_schema_valid_empty_and_default_fillers(tmp_path):
    s = session(tmp_path)
    s.history.append(HistorySegment(key="seg.1", title="cache", text="needle"))

    listed = json.loads(
        RecallContextTool(
            s,
            [{"action": "list", "keys": [], "query": "", "case_sensitive": False, "limit": 20}],
        ).call()
    )
    searched = RecallContextTool(
        s,
        [{"action": "search", "keys": [], "query": "needle", "case_sensitive": False, "limit": 20}],
    ).call()
    retrieved = RecallContextTool(
        s,
        [{"action": "get", "keys": ["seg.1"], "query": "", "case_sensitive": False, "limit": 20}],
    ).call()
    updated = json.loads(NoteTool(s, [{"action": "update", "fields": [], "set_goal": "ship"}]).call())
    viewed = json.loads(NoteTool(s, [{"action": "view", "fields": []}]).call())

    assert listed["segments"] == [{"key": "seg.1", "title": "cache"}]
    assert "needle" in searched
    assert "needle" in retrieved
    assert updated["changed"] == ["goal"]
    assert viewed["goal"] == "ship"


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"action": "view", "set_goal": "bad"}, "view does not accept"),
        ({"action": "view", "fields": ["goal", "summary"]}, "fields must contain only"),
        ({"action": "update", "fields": ["goal"], "set_goal": "bad"}, "fields is only valid"),
    ],
)
def test_note_actions_reject_conflicting_fields(tmp_path, payload, message):
    with pytest.raises(ToolError, match=message):
        NoteTool(session(tmp_path), [payload]).call()


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
    with pytest.raises(ToolError, match="inputs must be an array"):
        NextHintsTool(s, [{"inputs": [1, {"x": 1}]}]).call()  # non-string elements are rejected
    with pytest.raises(ToolError, match="unexpected field"):
        NextHintsTool(s, [{"inputs": ["a"], "extra": 1}]).call()
    assert s.quick_hints == ()


def test_suggest_tool_does_not_store_result():
    assert NextHintsTool.STORES_RESULT is False


def test_suggest_tool_merges_multiple_calls_in_one_batch(tmp_path):
    """Several legal NextHints calls in one batch accumulate their inputs in call order,
    deduplicated and capped at MAX_HINTS, instead of the last call replacing the rest."""
    s = session(tmp_path)
    runner = ToolRunner(s, ContextManager(s), input_fn=lambda *a: "", output_fn=lambda text: None)
    messages = runner.run(
        [
            ToolCall("n1", "NextHints", [{"inputs": ["run the tests", "show the diff"]}]),
            ToolCall("n2", "NextHints", [{"inputs": ["commit the work", "run the tests"]}]),
        ]
    )
    assert len(messages) == 2  # each call still gets its own tool result
    assert s.quick_hints == ("run the tests", "show the diff", "commit the work")

    # The cap applies to the merged total, not per call.
    s.clear_quick_hints()
    first = [f"a{index}" for index in range(4)]
    second = [f"b{index}" for index in range(4)]
    runner.run([ToolCall("n3", "NextHints", [{"inputs": first}]), ToolCall("n4", "NextHints", [{"inputs": second}])])
    assert s.quick_hints == (*first, *second)[: NextHintsTool.MAX_HINTS]


def test_suggest_tool_short_args(tmp_path):
    tool = NextHintsTool(session(tmp_path), [{"inputs": ["run the tests", "show the diff"]}])
    assert tool.short_args() == ['inputs: "run the tests", "show the diff"']


def test_quick_hints_are_transient_and_never_serialized(tmp_path):
    s = session(tmp_path)
    s.add_quick_hints(["run the tests", "show the diff"])
    s.next_hints_available = False
    assert s.quick_hints == ("run the tests", "show the diff")
    snapshot = SessionSnapshotCodec.snapshot(s, {})
    assert "quick_hints" not in snapshot
    assert "next_hints_available" not in snapshot
    assert "quick_hints" not in snapshot["state"]
    s.clear_quick_hints()
    assert s.quick_hints == ()


def test_runtime_settings_no_longer_exposes_quick_hints_config(tmp_path):
    settings = RuntimeSettings.from_dict({"runtime": {"quick_hints": False}})
    s = session(tmp_path)
    s.settings = settings

    assert not hasattr(settings, "quick_hints")
    assert "quick_hints" not in ConfigFile.DEFAULT_TEXT
    assert "NextHints" in {schema["function"]["name"] for schema in Tool.resolved_schemas(s)}


def test_legacy_config_quick_hints_key_loads_and_keeps_hints_enabled(tmp_path):
    """An old config file with `[runtime] quick_hints = false` still loads: the obsolete key is
    ignored (no runtime field, no crash) and the TUI capability stays on, so hints are not
    disabled by stale configuration."""
    cfg = tmp_path / "minacode.toml"
    cfg.write_text("[runtime]\nquick_hints = false\n", encoding="utf-8")
    s = Session.from_config_file(path=str(cfg))

    assert not hasattr(s.settings, "quick_hints")
    assert s.next_hints_available is True
    assert "NextHints" in {schema["function"]["name"] for schema in Tool.resolved_schemas(s)}


def test_resolved_schemas_follow_frontend_next_hints_capability(tmp_path):
    s = session(tmp_path)

    def names():
        return {schema["function"]["name"] for schema in Tool.resolved_schemas(s)}

    assert "NextHints" in names()
    s.next_hints_available = False
    assert "NextHints" not in names()
