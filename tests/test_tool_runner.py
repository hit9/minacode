"""tool runner (split from tests/test_agent_turn.py)."""

from agent_harness import call, session

from wizolt.context import ContextManager
from wizolt.runner import ToolRunner
from wizolt.tools import toolblocks, tooloutput


def test_tool_runner_refusal_stops_batch_and_invalid_args_are_not_stored(tmp_path):
    s = session(tmp_path)
    runner = ToolRunner(s, ContextManager(s), input_fn=lambda prompt: "skip it", output_fn=lambda text: None)
    runner.run([call("Bash", [":"]), call("Edit", ["second.txt", [{"op": "create", "content": "second"}]])])

    assert s.tool_records == []
    assert len(s.tool_errors) == 1
    assert "skip it" in s.tool_errors[0].error
    assert not (tmp_path / "second.txt").exists()

    outputs = []
    bad = session(tmp_path)
    ToolRunner(bad, ContextManager(bad), output_fn=lambda text: outputs.append(str(text))).run([call("Bash", [])])
    assert bad.tool_records == []
    assert len(bad.tool_errors) == 1
    assert outputs and "· rejected:" in outputs[0]  # argument errors collapse to a quiet line


def test_rejected_and_failed_calls_collapse_a_multiline_display_to_one_line(tmp_path):
    # A tool's display is whatever its short_args produced, and Note's is the whole rendered note so
    # a successful call can print it. A rejection is meant to be a quiet one-liner and a failure
    # leads with a red tag, so neither may inherit those lines: a rejected Note used to dim its
    # entire body and bury the reason at the end of the last line.
    from wizolt.base import LogRole
    from wizolt.tools.toolblocks import ToolDisplay

    s = session(tmp_path)
    note = call("Note", [{"replace_plan": [f"Task {index}" for index in range(1, 11)]}])
    display = ToolDisplay()
    display.display = tooloutput.short_call(s, note)
    assert len(display.display.splitlines()) > 1, "precondition: Note renders a multi-line display"

    rejected = list(toolblocks.reject_display(s, note, "ToolError: Note fields is only valid for view", d=display).walk())
    assert [item.role for item, _ in rejected] == [LogRole.MUTED]
    assert len(rejected[0][0].text.splitlines()) == 1
    assert rejected[0][0].text.endswith("· rejected: Note fields is only valid for view")

    failed = list(toolblocks.finish_display(s, note, "", "Note: disk is full", failed=True, d=display).walk())
    assert all(len((item.text or "").splitlines()) == 1 for item, _ in failed)

    # A successful Note is the one case that should keep every line: printing the note is the point.
    assert len(toolblocks.finish_display(s, note, "", "ok", failed=False, d=display).splitlines()) > 1


def test_tool_runner_refuses_without_reason_on_n(tmp_path):
    s = session(tmp_path)
    runner = ToolRunner(s, ContextManager(s), input_fn=lambda prompt: "n", output_fn=lambda text: None)

    runner.run([call("Bash", [":"])])

    assert s.tool_errors[0].error == "Cancelled: user refused tool call"


def test_tool_runner_refuses_with_direct_reason_input(tmp_path):
    s = session(tmp_path)
    runner = ToolRunner(s, ContextManager(s), input_fn=lambda prompt: "not now", output_fn=lambda text: None)

    runner.run([call("Bash", [":"])])

    assert s.tool_records == []
    assert len(s.tool_errors) == 1
    assert "not now" in s.tool_errors[0].error


def test_recall_tool_runner_does_not_create_new_result_keys(tmp_path):
    s = session(tmp_path)
    key = s.store_tool_result("Read", ["a.txt"], "result")
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    runner.run([call("Recall", [key])])
    assert [record.key for record in s.tool_records] == [key]


def test_replayed_delegate_keeps_its_call_line(tmp_path):
    # The finish block drops its own root when a worker_rule is wired, because live the closing
    # full-width rule carries the call instead. Transcript replay wires no rule and passes no
    # output, so the rule never fired and the saved call rendered as an orphaned `stored tr.N`
    # under nothing. Replay takes the fallback root; the live path still hands its rule over.
    from wizolt.base import ToolCall

    s = session(tmp_path)
    delegate = ToolCall(id="", name="Delegate", args=[{"action": "send", "order": "do it"}])

    replayed = str(toolblocks.finish_display(s, delegate, "tr.7", "", failed=False))
    assert replayed.splitlines()[0].strip().startswith("[worker]")
    assert "stored tr.7" in replayed

    # Live, a wired rule still takes the root away: the rule is the call line there.
    live = str(toolblocks.finish_display(s, delegate, "tr.7", "", failed=False, worker_rule=lambda text: None))
    assert "[worker]" not in live
