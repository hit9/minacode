"""bounded output (split from tests/test_context.py)."""
import os
import re

from agent_harness import call, session

from wizolt.base import (
    MAX_TOOL_OUTPUT_TOKENS,
)
from wizolt.context import ContextManager
from wizolt.runner import ToolRunner
from wizolt.tools import ReadTool


def test_session_tool_result_store_prunes_old_records(tmp_path):
    s = session(tmp_path)
    for index in range(405):
        s.store_tool_result("Bash", [str(index)], f"output {index}")

    assert len(s.tool_results) == 400
    assert len(s.tool_records) == 400
    assert "tr.1" not in s.tool_results
    assert s.tool_records[0].key == "tr.6"
    assert "tr.405" in s.tool_results

def test_bounded_output_marks_recall_key(tmp_path):
    s = session(tmp_path)
    context = ContextManager(s)
    large = "head\n" + "\n".join(f"line {index}" for index in range(20000)) + "\ntail\n"
    bounded = context.bound_output(large, "tr.large")

    assert "head" in bounded
    assert "tail" in bounded
    assert "<bounded_output" in bounded
    assert 'recall="tr.large"' in bounded

def test_bounded_output_keeps_head_and_tail_of_a_single_line_payload(tmp_path):
    """The reported failure: an MCP server returns one long line of compact JSON, and snapping the
    excerpts to a line boundary threw away everything but the wrapper tags -- the model saw a marker
    claiming ~9.5k omitted tokens out of ~9.5k, with no content on either side of it."""
    s = session(tmp_path)
    context = ContextManager(s)
    payload = '{"id": 1, "system_prompt": "' + "x" * 40000 + '", "tail_field": "last"}'
    large = f'<MCPCall server="orion" tool="get_agent">\n{payload}\n</MCPCall>'

    bounded = context.bound_output(large, "tr.1")

    assert '"system_prompt"' in bounded  # real head, not just the opening tag
    assert '"tail_field": "last"' in bounded  # real tail, not just the closing tag
    estimated = context.estimated_text_tokens(large)
    omitted = int(re.findall(r'omitted_tokens="(\d+)"', bounded)[0])
    # Roughly the whole budget is spent on content, rather than snapped away with it.
    assert omitted <= estimated - MAX_TOOL_OUTPUT_TOKENS // 2

def test_bounded_output_still_snaps_excerpts_to_line_boundaries(tmp_path):
    """Line-oriented output keeps whole lines: snapping is only abandoned when it would cost more
    than half the budget, so ordinary logs never gain a half-line at the cut."""
    s = session(tmp_path)
    context = ContextManager(s)
    large = "\n".join(f"line {index} " + "y" * 40 for index in range(20000))

    bounded = context.bound_output(large, "tr.1")
    head, _, rest = bounded.partition("<bounded_output")
    _, _, tail = rest.partition("/>")

    assert all(line in large.split("\n") for line in head.strip().split("\n"))
    assert all(line in large.split("\n") for line in tail.strip().split("\n"))

def test_bounded_output_materializes_full_output_to_asset_file(tmp_path):
    s = session(tmp_path)
    context = ContextManager(s)
    large = "head\n" + "\n".join(f"line {index}" for index in range(20000)) + "\ntail\n"
    bounded = context.bound_output(large, "tr.1")

    path = os.path.join(s.images.assets_dir(), "tr.1.txt")
    assert 'recall="tr.1"' in bounded
    assert f'file="{path}"' in bounded
    with open(path, encoding="utf-8") as file:
        assert file.read() == large

def test_bounded_output_marker_names_the_cheaper_way_to_read_the_rest(tmp_path):
    """`recall` and `file` say where the omitted middle went, not what to do about it, and the cheap
    move is the non-obvious one. The marker points at whichever one the result actually has."""
    s = session(tmp_path)
    context = ContextManager(s)
    large = "head\n" + "\n".join(f"line {index}" for index in range(20000)) + "\ntail\n"

    bounded = context.bound_output(large, "tr.1")
    assert f'hint="{ContextManager.OMITTED_OUTPUT_HINT}"' in bounded

    # No file to point at: the marker falls back to naming the Recall form that pages.
    (tmp_path / "blocked.txt").write_text("x", encoding="utf-8")
    s.images.assets_dir = lambda: str(tmp_path / "blocked.txt" / "sub")
    fileless = context.bound_output(large, "tr.2")
    assert 'file="' not in fileless
    assert f'hint="{ContextManager.OMITTED_OUTPUT_RECALL_HINT}"' in fileless

    # The compaction summary is bounded with no key at all: nothing to recall, nothing to advise.
    assert "hint=" not in context.bound_output(large, "")

def test_bounded_output_small_or_keyless_never_writes_asset_file(tmp_path):
    s = session(tmp_path)
    context = ContextManager(s)

    small = context.bound_output("small output", "tr.1")
    assert 'file="' not in small

    large = "head\n" + "\n".join(f"line {index}" for index in range(20000)) + "\ntail\n"
    keyless = context.bound_output(large, "")
    assert 'recall="' not in keyless
    assert 'file="' not in keyless

def test_bounded_output_survives_asset_write_failure(tmp_path, monkeypatch):
    s = session(tmp_path)
    context = ContextManager(s)
    large = "head\n" + "\n".join(f"line {index}" for index in range(20000)) + "\ntail\n"
    # A regular file where the assets dir would go: makedirs raises OSError.
    (tmp_path / "blocked.txt").write_text("x", encoding="utf-8")
    monkeypatch.setattr(s.images, "assets_dir", lambda: str(tmp_path / "blocked.txt" / "sub"))

    bounded = context.bound_output(large, "tr.1")
    assert 'recall="tr.1"' in bounded
    assert 'file="' not in bounded

def test_read_tool_message_inlines_bounded_output(tmp_path):
    path = tmp_path / "large.txt"
    path.write_text("first\n" + "\n".join(f"middle-{index}" for index in range(20000)) + "\nlast\n", encoding="utf-8")
    s = session(tmp_path)
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)
    call_obj = call("Read", [{"path": "large.txt", "ranges": [[1, 0]]}])
    output = ReadTool(s, call_obj.args).call()
    key = s.store_tool_result("Read", call_obj.args, output)

    message = runner.tool_message(call_obj, key, output)

    # A whole-file read prints no range: 1:0 is the "to the end of the file" sentinel, and
    # echoing it raw would show the model an empty-looking range it never wrote.
    assert message.startswith("tool tr.1 Read large.txt\noutput:\n")
    assert "<Read" in message
    assert "<bounded_output" in message
    assert 'recall="tr.1"' in message
    assert "-> FILE STATE" not in message

def test_tool_error_records_keep_recent_failures(tmp_path):
    s = session(tmp_path)
    for index in range(7):
        s.record_tool_error(f"tr.{index}", "Bash", [str(index)], f"error {index}")

    assert [record.key for record in s.tool_errors] == ["tr.2", "tr.3", "tr.4", "tr.5", "tr.6"]

def test_working_context_does_not_repeat_durable_tool_errors(tmp_path):
    s = session(tmp_path)
    for index in range(6):
        s.record_tool_error(f"tr.{index}", "Bash", [f"cmd {index}"], f"error {index}")

    context = "\n".join(str(message.get("content") or "") for message in ContextManager(s).model_messages("sys"))

    assert "Recent tool errors:" not in context
    assert "error 5" not in context
