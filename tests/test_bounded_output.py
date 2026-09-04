"""bounded output (split from tests/test_context.py)."""
import os
import re

from agent_harness import call, session

from wizolt.base import (
    MAX_TOOL_OUTPUT_TOKENS,
    ToolCall,
)
from wizolt.context import ContextManager
from wizolt.runner import ToolRunner
from wizolt.source import READ, SourceBlock, SourceSpan, SourceViewDraft, TextBlock, ToolOutput
from wizolt.tools import CodeIndex, ReadTool


def estimate(text: str) -> int:
    """One character per token: projection only needs a monotonic cost, and this makes the
    budget arithmetic in these tests readable."""
    return len(text)


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

    # The runner projects the source block, stores the retained text under tr.N, and inlines the
    # bounded marker (with the recall key) into the model-facing message.
    message = runner.finish(call_obj, output)

    # A whole-file read prints no range: 1:0 is the "to the end of the file" sentinel, and
    # echoing it raw would show the model an empty-looking range it never wrote.
    assert message.startswith("tool tr.1 Read large.txt\noutput:\n")
    assert "<Read" in message
    assert "<bounded_output" in message
    assert 'recall="tr.1"' in message
    assert "-> FILE STATE" not in message

def test_batched_read_projects_every_file_inside_one_output_budget(tmp_path):
    """The output budget covers the whole result, not each source block. A batched Read of several
    large files must not emit one full budget per file: source-bearing output skips the generic
    character bounding, so nothing downstream would catch the overflow."""
    big = "".join(f"line {index} of some moderately long content to burn tokens\n" for index in range(4000))
    for index in range(4):
        (tmp_path / f"f{index}.txt").write_text(big, encoding="utf-8")
    s = session(tmp_path)
    context = ContextManager(s)
    runner = ToolRunner(s, context, output_fn=lambda text: None)
    call_obj = call("Read", [{"path": f"f{index}.txt", "ranges": [[1, 0]]} for index in range(4)])

    message = runner.finish(call_obj, ReadTool(s, call_obj.args).call())

    assert message.count("<Read ") == 4  # every file still gets a view
    assert message.count("<bounded_output") == 4  # and every one of them says what it dropped
    assert context.estimated_text_tokens(message) < MAX_TOOL_OUTPUT_TOKENS * 1.2


def test_bounded_read_cannot_authorize_its_omitted_middle(tmp_path, monkeypatch):
    """Only projected lines become part of the view. A line number guessed from the omitted middle
    is refused as unseen rather than resolved against whatever now occupies that position."""
    path = tmp_path / "large.txt"
    path.write_text("".join(f"line-{index}\n" for index in range(20000)), encoding="utf-8")
    s = session(tmp_path)
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)
    call_obj = call("Read", [{"path": "large.txt", "ranges": [[1, 0]]}])

    message = runner.finish(call_obj, ReadTool(s, call_obj.args).call())
    view = s.get_source_view("view.1")

    assert "<bounded_output" in message
    assert len(view.spans) == 2  # a visible head and a visible tail, with nothing in between
    head, tail = view.spans
    assert head.end + 1 < tail.start
    guessed = head.end + 1  # a real line of the file, but one the model was never shown
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    runner.run_sync([ToolCall("edit", "Edit", ["large.txt", "view.1", [{"op": "replace", "start": guessed, "end": guessed, "content": "x\n"}]])])

    assert s.tool_errors and "source range unseen" in s.tool_errors[0].error
    assert path.read_text(encoding="utf-8") == "".join(f"line-{index}\n" for index in range(20000))


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

def _block(name, count, start=1):
    lines = tuple(f"{name} line {index}\n" for index in range(count))
    draft = SourceViewDraft(f"/w/{name}.py", f"{name}.py", start + count - 1, (SourceSpan(start, lines),), READ)
    return SourceBlock.plain(draft)


def test_projection_spends_one_budget_across_blocks_and_keeps_small_literal_parts(tmp_path):
    """Blocks share the result's budget, and a block that comes in under its share returns the
    rest: a small file beside a huge one is kept whole rather than clipped to an equal slice.
    Small literal wrappers stay whole; a large diff is bounded separately from the source blocks
    whose edit authority must remain line-granular."""
    small, large = _block("small", 3), _block("large", 400)
    output = ToolOutput.rendered(["<Search matches=2>", small, large, "</Search>"])
    budget = estimate(small.render()) + estimate(large.render()) // 4

    projected = output.project(max_tokens=budget, estimate=estimate)
    kept, clipped = projected.parts[1], projected.parts[2]

    assert [part for part in projected.parts if isinstance(part, str)] == ["<Search matches=2>", "</Search>"]
    assert kept == small and not kept.bounded  # under its share, so it survives intact
    assert clipped.bounded and clipped.draft.total_lines == large.draft.total_lines
    assert estimate(projected.render([None, None])) <= budget * 1.1
    assert len(clipped.draft.spans) == 2  # a visible head and a visible tail, middle dropped


def test_projection_keeps_evidence_when_literal_parts_alone_fill_the_budget(tmp_path):
    """A pathological result whose prose already exceeds the budget still returns some source:
    dropping the block entirely would leave the model an id it cannot see any lines behind."""
    block = _block("code", 200)
    output = ToolOutput.rendered(["x" * 4000, block])

    projected = output.project(max_tokens=100, estimate=estimate)

    assert isinstance(projected.parts[0], TextBlock)
    assert "<bounded_output" in projected.parts[0].render()
    assert projected.parts[1].bounded
    assert projected.parts[1].draft.line_count >= 1


def test_large_edit_diff_and_source_share_the_normal_output_budget(tmp_path, monkeypatch):
    """A source-bearing Edit must not bypass output bounding through its ordinary diff text."""
    from wizolt.tools import EditTool

    path = tmp_path / "large.py"
    path.write_text("old\n", encoding="utf-8")
    s = session(tmp_path)
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)
    read = ReadTool(s, [{"path": "large.py"}]).call()
    source = s.register_source_drafts(list(read.drafts))[0]
    body = "".join(f"line_{index} = {index}\n" for index in range(12000))
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")

    result = EditTool(s, ["large.py", source, [{"op": "replace", "start": 1, "end": 1, "content": body}]]).call()
    message = runner.finish(call("Edit", ["large.py", source, []]), result)

    assert message.count("<bounded_output") >= 2  # the diff and the large fresh source block
    assert 'recall="tr.1"' in message
    assert runner.context.estimated_text_tokens(message) < MAX_TOOL_OUTPUT_TOKENS * 1.2
    assert path.read_text(encoding="utf-8") == body


def test_projection_of_an_output_with_no_source_is_returned_unchanged(tmp_path):
    output = ToolOutput.of("plain bash output")

    assert output.project(max_tokens=1, estimate=estimate) is output
