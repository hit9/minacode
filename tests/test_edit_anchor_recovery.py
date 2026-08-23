"""edit anchor recovery (split from tests/test_edit_tool.py)."""
import pytest
from test_edit_tool import anchor, session

from minacode.base import ToolCall, ToolError
from minacode.context import ContextManager
from minacode.runner import ToolRunner
from minacode.tools import CodeIndex, EditTool, ReadTool


def test_single_anchor_stale_guides_content_check(tmp_path):
    path = tmp_path / "note.txt"
    path.write_text("old\n", encoding="utf-8")

    with pytest.raises(ToolError) as error:
        EditTool(session(tmp_path), ["note.txt", [{"op": "insert_before", "start": anchor(0, "wrong\n"), "content": "x\n"}]]).call()

    message = str(error.value)
    assert "stale anchor" in message
    assert "retry with a returned anchor only if its content is the line you meant" in message
    assert "prefer replace_unique" in message
    assert "<current-file-context hashline-numbered>" in message
    assert "not an inferred target" in message
    assert "anchor=1:" + ReadTool.line_hash("old\n") + " | old" in message

def test_range_stale_anchor_error_does_not_guess_current_range(tmp_path):
    path = tmp_path / "note.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")

    with pytest.raises(ToolError) as error:
        EditTool(session(tmp_path), ["note.txt", [{"op": "replace", "start": anchor(0, "wrong\n"), "end": anchor(2, "c\n"), "content": "x\n"}]]).call()

    message = str(error.value)
    assert "stale anchor" in message and "retry with a returned anchor" in message
    assert "<current-target-ranges hashline-numbered>" not in message
    assert "<current-file-context hashline-numbered>" in message
    assert path.read_text(encoding="utf-8") == "a\nb\nc\n"

def test_anchor_out_of_range_reports_file_length(tmp_path):
    path = tmp_path / "note.txt"
    path.write_text("a\nb\n", encoding="utf-8")

    with pytest.raises(ToolError, match="anchor line 10 out of range; file has 2 lines") as error:
        EditTool(session(tmp_path), ["note.txt", [{"op": "replace", "start": "10:abcde", "end": "10:abcde", "content": "x\n"}]]).call()

    assert "<current-target-ranges" not in str(error.value)
    assert "<current-file-context hashline-numbered>" in str(error.value)
    assert "anchor=1:" + ReadTool.line_hash("a\n") + " | a" in str(error.value)
    assert "anchor=2:" + ReadTool.line_hash("b\n") + " | b" in str(error.value)
    assert path.read_text(encoding="utf-8") == "a\nb\n"

def test_anchor_out_of_range_empty_file_returns_bounded_factual_context(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ToolError) as error:
        EditTool(
            session(tmp_path),
            ["empty.txt", [{"op": "insert_before", "start": "1:abcde", "content": "x\n"}]],
        ).call()

    message = str(error.value)
    assert "file has 0 lines" in message
    assert "not an inferred target" in message
    assert "(empty file)" in message
    assert path.read_text(encoding="utf-8") == ""

def test_stale_anchor_error_display_is_oneline_but_tool_result_keeps_guidance(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    path = tmp_path / "code.txt"
    path.write_text("a\nb\nc\nd\n", encoding="utf-8")
    out = []
    runner = ToolRunner(s, ContextManager(s), output_fn=out.append)

    runner.run([ToolCall("bad", "Edit", ["code.txt", [{"op": "replace", "start": anchor(0, "wrong\n"), "end": anchor(2, "wrong\n"), "content": "x\n"}]])])

    # Terminal side: the reject display collapses to one truncated line (no multi-line blowout).
    assert len(out) == 1
    items = list(out[0].walk())
    assert len(items) == 1
    rendered_line = items[0][0]
    assert "\n" not in rendered_line.text
    assert "..." in rendered_line.text
    # Model side: the full retry guidance is preserved without an untrusted guessed range.
    assert len(s.tool_errors) == 1
    message = s.tool_errors[0].error
    assert "retry with a returned anchor" in message
    assert "<current-target-ranges hashline-numbered>" not in message
    assert "<current-file-context hashline-numbered>" in message

def test_batch_stale_range_does_not_guess_after_prior_shift(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    path = tmp_path / "code.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    runner.run(
        [
            ToolCall(
                "first",
                "Edit",
                [
                    "code.txt",
                    [
                        {"op": "insert_before", "start": anchor(0, "a\n"), "content": "x\n"},
                        {"op": "replace", "start": anchor(1, "b\n"), "end": anchor(1, "b\n"), "content": "B\n"},
                    ],
                ],
            ),
            ToolCall(
                "second",
                "Edit",
                ["code.txt", [{"op": "replace", "start": anchor(1, "b\n"), "end": anchor(2, "c\n"), "content": "Y\n"}]],
            ),
        ]
    )

    assert path.read_text(encoding="utf-8") == "x\na\nB\nc\n"
    assert len(s.tool_errors) == 1
    assert "original line was changed in this batch" in s.tool_errors[0].error
    assert "<current-target-ranges" not in s.tool_errors[0].error
    assert "<current-file-context hashline-numbered>" in s.tool_errors[0].error

def test_stale_anchor_context_is_bounded_around_requested_line(tmp_path):
    path = tmp_path / "note.txt"
    path.write_text("".join(f"line{i}\n" for i in range(20)), encoding="utf-8")

    with pytest.raises(ToolError) as error:
        EditTool(
            session(tmp_path),
            ["note.txt", [{"op": "replace", "start": anchor(10, "wrong\n"), "end": anchor(10, "wrong\n"), "content": "x\n"}]],
        ).call()

    context = str(error.value).split("<current-file-context hashline-numbered>", 1)[1].split("</current-file-context>", 1)[0]
    anchor_rows = [line for line in context.splitlines() if line.startswith("anchor=")]
    assert len(anchor_rows) == 7
    assert "| line7" in anchor_rows[0]
    assert "| line13" in anchor_rows[-1]

def test_edit_refunds_neighborhood_anchors_around_change(tmp_path):
    path = tmp_path / "code.txt"
    path.write_text("".join(f"l{i}\n" for i in range(10)), encoding="utf-8")

    result = EditTool(session(tmp_path), ["code.txt", [{"op": "replace", "start": anchor(4, "l4\n"), "end": anchor(4, "l4\n"), "content": "L4\n"}]]).call()

    # Change on line 5 (1-based); the window adds three lines of context on each side: 2..8.
    assert "<invalidate>5:5</invalidate>" in result
    assert "anchor=2:" + ReadTool.line_hash("l1\n") + " | l1" in result
    assert "anchor=5:" + ReadTool.line_hash("L4\n") + " | L4" in result
    assert "anchor=8:" + ReadTool.line_hash("l7\n") + " | l7" in result
    assert "anchor=1:" + ReadTool.line_hash("l0\n") + " | l0" not in result
    assert "anchor=9:" + ReadTool.line_hash("l8\n") + " | l8" not in result

def test_edit_neighborhood_clamps_to_file_bounds(tmp_path):
    path = tmp_path / "code.txt"
    path.write_text("a\nb\nc\nd\n", encoding="utf-8")

    first = EditTool(session(tmp_path), ["code.txt", [{"op": "replace", "start": anchor(0, "a\n"), "end": anchor(0, "a\n"), "content": "A\n"}]]).call()
    assert "anchor=1:" + ReadTool.line_hash("A\n") + " | A" in first
    assert "anchor=4:" + ReadTool.line_hash("d\n") + " | d" in first

    last = EditTool(session(tmp_path), ["code.txt", [{"op": "replace", "start": anchor(3, "d\n"), "end": anchor(3, "d\n"), "content": "D\n"}]]).call()
    assert "anchor=1:" + ReadTool.line_hash("A\n") + " | A" in last
    assert "anchor=4:" + ReadTool.line_hash("D\n") + " | D" in last

def test_edit_anchor_description_notes_bash_view_carries_no_anchors():
    schema = EditTool.params_schema()
    edit = schema["properties"]["edits"]["items"]
    assert "replace_unique" in edit["properties"]["op"]["description"]
    for key in ("start", "end"):
        assert "a file viewed through Bash carries no anchors" in edit["properties"][key]["description"]
        assert "verifying its content, otherwise Read again" in edit["properties"][key]["description"]
    content_description = edit["properties"]["content"]["description"]
    assert "replace_all/replace_unique" in content_description
    assert "explicit empty string deletes the match" in content_description
    assert "new" not in edit["properties"]
    assert "lines before start and after end are preserved automatically" in content_description
    assert "do not copy it merely to keep it" in content_description
    assert "may independently equal neighboring text" in content_description
    assert "correspond exactly to the start/end anchor lines" not in content_description
    assert "replace_unique replaces text that occurs exactly once" in EditTool.DESCRIPTION
    assert "preserves it automatically" in EditTool.DESCRIPTION
    assert "Prefer replace_unique for a small edit" in EditTool.DESCRIPTION
    assert any("without copying the anchor as context" in example for example in EditTool.EXAMPLE)
    assert any("replace one exact block without anchors" in example for example in EditTool.EXAMPLE)
