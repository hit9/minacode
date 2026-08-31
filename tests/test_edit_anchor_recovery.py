"""edit failure recovery (split from tests/test_edit_tool.py)."""

import pytest
from test_edit_tool import session, view

from wizolt.base import ToolCall, ToolError
from wizolt.context import ContextManager
from wizolt.runner import ToolRunner
from wizolt.source import ToolOutput, render_tool_output
from wizolt.tools import CodeIndex, EditTool


def rendered(out, s):
    assert isinstance(out, ToolOutput)
    return render_tool_output(out, s.register_source_drafts(list(out.drafts)))


def test_stale_target_error_guides_fresh_view(tmp_path):
    s = session(tmp_path)
    path = tmp_path / "note.txt"
    path.write_text("old\n", encoding="utf-8")
    key = view(s, "note.txt")
    EditTool(s, ["note.txt", key, [{"op": "replace", "start": 1, "end": 1, "content": "changed\n"}]]).call()

    with pytest.raises(ToolError) as error:
        EditTool(s, ["note.txt", key, [{"op": "insert_before", "line": 1, "content": "x\n"}]]).call()

    message = str(error.value)
    assert "cannot relocate" in message
    assert "use the fresh view below or Read again" in message
    assert path.read_text(encoding="utf-8") == "changed\n"


def test_out_of_range_line_reports_view_bounds(tmp_path):
    s = session(tmp_path)
    path = tmp_path / "note.txt"
    path.write_text("a\nb\n", encoding="utf-8")
    key = view(s, "note.txt")

    with pytest.raises(ToolError, match="lines 10:10 are outside view"):
        EditTool(s, ["note.txt", key, [{"op": "replace", "start": 10, "end": 10, "content": "x\n"}]]).call()

    assert path.read_text(encoding="utf-8") == "a\nb\n"


def test_empty_file_insertion_nonzero_rejected(tmp_path):
    s = session(tmp_path)
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")
    key = view(s, "empty.txt")

    with pytest.raises(ToolError, match="only insert_after with line 0 is valid"):
        EditTool(s, ["empty.txt", key, [{"op": "insert_before", "line": 1, "content": "x\n"}]]).call()

    assert path.read_text(encoding="utf-8") == ""


def test_failure_recovery_is_bounded_fresh_view(tmp_path):
    s = session(tmp_path)
    path = tmp_path / "note.txt"
    path.write_text("".join(f"line{i}\n" for i in range(20)), encoding="utf-8")
    key = view(s, "note.txt")
    EditTool(s, ["note.txt", key, [{"op": "replace", "start": 11, "end": 11, "content": "CHANGED\n"}]]).call()

    with pytest.raises(ToolError) as error:
        EditTool(s, ["note.txt", key, [{"op": "replace", "start": 11, "end": 11, "content": "x\n"}]]).call()

    recovery = error.value.recovery
    assert isinstance(recovery, ToolOutput)
    text = rendered(recovery, s)
    assert "<source" in text
    rows = [line for line in text.splitlines() if " | " in line]
    assert 7 >= len(rows) >= 5  # at most seven current lines around the requested coordinate


def test_edit_description_does_not_mention_anchor():
    assert "source view" in EditTool.DESCRIPTION
    assert "anchor" not in EditTool.DESCRIPTION


def test_success_fresh_block_is_immediately_editable(tmp_path):
    s = session(tmp_path)
    path = tmp_path / "code.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")
    key = view(s, "code.txt")

    first = EditTool(s, ["code.txt", key, [{"op": "replace", "start": 2, "end": 2, "content": "B\n"}]]).call()
    fresh_key = s.register_source_drafts(list(first.drafts))[0]
    EditTool(s, ["code.txt", fresh_key, [{"op": "insert_after", "line": 2, "content": "x\n"}]]).call()

    assert path.read_text(encoding="utf-8") == "a\nB\nx\nc\n"


def test_success_fresh_block_clamps_to_file_bounds(tmp_path):
    s = session(tmp_path)
    path = tmp_path / "code.txt"
    path.write_text("a\nb\nc\nd\n", encoding="utf-8")
    key = view(s, "code.txt")

    first = EditTool(s, ["code.txt", key, [{"op": "replace", "start": 1, "end": 1, "content": "A\n"}]]).call()
    text = rendered(first, s)

    assert "1 | A" in text
    assert "4 | d" in text
    assert "5 |" not in text


def test_failed_edit_error_text_keeps_fresh_view_for_the_model(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    path = tmp_path / "code.txt"
    path.write_text("a\nb\nc\nd\n", encoding="utf-8")
    key = view(s, "code.txt")
    EditTool(s, ["code.txt", key, [{"op": "replace", "start": 1, "end": 1, "content": "A\n"}]]).call()
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    runner.run([ToolCall("bad", "Edit", ["code.txt", key, [{"op": "replace", "start": 1, "end": 1, "content": "x\n"}]])])

    assert len(s.tool_errors) == 1
    assert "source target changed" in s.tool_errors[0].error
    assert "Read again" in s.tool_errors[0].error


def test_batch_stale_range_does_not_guess_after_prior_shift(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    path = tmp_path / "code.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")
    key = view(s, "code.txt")
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    runner.run(
        [
            ToolCall(
                "first",
                "Edit",
                [
                    "code.txt",
                    key,
                    [
                        {"op": "insert_before", "line": 1, "content": "x\n"},
                        {"op": "replace", "start": 2, "end": 2, "content": "B\n"},
                    ],
                ],
            ),
            ToolCall(
                "second",
                "Edit",
                ["code.txt", key, [{"op": "replace", "start": 2, "end": 3, "content": "Y\n"}]],
            ),
        ]
    )

    assert path.read_text(encoding="utf-8") == "x\na\nB\nc\n"
    assert len(s.tool_errors) == 1
    assert "were replaced or deleted by an earlier edit in this batch" in s.tool_errors[0].error


def test_deletion_fresh_view_shows_the_seam_it_left(tmp_path, monkeypatch):
    # A deletion leaves no changed line to report, so a hunk-only view would come back empty and
    # force a Read just to keep working on the file the edit only just changed. The fresh view
    # covers the seam instead, and is a real view: the next edit can name it.
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    path = tmp_path / "code.txt"
    path.write_text("a\nb\nc\nd\n", encoding="utf-8")
    key = view(s, "code.txt")
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    runner.run([ToolCall("cut", "Edit", ["code.txt", key, [{"op": "delete", "start": 2, "end": 3}]])])

    fresh = s.get_source_view("view.2")
    assert fresh.total_lines == 2
    assert fresh.spans and [line for span in fresh.spans for line in span.lines] == ["a\n", "d\n"]

    runner.run([ToolCall("next", "Edit", ["code.txt", "view.2", [{"op": "replace", "start": 2, "end": 2, "content": "D\n"}]])])
    assert path.read_text(encoding="utf-8") == "a\nD\n"
    assert s.tool_errors == []


def test_deleting_the_whole_file_leaves_an_empty_file_view(tmp_path, monkeypatch):
    # The one legal shape for a view with no spans: an empty file, whose view carries its single
    # legal insertion boundary at line 0.
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    (tmp_path / "code.txt").write_text("a\nb\n", encoding="utf-8")
    key = view(s, "code.txt")
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    runner.run([ToolCall("cut", "Edit", ["code.txt", key, [{"op": "delete", "start": 1, "end": 2}]])])

    fresh = s.get_source_view("view.2")
    assert (fresh.total_lines, fresh.spans) == (0, ())
    runner.run([ToolCall("fill", "Edit", ["code.txt", "view.2", [{"op": "insert_after", "line": 0, "content": "new\n"}]])])
    assert (tmp_path / "code.txt").read_text(encoding="utf-8") == "new\n"
    assert s.tool_errors == []


def test_empty_file_insertion_rejects_once_another_writer_filled_it(tmp_path, monkeypatch):
    # An empty-file view carries one legal boundary and no content to validate, so it is the one
    # target that could be applied blind. It stays valid only while the file is still empty.
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")
    key = view(s, "empty.txt")
    path.write_text("written elsewhere\n", encoding="utf-8")
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    runner.run([ToolCall("fill", "Edit", ["empty.txt", key, [{"op": "insert_after", "line": 0, "content": "mine\n"}]])])

    assert path.read_text(encoding="utf-8") == "written elsewhere\n"
    assert s.tool_errors and "the empty file now has content" in s.tool_errors[0].error
