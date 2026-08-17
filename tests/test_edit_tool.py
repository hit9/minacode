import os
import re
import shutil

import pytest
from prompt_toolkit.utils import get_cwidth

from minacode.base import LogBlock, LogEdge, LogLine, LogRole, ToolCall, ToolError
from minacode.context import ContextManager
from minacode.model import ModelClient
from minacode.render import UiPrinter
from minacode.runner import EditBatchPlan, ToolRunner
from minacode.session import Session
from minacode.tools import CodeIndex, EditTool, ReadTool
from minacode.tools.files import Edit


def session(tmp_path):
    return Session(cwd=str(tmp_path))


def anchor(index, line):
    """Anchor for a 0-based line index, rendered the way the model sees it (1-based)."""
    return ReadTool.anchor(index, line)


def test_approval_segments_highlight_inline_edit_preview():
    preview = "--- foo.py\n+++ foo.py\n@@ -1,2 +1,2 @@\n def hello():\n-    pass\n+    return 42"
    block = LogBlock.hierarchy(
        LogLine("Edit", "foo.py", LogRole.TOOL),
        [
            LogLine("preview", role=LogRole.META, edge=LogEdge.BRANCH),
            *(LogLine("", line, LogRole.DIFF, LogEdge.CONTINUE) for line in preview.splitlines()),
        ],
    )
    segments = UiPrinter().log_segments(block)
    rendered = "".join(text for _style, text in segments)

    assert ("ansigreen", "Edit") in segments
    assert any(style == "fg:#ff7b72 bg:#003b00" and "return" in text for style, text in segments)
    assert any(style == "ansigreen bg:#003b00" and text == "+" for style, text in segments)
    assert any(style == "fg:default bg:#520000" and "pass" in text for style, text in segments)
    assert "\n\n" not in rendered


def test_auto_approved_edit_keeps_preview_pre_line(tmp_path, monkeypatch):
    # Edit's "auto …" pre-line carries the approval preview; the result line is tagged [auto].
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    (tmp_path / "a.txt").write_text("hello\nworld\n", encoding="utf-8")
    out = []
    runner = ToolRunner(s, ContextManager(s), output_fn=out.append)
    runner.run([ToolCall("e0", "Edit", ["a.txt", [{"op": "insert_after", "start": anchor(0, "hello\n"), "content": "NEW\n"}]])])
    assert len(out) == 2
    assert isinstance(out[0], LogBlock)
    root, _level = next(out[0].walk())
    assert root.role is LogRole.AUTO
    assert "preview" in str(out[0])
    assert str(out[1]).rstrip().endswith("[auto]")


def test_batch_edit_no_change_reports_current_target_range(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    path = tmp_path / "code.txt"
    path.write_text("a\nb\n", encoding="utf-8")
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    runner.run([ToolCall("noop", "Edit", ["code.txt", [{"op": "replace", "start": anchor(1, "b\n"), "end": anchor(1, "b\n"), "content": "b\n"}]])])

    assert s.tool_errors
    message = s.tool_errors[0].error
    assert "edit produced no changes; requested content already matches target range" in message
    assert "anchor=2:" + ReadTool.line_hash("b\n") + " | b" in message
    assert path.read_text(encoding="utf-8") == "a\nb\n"


def test_batch_edit_stale_anchor_reports_current_line(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    path = tmp_path / "code.txt"
    path.write_text("a\nb\n", encoding="utf-8")
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    runner.run([ToolCall("bad", "Edit", ["code.txt", [{"op": "replace", "start": anchor(1, "wrong\n"), "end": anchor(1, "wrong\n"), "content": "B\n"}]])])

    assert s.tool_errors
    assert "current is anchor=2:" + ReadTool.line_hash("b\n") + " | b" in s.tool_errors[0].error
    assert path.read_text(encoding="utf-8") == "a\nb\n"


def test_code_index_updates_after_file_mutation_tools(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    updated = []
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: updated.extend(paths) or "")
    runner = ToolRunner(s, ContextManager(s), input_fn=lambda prompt: (_ for _ in ()).throw(AssertionError("unexpected prompt")), output_fn=lambda text: None)

    runner.run([ToolCall("empty", "Edit", ["empty.py", [{"op": "create", "content": ""}]])])
    runner.run([ToolCall("create", "Edit", ["made.py", [{"op": "create", "content": "print(1)\n"}]])])
    runner.run([ToolCall("edit", "Edit", ["made.py", [{"op": "replace_all", "old": "1", "new": "2"}]])])

    assert (tmp_path / "made.py").read_text(encoding="utf-8") == "print(2)\n"
    assert updated == ["empty.py", "made.py", "made.py"]


def test_diff_segments_gracefully_degrades_without_header_path(tmp_path):
    ui = UiPrinter()
    # No +++ line, so pygments cannot pick a lexer.
    diff = "@@ -1,1 +1,1 @@\n- old\n+ new\n"
    segments = ui.diff_segments(diff)

    assert any(t == "-" and s == "ansired bg:#520000" for s, t in segments)
    assert any(t == "+" and s == "ansigreen bg:#003b00" for s, t in segments)


def test_diff_segments_gracefully_degrades_without_lexer(tmp_path):
    ui = UiPrinter()
    diff = "--- foo.unknownxyz\n+++ foo.unknownxyz\n@@ -1,1 +1,1 @@\n- old\n+ new\n"
    segments = ui.diff_segments(diff)

    assert any(t == "-" and s == "ansired bg:#520000" for s, t in segments)
    assert any("old" in t and s == "fg:default bg:#520000" for s, t in segments)
    assert any(t == "+" and s == "ansigreen bg:#003b00" for s, t in segments)


def test_diff_segments_syntax_highlights_python(tmp_path):
    ui = UiPrinter()
    diff = "--- foo.py\n+++ foo.py\n@@ -1,2 +1,2 @@\n def hello():\n-    pass\n+    return 42\n"
    segments = ui.diff_segments(diff)

    assert any(t == "+" and s == "ansigreen bg:#003b00" for s, t in segments)
    assert any(t == "return" and s == "fg:#ff7b72 bg:#003b00" for s, t in segments)

    assert any(t == "-" and s == "ansired bg:#520000" for s, t in segments)
    assert any("pass" in t and s == "fg:default bg:#520000" for s, t in segments)

    # Changed-line gutters join the background band; context stays unfilled.
    assert any("|" in text and style == "ansibrightblack bg:#003b00" for style, text in segments)
    assert any("|" in text and style == "ansibrightblack bg:#520000" for style, text in segments)
    assert any("1" in text and "|" in text and "bg:" not in style for style, text in segments)
    assert any(text == "def" and "bg:" not in style for style, text in segments)

    live = ui.segment_lines(ui.diff_segments_live(diff, row_width=40))
    changed = [line for line in live if any("bg:" in style for style, _text in line)]
    widths = [sum(get_cwidth(text.rstrip("\n")) for _style, text in line) for line in changed]
    assert set(widths) == {40}


def test_approval_diff_background_fills_every_wrapped_row(monkeypatch):
    preview = "--- foo.py\n+++ foo.py\n@@ -1,3 +1,3 @@\n-short\n+a\n+this is a much longer changed line that forces wrapping across several terminal rows"
    block = LogBlock.hierarchy(
        LogLine("Edit", "foo.py", LogRole.TOOL),
        [
            LogLine("preview", role=LogRole.META, edge=LogEdge.BRANCH),
            *(LogLine("", line, LogRole.DIFF, LogEdge.CONTINUE) for line in preview.splitlines()),
        ],
    )

    with monkeypatch.context() as patch:
        patch.setattr(shutil, "get_terminal_size", lambda fallback=(80, 24): os.terminal_size((50, 24)))
        lines = UiPrinter.segment_lines(UiPrinter().log_segments(block))

    spans = []
    for line in lines:
        column = 0
        background_columns = []
        for style, text in line:
            width = get_cwidth(text.rstrip("\n"))
            if "bg:" in style:
                background_columns.extend(range(column, column + width))
            column += width
        if background_columns:
            spans.append((min(background_columns), max(background_columns) + 1))

    expected_start = get_cwidth(LogBlock.prefix(2, LogEdge.CONTINUE))
    assert len(spans) >= 5
    assert set(spans) == {(expected_start, 49)}


def test_edit_accepts_inspect_code_anchor(tmp_path):
    s = session(tmp_path)
    path = tmp_path / "note.txt"
    path.write_text("old\n", encoding="utf-8")
    inspect_anchor = "anchor=1:" + ReadTool.indexed_line_hash("old\n")

    result = EditTool(s, ["note.txt", [{"op": "replace", "start": inspect_anchor, "end": inspect_anchor, "content": "new\n"}]]).call()

    assert '<Edit path="note.txt">' in result
    assert path.read_text(encoding="utf-8") == "new\n"


def test_edit_anchor_consistent_with_read_on_exotic_line_boundary(tmp_path):
    # Regression: Edit split lines with str.splitlines(True) while Read uses readlines, so a file
    # containing a form-feed numbered lines differently and a valid Read anchor went stale in Edit.
    s = session(tmp_path)
    path = tmp_path / "ff.txt"
    path.write_text("a\nb\x0cc\nd\n", encoding="utf-8")  # form-feed inside the middle line
    read = ReadTool(s, [{"path": "ff.txt"}]).call()
    assert f"anchor=3:{ReadTool.line_hash('d')} | d" in read  # Read numbers "d" as line 3
    EditTool(s, ["ff.txt", [{"op": "replace", "start": anchor(2, "d\n"), "end": anchor(2, "d\n"), "content": "D\n"}]]).call()
    assert path.read_text(encoding="utf-8") == "a\nb\x0cc\nD\n"


def test_edit_anchor_survives_trailing_newline_change(tmp_path):
    # Regression: line_hash used to fold the trailing newline into the hash, so an anchor captured
    # for a last line without a newline went stale once an edit gave the file a trailing newline,
    # even though the line's visible text never changed.
    s = session(tmp_path)
    path = tmp_path / "note.txt"
    path.write_text("a\nb\n", encoding="utf-8")
    # Anchor built from the newline-less form of the line (as captured when "b" was the last line
    # before a trailing newline was added). It must still resolve against the current "b\n".
    anc = anchor(1, "b")
    EditTool(s, ["note.txt", [{"op": "replace", "start": anc, "end": anc, "content": "B\n"}]]).call()
    assert path.read_text(encoding="utf-8") == "a\nB\n"


def test_edit_preserves_literal_escape_sequences_in_content_and_new(tmp_path):
    s = session(tmp_path)
    literal_line = r'pattern = "\n\t"'
    tool = EditTool(s, ["script.py", [{"op": "create", "content": literal_line}]])

    preview = tool.preview()
    tool.call()

    assert literal_line in preview
    assert (tmp_path / "script.py").read_text(encoding="utf-8") == literal_line

    path = tmp_path / "replace.py"
    path.write_text("old\nnext\n", encoding="utf-8")
    EditTool(s, ["replace.py", [{"op": "replace", "start": anchor(0, "old\n"), "end": anchor(0, "old\n"), "content": literal_line}]]).call()
    assert path.read_text(encoding="utf-8") == literal_line + "\nnext\n"

    path = tmp_path / "unique.py"
    path.write_text("value = OLD\n", encoding="utf-8")
    EditTool(s, ["unique.py", [{"op": "replace_unique", "old": "OLD", "new": r'"\n"'}]]).call()
    assert path.read_text(encoding="utf-8") == 'value = "\\n"\n'


def test_edit_accepts_redundant_matching_path_in_model_operation(tmp_path):
    payload = {
        "path": "script.py",
        "edits": [{"op": "create", "content": "print(1)\n", "path": "script.py"}],
    }

    call = ModelClient.tool_call("edit", "Edit", payload)
    EditTool(session(tmp_path), call.args).call()

    assert call.error == ""
    assert (tmp_path / "script.py").read_text(encoding="utf-8") == "print(1)\n"
    assert payload["edits"][0]["path"] == "script.py"


def test_edit_rejects_different_nested_path_in_model_operation(tmp_path):
    call = ModelClient.tool_call(
        "edit",
        "Edit",
        {
            "path": "script.py",
            "edits": [{"op": "create", "content": "print(1)\n", "path": "other.py"}],
        },
    )

    with pytest.raises(ToolError, match="Edit unexpected field: path"):
        EditTool(session(tmp_path), call.args).call()

    assert not (tmp_path / "script.py").exists()
    assert not (tmp_path / "other.py").exists()


def test_edit_creates_and_patches_file(tmp_path):
    s = session(tmp_path)
    EditTool(s, ["empty/keep.txt", [{"op": "create", "content": ""}]]).call()
    assert (tmp_path / "empty" / "keep.txt").read_text(encoding="utf-8") == ""
    with pytest.raises(ToolError):
        EditTool(s, ["empty/keep.txt", [{"op": "create", "content": ""}]]).call()
    EditTool(s, ["empty/keep.txt", [{"op": "replace_all", "old": "", "new": "kept\n"}]]).call()
    assert (tmp_path / "empty" / "keep.txt").read_text(encoding="utf-8") == "kept\n"

    EditTool(s, ["nested/note.txt", [{"op": "create", "content": "one\ntwo\nthree\n"}]]).call()
    path = tmp_path / "nested" / "note.txt"
    assert path.read_text(encoding="utf-8") == "one\ntwo\nthree\n"

    with pytest.raises(ToolError):
        EditTool(s, ["missing.txt", [{"op": "replace_all", "old": "", "new": "again\n"}]]).call()

    EditTool(
        s,
        [
            "nested/note.txt",
            [
                {"op": "replace", "start": anchor(0, "one\n"), "end": anchor(0, "one\n"), "content": "ONE\n"},
                {"op": "insert_after", "start": anchor(1, "two\n"), "content": "TWO-AND-HALF\n"},
                {"op": "delete", "start": anchor(2, "three\n"), "end": anchor(2, "three\n")},
            ],
        ],
    ).call()
    assert path.read_text(encoding="utf-8") == "ONE\ntwo\nTWO-AND-HALF\n"

    EditTool(s, ["nested/note.txt", [{"op": "replace_all", "old": "TWO", "new": "two"}]]).call()
    assert path.read_text(encoding="utf-8") == "ONE\ntwo\ntwo-AND-HALF\n"

    with pytest.raises(ToolError):
        EditTool(s, ["nested/note.txt", [{"op": "replace_all", "old": "", "new": "bad\n"}]]).call()
    with pytest.raises(ToolError):
        EditTool(s, ["nested/note.txt", [{"op": "replace", "start": anchor(0, "one\n"), "end": anchor(0, "one\n"), "content": "bad\n"}]]).call()


def test_edit_index_update_uses_call_path_when_output_path_is_unparseable(tmp_path, monkeypatch):
    s = session(tmp_path)
    updated = []
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: updated.extend(paths) or "")

    ToolRunner(s, ContextManager(s), output_fn=lambda text: None).update_code_index(
        ToolCall("edit", "Edit", ["made.py", [{"op": "create", "content": "x\n"}]]),
        "<Edit path=bad />",
    )

    assert updated == ["made.py"]


def test_edit_inserts_before_existing_line_with_needed_newline(tmp_path):
    s = session(tmp_path)
    path = tmp_path / "code.txt"
    path.write_text("a\nb\n", encoding="utf-8")

    EditTool(s, ["code.txt", [{"op": "insert_before", "start": anchor(1, "b\n"), "content": "inserted"}]]).call()
    assert path.read_text(encoding="utf-8") == "a\ninserted\nb\n"


def test_edit_no_change_replace_all_reports_identical_file(tmp_path):
    s = session(tmp_path)
    path = tmp_path / "note.txt"
    path.write_text("old\n", encoding="utf-8")

    with pytest.raises(ToolError) as error:
        EditTool(s, ["note.txt", [{"op": "replace_all", "old": "old", "new": "old"}]]).call()

    assert str(error.value) == "edit produced no changes; replace_all result is identical to current file"


def test_edit_no_change_reports_current_target_range(tmp_path):
    s = session(tmp_path)
    path = tmp_path / "note.txt"
    path.write_text("old\n", encoding="utf-8")

    with pytest.raises(ToolError) as error:
        EditTool(s, ["note.txt", [{"op": "replace", "start": anchor(0, "old\n"), "end": anchor(0, "old\n"), "content": "old\n"}]]).call()

    message = str(error.value)
    assert "edit produced no changes; requested content already matches target range" in message
    assert "<current-target-ranges hashline-numbered>" in message
    assert "anchor=1:" + ReadTool.line_hash("old\n") + " | old" in message


def test_edit_rejects_directory_target(tmp_path):
    s = session(tmp_path)
    (tmp_path / "pkg").mkdir()

    with pytest.raises(ToolError, match="path is a directory"):
        EditTool(s, ["pkg", [{"op": "replace_all", "old": "", "new": "x\n"}]]).call()


def test_edit_rejects_overlaps_and_mixed_modes(tmp_path):
    s = session(tmp_path)
    path = tmp_path / "code.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")

    with pytest.raises(ToolError):
        EditTool(
            s,
            [
                "code.txt",
                [
                    {"op": "replace", "start": anchor(0, "a\n"), "end": anchor(1, "b\n"), "content": "x\n"},
                    {"op": "delete", "start": anchor(1, "b\n"), "end": anchor(1, "b\n")},
                ],
            ],
        ).call()
    assert path.read_text(encoding="utf-8") == "a\nb\nc\n"

    with pytest.raises(ToolError):
        EditTool(
            s,
            [
                "code.txt",
                [
                    {"op": "replace_all", "old": "a", "new": "A"},
                    {"op": "insert_before", "start": anchor(1, "b\n"), "content": "x\n"},
                ],
            ],
        ).call()
    assert path.read_text(encoding="utf-8") == "a\nb\nc\n"


def test_edit_stale_anchor_reports_current_line(tmp_path):
    s = session(tmp_path)
    path = tmp_path / "note.txt"
    path.write_text("old\n", encoding="utf-8")

    with pytest.raises(ToolError) as error:
        EditTool(s, ["note.txt", [{"op": "replace", "start": anchor(0, "wrong\n"), "end": anchor(0, "wrong\n"), "content": "new\n"}]]).call()

    assert "stale anchor" in str(error.value)
    assert "current is anchor=1:" + ReadTool.line_hash("old\n") + " | old" in str(error.value)


@pytest.mark.parametrize(
    ("stale", "current", "expected"),
    [
        (anchor(1, "target\n"), "x\na\ntarget\nc\n", "x\na\nupdated\nc\n"),
        (anchor(2, "target\n"), "a\ntarget\nc\n", "a\nupdated\nc\n"),
    ],
)
def test_edit_relocates_unique_nearby_anchor(tmp_path, stale, current, expected):
    path = tmp_path / "note.txt"
    path.write_text(current, encoding="utf-8")

    EditTool(session(tmp_path), ["note.txt", [{"op": "replace", "start": stale, "end": stale, "content": "updated\n"}]]).call()

    assert path.read_text(encoding="utf-8") == expected


def test_edit_relocates_both_range_anchors(tmp_path):
    path = tmp_path / "note.txt"
    path.write_text("x\na\nb\nc\nd\n", encoding="utf-8")

    EditTool(
        session(tmp_path),
        ["note.txt", [{"op": "replace", "start": anchor(1, "b\n"), "end": anchor(2, "c\n"), "content": "updated\n"}]],
    ).call()

    assert path.read_text(encoding="utf-8") == "x\na\nupdated\nd\n"


@pytest.mark.parametrize(
    "current",
    [
        "x\na\ntarget\n" + "filler\n" * 60 + "target\n",
        "filler\n" * (ReadTool.MAX_ANCHOR_DRIFT + 2) + "target\n",
        "a\nchanged\n",
    ],
    ids=("duplicate-anywhere", "beyond-drift-limit", "content-changed"),
)
def test_edit_does_not_guess_unsafe_anchor_relocation(tmp_path, current):
    path = tmp_path / "note.txt"
    path.write_text(current, encoding="utf-8")

    with pytest.raises(ToolError, match="stale anchor"):
        EditTool(
            session(tmp_path),
            ["note.txt", [{"op": "replace", "start": anchor(1, "target\n"), "end": anchor(1, "target\n"), "content": "updated\n"}]],
        ).call()

    assert path.read_text(encoding="utf-8") == current


def test_line_hash_ignores_trailing_newline():
    # An anchor must depend only on the visible content, so a line's anchor stays stable when only
    # the trailing newline changes (e.g. the last line gaining/losing the final "\n"). It must also
    # agree with the newline-stripping indexed hash the anchor matcher accepts.
    assert ReadTool.line_hash("code") == ReadTool.line_hash("code\n") == ReadTool.line_hash("code\n\n")
    assert ReadTool.anchor_matches("code\n", ReadTool.line_hash("code"))
    assert ReadTool.anchor_matches("code", ReadTool.line_hash("code\n"))


def test_line_hash_is_short_lowercase_base36(tmp_path):
    line_hash = ReadTool.line_hash("alpha\n")
    assert len(line_hash) == 5
    assert line_hash == line_hash.lower()
    assert set(line_hash) <= set("0123456789abcdefghijklmnopqrstuvwxyz")


def test_planned_edit_refuses_to_overwrite_external_change(tmp_path):
    s = session(tmp_path)
    path = tmp_path / "code.txt"
    path.write_text("a\nb\n", encoding="utf-8")
    call = ToolCall("edit", "Edit", ["code.txt", [{"op": "replace", "start": anchor(1, "b\n"), "end": anchor(1, "b\n"), "content": "B\n"}]])
    plan = EditBatchPlan(s).build([call])
    path.write_text("external\n", encoding="utf-8")

    with pytest.raises(ToolError, match="planned edit is stale"):
        plan.planned[call.id].call(EditTool(s, call.args))

    assert path.read_text(encoding="utf-8") == "external\n"


@pytest.mark.parametrize(
    ("original", "raw_edits"),
    [
        ("", [{"op": "create", "content": "a\nb"}]),
        ("aba\n", [{"op": "replace_all", "old": "a", "new": "A"}]),
        (
            "a\nb\nc\n",
            [
                {"op": "replace", "start": anchor(0, "a\n"), "end": anchor(0, "a\n"), "content": "A\n"},
                {"op": "insert_after", "start": anchor(1, "b\n"), "content": "x\n"},
                {"op": "delete", "start": anchor(2, "c\n"), "end": anchor(2, "c\n")},
            ],
        ),
        ("a\nb\n", [{"op": "insert_before", "start": anchor(1, "b\n"), "content": "inserted"}]),
        ("a\nb", [{"op": "delete", "start": anchor(1, "b"), "end": anchor(1, "b")}]),
        ("a\nb\nc\n", [{"op": "replace_unique", "old": "b\n", "new": "B"}]),
    ],
)
def test_single_and_batch_edit_application_are_equivalent(tmp_path, original, raw_edits):
    tool = EditTool(session(tmp_path), ["code.txt", raw_edits])
    _path, edits = tool.parse()
    single = tool.apply(original, edits)
    original_lines = ReadTool.split_lines(original)
    plan = EditBatchPlan(tool.session)
    state = plan.FileState(
        "code.txt",
        [plan.Line(line, index) for index, line in enumerate(original_lines)],
        original_lines,
        edits[0].op != "create",
    )

    batch = plan.apply(tool, state, edits)

    assert "".join(line.text for line in batch.lines) == single.content
    assert batch.changes == single.changes
    assert batch.replacements == single.replacements
    assert batch.replace_all == single.replace_all


@pytest.mark.parametrize(
    ("original", "raw_edits"),
    [
        (
            "a\nb\nc\n",
            [
                {"op": "replace", "start": anchor(0, "a\n"), "end": anchor(1, "b\n"), "content": "x\n"},
                {"op": "delete", "start": anchor(1, "b\n"), "end": anchor(1, "b\n")},
            ],
        ),
        (
            "a\nb\n",
            [
                {"op": "replace_all", "old": "a", "new": "A"},
                {"op": "insert_before", "start": anchor(1, "b\n"), "content": "x\n"},
            ],
        ),
        ("a\n", [{"op": "replace_all", "old": "", "new": "x"}]),
        ("a\nb\n", [{"op": "delete", "start": anchor(1, "b\n"), "end": anchor(0, "a\n")}]),
    ],
)
def test_single_and_batch_edit_application_raise_the_same_error(tmp_path, original, raw_edits):
    tool = EditTool(session(tmp_path), ["code.txt", raw_edits])
    _path, edits = tool.parse()
    original_lines = ReadTool.split_lines(original)
    plan = EditBatchPlan(tool.session)
    state = plan.FileState("code.txt", [plan.Line(line, index) for index, line in enumerate(original_lines)], original_lines, True)

    with pytest.raises(ToolError) as single_error:
        tool.apply(original, edits)
    with pytest.raises(ToolError) as batch_error:
        plan.apply(tool, state, edits)

    assert str(batch_error.value) == str(single_error.value)


def test_split_lines_matches_readlines_only_on_newline():
    # Edit's line model must number lines exactly like Read (file.readlines), i.e. split on "\n"
    # only. str.splitlines(True) also breaks on \x0c and friends, which would desync anchors.
    assert ReadTool.split_lines("a\nb\x0cc\nd\n") == ["a\n", "b\x0cc\n", "d\n"]
    assert ReadTool.split_lines("a\nb") == ["a\n", "b"]
    assert ReadTool.split_lines("") == []
    assert ReadTool.split_lines("a\nb\x0cc\nd\n") != "a\nb\x0cc\nd\n".splitlines(True)


def test_tool_runner_batch_edit_accepts_drifted_anchor(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    path = tmp_path / "code.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    runner.run(
        [
            ToolCall("insert", "Edit", ["code.txt", [{"op": "insert_before", "start": anchor(1, "b\n"), "content": "x\n"}]]),
            ToolCall("replace", "Edit", ["code.txt", [{"op": "replace", "start": anchor(3, "c\n"), "end": anchor(3, "c\n"), "content": "C\n"}]]),
        ]
    )

    assert path.read_text(encoding="utf-8") == "a\nx\nb\nC\n"
    assert s.tool_errors == []


def test_tool_runner_relocates_anchor_drifted_before_batch(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    path = tmp_path / "code.txt"
    path.write_text("x\na\nb\nc\n", encoding="utf-8")
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    runner.run([ToolCall("replace", "Edit", ["code.txt", [{"op": "replace", "start": anchor(1, "b\n"), "end": anchor(1, "b\n"), "content": "B\n"}]])])

    assert path.read_text(encoding="utf-8") == "x\na\nB\nc\n"
    assert s.tool_errors == []


def test_tool_runner_batch_edit_barrier_rejects_ambiguous_relocation(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    path = tmp_path / "code.txt"
    path.write_text("a\nb\nc\nc\n", encoding="utf-8")
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    runner.run(
        [
            ToolCall("insert", "Edit", ["code.txt", [{"op": "insert_before", "start": anchor(1, "b\n"), "content": "x\n"}]]),
            ToolCall("barrier", "Bash", [":"]),
            ToolCall("replace", "Edit", ["code.txt", [{"op": "replace", "start": anchor(2, "c\n"), "end": anchor(2, "c\n"), "content": "C\n"}]]),
        ]
    )

    assert path.read_text(encoding="utf-8") == "a\nx\nb\nc\nc\n"
    assert len([record for record in s.tool_records if record.name == "Edit"]) == 1
    assert s.tool_errors


def test_tool_runner_batch_edit_can_create_empty_then_patch_same_file(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    runner.run(
        [
            ToolCall("create", "Edit", ["empty.txt", [{"op": "create", "content": ""}]]),
            ToolCall("patch", "Edit", ["empty.txt", [{"op": "replace_all", "old": "", "new": "filled\n"}]]),
        ]
    )

    assert (tmp_path / "empty.txt").read_text(encoding="utf-8") == "filled\n"
    assert len([record for record in s.tool_records if record.name == "Edit"]) == 2
    assert s.tool_errors == []


def test_tool_runner_batch_edit_can_create_then_patch_same_file(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    runner.run(
        [
            ToolCall("create", "Edit", ["new.txt", [{"op": "create", "content": "a\nb\n"}]]),
            ToolCall("patch", "Edit", ["new.txt", [{"op": "replace", "start": anchor(1, "b\n"), "end": anchor(1, "b\n"), "content": "B\n"}]]),
        ]
    )

    assert (tmp_path / "new.txt").read_text(encoding="utf-8") == "a\nB\n"
    assert len([record for record in s.tool_records if record.name == "Edit"]) == 2
    assert s.tool_errors == []


def test_tool_runner_batch_edit_create_and_existing_file_edit_are_independent(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    (tmp_path / "old.txt").write_text("a\nb\n", encoding="utf-8")
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    runner.run(
        [
            ToolCall("create", "Edit", ["new.txt", [{"op": "create", "content": "n\n"}]]),
            ToolCall("edit", "Edit", ["old.txt", [{"op": "replace", "start": anchor(1, "b\n"), "end": anchor(1, "b\n"), "content": "B\n"}]]),
        ]
    )

    assert (tmp_path / "new.txt").read_text(encoding="utf-8") == "n\n"
    assert (tmp_path / "old.txt").read_text(encoding="utf-8") == "a\nB\n"
    assert len([record for record in s.tool_records if record.name == "Edit"]) == 2
    assert s.tool_errors == []


def test_tool_runner_batch_edit_maps_original_anchor_after_delete(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    path = tmp_path / "code.txt"
    path.write_text("a\nb\nc\nd\n", encoding="utf-8")
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    runner.run(
        [
            ToolCall("delete", "Edit", ["code.txt", [{"op": "delete", "start": anchor(1, "b\n"), "end": anchor(1, "b\n")}]]),
            ToolCall("replace", "Edit", ["code.txt", [{"op": "replace", "start": anchor(3, "d\n"), "end": anchor(3, "d\n"), "content": "D\n"}]]),
        ]
    )

    assert path.read_text(encoding="utf-8") == "a\nc\nD\n"
    assert s.tool_errors == []


def test_tool_runner_batch_edit_maps_original_anchor_after_insert(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    path = tmp_path / "code.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    runner.run(
        [
            ToolCall("insert", "Edit", ["code.txt", [{"op": "insert_before", "start": anchor(1, "b\n"), "content": "x\n"}]]),
            ToolCall("replace", "Edit", ["code.txt", [{"op": "replace", "start": anchor(2, "c\n"), "end": anchor(2, "c\n"), "content": "C\n"}]]),
        ]
    )

    assert path.read_text(encoding="utf-8") == "a\nx\nb\nC\n"
    assert s.tool_errors == []


def test_tool_runner_batch_edit_plans_files_independently(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    (tmp_path / "a.txt").write_text("a\nb\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("x\ny\n", encoding="utf-8")
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    runner.run(
        [
            ToolCall("edit-a", "Edit", ["a.txt", [{"op": "insert_after", "start": anchor(0, "a\n"), "content": "A\n"}]]),
            ToolCall("edit-b", "Edit", ["b.txt", [{"op": "replace", "start": anchor(1, "y\n"), "end": anchor(1, "y\n"), "content": "Y\n"}]]),
        ]
    )

    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "a\nA\nb\n"
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "x\nY\n"
    assert s.tool_errors == []


def test_tool_runner_batch_edit_read_between_edits_sees_intermediate_file(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    path = tmp_path / "code.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    runner.run(
        [
            ToolCall("insert", "Edit", ["code.txt", [{"op": "insert_before", "start": anchor(1, "b\n"), "content": "x\n"}]]),
            ToolCall("read", "Read", [{"path": "code.txt", "ranges": [[1, 0]]}]),
            ToolCall("replace", "Edit", ["code.txt", [{"op": "replace", "start": anchor(3, "c\n"), "end": anchor(3, "c\n"), "content": "C\n"}]]),
        ]
    )

    read_record = next(record for record in s.tool_records if record.name == "Read")
    assert "| x" in read_record.output
    assert "| c" in read_record.output
    assert "| C" not in read_record.output
    assert path.read_text(encoding="utf-8") == "a\nx\nb\nC\n"


def test_tool_runner_batch_edit_rejects_anchor_for_line_changed_in_batch(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    path = tmp_path / "code.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    runner.run(
        [
            ToolCall("first", "Edit", ["code.txt", [{"op": "replace", "start": anchor(2, "c\n"), "end": anchor(2, "c\n"), "content": "C\n"}]]),
            ToolCall("second", "Edit", ["code.txt", [{"op": "replace", "start": anchor(2, "c\n"), "end": anchor(2, "c\n"), "content": "D\n"}]]),
        ]
    )

    assert path.read_text(encoding="utf-8") == "a\nb\nC\n"
    assert len([record for record in s.tool_records if record.name == "Edit"]) == 1
    assert s.tool_errors


def test_tool_runner_batch_edit_rejects_create_mixed_with_patch_ops(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    runner.run(
        [
            ToolCall(
                "bad",
                "Edit",
                ["bad.txt", [{"op": "create", "content": "one\n"}, {"op": "replace_all", "old": "one", "new": "two"}]],
            )
        ]
    )

    assert not (tmp_path / "bad.txt").exists()
    assert s.tool_records == []
    assert s.tool_errors and "create cannot be mixed" in s.tool_errors[0].error


def test_tool_runner_batch_edit_rejects_directory_target(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    (tmp_path / "pkg").mkdir()
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    runner.run([ToolCall("patch", "Edit", ["pkg", [{"op": "replace_all", "old": "", "new": "x\n"}]])])

    assert s.tool_records == []
    assert s.tool_errors and "path is a directory" in s.tool_errors[0].error


def test_tool_runner_batch_edit_rejects_duplicate_create_same_file(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    runner.run(
        [
            ToolCall("create", "Edit", ["dup.txt", [{"op": "create", "content": "one\n"}]]),
            ToolCall("again", "Edit", ["dup.txt", [{"op": "create", "content": "two\n"}]]),
        ]
    )

    assert (tmp_path / "dup.txt").read_text(encoding="utf-8") == "one\n"
    assert len([record for record in s.tool_records if record.name == "Edit"]) == 1
    assert s.tool_errors and "file already exists" in s.tool_errors[0].error


def test_tool_runner_batch_edit_rejects_patch_missing_file_without_create(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    runner.run([ToolCall("patch", "Edit", ["missing.txt", [{"op": "replace_all", "old": "", "new": "x\n"}]])])

    assert not (tmp_path / "missing.txt").exists()
    assert s.tool_records == []
    assert s.tool_errors and "use op=create" in s.tool_errors[0].error


def test_validate_edit_target_branches(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    s = session(workspace)
    tool = EditTool(s, [])
    (workspace / "a.py").write_text("x", encoding="utf-8")
    (workspace / "sub").mkdir()
    external = tmp_path / "external"
    external.mkdir()
    external_parent_file = tmp_path / "not-a-directory"
    external_parent_file.write_text("x", encoding="utf-8")

    # Existing file, editing -> True (caller should read it).
    assert tool._validate_target(str(workspace / "a.py"), creating=False) is True
    # Missing file, creating inside the workspace -> False (create fresh).
    assert tool._validate_target(str(workspace / "new.py"), creating=True) is False
    # A missing file may be created in an existing external directory; only implicit
    # creation of external parent directories is forbidden.
    assert tool._validate_target(str(external / "new.py"), creating=True) is False
    # Each invalid state raises the same ToolError both edit paths relied on.
    with pytest.raises(ToolError, match="file already exists"):
        tool._validate_target(str(workspace / "a.py"), creating=True)
    with pytest.raises(ToolError, match="path is a directory"):
        tool._validate_target(str(workspace / "sub"), creating=False)
    with pytest.raises(ToolError, match="does not exist"):
        tool._validate_target(str(workspace / "missing.py"), creating=False)
    with pytest.raises(ToolError, match="parent path is not a directory"):
        tool._validate_target(str(external_parent_file / "new.py"), creating=True)
    with pytest.raises(ToolError, match="create it with an approved Bash mkdir"):
        tool._validate_target(str(tmp_path / "missing-external" / "new.py"), creating=True)


def test_edit_creates_file_in_existing_external_directory(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    s = session(workspace)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    runner.run([ToolCall("create", "Edit", ["../external/new.py", [{"op": "create", "content": "value = 1\n"}]])])

    assert (external / "new.py").read_text(encoding="utf-8") == "value = 1\n"
    assert not s.tool_errors


def test_yolo_approves_mutating_tools_without_prompt(tmp_path):
    s = session(tmp_path)
    s.settings.yolo = True
    runner = ToolRunner(s, ContextManager(s), input_fn=lambda prompt: (_ for _ in ()).throw(AssertionError("unexpected prompt")), output_fn=lambda text: None)

    runner.run([ToolCall("create", "Edit", ["auto.txt", [{"op": "create", "content": "ok\n"}]])])

    assert (tmp_path / "auto.txt").read_text(encoding="utf-8") == "ok\n"
    assert len(s.tool_records) == 1


def test_edit_refunds_anchors_for_immediate_followup(tmp_path):
    """The core reflux contract: a second edit that uses only the anchors returned by the first
    edit's output must succeed, so long same-file edit runs never need a fresh Read."""
    s = session(tmp_path)
    path = tmp_path / "code.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")
    first = EditTool(s, ["code.txt", [{"op": "replace", "start": anchor(1, "b\n"), "end": anchor(1, "b\n"), "content": "B\n"}]]).call()
    refunded = re.search(r"(anchor=2:[0-9a-z]+ \| B)", first)
    assert refunded, first

    EditTool(s, ["code.txt", [{"op": "insert_after", "start": refunded.group(1), "content": "x\n"}]]).call()

    assert path.read_text(encoding="utf-8") == "a\nB\nx\nc\n"


def test_edit_refunded_anchor_matches_followup_read(tmp_path):
    """An anchor refunded by a successful edit is exactly the anchor Read reports for the same line."""
    s = session(tmp_path)
    path = tmp_path / "code.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")
    first = EditTool(s, ["code.txt", [{"op": "replace", "start": anchor(1, "b\n"), "end": anchor(1, "b\n"), "content": "B\n"}]]).call()
    refunded = re.search(r"anchor=2:[0-9a-z]+ \| B", first)
    assert refunded, first

    read = ReadTool(s, [{"path": "code.txt"}]).call()
    assert refunded.group(0) in read


def test_edit_old_anchor_still_rejected_after_edit(tmp_path):
    """Refluxing new anchors must not soften the guard: an anchor captured before an edit still
    fails, and the error still echoes the current anchor for the line."""
    s = session(tmp_path)
    path = tmp_path / "code.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")
    old = anchor(1, "b\n")
    EditTool(s, ["code.txt", [{"op": "replace", "start": old, "end": old, "content": "B\n"}]]).call()

    with pytest.raises(ToolError, match="stale anchor"):
        EditTool(s, ["code.txt", [{"op": "replace", "start": old, "end": old, "content": "x\n"}]]).call()
    assert path.read_text(encoding="utf-8") == "a\nB\nc\n"


def test_edit_whole_file_ops_do_not_refund_full_anchor_table(tmp_path):
    """create and replace_all have no per-hunk change to describe; refunding the whole file's
    anchor table would turn one small edit into a large context, so the refund block is omitted."""
    s = session(tmp_path)
    path = tmp_path / "big.txt"
    path.write_text("".join(f"line{i}\n" for i in range(50)), encoding="utf-8")

    replaced = EditTool(s, ["big.txt", [{"op": "replace_all", "old": "line1\n", "new": "LINE1\n"}]]).call()
    assert "<content hashline-numbered>" not in replaced
    assert "anchor=" not in replaced
    assert path.read_text(encoding="utf-8") == "".join(f"LINE{i}\n" if i == 1 else f"line{i}\n" for i in range(50))

    created = EditTool(s, ["fresh.py", [{"op": "create", "content": "".join(f"x{i}\n" for i in range(50))}]]).call()
    assert "<content hashline-numbered>" not in created
    assert "anchor=" not in created
    assert (tmp_path / "fresh.py").read_text(encoding="utf-8") == "".join(f"x{i}\n" for i in range(50))


def test_edit_warns_on_adjacent_duplicate_introduced(tmp_path):
    s = session(tmp_path)
    path = tmp_path / "dup.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")

    result = EditTool(s, ["dup.txt", [{"op": "replace", "start": anchor(1, "b\n"), "end": anchor(1, "b\n"), "content": "c\n"}]]).call()

    assert "<warnings>" in result
    assert "duplicate-lines: adjacent identical lines after this edit; confirm intended" in result
    # The two anchors name the duplicate pair in the edited file (indexes 1 and 2).
    a1 = anchor(1, "c\n")
    a2 = anchor(2, "c\n")
    assert f"anchor={a1} | c" in result
    assert f"anchor={a2} | c" in result
    assert result.index("<warnings>") > result.index("@@")  # after the diff, before the refund block
    assert result.index("</warnings>") < result.index("<invalidate>")
    assert path.read_text(encoding="utf-8") == "a\nc\nc\n"  # the file was still written


def test_edit_no_warning_on_blank_line_duplicates(tmp_path):
    s = session(tmp_path)
    path = tmp_path / "blank.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")

    result = EditTool(s, ["blank.txt", [{"op": "replace", "start": anchor(1, "b\n"), "end": anchor(1, "b\n"), "content": "\n\n"}]]).call()

    assert "<warnings>" not in result
    assert path.read_text(encoding="utf-8") == "a\n\n\nc\n"


def test_edit_no_warning_on_pre_existing_duplicates(tmp_path):
    s = session(tmp_path)
    path = tmp_path / "existing.txt"
    path.write_text("a\nb\nb\nc\n", encoding="utf-8")  # (b, b) pair already present

    result = EditTool(s, ["existing.txt", [{"op": "replace", "start": anchor(0, "a\n"), "end": anchor(0, "a\n"), "content": "A\n"}]]).call()

    assert "<warnings>" not in result
    assert path.read_text(encoding="utf-8") == "A\nb\nb\nc\n"


def test_edit_no_warnings_output_unchanged(tmp_path):
    s = session(tmp_path)
    path = tmp_path / "plain.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")

    result = EditTool(s, ["plain.txt", [{"op": "replace", "start": anchor(0, "a\n"), "end": anchor(0, "a\n"), "content": "A\n"}]]).call()

    assert "<warnings>" not in result
    assert "<Edit path=\"plain.txt\">" in result
    assert path.read_text(encoding="utf-8") == "A\nb\nc\n"


def test_edit_warnings_truncated_at_limit(tmp_path):
    s = session(tmp_path)
    path = tmp_path / "many.txt"
    path.write_text("\n".join(f"l{i}" for i in range(8)) + "\n", encoding="utf-8")
    duplicate_block = "".join(f"d{i}\nd{i}\n" for i in range(8))  # 8 distinct adjacent pairs -> 16 anchors

    result = EditTool(s, ["many.txt", [{"op": "replace", "start": anchor(0, "l0\n"), "end": anchor(7, "l7\n"), "content": duplicate_block}]]).call()

    assert "<warnings>" in result
    assert result.count("duplicate-lines:") == 1  # one rule fired; the cap is on warnings and anchors
    anchor_rows = [line for line in result.splitlines() if line.startswith("anchor=")]
    assert len(anchor_rows) <= 12
    assert "..." in result
    assert result.endswith("</Edit>")
    assert path.read_text(encoding="utf-8") == duplicate_block


def test_edit_warnings_do_not_affect_apply(tmp_path):
    s = session(tmp_path)
    tool = EditTool(s, ["x.txt", [{"op": "replace", "start": anchor(1, "b\n"), "end": anchor(1, "b\n"), "content": "c\n"}]])
    original = "a\nb\nc\n"

    result = tool.apply(original, [tool.parse()[1][0]])
    assert result.content == "a\nc\nc\n"  # apply output is untouched by warnings
    assert len(result.changes) == 1

    # warnings_block is a pure string method: it observes before/after and changes nothing.
    block = tool.warnings_block(original, result.content)
    assert "duplicate-lines" in block and block.startswith("<warnings>") and block.endswith("</warnings>")
    assert tool.warnings_block(original, original) == ""

    # Error behavior is unchanged too: overlapping edits still raise.
    with pytest.raises(ToolError, match="overlap"):
        tool.apply("a\nb\nc\n", [
            Edit(op="replace", start=anchor(0, "a\n"), end=anchor(1, "b\n"), content="x\n"),
            Edit(op="replace", start=anchor(1, "b\n"), end=anchor(2, "c\n"), content="y\n"),
        ])


def test_duplicate_lines_rule_branches():
    """_duplicate_lines only fires on pairs that are new in `after` and neither blank nor existing
    in `before`; it returns the warning object or None, never raising."""
    from minacode.tools.files import _duplicate_lines

    # ① an edit introducing adjacent identical non-blank lines reports duplicate-lines.
    warning = _duplicate_lines("a\nb\n", "a\nb\nb\n")
    assert warning is not None
    assert warning.code == "duplicate-lines"
    assert warning.message == "adjacent identical lines after this edit; confirm intended"

    # ② a pair already adjacent in `before` is not reported.
    assert _duplicate_lines("a\na\n", "a\na\n") is None

    # ③ blank lines never report, even when the edit piles them up.
    assert _duplicate_lines("x\n", "x\n\n\n") is None

    # ④ no adjacent duplicates at all: None.
    assert _duplicate_lines("a\nb\n", "a\nc\n") is None


# --- stale/out-of-range anchor recovery (A) ---


def test_single_anchor_stale_guides_content_check(tmp_path):
    path = tmp_path / "note.txt"
    path.write_text("old\n", encoding="utf-8")

    with pytest.raises(ToolError) as error:
        EditTool(session(tmp_path), ["note.txt", [{"op": "insert_before", "start": anchor(0, "wrong\n"), "content": "x\n"}]]).call()

    message = str(error.value)
    assert "stale anchor" in message
    assert "retry with the current anchor only if its content is the line you meant, otherwise re-read" in message


def test_range_stale_anchor_error_does_not_guess_current_range(tmp_path):
    path = tmp_path / "note.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")

    with pytest.raises(ToolError) as error:
        EditTool(session(tmp_path), ["note.txt", [{"op": "replace", "start": anchor(0, "wrong\n"), "end": anchor(2, "c\n"), "content": "x\n"}]]).call()

    message = str(error.value)
    assert "stale anchor" in message and "retry with the current anchor" in message
    assert "<current-target-ranges hashline-numbered>" not in message
    assert path.read_text(encoding="utf-8") == "a\nb\nc\n"


def test_anchor_out_of_range_reports_file_length(tmp_path):
    path = tmp_path / "note.txt"
    path.write_text("a\nb\n", encoding="utf-8")

    with pytest.raises(ToolError, match="anchor line 10 out of range; file has 2 lines") as error:
        EditTool(session(tmp_path), ["note.txt", [{"op": "replace", "start": "10:abcde", "end": "10:abcde", "content": "x\n"}]]).call()

    assert "<current-target-ranges" not in str(error.value)
    assert path.read_text(encoding="utf-8") == "a\nb\n"


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
    assert "retry with the current anchor" in message
    assert "<current-target-ranges hashline-numbered>" not in message


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


# --- neighborhood anchors on success (B) ---


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


# --- description (C) ---


def test_edit_anchor_description_notes_bash_view_carries_no_anchors():
    schema = EditTool.params_schema()
    edit = schema["properties"]["edits"]["items"]
    assert "replace_unique" in edit["properties"]["op"]["description"]
    for key in ("start", "end"):
        assert "a file viewed through Bash carries no anchors" in edit["properties"][key]["description"]
    assert "replace_unique replaces text that occurs exactly once" in EditTool.DESCRIPTION


# --- replace_unique op (D) ---


def test_replace_unique_replaces_exact_single_hit(tmp_path):
    path = tmp_path / "note.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")

    result = EditTool(session(tmp_path), ["note.txt", [{"op": "replace_unique", "old": "b\n", "new": "B\n"}]]).call()

    assert path.read_text(encoding="utf-8") == "a\nB\nc\n"
    # The hit position is visible like any other change: invalidate plus refunded anchors.
    assert "<invalidate>2:2</invalidate>" in result
    assert "anchor=2:" + ReadTool.line_hash("B\n") + " | B" in result


def test_replace_unique_rejects_multiple_hits_keeps_file_unchanged(tmp_path):
    path = tmp_path / "note.txt"
    original = "a\nb\nb\nc\n"
    path.write_text(original, encoding="utf-8")

    with pytest.raises(ToolError) as error:
        EditTool(session(tmp_path), ["note.txt", [{"op": "replace_unique", "old": "b\n", "new": "B\n"}]]).call()

    message = str(error.value)
    assert "occurs 2 times at lines 2, 3" in message
    assert "must occur exactly once" in message
    assert path.read_text(encoding="utf-8") == original


def test_replace_unique_rejects_missing_text(tmp_path):
    path = tmp_path / "note.txt"
    original = "a\nb\nc\n"
    path.write_text(original, encoding="utf-8")

    with pytest.raises(ToolError, match="replace_unique old text not found"):
        EditTool(session(tmp_path), ["note.txt", [{"op": "replace_unique", "old": "z\n", "new": "Z\n"}]]).call()

    assert path.read_text(encoding="utf-8") == original


def test_replace_unique_matches_indentation_and_special_chars_verbatim(tmp_path):
    path = tmp_path / "code.py"
    path.write_text("def f():\n    x = 1\n    y = 2\n", encoding="utf-8")

    EditTool(session(tmp_path), ["code.py", [{"op": "replace_unique", "old": "    x = 1\n", "new": "    x = 10\n"}]]).call()
    assert path.read_text(encoding="utf-8") == "def f():\n    x = 10\n    y = 2\n"

    quoted = EditTool(session(tmp_path), ["code.py", [{"op": "replace_unique", "old": "x = 10\n", "new": 'x = "ten"\n'}]]).call()
    assert path.read_text(encoding="utf-8") == 'def f():\n    x = "ten"\n    y = 2\n'
    assert quoted.endswith("</Edit>")


def test_replace_unique_replaces_text_spanning_lines(tmp_path):
    path = tmp_path / "note.txt"
    path.write_text("a\nb\nc\nd\n", encoding="utf-8")

    EditTool(session(tmp_path), ["note.txt", [{"op": "replace_unique", "old": "b\nc\n", "new": "BC\n"}]]).call()

    assert path.read_text(encoding="utf-8") == "a\nBC\nd\n"


@pytest.mark.parametrize(
    ("original", "old", "new", "expected"),
    [
        ("a\nb\nc\n", "b\n", "B", "a\nBc\n"),
        ("a\nb", "\n", "", "ab"),
        ("abc\ndef\n", "bc\n", "X", "aXdef\n"),
        ("a\nb\n", "a\nb\n", "", ""),
    ],
)
def test_replace_unique_is_exact_across_line_boundaries(tmp_path, original, old, new, expected):
    path = tmp_path / "note.txt"
    path.write_text(original, encoding="utf-8")

    EditTool(session(tmp_path), ["note.txt", [{"op": "replace_unique", "old": old, "new": new}]]).call()

    assert path.read_text(encoding="utf-8") == expected


def test_replace_unique_multiple_hit_report_is_bounded(tmp_path):
    path = tmp_path / "note.txt"
    original = "hit\n" * 7
    path.write_text(original, encoding="utf-8")

    with pytest.raises(ToolError) as error:
        EditTool(session(tmp_path), ["note.txt", [{"op": "replace_unique", "old": "hit", "new": "miss"}]]).call()

    assert "occurs 7 times at lines 1, 2, 3, 4, 5, ..." in str(error.value)
    assert path.read_text(encoding="utf-8") == original


def test_replace_unique_mixes_with_anchored_ops_in_one_call(tmp_path):
    path = tmp_path / "code.txt"
    path.write_text("a\nb\nc\nd\n", encoding="utf-8")

    EditTool(
        session(tmp_path),
        [
            "code.txt",
            [
                {"op": "replace_unique", "old": "b\n", "new": "B\n"},
                {"op": "delete", "start": anchor(3, "d\n"), "end": anchor(3, "d\n")},
            ],
        ],
    ).call()

    assert path.read_text(encoding="utf-8") == "a\nB\nc\n"


def test_replace_unique_then_anchored_edit_in_batch(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    path = tmp_path / "code.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    runner.run(
        [
            ToolCall("ru", "Edit", ["code.txt", [{"op": "replace_unique", "old": "b\n", "new": "B\nB\n"}]]),
            ToolCall("ins", "Edit", ["code.txt", [{"op": "insert_before", "start": anchor(2, "c\n"), "content": "x\n"}]]),
        ]
    )

    # The second call's anchor was captured against the original file; the batch maps it through the
    # lines the replace_unique edit shifted down.
    assert path.read_text(encoding="utf-8") == "a\nB\nB\nx\nc\n"
    assert s.tool_errors == []


def test_replace_unique_line_join_keeps_later_batch_anchor(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    path = tmp_path / "code.txt"
    path.write_text("a\nb\nc\nd\n", encoding="utf-8")
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    runner.run(
        [
            ToolCall("ru", "Edit", ["code.txt", [{"op": "replace_unique", "old": "b\n", "new": "B"}]]),
            ToolCall("ins", "Edit", ["code.txt", [{"op": "insert_before", "start": anchor(3, "d\n"), "content": "x\n"}]]),
        ]
    )

    assert path.read_text(encoding="utf-8") == "a\nBc\nx\nd\n"
    assert s.tool_errors == []
