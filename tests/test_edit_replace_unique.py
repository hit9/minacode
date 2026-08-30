"""edit replace unique (split from tests/test_edit_tool.py)."""
import pytest
from test_edit_tool import anchor, session

from wizolt.base import ToolCall, ToolError
from wizolt.context import ContextManager
from wizolt.runner import ToolRunner
from wizolt.tools import CodeIndex, EditTool, ReadTool
from wizolt.tools.files import Edit


def test_replace_unique_replaces_exact_single_hit(tmp_path):
    path = tmp_path / "note.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")
    payload = {"path": "note.txt", "edits": [{"op": "replace_unique", "old": "b\n", "content": "B\n"}]}

    result = EditTool(session(tmp_path), EditTool.payload_args(payload)).call()

    assert path.read_text(encoding="utf-8") == "a\nB\nc\n"
    # The hit position is visible like any other change: invalidate plus refunded anchors.
    assert "<invalidate>2:2</invalidate>" in result
    assert "anchor=2:" + ReadTool.line_hash("B\n") + " | B" in result

def test_replace_unique_rejects_multiple_hits_keeps_file_unchanged(tmp_path):
    path = tmp_path / "note.txt"
    original = "a\nb\nb\nc\n"
    path.write_text(original, encoding="utf-8")

    with pytest.raises(ToolError) as error:
        EditTool(session(tmp_path), ["note.txt", [{"op": "replace_unique", "old": "b\n", "content": "B\n"}]]).call()

    message = str(error.value)
    assert "occurs 2 times at lines 2, 3" in message
    assert "must occur exactly once" in message
    assert path.read_text(encoding="utf-8") == original

def test_replace_unique_rejects_missing_text(tmp_path):
    path = tmp_path / "note.txt"
    original = "a\nb\nc\n"
    path.write_text(original, encoding="utf-8")

    with pytest.raises(ToolError, match="replace_unique old text not found"):
        EditTool(session(tmp_path), ["note.txt", [{"op": "replace_unique", "old": "z\n", "content": "Z\n"}]]).call()

    assert path.read_text(encoding="utf-8") == original

@pytest.mark.parametrize("op", ["replace_all", "replace_unique"])
def test_exact_replacement_requires_explicit_content_instead_of_deleting(tmp_path, op):
    path = tmp_path / "note.txt"
    original = "a\nb\nc\n"
    path.write_text(original, encoding="utf-8")

    with pytest.raises(ToolError, match=rf"{op} requires content; use an explicit empty string"):
        EditTool(session(tmp_path), ["note.txt", [{"op": op, "old": "b\n"}]]).call()

    assert path.read_text(encoding="utf-8") == original

@pytest.mark.parametrize("op", ["replace_all", "replace_unique"])
def test_exact_replacement_explicit_empty_content_deletes_match(tmp_path, op):
    path = tmp_path / "note.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")

    EditTool(session(tmp_path), ["note.txt", [{"op": op, "old": "b\n", "content": ""}]]).call()

    assert path.read_text(encoding="utf-8") == "a\nc\n"

def test_replace_unique_rejects_removed_new_field_without_touching_file(tmp_path):
    path = tmp_path / "note.txt"
    original = "a\nb\nc\n"
    path.write_text(original, encoding="utf-8")

    with pytest.raises(ToolError, match="Edit unexpected field: new"):
        EditTool(session(tmp_path), ["note.txt", [{"op": "replace_unique", "old": "b\n", "new": "B\n"}]]).call()

    assert path.read_text(encoding="utf-8") == original

def test_replace_unique_matches_indentation_and_special_chars_verbatim(tmp_path):
    path = tmp_path / "code.py"
    path.write_text("def f():\n    x = 1\n    y = 2\n", encoding="utf-8")

    EditTool(session(tmp_path), ["code.py", [{"op": "replace_unique", "old": "    x = 1\n", "content": "    x = 10\n"}]]).call()
    assert path.read_text(encoding="utf-8") == "def f():\n    x = 10\n    y = 2\n"

    quoted = EditTool(session(tmp_path), ["code.py", [{"op": "replace_unique", "old": "x = 10\n", "content": 'x = "ten"\n'}]]).call()
    assert path.read_text(encoding="utf-8") == 'def f():\n    x = "ten"\n    y = 2\n'
    assert quoted.endswith("</Edit>")

def test_replace_unique_replaces_text_spanning_lines(tmp_path):
    path = tmp_path / "note.txt"
    path.write_text("a\nb\nc\nd\n", encoding="utf-8")

    EditTool(session(tmp_path), ["note.txt", [{"op": "replace_unique", "old": "b\nc\n", "content": "BC\n"}]]).call()

    assert path.read_text(encoding="utf-8") == "a\nBC\nd\n"

@pytest.mark.parametrize(
    ("original", "old", "replacement", "expected"),
    [
        ("a\nb\nc\n", "b\n", "B", "a\nBc\n"),
        ("a\nb", "\n", "", "ab"),
        ("abc\ndef\n", "bc\n", "X", "aXdef\n"),
        ("a\nb\n", "a\nb\n", "", ""),
    ],
)
def test_replace_unique_is_exact_across_line_boundaries(tmp_path, original, old, replacement, expected):
    path = tmp_path / "note.txt"
    path.write_text(original, encoding="utf-8")

    EditTool(session(tmp_path), ["note.txt", [{"op": "replace_unique", "old": old, "content": replacement}]]).call()

    assert path.read_text(encoding="utf-8") == expected

def test_replace_unique_multiple_hit_report_is_bounded(tmp_path):
    path = tmp_path / "note.txt"
    original = "hit\n" * 7
    path.write_text(original, encoding="utf-8")

    with pytest.raises(ToolError) as error:
        EditTool(session(tmp_path), ["note.txt", [{"op": "replace_unique", "old": "hit", "content": "miss"}]]).call()

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
                {"op": "replace_unique", "old": "b\n", "content": "B\n"},
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
            ToolCall("ru", "Edit", ["code.txt", [{"op": "replace_unique", "old": "b\n", "content": "B\nB\n"}]]),
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
            ToolCall("ru", "Edit", ["code.txt", [{"op": "replace_unique", "old": "b\n", "content": "B"}]]),
            ToolCall("ins", "Edit", ["code.txt", [{"op": "insert_before", "start": anchor(3, "d\n"), "content": "x\n"}]]),
        ]
    )

    assert path.read_text(encoding="utf-8") == "a\nBc\nx\nd\n"
    assert s.tool_errors == []

def test_large_edit_warns_on_what_the_call_wrote_not_on_the_file(tmp_path):
    """The subject is the assistant message the call arrived in: one call that writes a lot is one
    message that loses a lot when it times out. So the measure is the payload, and an ordinary edit
    to a large file says nothing."""
    from wizolt.tools.files import LARGE_EDIT_CHARS, _large_edit

    small = [Edit(op="replace", start=anchor(0, "a\n"), end=anchor(0, "a\n"), content="b\n")]
    assert _large_edit(small) is None

    big = [Edit(op="create", content="x" * LARGE_EDIT_CHARS)]
    warning = _large_edit(big)
    assert warning is not None and warning.code == "large-edit"
    assert str(LARGE_EDIT_CHARS) in warning.message and "several" in warning.message

    # Counted across the whole batch: several edits in one call are still one message, and every
    # operation contributes through the same content field.
    half = "y" * (LARGE_EDIT_CHARS // 2 + 1)
    assert _large_edit([Edit(op="create", content=half), Edit(op="replace_all", old="q", content=half)]) is not None

def test_large_edit_warning_rides_the_edit_result(tmp_path):
    """It reaches the model the same way every other post-edit observation does, so a call that was
    too big says so in its own result rather than only in the tool description."""
    from wizolt.tools.files import LARGE_EDIT_CHARS

    s = session(tmp_path)
    body = "".join(f"line {index}\n" for index in range(LARGE_EDIT_CHARS // 8))
    result = EditTool(s, ["big.py", [{"op": "create", "content": body}]]).call()

    assert "<warnings>" in result and "large-edit" in result
    assert (tmp_path / "big.py").read_text(encoding="utf-8") == body  # advisory only: the edit stands
