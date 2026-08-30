"""edit anchors (split from tests/test_edit_tool.py)."""

import pytest
from test_edit_tool import anchor, session

from wizolt.base import ToolCall, ToolError
from wizolt.context import ContextManager
from wizolt.model import ModelClient
from wizolt.runner import ToolRunner
from wizolt.tools import CodeIndex, EditTool, ReadTool
from wizolt.tools.editplan import EditBatchPlan


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


def test_edit_preserves_literal_escape_sequences_in_content(tmp_path):
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
    EditTool(s, ["unique.py", [{"op": "replace_unique", "old": "OLD", "content": r'"\n"'}]]).call()
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
    EditTool(s, ["empty/keep.txt", [{"op": "replace_all", "old": "", "content": "kept\n"}]]).call()
    assert (tmp_path / "empty" / "keep.txt").read_text(encoding="utf-8") == "kept\n"

    EditTool(s, ["nested/note.txt", [{"op": "create", "content": "one\ntwo\nthree\n"}]]).call()
    path = tmp_path / "nested" / "note.txt"
    assert path.read_text(encoding="utf-8") == "one\ntwo\nthree\n"

    with pytest.raises(ToolError):
        EditTool(s, ["missing.txt", [{"op": "replace_all", "old": "", "content": "again\n"}]]).call()

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

    EditTool(s, ["nested/note.txt", [{"op": "replace_all", "old": "TWO", "content": "two"}]]).call()
    assert path.read_text(encoding="utf-8") == "ONE\ntwo\ntwo-AND-HALF\n"

    with pytest.raises(ToolError):
        EditTool(s, ["nested/note.txt", [{"op": "replace_all", "old": "", "content": "bad\n"}]]).call()
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


def test_edit_allows_repeated_structural_boundary_lines(tmp_path):
    path = tmp_path / "code.txt"
    path.write_text("#if A\nold\n#endif\n#endif\n", encoding="utf-8")

    EditTool(
        session(tmp_path),
        [
            "code.txt",
            [
                {
                    "op": "replace",
                    "start": anchor(1, "old\n"),
                    "end": anchor(2, "#endif\n"),
                    "content": "new\n#endif\n",
                }
            ],
        ],
    ).call()

    assert path.read_text(encoding="utf-8") == "#if A\nnew\n#endif\n#endif\n"

    EditTool(
        session(tmp_path),
        ["code.txt", [{"op": "insert_after", "start": anchor(3, "#endif\n"), "content": "#endif\n"}]],
    ).call()
    assert path.read_text(encoding="utf-8") == "#if A\nnew\n#endif\n#endif\n#endif\n"


def test_edit_no_change_replace_all_reports_identical_file(tmp_path):
    s = session(tmp_path)
    path = tmp_path / "note.txt"
    path.write_text("old\n", encoding="utf-8")

    with pytest.raises(ToolError) as error:
        EditTool(s, ["note.txt", [{"op": "replace_all", "old": "old", "content": "old"}]]).call()

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
        EditTool(s, ["pkg", [{"op": "replace_all", "old": "", "content": "x\n"}]]).call()


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
                    {"op": "replace_all", "old": "a", "content": "A"},
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
    assert "<current-file-context hashline-numbered>" in str(error.value)
    assert "prefer replace_unique" in str(error.value)


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
