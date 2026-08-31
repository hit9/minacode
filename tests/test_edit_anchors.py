"""source-view editing (split from tests/test_edit_tool.py)."""

import pytest
from test_edit_tool import session, view

from wizolt.base import ToolCall, ToolError
from wizolt.context import ContextManager
from wizolt.model import ModelClient
from wizolt.runner import ToolRunner
from wizolt.source import MAX_VIEW_DRIFT, ToolOutput, render_tool_output
from wizolt.tools import CodeIndex, EditTool, ReadTool
from wizolt.tools.editplan import EditBatchPlan


def rendered(out, s):
    """Render a ToolOutput with fresh view keys, the way the runner presents it to the model."""
    assert isinstance(out, ToolOutput)
    return render_tool_output(out, s.register_source_drafts(list(out.drafts)))


def test_edit_accepts_read_view_evidence(tmp_path):
    # The old test_edit_accepts_inspect_code_anchor: an Edit against a source view produced by
    # Read (here standing in for InspectCode, which hydrates the same kind of block) applies.
    s = session(tmp_path)
    path = tmp_path / "note.txt"
    path.write_text("old\n", encoding="utf-8")
    key = view(s, "note.txt")

    result = EditTool(s, ["note.txt", key, [{"op": "replace", "start": 1, "end": 1, "content": "new\n"}]]).call()

    assert '<Edit path="note.txt">' in result.retained_text
    assert path.read_text(encoding="utf-8") == "new\n"


def test_edit_read_and_edit_agree_on_exotic_line_boundary(tmp_path):
    # Regression: Read numbers lines with readlines (split on "\n" only), so a form-feed inside
    # the middle line must not shift the numbers Edit resolves. "d" is line 3 in both.
    s = session(tmp_path)
    path = tmp_path / "ff.txt"
    path.write_text("a\nb\x0cc\nd\n", encoding="utf-8")
    out = ReadTool(s, [{"path": "ff.txt"}]).call()
    assert "3 | d" in out.retained_text
    key = view(s, "ff.txt")
    EditTool(s, ["ff.txt", key, [{"op": "replace", "start": 3, "end": 3, "content": "D\n"}]]).call()
    assert path.read_text(encoding="utf-8") == "a\nb\x0cc\nD\n"


def test_edit_last_line_without_newline_resolves_identically(tmp_path):
    # A last line with or without a trailing newline is captured exactly by the view; the line
    # number stays stable and the edit resolves against the captured text either way.
    s = session(tmp_path)
    path = tmp_path / "note.txt"
    path.write_text("a\nb\n", encoding="utf-8")
    key = view(s, "note.txt")
    EditTool(s, ["note.txt", key, [{"op": "replace", "start": 2, "end": 2, "content": "B\n"}]]).call()
    assert path.read_text(encoding="utf-8") == "a\nB\n"

    path.write_text("a\nb", encoding="utf-8")  # no final newline
    key = view(s, "note.txt")
    EditTool(s, ["note.txt", key, [{"op": "replace", "start": 2, "end": 2, "content": "B"}]]).call()
    assert path.read_text(encoding="utf-8") == "a\nB"


def test_edit_preserves_literal_escape_sequences_in_content(tmp_path):
    s = session(tmp_path)
    literal_line = r'pattern = "\n\t"'
    tool = EditTool(s, ["script.py", "", [{"op": "create", "content": literal_line}]])

    preview = tool.preview()
    tool.call()

    assert literal_line in preview
    assert (tmp_path / "script.py").read_text(encoding="utf-8") == literal_line

    path = tmp_path / "replace.py"
    path.write_text("old\nnext\n", encoding="utf-8")
    key = view(s, "replace.py")
    EditTool(s, ["replace.py", key, [{"op": "replace", "start": 1, "end": 1, "content": literal_line}]]).call()
    assert path.read_text(encoding="utf-8") == literal_line + "\nnext\n"


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
    EditTool(s, ["empty/keep.txt", "", [{"op": "create", "content": ""}]]).call()
    assert (tmp_path / "empty" / "keep.txt").read_text(encoding="utf-8") == ""
    with pytest.raises(ToolError, match="file already exists"):
        EditTool(s, ["empty/keep.txt", "", [{"op": "create", "content": ""}]]).call()
    # The empty file edits through its fresh empty-file view: insert_after line 0.
    key = view(s, "empty/keep.txt")
    EditTool(s, ["empty/keep.txt", key, [{"op": "insert_after", "line": 0, "content": "kept\n"}]]).call()
    assert (tmp_path / "empty" / "keep.txt").read_text(encoding="utf-8") == "kept\n"

    EditTool(s, ["nested/note.txt", "", [{"op": "create", "content": "one\ntwo\nthree\n"}]]).call()
    path = tmp_path / "nested" / "note.txt"
    assert path.read_text(encoding="utf-8") == "one\ntwo\nthree\n"

    missing = tmp_path / "missing.txt"
    missing.write_text("gone\n", encoding="utf-8")
    missing_key = view(s, "missing.txt")
    missing.unlink()
    with pytest.raises(ToolError, match="use op=create"):
        EditTool(s, ["missing.txt", missing_key, [{"op": "replace", "start": 1, "end": 1, "content": "again\n"}]]).call()

    key = view(s, "nested/note.txt")
    EditTool(
        s,
        [
            "nested/note.txt",
            key,
            [
                {"op": "replace", "start": 1, "end": 1, "content": "ONE\n"},
                {"op": "insert_after", "line": 2, "content": "TWO-AND-HALF\n"},
                {"op": "delete", "start": 3, "end": 3},
            ],
        ],
    ).call()
    assert path.read_text(encoding="utf-8") == "ONE\ntwo\nTWO-AND-HALF\n"


def test_edit_index_update_uses_call_path_when_output_path_is_unparseable(tmp_path, monkeypatch):
    s = session(tmp_path)
    updated = []
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: updated.extend(paths) or "")

    ToolRunner(s, ContextManager(s), output_fn=lambda text: None).update_code_index(
        ToolCall("edit", "Edit", ["made.py", "", [{"op": "create", "content": "x\n"}]]),
        "<Edit path=bad />",
    )

    assert updated == ["made.py"]


def test_edit_inserts_before_existing_line_with_needed_newline(tmp_path):
    s = session(tmp_path)
    path = tmp_path / "code.txt"
    path.write_text("a\nb\n", encoding="utf-8")
    key = view(s, "code.txt")

    EditTool(s, ["code.txt", key, [{"op": "insert_before", "line": 2, "content": "inserted"}]]).call()
    assert path.read_text(encoding="utf-8") == "a\ninserted\nb\n"


def test_edit_allows_repeated_structural_boundary_lines(tmp_path):
    s = session(tmp_path)
    path = tmp_path / "code.txt"
    path.write_text("#if A\nold\n#endif\n#endif\n", encoding="utf-8")

    key = view(s, "code.txt")
    EditTool(
        s,
        ["code.txt", key, [{"op": "replace", "start": 2, "end": 3, "content": "new\n#endif\n"}]],
    ).call()
    assert path.read_text(encoding="utf-8") == "#if A\nnew\n#endif\n#endif\n"

    key = view(s, "code.txt")
    EditTool(s, ["code.txt", key, [{"op": "insert_after", "line": 4, "content": "#endif\n"}]]).call()
    assert path.read_text(encoding="utf-8") == "#if A\nnew\n#endif\n#endif\n#endif\n"


def test_edit_no_change_reports_fresh_view_recovery(tmp_path):
    s = session(tmp_path)
    path = tmp_path / "note.txt"
    path.write_text("old\n", encoding="utf-8")
    key = view(s, "note.txt")

    with pytest.raises(ToolError) as error:
        EditTool(s, ["note.txt", key, [{"op": "replace", "start": 1, "end": 1, "content": "old\n"}]]).call()

    message = str(error.value)
    assert "edit produced no changes; requested content already matches target range" in message
    recovery = error.value.recovery
    assert isinstance(recovery, ToolOutput)
    assert "<source" in rendered(recovery, s)


def test_edit_rejects_directory_target(tmp_path):
    s = session(tmp_path)
    path = tmp_path / "pkg"
    path.write_text("x\n", encoding="utf-8")
    key = view(s, "pkg")
    path.unlink()
    path.mkdir()

    with pytest.raises(ToolError, match="path is a directory"):
        EditTool(s, ["pkg", key, [{"op": "replace", "start": 1, "end": 1, "content": "y\n"}]]).call()


def test_edit_rejects_overlaps_and_mixed_modes(tmp_path):
    s = session(tmp_path)
    path = tmp_path / "code.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")
    key = view(s, "code.txt")

    with pytest.raises(ToolError, match="overlap"):
        EditTool(
            s,
            [
                "code.txt",
                key,
                [
                    {"op": "replace", "start": 1, "end": 2, "content": "x\n"},
                    {"op": "delete", "start": 2, "end": 2},
                ],
            ],
        ).call()
    assert path.read_text(encoding="utf-8") == "a\nb\nc\n"

    with pytest.raises(ToolError, match="create cannot be mixed"):
        EditTool(
            s,
            [
                "code.txt",
                key,
                [
                    {"op": "create", "content": "x\n"},
                    {"op": "insert_before", "line": 2, "content": "y\n"},
                ],
            ],
        ).call()
    assert path.read_text(encoding="utf-8") == "a\nb\nc\n"


def test_edit_stale_target_reports_fresh_view(tmp_path):
    s = session(tmp_path)
    path = tmp_path / "note.txt"
    path.write_text("old\n", encoding="utf-8")
    key = view(s, "note.txt")
    EditTool(s, ["note.txt", key, [{"op": "replace", "start": 1, "end": 1, "content": "changed\n"}]]).call()

    with pytest.raises(ToolError) as error:
        EditTool(s, ["note.txt", key, [{"op": "replace", "start": 1, "end": 1, "content": "new\n"}]]).call()

    message = str(error.value)
    assert "cannot relocate" in message
    assert "use the fresh view below or Read again" in message
    assert "<source" in rendered(error.value.recovery, s)


def test_edit_relocates_unique_nearby_target(tmp_path):
    s = session(tmp_path)
    path = tmp_path / "note.txt"
    path.write_text("x\na\ntarget\nc\n", encoding="utf-8")
    key = view(s, "note.txt")

    # A shift above the target pushes it down one line; the old view still resolves it.
    EditTool(s, ["note.txt", key, [{"op": "insert_after", "line": 1, "content": "INS\n"}]]).call()
    result = EditTool(s, ["note.txt", key, [{"op": "replace", "start": 3, "end": 3, "content": "updated\n"}]]).call()

    assert path.read_text(encoding="utf-8") == "x\nINS\na\nupdated\nc\n"
    assert "relocated view.1 lines 3:3 -> current lines 4:4" in result.retained_text


def test_edit_relocates_both_range_anchors(tmp_path):
    s = session(tmp_path)
    path = tmp_path / "note.txt"
    path.write_text("x\na\nb\nc\nd\n", encoding="utf-8")
    key = view(s, "note.txt")

    EditTool(s, ["note.txt", key, [{"op": "insert_after", "line": 1, "content": "INS\n"}]]).call()
    result = EditTool(s, ["note.txt", key, [{"op": "replace", "start": 3, "end": 4, "content": "updated\n"}]]).call()

    assert path.read_text(encoding="utf-8") == "x\nINS\na\nupdated\nd\n"
    assert "relocated view.1 lines 3:4 -> current lines 4:5" in result.retained_text


@pytest.mark.parametrize(
    ("current",),
    [
        ("x\np\na\ntarget\nfiller\ntarget\nc\n",),  # duplicate target inside the drift window
        (("filler\n" * (MAX_VIEW_DRIFT + 5)) + "target\n",),  # beyond drift limit
        ("a\nchanged\n",),  # content changed
    ],
    ids=("duplicate", "beyond-drift", "content-changed"),
)
def test_edit_does_not_guess_unsafe_relocation(tmp_path, current):
    s = session(tmp_path)
    path = tmp_path / "note.txt"
    path.write_text("x\na\ntarget\nc\n", encoding="utf-8")  # target uniquely at line 3
    key = view(s, "note.txt")
    path.write_text(current, encoding="utf-8")  # external rewrite moves/duplicates/removes it

    with pytest.raises(ToolError, match="cannot relocate"):
        EditTool(s, ["note.txt", key, [{"op": "replace", "start": 3, "end": 3, "content": "updated\n"}]]).call()

    assert path.read_text(encoding="utf-8") == current


def test_planned_edit_refuses_to_overwrite_external_change(tmp_path):
    s = session(tmp_path)
    path = tmp_path / "code.txt"
    path.write_text("a\nb\n", encoding="utf-8")
    key = view(s, "code.txt")
    call = ToolCall("edit", "Edit", ["code.txt", key, [{"op": "replace", "start": 2, "end": 2, "content": "B\n"}]])
    plan = EditBatchPlan(s).build([call])
    path.write_text("external\n", encoding="utf-8")

    with pytest.raises(ToolError, match="planned edit is stale"):
        plan.planned[call.id].call(EditTool(s, call.args))

    assert path.read_text(encoding="utf-8") == "external\n"
