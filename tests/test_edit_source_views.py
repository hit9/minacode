"""source-view editing (split from tests/test_edit_tool.py)."""

import asyncio
import threading

import pytest
from test_edit_tool import session, view

from wizolt.base import ToolCall, ToolError
from wizolt.context import ContextManager
from wizolt.model import ModelClient
from wizolt.runner import ToolRunner
from wizolt.source import MAX_VIEW_DRIFT, ToolOutput
from wizolt.tools import CodeIndex, EditTool, ReadTool
from wizolt.tools.editplan import EditBatchPlan


async def ignore_index_update(_index, _paths):
    return ""


def rendered(out, s):
    """Render a ToolOutput with fresh view keys, the way the runner presents it to the model."""
    assert isinstance(out, ToolOutput)
    return out.render(s.register_source_drafts(list(out.drafts)))


def test_edit_accepts_read_view_evidence(tmp_path):
    # An Edit against a source view produced by
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


def test_edit_infers_only_the_complete_missing_replace_shape(tmp_path):
    """Providers sometimes omit only `op`; source + range + explicit text is safely replace."""
    s = session(tmp_path)
    path = tmp_path / "script.py"
    path.write_text("old\nkeep\n", encoding="utf-8")
    key = view(s, "script.py")

    EditTool(s, ["script.py", key, [{"start": 1, "end": 1, "content": "new\n"}]]).call()

    assert path.read_text(encoding="utf-8") == "new\nkeep\n"


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
    # An existing zero-byte file has nothing to preserve, so create may rewrite it...
    EditTool(s, ["empty/keep.txt", "", [{"op": "create", "content": "kept\n"}]]).call()
    assert (tmp_path / "empty" / "keep.txt").read_text(encoding="utf-8") == "kept\n"
    # ...but a non-empty existing file is still refused.
    with pytest.raises(ToolError, match="file already exists"):
        EditTool(s, ["empty/keep.txt", "", [{"op": "create", "content": "again\n"}]]).call()
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
                {"op": "replace", "start": 2, "end": 2, "content": "two\nTWO-AND-HALF\n"},
                {"op": "delete", "start": 3, "end": 3},
            ],
        ],
    ).call()
    assert path.read_text(encoding="utf-8") == "ONE\ntwo\nTWO-AND-HALF\n"


async def test_edit_index_update_uses_call_path_when_output_path_is_unparseable(tmp_path, monkeypatch):
    s = session(tmp_path)
    updated = []

    async def record_update(_index, paths):
        updated.extend(paths)
        return ""

    monkeypatch.setattr(CodeIndex, "update", record_update)

    await ToolRunner(s, ContextManager(s), output_fn=lambda text: None).update_code_index(
        ToolCall("edit", "Edit", ["made.py", "", [{"op": "create", "content": "x\n"}]]),
        "<Edit path=bad />",
    )

    assert updated == ["made.py"]


def test_edit_inserts_before_existing_line_with_needed_newline(tmp_path):
    s = session(tmp_path)
    path = tmp_path / "code.txt"
    path.write_text("a\nb\n", encoding="utf-8")
    key = view(s, "code.txt")

    EditTool(s, ["code.txt", key, [{"op": "replace", "start": 2, "end": 2, "content": "inserted\nb\n"}]]).call()
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
    EditTool(s, ["code.txt", key, [{"op": "replace", "start": 4, "end": 4, "content": "#endif\n#endif\n"}]]).call()
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
                    {"op": "replace", "start": 2, "end": 2, "content": "b\ny\n"},
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
    assert "use the fresh view below, or Read again" in message
    assert "<source" in rendered(error.value.recovery, s)


def test_edit_relocates_unique_nearby_target(tmp_path):
    s = session(tmp_path)
    path = tmp_path / "note.txt"
    path.write_text("x\na\ntarget\nc\n", encoding="utf-8")
    key = view(s, "note.txt")

    # A shift above the target pushes it down one line; the old view still resolves it.
    EditTool(s, ["note.txt", key, [{"op": "replace", "start": 1, "end": 1, "content": "x\nINS\n"}]]).call()
    result = EditTool(s, ["note.txt", key, [{"op": "replace", "start": 3, "end": 3, "content": "updated\n"}]]).call()

    assert path.read_text(encoding="utf-8") == "x\nINS\na\nupdated\nc\n"
    assert "relocated view.1 lines 3:3 -> current lines 4:4" in result.retained_text


def test_edit_relocates_a_multi_line_range(tmp_path):
    s = session(tmp_path)
    path = tmp_path / "note.txt"
    path.write_text("x\na\nb\nc\nd\n", encoding="utf-8")
    key = view(s, "note.txt")

    EditTool(s, ["note.txt", key, [{"op": "replace", "start": 1, "end": 1, "content": "x\nINS\n"}]]).call()
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


@pytest.mark.parametrize(
    ("disturb", "leftover"),
    [
        (lambda path: path.write_text("external\n", encoding="utf-8"), "external\n"),
        (lambda path: path.unlink(), None),
        (lambda path: (path.unlink(), path.mkdir())[0], None),
    ],
    ids=("rewritten", "deleted", "replaced-by-a-directory"),
)
async def test_planned_edit_refuses_to_overwrite_what_changed_after_planning(tmp_path, disturb, leftover):
    """Planning is side-effect free, so the file it computed against can change before the write.
    The last thing a planned edit does is re-read the file and require it to be exactly what it
    planned from -- rewritten, deleted, or turned into a directory, none of them get written."""
    s = session(tmp_path)
    path = tmp_path / "code.txt"
    path.write_text("a\nb\n", encoding="utf-8")
    key = view(s, "code.txt")
    call = ToolCall("edit", "Edit", ["code.txt", key, [{"op": "replace", "start": 2, "end": 2, "content": "B\n"}]])
    plan = await EditBatchPlan(s).build([call])
    disturb(path)

    with pytest.raises(ToolError, match="planned edit is stale"):
        await plan.planned[call.id].apply(EditTool(s, call.args))

    assert (path.read_text(encoding="utf-8") if leftover else None) == leftover


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["code.txt", "$VIEW"], "Edit requires path, source, and edits"),
        ([1, "$VIEW", [{"op": "replace", "start": 1, "end": 1, "content": "x\n"}]], "Edit path must be a string"),
        (["code.txt", 12, [{"op": "replace", "start": 1, "end": 1, "content": "x\n"}]], "Edit source must be a string"),
        (["code.txt", "$VIEW", "replace line 1"], "Edit edits must be a non-empty array"),
        (["code.txt", "$VIEW", []], "Edit edits must be a non-empty array"),
        (["code.txt", "$VIEW", ["replace 1"]], "each edit must be an object"),
        (["code.txt", "$VIEW", [{"op": "replace", "start": 1, "end": 1, "content": "x\n", "old": "a"}]], "mixed edit evidence modes"),
        (["code.txt", "$VIEW", [{"op": "rewrite", "start": 1, "end": 1}]], "Edit op must be create"),
        (["code.txt", "$VIEW", [{"start": 1, "end": 1}]], "Edit op is required"),
        (["code.txt", "$VIEW", [{"start": 1, "end": 1, "content": 7}]], "Edit op is required"),
        (["code.txt", "", [{"start": 1, "end": 1, "content": "x\n"}]], "Edit op is required"),
        (["code.txt", "$VIEW", [{"op": "create", "content": "x\n"}]], "source is forbidden for create"),
        (["code.txt", "", [{"op": "replace", "start": 1, "end": 1, "content": "x\n"}]], "replace needs evidence"),
        (["code.txt", "$VIEW", [{"op": "create", "content": "x"}, {"op": "delete", "start": 1, "end": 1}]], "create cannot be mixed"),
        (["code.txt", "$VIEW", [{"op": "replace", "start": "1", "end": 1, "content": "x\n"}]], "replace requires integer start"),
        (["code.txt", "$VIEW", [{"op": "replace", "start": True, "end": 1, "content": "x\n"}]], "replace requires integer start"),
        (["code.txt", "$VIEW", [{"op": "delete", "start": 1}]], "delete requires integer end"),
        (["code.txt", "$VIEW", [{"op": "replace", "start": 2, "end": 1, "content": "x\n"}]], "replace requires 1 <= start <= end"),
        (["code.txt", "$VIEW", [{"op": "replace", "start": 0, "end": 1, "content": "x\n"}]], "replace requires 1 <= start <= end"),
        (["code.txt", "$VIEW", [{"op": "insert_after", "line": -1, "content": "x\n"}]], "Edit unexpected field: line"),
        (["code.txt", "$VIEW", [{"op": "insert_after", "line": 1}]], "Edit unexpected field: line"),
        (["code.txt", "$VIEW", [{"op": "insert_before", "content": "x\n"}]], "Edit op must be create"),
        (["code.txt", "", [{"op": "create"}]], "create requires content"),
        (["code.txt", "view.99", [{"op": "replace", "start": 1, "end": 1, "content": "x\n"}]], "source missing view.99 is unknown or expired"),
        (["other.txt", "$VIEW", [{"op": "replace", "start": 1, "end": 1, "content": "x\n"}]], "source path mismatch"),
    ],
)
def test_edit_rejects_malformed_calls_before_touching_the_file(tmp_path, args, message):
    """Edit is the one tool that writes, so every malformed call must be refused by name before
    any file is opened -- a half-understood call is not a smaller edit, it is a wrong one."""
    s = session(tmp_path)
    path = tmp_path / "code.txt"
    path.write_text("a\nb\n", encoding="utf-8")
    (tmp_path / "other.txt").write_text("a\nb\n", encoding="utf-8")
    key = view(s, "code.txt")
    resolved = [key if arg == "$VIEW" else arg for arg in args]

    with pytest.raises(ToolError, match=message):
        EditTool(s, resolved).call()

    assert path.read_text(encoding="utf-8") == "a\nb\n"
    assert (tmp_path / "other.txt").read_text(encoding="utf-8") == "a\nb\n"


def test_edit_path_mismatch_does_not_reveal_the_other_view(tmp_path):
    """A view id is not an authorization token: naming the wrong path is refused, and the error
    says nothing about what the mismatched view holds."""
    s = session(tmp_path)
    (tmp_path / "secret.txt").write_text("private line\n", encoding="utf-8")
    (tmp_path / "other.txt").write_text("other line\n", encoding="utf-8")
    key = view(s, "secret.txt")

    with pytest.raises(ToolError) as error:
        EditTool(s, ["other.txt", key, [{"op": "replace", "start": 1, "end": 1, "content": "x\n"}]]).call()

    assert "private line" not in str(error.value)
    assert error.value.recovery is None


@pytest.mark.parametrize(
    ("edits", "message"),
    [
        ([{"op": "replace", "start": 2, "end": 2, "content": "b\n"}], "requested content already matches target range"),
        (
            # Delete line 2 and put the same text back beside it: each half is a real change, the
            # pair is not, and saying "already matches" would misdescribe what happened.
            [{"op": "delete", "start": 2, "end": 2}, {"op": "replace", "start": 3, "end": 3, "content": "b\nc\n"}],
            "edits cancel out; check requested content",
        ),
    ],
    ids=("already-matches", "cancel-out"),
)
def test_an_edit_that_would_change_nothing_is_an_error_with_a_fresh_view(tmp_path, edits, message):
    """A call that leaves the file identical is a misunderstanding, not a success: reporting it as
    done would tell the model its change landed. The error names why and returns the current text
    of the targets as a view, so the next attempt can be aimed rather than re-read."""
    s = session(tmp_path)
    path = tmp_path / "code.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")
    key = view(s, "code.txt")

    with pytest.raises(ToolError, match=message) as error:
        EditTool(s, ["code.txt", key, edits]).call()

    assert path.read_text(encoding="utf-8") == "a\nb\nc\n"
    assert "<source" in rendered(error.value.recovery, s)


def test_preview_reports_a_no_op_without_writing(tmp_path):
    """The confirmation preview runs the same resolution as the write, so a call that changes
    nothing is caught before the user is ever asked to approve it."""
    s = session(tmp_path)
    path = tmp_path / "code.txt"
    path.write_text("a\nb\n", encoding="utf-8")
    key = view(s, "code.txt")

    with pytest.raises(ToolError, match="edit produced no changes"):
        EditTool(s, ["code.txt", key, [{"op": "replace", "start": 1, "end": 1, "content": "a\n"}]]).preview()

    assert path.read_text(encoding="utf-8") == "a\nb\n"


async def test_a_cancelling_batch_edit_fails_cleanly_instead_of_crashing(tmp_path, monkeypatch):
    """Regression: the no-op recovery view was built only from targets whose content already
    matched, so a pair of edits that cancelled out produced a view with no spans over a file that
    has content -- and rendering that raised ValueError out of the batch planner, past every
    ToolError handler, taking the turn with it."""
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", ignore_index_update)
    path = tmp_path / "code.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")
    key = view(s, "code.txt")
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    messages = await runner.run(
        [
            ToolCall(
                "wash",
                "Edit",
                ["code.txt", key, [{"op": "delete", "start": 2, "end": 2}, {"op": "replace", "start": 3, "end": 3, "content": "b\nc\n"}]],
            )
        ]
    )

    assert path.read_text(encoding="utf-8") == "a\nb\nc\n"
    assert len(messages) == 1  # the call still gets exactly one result
    assert "edits cancel out" in s.tool_errors[0].error
    assert "2 | b" in s.tool_errors[0].error  # and a view of what the target actually holds


async def test_batch_relocates_each_edit_and_reports_it(tmp_path, monkeypatch):
    """A batch of edits after an external shift relocates every operation under the same rules as a
    single edit, and reports each relocation separately in the one result. The batch planner maps
    each op's view line to where it now sits after earlier ops, so each replace still finds its
    target by exact text."""
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", ignore_index_update)
    path = tmp_path / "code.txt"
    path.write_text("x\na\ntarget\nc\nd\n", encoding="utf-8")
    key = view(s, "code.txt")
    path.write_text("HEAD\nx\na\ntarget\nc\nd\n", encoding="utf-8")  # every line shifts down one
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    await runner.run(
        [
            ToolCall(
                "batch",
                "Edit",
                [
                    "code.txt",
                    key,
                    [
                        {"op": "replace", "start": 4, "end": 4, "content": "c\nTAIL2\n"},
                        {"op": "replace", "start": 3, "end": 3, "content": "TARGET2\n"},
                    ],
                ],
            )
        ]
    )

    assert path.read_text(encoding="utf-8") == "HEAD\nx\na\nTARGET2\nc\nTAIL2\nd\n"
    assert s.tool_errors == []
    record = next(record for record in s.tool_records if record.name == "Edit")
    assert f"relocated {key} lines 3:3 -> current lines 4:4" in record.output
    assert f"relocated {key} lines 4:4 -> current lines 5:5" in record.output
    assert record.output.count("relocated ") == 2


def test_edit_create_overwrites_zero_byte_file_only(tmp_path):
    """An existing zero-byte file has nothing to preserve, so create may rewrite it; a non-empty
    existing file is still refused rather than overwritten."""
    s = session(tmp_path)
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")

    EditTool(s, ["empty.txt", "", [{"op": "create", "content": "first\n"}]]).call()
    assert path.read_text(encoding="utf-8") == "first\n"

    # The file has content now, so another create is refused.
    with pytest.raises(ToolError, match="file already exists"):
        EditTool(s, ["empty.txt", "", [{"op": "create", "content": "second\n"}]]).call()
    assert path.read_text(encoding="utf-8") == "first\n"


def test_single_line_replace_resolves_by_view_context_when_target_repeats(tmp_path):
    """A drifted single-line target that repeats within the window is disambiguated by the lines
    the view showed around it: the neighbours single out one occurrence, and the edit lands there
    with a relocation report instead of being refused as ambiguous."""
    s = session(tmp_path)
    path = tmp_path / "code.py"
    path.write_text("x\n#endif\ny\n#endif\nz\n", encoding="utf-8")
    key = view(s, "code.py")
    path.write_text("HEAD\nx\n#endif\ny\n#endif\nz\n", encoding="utf-8")

    result = EditTool(s, ["code.py", key, [{"op": "replace", "start": 2, "end": 2, "content": "#else\n"}]]).call()

    assert path.read_text(encoding="utf-8") == "HEAD\nx\n#else\ny\n#endif\nz\n"
    assert f"relocated {key} lines 2:2 -> current lines 3:3" in result.retained_text


def test_repeated_context_still_refuses_ambiguous_relocation(tmp_path):
    """Context narrows but never fabricates: when the whole three-line neighbourhood repeats,
    both candidates survive the filter and the edit is still refused as ambiguous, with widening
    the range named as the retry."""
    s = session(tmp_path)
    path = tmp_path / "code.py"
    path.write_text("p\n#endif\nq\n", encoding="utf-8")
    key = view(s, "code.py")
    path.write_text("w\np\n#endif\nq\np\n#endif\nq\n", encoding="utf-8")

    with pytest.raises(ToolError, match="widen the range"):
        EditTool(s, ["code.py", key, [{"op": "replace", "start": 2, "end": 2, "content": "#else\n"}]]).call()

    assert path.read_text(encoding="utf-8") == "w\np\n#endif\nq\np\n#endif\nq\n"


def test_context_is_a_tiebreaker_not_a_requirement(tmp_path):
    """A unique candidate relocates even though its neighbours have since changed: the surrounding
    lines only narrow a set of several, so one exact match is never rejected because a neighbour
    moved. This is the property the batch path depends on."""
    s = session(tmp_path)
    path = tmp_path / "code.txt"
    path.write_text("w\nx\nz\n", encoding="utf-8")
    key = view(s, "code.txt")
    path.write_text("a\nb\nx\nc\n", encoding="utf-8")  # x moved; neither neighbour matches the view's

    result = EditTool(s, ["code.txt", key, [{"op": "replace", "start": 2, "end": 2, "content": "y\n"}]]).call()

    assert path.read_text(encoding="utf-8") == "a\nb\ny\nc\n"
    assert f"relocated {key} lines 2:2 -> current lines 3:3" in result.retained_text


def test_matching_text_in_place_is_not_trusted_when_the_neighbours_moved(tmp_path):
    """The regression the boundary witness used to catch. `pass` repeats, and after a line is
    prepended the view's line 4 still finds `pass` at its own index -- but that is the *previous*
    occurrence, not the one the model saw. Matching text alone is a coincidence here, so the edit
    is refused rather than written a line off target."""
    s = session(tmp_path)
    path = tmp_path / "code.py"
    path.write_text("x\npass\npass\npass\npass\npass\n", encoding="utf-8")
    key = view(s, "code.py")
    path.write_text("x\ny\npass\npass\npass\npass\npass\npass\n", encoding="utf-8")

    with pytest.raises(ToolError, match="cannot relocate"):
        EditTool(s, ["code.py", key, [{"op": "replace", "start": 4, "end": 4, "content": "pass\nNEW\n"}]]).call()

    assert path.read_text(encoding="utf-8") == "x\ny\npass\npass\npass\npass\npass\npass\n"


def test_neighbours_single_out_the_intended_repeat_in_place(tmp_path):
    """The same shape with distinguishable neighbours: view line 4 is b()'s `pass`, one of three
    identical lines, and after the shift its own index holds a different `pass`. The lines the
    view showed beside it name the intended one, and the edit lands there."""
    s = session(tmp_path)
    path = tmp_path / "code.py"
    path.write_text("def a():\n    pass\ndef b():\n    pass\ndef c():\n    pass\n", encoding="utf-8")
    key = view(s, "code.py")
    path.write_text("import os\ndef a():\n    pass\ndef b():\n    pass\ndef c():\n    pass\n", encoding="utf-8")

    EditTool(s, ["code.py", key, [{"op": "replace", "start": 4, "end": 4, "content": "    pass\n    # b\n"}]]).call()

    assert path.read_text(encoding="utf-8") == "import os\ndef a():\n    pass\ndef b():\n    pass\n    # b\ndef c():\n    pass\n"


def test_a_unique_target_resolves_in_place_though_its_neighbours_changed(tmp_path):
    """Requiring the neighbours in place must not cost a working edit: when the target is the only
    match in the window, it resolves at its own index even with both neighbours rewritten, and the
    result reports no relocation because nothing moved."""
    s = session(tmp_path)
    path = tmp_path / "code.txt"
    path.write_text("a\nUNIQUE\nc\n", encoding="utf-8")
    key = view(s, "code.txt")
    path.write_text("A\nUNIQUE\nC\n", encoding="utf-8")  # both neighbours rewritten, target intact

    result = EditTool(s, ["code.txt", key, [{"op": "replace", "start": 2, "end": 2, "content": "edited\n"}]]).call()

    assert path.read_text(encoding="utf-8") == "A\nedited\nC\n"
    assert "relocated" not in result.retained_text


async def test_edit_planning_reads_files_without_blocking_the_loop(tmp_path, monkeypatch):
    s = session(tmp_path)
    path = tmp_path / "code.txt"
    path.write_text("a\nb\n", encoding="utf-8")
    key = view(s, "code.txt")
    call = ToolCall("edit", "Edit", ["code.txt", key, [{"op": "replace", "start": 2, "end": 2, "content": "B\n"}]])
    entered, release = threading.Event(), threading.Event()
    original = EditBatchPlan.snapshot

    def blocked(target):
        entered.set()
        release.wait()
        return original(target)

    monkeypatch.setattr(EditBatchPlan, "snapshot", staticmethod(blocked))
    task = asyncio.create_task(EditBatchPlan(s).build([call]))
    while not entered.is_set():
        await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert not task.done()
    release.set()

    assert call.id in (await task).planned


async def test_cancelling_an_edit_waits_for_the_write_receipt(tmp_path, monkeypatch):
    s = session(tmp_path)
    path = tmp_path / "code.txt"
    path.write_text("a\nb\n", encoding="utf-8")
    key = view(s, "code.txt")
    call = ToolCall("edit", "Edit", ["code.txt", key, [{"op": "replace", "start": 2, "end": 2, "content": "B\n"}]])
    planned = (await EditBatchPlan(s).build([call])).planned[call.id]
    written, release = threading.Event(), threading.Event()
    original = EditBatchPlan.PlannedEdit.transact

    def blocked(receipt):
        result = original(receipt)
        written.set()
        release.wait()
        return result

    monkeypatch.setattr(EditBatchPlan.PlannedEdit, "transact", blocked)
    tool = EditTool(s, call.args)
    task = asyncio.create_task(planned.apply(tool))
    while not written.is_set():
        await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert path.read_text(encoding="utf-8") == "a\nB\n"
    assert tool.turn_diff() is not None
