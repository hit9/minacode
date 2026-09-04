"""edit batch (split from tests/test_edit_tool.py)."""

import pytest
from test_edit_tool import session, view

from wizolt.base import ToolCall, ToolError
from wizolt.context import ContextManager
from wizolt.runner import ToolRunner
from wizolt.tools import CodeIndex, EditTool
from wizolt.tools.files import Edit


async def ignore_index_update(_index, _paths):
    return ""


def runner(s):
    return ToolRunner(s, ContextManager(s), output_fn=lambda text: None)


def rendered(out, s):
    return out.render(s.register_source_drafts(list(out.drafts)))


async def test_tool_runner_batch_edit_rejects_consumed_target(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", ignore_index_update)
    path = tmp_path / "code.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")
    key = view(s, "code.txt")

    await runner(s).run(
        [
            ToolCall("first", "Edit", ["code.txt", key, [{"op": "replace", "start": 3, "end": 3, "content": "C\n"}]]),
            ToolCall("second", "Edit", ["code.txt", key, [{"op": "replace", "start": 3, "end": 3, "content": "D\n"}]]),
        ]
    )

    assert path.read_text(encoding="utf-8") == "a\nb\nC\n"
    assert len([record for record in s.tool_records if record.name == "Edit"]) == 1
    assert s.tool_errors and "source target consumed" in s.tool_errors[0].error


async def test_tool_runner_planned_edit_writes_adjacent_duplicates_without_warning(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", ignore_index_update)
    path = tmp_path / "code.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")
    key = view(s, "code.txt")

    await runner(s).run([ToolCall("duplicate", "Edit", ["code.txt", key, [{"op": "replace", "start": 2, "end": 2, "content": "x\nx\n"}]])])

    record = next(record for record in s.tool_records if record.name == "Edit")
    assert "<warnings>" not in record.output
    assert path.read_text(encoding="utf-8") == "a\nx\nx\nc\n"


async def test_tool_runner_batch_edit_rejects_create_mixed_with_patch_ops(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", ignore_index_update)
    await runner(s).run(
        [
            ToolCall(
                "bad",
                "Edit",
                ["bad.txt", "", [{"op": "create", "content": "one\n"}, {"op": "replace", "start": 1, "end": 1, "content": "two\n"}]],
            )
        ]
    )

    assert not (tmp_path / "bad.txt").exists()
    assert s.tool_records == []
    assert s.tool_errors and "create cannot be mixed" in s.tool_errors[0].error


async def test_tool_runner_batch_edit_rejects_directory_target(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", ignore_index_update)
    path = tmp_path / "pkg"
    path.write_text("x\n", encoding="utf-8")
    key = view(s, "pkg")
    path.unlink()
    path.mkdir()
    await runner(s).run([ToolCall("patch", "Edit", ["pkg", key, [{"op": "replace", "start": 1, "end": 1, "content": "y\n"}]])])

    assert s.tool_records == []
    assert s.tool_errors and "path is a directory" in s.tool_errors[0].error


async def test_tool_runner_batch_edit_rejects_duplicate_create_same_file(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", ignore_index_update)
    await runner(s).run(
        [
            ToolCall("create", "Edit", ["dup.txt", "", [{"op": "create", "content": "one\n"}]]),
            ToolCall("again", "Edit", ["dup.txt", "", [{"op": "create", "content": "two\n"}]]),
        ]
    )

    assert (tmp_path / "dup.txt").read_text(encoding="utf-8") == "one\n"
    assert len([record for record in s.tool_records if record.name == "Edit"]) == 1
    assert s.tool_errors and "file already exists" in s.tool_errors[0].error


async def test_tool_runner_batch_edit_rejects_patch_missing_file_without_create(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", ignore_index_update)
    path = tmp_path / "missing.txt"
    path.write_text("x\n", encoding="utf-8")
    key = view(s, "missing.txt")
    path.unlink()
    await runner(s).run([ToolCall("patch", "Edit", ["missing.txt", key, [{"op": "replace", "start": 1, "end": 1, "content": "x\n"}]])])

    assert not path.exists()
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


async def test_edit_creates_file_in_existing_external_directory(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    s = session(workspace)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", ignore_index_update)
    await runner(s).run([ToolCall("create", "Edit", ["../external/new.py", "", [{"op": "create", "content": "value = 1\n"}]])])

    assert (external / "new.py").read_text(encoding="utf-8") == "value = 1\n"
    assert not s.tool_errors


async def test_yolo_approves_mutating_tools_without_prompt(tmp_path):
    s = session(tmp_path)
    s.settings.yolo = True
    r = ToolRunner(
        s,
        ContextManager(s),
        input_fn=lambda prompt: (_ for _ in ()).throw(AssertionError("unexpected prompt")),
        output_fn=lambda text: None,
    )
    await r.run([ToolCall("create", "Edit", ["auto.txt", "", [{"op": "create", "content": "ok\n"}]])])

    assert (tmp_path / "auto.txt").read_text(encoding="utf-8") == "ok\n"
    assert len(s.tool_records) == 1


def test_edit_fresh_view_for_immediate_followup(tmp_path):
    """The fresh block of a successful edit is a registered view the next edit can use directly."""
    s = session(tmp_path)
    path = tmp_path / "code.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")
    key = view(s, "code.txt")
    first = EditTool(s, ["code.txt", key, [{"op": "replace", "start": 2, "end": 2, "content": "B\n"}]]).call()
    fresh_key = s.register_source_drafts(list(first.drafts))[0]

    EditTool(s, ["code.txt", fresh_key, [{"op": "replace", "start": 2, "end": 2, "content": "B\nx\n"}]]).call()

    assert path.read_text(encoding="utf-8") == "a\nB\nx\nc\n"


def test_edit_old_view_still_rejected_after_edit(tmp_path):
    """A view captured before an edit still fails against the edited file: the target moved or the
    content changed, so the runtime refuses rather than relocating against an untrusted position."""
    s = session(tmp_path)
    path = tmp_path / "code.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")
    key = view(s, "code.txt")
    EditTool(s, ["code.txt", key, [{"op": "replace", "start": 2, "end": 2, "content": "B\n"}]]).call()

    with pytest.raises(ToolError, match="cannot relocate"):
        EditTool(s, ["code.txt", key, [{"op": "replace", "start": 2, "end": 2, "content": "x\n"}]]).call()
    assert path.read_text(encoding="utf-8") == "a\nB\nc\n"


def test_edit_whole_file_ops_do_not_refund_full_view(tmp_path):
    """A one-line replacement on a large file returns a fresh view covering only the changed hunk
    plus a few context lines, never the whole file."""
    s = session(tmp_path)
    path = tmp_path / "big.txt"
    path.write_text("".join(f"line{i}\n" for i in range(50)), encoding="utf-8")
    key = view(s, "big.txt")

    replaced = EditTool(s, ["big.txt", key, [{"op": "replace", "start": 2, "end": 2, "content": "LINE1\n"}]]).call()
    text = rendered(replaced, s)
    assert 'lines="1:5"' in text
    assert 'lines="1:50"' not in text
    assert path.read_text(encoding="utf-8") == "".join(f"LINE{i}\n" if i == 1 else f"line{i}\n" for i in range(50))


async def test_batch_insert_far_from_later_target_keeps_edit_alive(tmp_path, monkeypatch):
    """An insertion far from a later edit's target leaves that edit resolvable through the batch's
    origin mapping: insert into line 4, then replace original line 2."""
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", ignore_index_update)
    path = tmp_path / "code.txt"
    path.write_text("a\nb\nc\nd\ne\n", encoding="utf-8")
    key = view(s, "code.txt")

    await runner(s).run(
        [
            ToolCall("ins", "Edit", ["code.txt", key, [{"op": "replace", "start": 4, "end": 4, "content": "d\nINS\n"}]]),
            ToolCall("rep", "Edit", ["code.txt", key, [{"op": "replace", "start": 2, "end": 2, "content": "B\n"}]]),
        ]
    )

    assert path.read_text(encoding="utf-8") == "a\nB\nc\nd\nINS\ne\n"
    assert s.tool_errors == []


def test_edit_no_warnings_output_unchanged(tmp_path):
    s = session(tmp_path)
    path = tmp_path / "plain.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")
    key = view(s, "plain.txt")

    result = EditTool(s, ["plain.txt", key, [{"op": "replace", "start": 1, "end": 1, "content": "A\n"}]]).call()

    assert "<warnings>" not in result.retained_text
    assert '<Edit path="plain.txt">' in result.retained_text
    assert path.read_text(encoding="utf-8") == "A\nb\nc\n"


def test_edit_writes_an_intentional_adjacent_duplicate_without_warning(tmp_path):
    # An intentional adjacent duplicate (#endif closing nested guards) is a legal edit under
    # range-only editing: nothing inspects the resulting text, so no <warnings> block is emitted.
    s = session(tmp_path)
    path = tmp_path / "guards.py"
    path.write_text("".join(f"line {index}\n" for index in range(1, 567)) + "#endif\n", encoding="utf-8")
    key = view(s, "guards.py")

    result = EditTool(s, ["guards.py", key, [{"op": "replace", "start": 567, "end": 567, "content": "#endif\n#endif\n"}]]).call()

    assert "<warnings>" not in result.retained_text
    assert path.read_text(encoding="utf-8") == "".join(f"line {index}\n" for index in range(1, 567)) + "#endif\n#endif\n"


def test_edit_warnings_do_not_affect_apply(tmp_path):
    s = session(tmp_path)
    path = tmp_path / "x.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")
    key = view(s, "x.txt")
    tool = EditTool(s, ["x.txt", key, [{"op": "replace", "start": 2, "end": 2, "content": "z\n"}]])
    original = "a\nb\nc\n"
    view_obj = s.get_source_view(key)
    edits = tool.parse()[2]

    result = tool.apply(original, edits, view_obj)
    assert result.content == "a\nz\nc\n"  # apply output is untouched by warnings
    assert len(result.changes) == 1

    # A small edit that touches neither seam carries no warning: the large-edit advisory reads only
    # what the call wrote, and nothing here repeats a preserved neighbour. The boundary-duplicate
    # advisories that do read the file's before and after live in test_edit_apply.py.
    assert result.seam_duplicates == []
    assert tool.warnings_block(edits, result.seam_duplicates) == ""

    # Error behavior is unchanged too: overlapping edits still raise.
    with pytest.raises(ToolError, match="overlap"):
        tool.apply(
            "a\nb\nc\n",
            [
                Edit(op="replace", start=1, end=2, content="x\n"),
                Edit(op="replace", start=2, end=3, content="y\n"),
            ],
            view_obj,
        )


def test_large_edit_warns_on_what_the_call_wrote_not_on_the_file():
    """The subject is the assistant message the call arrived in: one call that writes a lot is one
    message that loses a lot when it times out. So the measure is the payload, and an ordinary edit
    to a large file says nothing."""
    from wizolt.tools.files import LARGE_EDIT_CHARS, _large_edit

    small = [Edit(op="replace", start=1, end=1, content="b\n")]
    assert _large_edit(small) is None

    big = [Edit(op="create", content="x" * LARGE_EDIT_CHARS)]
    warning = _large_edit(big)
    assert warning is not None and "large-edit" in warning
    assert str(LARGE_EDIT_CHARS) in warning and "several" in warning

    # Counted across the whole batch: several edits in one call are still one message, and every
    # operation contributes through the same content field.
    half = "y" * (LARGE_EDIT_CHARS // 2 + 1)
    assert _large_edit([Edit(op="create", content=half), Edit(op="replace", start=1, end=1, content=half)]) is not None


def test_large_edit_warning_rides_the_edit_result(tmp_path):
    """It reaches the model the same way every other post-edit observation does, so a call that was
    too big says so in its own result rather than only in the tool description."""
    from wizolt.tools.files import LARGE_EDIT_CHARS

    s = session(tmp_path)
    body = "".join(f"line {index}\n" for index in range(LARGE_EDIT_CHARS // 8))
    result = EditTool(s, ["big.py", "", [{"op": "create", "content": body}]]).call()

    assert "<warnings>" in result.retained_text and "large-edit" in result.retained_text
    assert (tmp_path / "big.py").read_text(encoding="utf-8") == body  # advisory only: the edit stands


async def test_batch_refuses_two_insertions_at_one_point(tmp_path, monkeypatch):
    """Two insertions at one point used to be refused as a shared insertion point; now the same
    intent is two replace N:N over the same line, which collide as identical overlapping ranges
    and are refused before anything is written."""
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", ignore_index_update)
    path = tmp_path / "code.txt"
    path.write_text("a\nb\nc\nd\n", encoding="utf-8")
    key = view(s, "code.txt")

    await runner(s).run(
        [
            ToolCall(
                "both",
                "Edit",
                [
                    "code.txt",
                    key,
                    [
                        {"op": "replace", "start": 2, "end": 2, "content": "b\nX\n"},
                        {"op": "replace", "start": 2, "end": 2, "content": "b\nY\n"},
                    ],
                ],
            )
        ]
    )

    assert path.read_text(encoding="utf-8") == "a\nb\nc\nd\n"  # nothing was written
    assert s.tool_errors and "edits overlap or are identical ranges" in s.tool_errors[0].error


async def test_batch_accepts_two_views_of_one_path(tmp_path, monkeypatch):
    """Two reads of one file give two ids. The plan indexes its lines by the first view it saw, so
    a later call naming the other one is aligned to the same lines rather than refused -- as long
    as the text behind it still matches what the file held when planning began."""
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", ignore_index_update)
    path = tmp_path / "code.txt"
    path.write_text("a\nb\nc\nd\n", encoding="utf-8")
    whole = view(s, "code.txt")
    tail = view(s, "code.txt", ranges=[[3, 4]])

    assert whole != tail
    await runner(s).run(
        [
            ToolCall("first", "Edit", ["code.txt", whole, [{"op": "replace", "start": 1, "end": 1, "content": "a\nX\n"}]]),
            ToolCall("second", "Edit", ["code.txt", tail, [{"op": "replace", "start": 4, "end": 4, "content": "D\n"}]]),
        ]
    )

    assert path.read_text(encoding="utf-8") == "a\nX\nb\nc\nD\n"
    assert s.tool_errors == []


async def test_batch_relocates_a_view_line_the_file_no_longer_has_room_for(tmp_path, monkeypatch):
    """The file shrank before the batch, so the view's line 5 is past its end and no line origin
    can describe it. The planned index is then only a starting guess, and the exact target -- still
    present, still unique -- is found where it actually is."""
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", ignore_index_update)
    path = tmp_path / "code.txt"
    path.write_text("a\nb\nc\nd\ne\n", encoding="utf-8")
    key = view(s, "code.txt")
    path.write_text("a\nb\ne\n", encoding="utf-8")  # c and d removed elsewhere; e moves up

    await runner(s).run([ToolCall("edit", "Edit", ["code.txt", key, [{"op": "replace", "start": 5, "end": 5, "content": "E\n"}]])])

    assert path.read_text(encoding="utf-8") == "a\nb\nE\n"
    assert s.tool_errors == []
    record = next(record for record in s.tool_records if record.name == "Edit")
    assert f"relocated {key} lines 5:5 -> current lines 3:3" in record.output


async def test_batch_trusts_a_tracked_index_when_an_earlier_edit_changed_a_neighbour(tmp_path, monkeypatch):
    """A tracked index is identity, not a guess. The first call rewrites line 2; the second edits
    line 3, whose text repeats through the file and whose neighbour the first call just changed.
    The plan followed line 3 through that edit, so its position is not re-derived from surrounding
    text and the second call lands -- the neighbour check applies only to an assumed coordinate."""
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", ignore_index_update)
    path = tmp_path / "code.py"
    path.write_text("a\npass\npass\npass\nb\npass\n", encoding="utf-8")
    key = view(s, "code.py")

    await runner(s).run(
        [
            ToolCall("first", "Edit", ["code.py", key, [{"op": "replace", "start": 2, "end": 2, "content": "TWO\n"}]]),
            ToolCall("second", "Edit", ["code.py", key, [{"op": "replace", "start": 3, "end": 3, "content": "THREE\n"}]]),
        ]
    )

    assert path.read_text(encoding="utf-8") == "a\nTWO\nTHREE\npass\nb\npass\n"
    assert s.tool_errors == []


async def test_batch_untracked_target_still_needs_its_neighbours(tmp_path, monkeypatch):
    """The other half of the same rule: when the file drifted before the batch began, the plan
    tracked nothing, so the view's coordinate is an assumption again. Matching text there is not
    enough while the target repeats and the neighbours disagree, and the call is refused."""
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", ignore_index_update)
    path = tmp_path / "code.py"
    path.write_text("x\npass\npass\npass\npass\npass\n", encoding="utf-8")
    key = view(s, "code.py")
    path.write_text("x\ny\npass\npass\npass\npass\npass\npass\n", encoding="utf-8")  # drifted before the batch

    await runner(s).run([ToolCall("ins", "Edit", ["code.py", key, [{"op": "replace", "start": 4, "end": 4, "content": "pass\nNEW\n"}]])])

    assert path.read_text(encoding="utf-8") == "x\ny\npass\npass\npass\npass\npass\npass\n"
    assert s.tool_errors and "cannot relocate" in s.tool_errors[0].error
