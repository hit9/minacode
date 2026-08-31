"""edit batch (split from tests/test_edit_tool.py)."""

import pytest
from test_edit_tool import session, view

from wizolt.base import ToolCall, ToolError
from wizolt.context import ContextManager
from wizolt.runner import ToolRunner
from wizolt.tools import CodeIndex, EditTool
from wizolt.tools.files import Edit


def runner(s):
    return ToolRunner(s, ContextManager(s), output_fn=lambda text: None)


def rendered(out, s):
    return out.render(s.register_source_drafts(list(out.drafts)))


def test_tool_runner_batch_edit_rejects_consumed_target(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    path = tmp_path / "code.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")
    key = view(s, "code.txt")

    runner(s).run(
        [
            ToolCall("first", "Edit", ["code.txt", key, [{"op": "replace", "start": 3, "end": 3, "content": "C\n"}]]),
            ToolCall("second", "Edit", ["code.txt", key, [{"op": "replace", "start": 3, "end": 3, "content": "D\n"}]]),
        ]
    )

    assert path.read_text(encoding="utf-8") == "a\nb\nC\n"
    assert len([record for record in s.tool_records if record.name == "Edit"]) == 1
    assert s.tool_errors and "source target consumed" in s.tool_errors[0].error


def test_tool_runner_planned_edit_keeps_warnings(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    path = tmp_path / "code.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")
    key = view(s, "code.txt")

    runner(s).run([ToolCall("duplicate", "Edit", ["code.txt", key, [{"op": "replace", "start": 2, "end": 2, "content": "x\nx\n"}]])])

    record = next(record for record in s.tool_records if record.name == "Edit")
    assert "<warnings>" in record.output
    assert "duplicate-lines" in record.output
    assert path.read_text(encoding="utf-8") == "a\nx\nx\nc\n"


def test_tool_runner_batch_edit_rejects_create_mixed_with_patch_ops(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    runner(s).run(
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


def test_tool_runner_batch_edit_rejects_directory_target(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    path = tmp_path / "pkg"
    path.write_text("x\n", encoding="utf-8")
    key = view(s, "pkg")
    path.unlink()
    path.mkdir()
    runner(s).run([ToolCall("patch", "Edit", ["pkg", key, [{"op": "replace", "start": 1, "end": 1, "content": "y\n"}]])])

    assert s.tool_records == []
    assert s.tool_errors and "path is a directory" in s.tool_errors[0].error


def test_tool_runner_batch_edit_rejects_duplicate_create_same_file(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    runner(s).run(
        [
            ToolCall("create", "Edit", ["dup.txt", "", [{"op": "create", "content": "one\n"}]]),
            ToolCall("again", "Edit", ["dup.txt", "", [{"op": "create", "content": "two\n"}]]),
        ]
    )

    assert (tmp_path / "dup.txt").read_text(encoding="utf-8") == "one\n"
    assert len([record for record in s.tool_records if record.name == "Edit"]) == 1
    assert s.tool_errors and "file already exists" in s.tool_errors[0].error


def test_tool_runner_batch_edit_rejects_patch_missing_file_without_create(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    path = tmp_path / "missing.txt"
    path.write_text("x\n", encoding="utf-8")
    key = view(s, "missing.txt")
    path.unlink()
    runner(s).run([ToolCall("patch", "Edit", ["missing.txt", key, [{"op": "replace", "start": 1, "end": 1, "content": "x\n"}]])])

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


def test_edit_creates_file_in_existing_external_directory(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    s = session(workspace)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    runner(s).run([ToolCall("create", "Edit", ["../external/new.py", "", [{"op": "create", "content": "value = 1\n"}]])])

    assert (external / "new.py").read_text(encoding="utf-8") == "value = 1\n"
    assert not s.tool_errors


def test_yolo_approves_mutating_tools_without_prompt(tmp_path):
    s = session(tmp_path)
    s.settings.yolo = True
    r = ToolRunner(
        s,
        ContextManager(s),
        input_fn=lambda prompt: (_ for _ in ()).throw(AssertionError("unexpected prompt")),
        output_fn=lambda text: None,
    )
    r.run([ToolCall("create", "Edit", ["auto.txt", "", [{"op": "create", "content": "ok\n"}]])])

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

    EditTool(s, ["code.txt", fresh_key, [{"op": "insert_after", "line": 2, "content": "x\n"}]]).call()

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


def test_batch_insert_outside_witness_keeps_later_edit_alive(tmp_path, monkeypatch):
    """An insertion far from a later edit's target leaves that edit resolvable through the batch's
    origin mapping: insert_after line 4, then replace original line 2."""
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    path = tmp_path / "code.txt"
    path.write_text("a\nb\nc\nd\ne\n", encoding="utf-8")
    key = view(s, "code.txt")

    runner(s).run(
        [
            ToolCall("ins", "Edit", ["code.txt", key, [{"op": "insert_after", "line": 4, "content": "INS\n"}]]),
            ToolCall("rep", "Edit", ["code.txt", key, [{"op": "replace", "start": 2, "end": 2, "content": "B\n"}]]),
        ]
    )

    assert path.read_text(encoding="utf-8") == "a\nB\nc\nd\nINS\ne\n"
    assert s.tool_errors == []


def test_edit_warns_on_adjacent_duplicate_introduced(tmp_path):
    s = session(tmp_path)
    path = tmp_path / "dup.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")
    key = view(s, "dup.txt")

    result = EditTool(s, ["dup.txt", key, [{"op": "replace", "start": 2, "end": 2, "content": "c\n"}]]).call()

    assert "<warnings>" in result.retained_text
    assert "duplicate-lines: adjacent identical lines after this edit; confirm intended" in result.retained_text
    assert result.retained_text.index("<warnings>") > result.retained_text.index("@@")  # after the diff
    assert result.retained_text.index("</warnings>") < result.retained_text.index("</Edit>")
    assert path.read_text(encoding="utf-8") == "a\nc\nc\n"  # the file was still written


def test_edit_no_warning_on_blank_line_duplicates(tmp_path):
    s = session(tmp_path)
    path = tmp_path / "blank.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")
    key = view(s, "blank.txt")

    result = EditTool(s, ["blank.txt", key, [{"op": "replace", "start": 2, "end": 2, "content": "\n\n"}]]).call()

    assert "<warnings>" not in result.retained_text
    assert path.read_text(encoding="utf-8") == "a\n\n\nc\n"


def test_edit_no_warning_on_pre_existing_duplicates(tmp_path):
    s = session(tmp_path)
    path = tmp_path / "existing.txt"
    path.write_text("a\nb\nb\nc\n", encoding="utf-8")  # (b, b) pair already present
    key = view(s, "existing.txt")

    result = EditTool(s, ["existing.txt", key, [{"op": "replace", "start": 1, "end": 1, "content": "A\n"}]]).call()

    assert "<warnings>" not in result.retained_text
    assert path.read_text(encoding="utf-8") == "A\nb\nb\nc\n"


def test_edit_no_warnings_output_unchanged(tmp_path):
    s = session(tmp_path)
    path = tmp_path / "plain.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")
    key = view(s, "plain.txt")

    result = EditTool(s, ["plain.txt", key, [{"op": "replace", "start": 1, "end": 1, "content": "A\n"}]]).call()

    assert "<warnings>" not in result.retained_text
    assert '<Edit path="plain.txt">' in result.retained_text
    assert path.read_text(encoding="utf-8") == "A\nb\nc\n"


def test_edit_warnings_truncated_at_limit(tmp_path):
    s = session(tmp_path)
    path = tmp_path / "many.txt"
    path.write_text("\n".join(f"l{i}" for i in range(8)) + "\n", encoding="utf-8")
    duplicate_block = "".join(f"d{i}\nd{i}\n" for i in range(8))  # 8 adjacent pairs -> 16 lines
    key = view(s, "many.txt")

    result = EditTool(s, ["many.txt", key, [{"op": "replace", "start": 1, "end": 8, "content": duplicate_block}]]).call()

    assert "<warnings>" in result.retained_text
    assert result.retained_text.count("duplicate-lines:") == 1  # one rule fired
    warnings_section = result.retained_text.split("<warnings>")[1].split("</warnings>")[0]
    numbered = [line for line in warnings_section.splitlines() if " | " in line]
    assert len(numbered) <= 12
    assert "..." in result.retained_text
    assert result.retained_text.rstrip().endswith("</Edit>")
    assert path.read_text(encoding="utf-8") == duplicate_block


def test_edit_warnings_do_not_affect_apply(tmp_path):
    s = session(tmp_path)
    path = tmp_path / "x.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")
    key = view(s, "x.txt")
    tool = EditTool(s, ["x.txt", key, [{"op": "replace", "start": 2, "end": 2, "content": "c\n"}]])
    original = "a\nb\nc\n"
    view_obj = s.get_source_view(key)
    edits = tool.parse()[2]

    result = tool.apply(original, edits, view_obj)
    assert result.content == "a\nc\nc\n"  # apply output is untouched by warnings
    assert len(result.changes) == 1

    # warnings_block is a pure string method: it observes before/after and changes nothing.
    block = tool.warnings_block(original, result.content, edits)
    assert "duplicate-lines" in block and block.startswith("<warnings>") and block.endswith("</warnings>")
    assert tool.warnings_block(original, original, edits) == ""

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


def test_duplicate_lines_rule_branches():
    """_duplicate_lines only fires on pairs that are new in `after` and neither blank nor existing
    in `before`; it returns the warning object or None, never raising."""
    from wizolt.tools.files import _duplicate_lines

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


def test_large_edit_warns_on_what_the_call_wrote_not_on_the_file():
    """The subject is the assistant message the call arrived in: one call that writes a lot is one
    message that loses a lot when it times out. So the measure is the payload, and an ordinary edit
    to a large file says nothing."""
    from wizolt.tools.files import LARGE_EDIT_CHARS, _large_edit

    small = [Edit(op="replace", start=1, end=1, content="b\n")]
    assert _large_edit(small) is None

    big = [Edit(op="create", content="x" * LARGE_EDIT_CHARS)]
    warning = _large_edit(big)
    assert warning is not None and warning.code == "large-edit"
    assert str(LARGE_EDIT_CHARS) in warning.message and "several" in warning.message

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
