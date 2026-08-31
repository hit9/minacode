"""edit apply (split from tests/test_edit_tool.py)."""

import pytest
from test_edit_tool import session, view

from wizolt.base import ToolCall, ToolError, split_lines
from wizolt.context import ContextManager
from wizolt.runner import ToolRunner
from wizolt.tools import CodeIndex, EditTool
from wizolt.tools.editplan import EditBatchPlan


@pytest.mark.parametrize(
    ("original", "raw_edits"),
    [
        ("", [{"op": "create", "content": "a\nb"}]),
        (
            "a\nb\nc\n",
            [
                {"op": "replace", "start": 1, "end": 1, "content": "A\n"},
                {"op": "insert_after", "line": 2, "content": "x\n"},
                {"op": "delete", "start": 3, "end": 3},
            ],
        ),
        ("a\nb\n", [{"op": "insert_before", "line": 2, "content": "inserted"}]),
        ("a\nb", [{"op": "delete", "start": 2, "end": 2}]),
    ],
)
def test_single_and_batch_edit_application_are_equivalent(tmp_path, original, raw_edits):
    s = session(tmp_path)
    path = tmp_path / "code.txt"
    creating = raw_edits[0]["op"] == "create"
    if not creating:
        path.write_text(original, encoding="utf-8")
    key = "" if creating else view(s, "code.txt")
    tool = EditTool(s, ["code.txt", key, raw_edits])
    _, _, edits = tool.parse()
    view_obj = s.get_source_view(key) if key else None
    single = tool.apply(original, edits, view_obj)
    original_lines = split_lines(original)
    plan = EditBatchPlan(tool.session)
    state = plan.FileState(
        "code.txt",
        [] if creating else [plan.Line(line, (key, i + 1)) for i, line in enumerate(original_lines)],
        original_lines,
        not creating,
    )

    batch = plan.apply(tool, state, edits, view_obj)

    assert "".join(line.text for line in batch.lines) == single.content
    assert batch.changes == single.changes
    assert batch.replacements == single.replacements


def test_single_and_batch_edit_application_raise_the_same_error(tmp_path):
    original = "a\nb\nc\n"
    raw_edits = [
        {"op": "replace", "start": 1, "end": 2, "content": "x\n"},
        {"op": "delete", "start": 2, "end": 2},
    ]
    s = session(tmp_path)
    (tmp_path / "code.txt").write_text(original, encoding="utf-8")
    key = view(s, "code.txt")
    tool = EditTool(s, ["code.txt", key, raw_edits])
    _, _, edits = tool.parse()
    view_obj = s.get_source_view(key)
    original_lines = split_lines(original)
    plan = EditBatchPlan(tool.session)
    state = plan.FileState("code.txt", [plan.Line(line, (key, i + 1)) for i, line in enumerate(original_lines)], original_lines, True)

    with pytest.raises(ToolError) as single_error:
        tool.apply(original, edits, view_obj)
    with pytest.raises(ToolError) as batch_error:
        plan.apply(tool, state, edits, view_obj)

    assert str(batch_error.value) == str(single_error.value)


def test_split_lines_matches_readlines_only_on_newline():
    # Edit's line model must number lines exactly like Read (file.readlines), i.e. split on "\n"
    # only. str.splitlines(True) also breaks on \x0c and friends, which would desync line numbers.
    assert split_lines("a\nb\x0cc\nd\n") == ["a\n", "b\x0cc\n", "d\n"]
    assert split_lines("a\nb") == ["a\n", "b"]
    assert split_lines("") == []
    assert split_lines("a\nb\x0cc\nd\n") != "a\nb\x0cc\nd\n".splitlines(True)


def test_tool_runner_batch_edit_accepts_drifted_view(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    path = tmp_path / "code.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")
    key = view(s, "code.txt")
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    runner.run(
        [
            ToolCall("insert", "Edit", ["code.txt", key, [{"op": "insert_before", "line": 2, "content": "x\n"}]]),
            ToolCall("replace", "Edit", ["code.txt", key, [{"op": "replace", "start": 3, "end": 3, "content": "C\n"}]]),
        ]
    )

    assert path.read_text(encoding="utf-8") == "a\nx\nb\nC\n"
    assert s.tool_errors == []


def test_tool_runner_rejects_view_drifted_before_batch(tmp_path, monkeypatch):
    # An external rewrite between the read and the batch renumbers the file underneath the view; the
    # plan resolves by (view, line) origin against the new content, sees a changed target, and
    # refuses rather than relocating against an untrusted position.
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    path = tmp_path / "code.txt"
    path.write_text("x\na\nb\nc\n", encoding="utf-8")
    key = view(s, "code.txt")
    path.write_text("x\nINS\na\nb\nc\n", encoding="utf-8")
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    runner.run([ToolCall("replace", "Edit", ["code.txt", key, [{"op": "replace", "start": 3, "end": 3, "content": "B\n"}]])])

    assert path.read_text(encoding="utf-8") == "x\nINS\na\nb\nc\n"
    assert s.tool_errors and "source target changed" in s.tool_errors[0].error


def test_tool_runner_batch_edit_barrier_rejects_ambiguous_relocation(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    path = tmp_path / "code.txt"
    path.write_text("a\nb\nc\nc\n", encoding="utf-8")
    key = view(s, "code.txt")
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    runner.run(
        [
            ToolCall("insert", "Edit", ["code.txt", key, [{"op": "insert_before", "line": 2, "content": "x\n"}]]),
            ToolCall("barrier", "Bash", [":"]),
            ToolCall("replace", "Edit", ["code.txt", key, [{"op": "replace", "start": 3, "end": 3, "content": "C\n"}]]),
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

    # create the empty file, then re-read it so the follow-up edit has a current empty-file view.
    runner.run([ToolCall("create", "Edit", ["empty.txt", "", [{"op": "create", "content": ""}]])])
    key = view(s, "empty.txt")
    runner.run([ToolCall("patch", "Edit", ["empty.txt", key, [{"op": "insert_after", "line": 0, "content": "filled\n"}]])])

    assert (tmp_path / "empty.txt").read_text(encoding="utf-8") == "filled\n"
    assert len([record for record in s.tool_records if record.name == "Edit"]) == 2
    assert s.tool_errors == []


def test_tool_runner_batch_edit_rejects_inline_patch_without_read(tmp_path, monkeypatch):
    # In one batch the plan resolves every call against views registered before the batch runs, so
    # a second call that tries to use the just-created file's fresh view inline cannot: the view
    # does not exist yet and the call is refused instead of guessing at line numbers.
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    runner.run(
        [
            ToolCall("create", "Edit", ["empty.txt", "", [{"op": "create", "content": ""}]]),
            ToolCall("patch", "Edit", ["empty.txt", "view.1", [{"op": "insert_after", "line": 0, "content": "filled\n"}]]),
        ]
    )

    assert (tmp_path / "empty.txt").read_text(encoding="utf-8") == ""
    assert len([record for record in s.tool_records if record.name == "Edit"]) == 1
    assert s.tool_errors and "source missing" in s.tool_errors[0].error


def test_tool_runner_batch_edit_can_create_then_patch_same_file(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    runner.run([ToolCall("create", "Edit", ["new.txt", "", [{"op": "create", "content": "a\nb\n"}]])])
    key = view(s, "new.txt")
    runner.run([ToolCall("patch", "Edit", ["new.txt", key, [{"op": "replace", "start": 2, "end": 2, "content": "B\n"}]])])

    assert (tmp_path / "new.txt").read_text(encoding="utf-8") == "a\nB\n"
    assert len([record for record in s.tool_records if record.name == "Edit"]) == 2
    assert s.tool_errors == []


def test_tool_runner_batch_edit_create_and_existing_file_edit_are_independent(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    (tmp_path / "old.txt").write_text("a\nb\n", encoding="utf-8")
    key = view(s, "old.txt")
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    runner.run(
        [
            ToolCall("create", "Edit", ["new.txt", "", [{"op": "create", "content": "n\n"}]]),
            ToolCall("edit", "Edit", ["old.txt", key, [{"op": "replace", "start": 2, "end": 2, "content": "B\n"}]]),
        ]
    )

    assert (tmp_path / "new.txt").read_text(encoding="utf-8") == "n\n"
    assert (tmp_path / "old.txt").read_text(encoding="utf-8") == "a\nB\n"
    assert len([record for record in s.tool_records if record.name == "Edit"]) == 2
    assert s.tool_errors == []


def test_tool_runner_batch_edit_maps_original_view_after_delete(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    path = tmp_path / "code.txt"
    path.write_text("a\nb\nc\nd\n", encoding="utf-8")
    key = view(s, "code.txt")
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    runner.run(
        [
            ToolCall("delete", "Edit", ["code.txt", key, [{"op": "delete", "start": 2, "end": 2}]]),
            ToolCall("replace", "Edit", ["code.txt", key, [{"op": "replace", "start": 4, "end": 4, "content": "D\n"}]]),
        ]
    )

    assert path.read_text(encoding="utf-8") == "a\nc\nD\n"
    assert s.tool_errors == []


def test_tool_runner_batch_edit_maps_original_view_after_insert(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    path = tmp_path / "code.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")
    key = view(s, "code.txt")
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    runner.run(
        [
            ToolCall("insert", "Edit", ["code.txt", key, [{"op": "insert_before", "line": 2, "content": "x\n"}]]),
            ToolCall("replace", "Edit", ["code.txt", key, [{"op": "replace", "start": 3, "end": 3, "content": "C\n"}]]),
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
    key_a = view(s, "a.txt")
    key_b = view(s, "b.txt")
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    runner.run(
        [
            ToolCall("edit-a", "Edit", ["a.txt", key_a, [{"op": "insert_after", "line": 1, "content": "A\n"}]]),
            ToolCall("edit-b", "Edit", ["b.txt", key_b, [{"op": "replace", "start": 2, "end": 2, "content": "Y\n"}]]),
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
    key = view(s, "code.txt")
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    runner.run(
        [
            ToolCall("insert", "Edit", ["code.txt", key, [{"op": "insert_before", "line": 2, "content": "x\n"}]]),
            ToolCall("read", "Read", [{"path": "code.txt", "ranges": [[1, 0]]}]),
            ToolCall("replace", "Edit", ["code.txt", key, [{"op": "replace", "start": 3, "end": 3, "content": "C\n"}]]),
        ]
    )

    read_record = next(record for record in s.tool_records if record.name == "Read")
    assert "| x" in read_record.output
    assert "| c" in read_record.output
    assert "| C" not in read_record.output
    assert path.read_text(encoding="utf-8") == "a\nx\nb\nC\n"
