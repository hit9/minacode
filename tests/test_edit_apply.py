"""edit apply (split from tests/test_edit_tool.py)."""
import pytest
from test_edit_tool import anchor, session

from minacode.base import ToolCall, ToolError
from minacode.context import ContextManager
from minacode.runner import EditBatchPlan, ToolRunner
from minacode.tools import CodeIndex, EditTool, ReadTool


@pytest.mark.parametrize(
    ("original", "raw_edits"),
    [
        ("", [{"op": "create", "content": "a\nb"}]),
        ("aba\n", [{"op": "replace_all", "old": "a", "content": "A"}]),
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
        ("a\nb\nc\n", [{"op": "replace_unique", "old": "b\n", "content": "B"}]),
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
                {"op": "replace_all", "old": "a", "content": "A"},
                {"op": "insert_before", "start": anchor(1, "b\n"), "content": "x\n"},
            ],
        ),
        ("a\n", [{"op": "replace_all", "old": "", "content": "x"}]),
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
            ToolCall("patch", "Edit", ["empty.txt", [{"op": "replace_all", "old": "", "content": "filled\n"}]]),
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
