"""edit batch (split from tests/test_edit_tool.py)."""
import re

import pytest
from test_edit_tool import anchor, session

from minacode.base import ToolCall, ToolError
from minacode.context import ContextManager
from minacode.runner import ToolRunner
from minacode.tools import CodeIndex, EditTool, ReadTool
from minacode.tools.files import Edit


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

def test_tool_runner_planned_edit_keeps_warnings(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    path = tmp_path / "code.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    runner.run(
        [
            ToolCall(
                "duplicate",
                "Edit",
                ["code.txt", [{"op": "replace", "start": anchor(1, "b\n"), "end": anchor(1, "b\n"), "content": "x\nx\n"}]],
            )
        ]
    )

    record = next(record for record in s.tool_records if record.name == "Edit")
    assert "<warnings>" in record.output
    assert "duplicate-lines" in record.output
    assert path.read_text(encoding="utf-8") == "a\nx\nx\nc\n"

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
                ["bad.txt", [{"op": "create", "content": "one\n"}, {"op": "replace_all", "old": "one", "content": "two"}]],
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

    runner.run([ToolCall("patch", "Edit", ["pkg", [{"op": "replace_all", "old": "", "content": "x\n"}]])])

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

    runner.run([ToolCall("patch", "Edit", ["missing.txt", [{"op": "replace_all", "old": "", "content": "x\n"}]])])

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

    replaced = EditTool(s, ["big.txt", [{"op": "replace_all", "old": "line1\n", "content": "LINE1\n"}]]).call()
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
    assert '<Edit path="plain.txt">' in result
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
    edits = tool.parse()[1]
    block = tool.warnings_block(original, result.content, edits)
    assert "duplicate-lines" in block and block.startswith("<warnings>") and block.endswith("</warnings>")
    assert tool.warnings_block(original, original, edits) == ""

    # Error behavior is unchanged too: overlapping edits still raise.
    with pytest.raises(ToolError, match="overlap"):
        tool.apply(
            "a\nb\nc\n",
            [
                Edit(op="replace", start=anchor(0, "a\n"), end=anchor(1, "b\n"), content="x\n"),
                Edit(op="replace", start=anchor(1, "b\n"), end=anchor(2, "c\n"), content="y\n"),
            ],
        )

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
