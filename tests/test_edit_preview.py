"""edit preview (split from tests/test_edit_tool.py)."""
from prompt_toolkit.utils import get_cwidth
from test_edit_tool import anchor, session

from minacode.base import LogBlock, LogEdge, LogLine, LogRole, ToolCall
from minacode.context import ContextManager
from minacode.render import UiPrinter
from minacode.runner import ToolRunner
from minacode.tools import CodeIndex, ReadTool


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
    rendered = "".join(text for _, text in segments)

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
    root, _ = next(out[0].walk())
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
    runner.run([ToolCall("edit", "Edit", ["made.py", [{"op": "replace_all", "old": "1", "content": "2"}]])])

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
    changed = [line for line in live if any("bg:" in style for style, _ in line)]
    widths = [sum(get_cwidth(text.rstrip("\n")) for _, text in line) for line in changed]
    assert set(widths) == {40}
