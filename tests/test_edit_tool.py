import os
import re
import shutil

import pytest
from prompt_toolkit.utils import get_cwidth

from minacode.base import LogBlock, LogEdge, LogLine, LogRole, ToolCall, ToolError
from minacode.context import ContextManager
from minacode.model import ModelClient
from minacode.render import UiPrinter
from minacode.runner import EditBatchPlan, ToolRunner
from minacode.session import Session
from minacode.tools import CodeIndex, EditTool, ReadTool
from minacode.tools.files import Edit


def session(tmp_path):
    return Session(cwd=str(tmp_path))


def anchor(index, line):
    """Anchor for a 0-based line index, rendered the way the model sees it (1-based)."""
    return ReadTool.anchor(index, line)


















def test_approval_diff_background_fills_every_wrapped_row(monkeypatch):
    preview = "--- foo.py\n+++ foo.py\n@@ -1,3 +1,3 @@\n-short\n+a\n+this is a much longer changed line that forces wrapping across several terminal rows"
    block = LogBlock.hierarchy(
        LogLine("Edit", "foo.py", LogRole.TOOL),
        [
            LogLine("preview", role=LogRole.META, edge=LogEdge.BRANCH),
            *(LogLine("", line, LogRole.DIFF, LogEdge.CONTINUE) for line in preview.splitlines()),
        ],
    )

    with monkeypatch.context() as patch:
        patch.setattr(shutil, "get_terminal_size", lambda fallback=(80, 24): os.terminal_size((50, 24)))
        lines = UiPrinter.segment_lines(UiPrinter().log_segments(block))

    spans = []
    for line in lines:
        column = 0
        background_columns = []
        for style, text in line:
            width = get_cwidth(text.rstrip("\n"))
            if "bg:" in style:
                background_columns.extend(range(column, column + width))
            column += width
        if background_columns:
            spans.append((min(background_columns), max(background_columns) + 1))

    expected_start = get_cwidth(LogBlock.prefix(2, LogEdge.CONTINUE))
    assert len(spans) >= 5
    assert set(spans) == {(expected_start, 49)}














































































































# --- stale/out-of-range anchor recovery (A) ---
















# --- neighborhood anchors on success (B) ---






# --- description (C) ---




# --- replace_unique op (D) ---






























