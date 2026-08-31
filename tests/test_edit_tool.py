import os
import shutil

from prompt_toolkit.utils import get_cwidth

from wizolt.base import LogBlock, LogEdge, LogLine, LogRole
from wizolt.render import UiPrinter
from wizolt.session import Session
from wizolt.tools import ReadTool


def session(tmp_path):
    return Session(cwd=str(tmp_path))


def view(s, relpath, ranges=None):
    """Read `relpath` through the real ReadTool and register its source draft, returning the
    session-scoped view.N key the runner would allocate on the main thread."""
    args = {"path": relpath}
    if ranges:
        args["ranges"] = ranges
    out = ReadTool(s, [args]).call()
    return s.register_source_drafts(list(out.drafts))[0]


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
