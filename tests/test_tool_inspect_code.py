"""tool inspect code (split from tests/test_tools.py)."""
import os
import shutil

import code_symbol_index as csi
import pytest
from test_tools import session

from minacode.base import (
    LogBlock,
    LogEdge,
    LogLine,
    LogRole,
    ToolError,
)
from minacode.render import UiPrinter
from minacode.tools import (
    CodeIndex,
    InspectCodeTool,
)


def test_inspect_code_api_errors_return_failed_result(tmp_path, monkeypatch):
    s = session(tmp_path)
    monkeypatch.setattr(CodeIndex, "available", lambda self: True)
    monkeypatch.setattr(csi, "search", lambda *args, **kwargs: (_ for _ in ()).throw(csi.CodeSymbolIndexError("bad query")))

    result = InspectCodeTool(s, ["find", "Missing"]).call()

    assert "* exit_code: 1" in result
    assert "bad query" in result

def test_inspect_code_modes_call_symbol_index_api(tmp_path, monkeypatch):
    s = session(tmp_path)
    (tmp_path / "sample.py").write_text("class Example:\n    pass\n", encoding="utf-8")
    calls = []

    monkeypatch.setattr(CodeIndex, "available", lambda self: True)
    monkeypatch.setattr(csi, "search", lambda query, **kwargs: calls.append(("search", query, kwargs)) or "search ok")
    monkeypatch.setattr(csi, "inspect", lambda query, **kwargs: calls.append(("inspect", query, kwargs)) or "inspect ok")
    monkeypatch.setattr(csi, "outline", lambda path, **kwargs: calls.append(("outline", path, kwargs)) or "outline ok")
    monkeypatch.setattr(csi, "refs", lambda query, **kwargs: calls.append(("refs", query, kwargs)) or "refs ok")
    monkeypatch.setattr(csi, "impls", lambda query, **kwargs: calls.append(("impls", query, kwargs)) or "impls ok")
    monkeypatch.setattr(csi, "callers", lambda query, **kwargs: calls.append(("callers", query, kwargs)) or "callers ok")
    monkeypatch.setattr(csi, "callees", lambda query, **kwargs: calls.append(("callees", query, kwargs)) or "callees ok")

    assert "search ok" in InspectCodeTool(s, ["find", "Example", {"kind": "class,function", "limit": 10, "exact_only": True}]).call()
    assert "inspect ok" in InspectCodeTool(s, ["inspect", "Example", {"path": "sample.py"}]).call()
    assert "outline ok" in InspectCodeTool(s, ["outline", "sample.py"]).call()
    assert "outline ok" in InspectCodeTool(s, ["outline", "sample.py", {"limit": 300}]).call()
    assert "refs ok" in InspectCodeTool(s, ["refs", "Example", {"all_kinds": True, "offset": 5}]).call()
    assert "impls ok" in InspectCodeTool(s, ["impls", "Example", {"kind": "class"}]).call()
    assert "callers ok" in InspectCodeTool(s, ["callers", "Example", {"depth": 2}]).call()
    assert "callees ok" in InspectCodeTool(s, ["callees", "Example"]).call()

    assert calls[0] == (
        "search",
        "Example",
        {"root": str(tmp_path), "kind": "class,function", "path": None, "exact_only": True, "format": "text", "limit": 10},
    )
    assert calls[1] == (
        "inspect",
        "Example",
        {
            "root": str(tmp_path),
            "kind": None,
            "path": "sample.py",
            "exact_only": False,
            "format": "text",
            "limit": csi.DEFAULT_PAGE_LIMIT,
            "anchors": True,
            "anchor_format": "explicit",
        },
    )
    assert calls[2] == (
        "outline",
        "sample.py",
        {"root": str(tmp_path), "symbol": None, "max_symbols": csi.DEFAULT_MAX_OUTLINE_SYMBOLS, "format": "text"},
    )
    assert calls[3] == (
        "outline",
        "sample.py",
        {"root": str(tmp_path), "symbol": None, "max_symbols": 300, "format": "text"},
    )
    assert calls[4] == (
        "refs",
        "Example",
        {
            "root": str(tmp_path),
            "kind": None,
            "path": None,
            "exact_only": False,
            "format": "text",
            "limit": csi.DEFAULT_MAX_REFERENCES,
            "offset": 5,
            "ref_kinds": "all",
        },
    )
    assert calls[5] == (
        "impls",
        "Example",
        {"root": str(tmp_path), "kind": "class", "path": None, "exact_only": False, "format": "text", "limit": csi.DEFAULT_MAX_IMPLEMENTORS, "offset": 0},
    )
    assert calls[6] == (
        "callers",
        "Example",
        {"root": str(tmp_path), "kind": None, "path": None, "exact_only": False, "format": "text", "limit": csi.DEFAULT_MAX_CALLERS, "depth": 2},
    )
    assert calls[7] == (
        "callees",
        "Example",
        {
            "root": str(tmp_path),
            "kind": None,
            "path": None,
            "exact_only": False,
            "format": "text",
            "limit": csi.DEFAULT_MAX_CALLEES,
            "depth": 3,
            "loose": False,
        },
    )

    assert "refs ok" in InspectCodeTool(s, ["refs", "Example", {"ref_kind": "call,write"}]).call()
    assert calls[8][2]["ref_kinds"] == "call,write"
    assert "callees ok" in InspectCodeTool(s, ["callees", "Example", {"loose": True}]).call()
    assert calls[9][2]["loose"] is True

    with pytest.raises(ToolError):
        InspectCodeTool(s, ["outline", "missing.py"]).call()
    with pytest.raises(ToolError):
        InspectCodeTool(s, ["inspect", "sample.py"]).call()
    with pytest.raises(ToolError):
        InspectCodeTool(s, ["outline", "sample.py", {"limit": 1001}]).call()
    with pytest.raises(ToolError):
        InspectCodeTool(s, ["refs", "sample.py"]).call()
    with pytest.raises(ToolError):
        InspectCodeTool(s, ["callers", "Example", {"depth": 9}]).call()
    with pytest.raises(ToolError):
        InspectCodeTool(s, ["refs", "Example", {"ref_kind": "bogus"}]).call()
    with pytest.raises(ToolError):
        InspectCodeTool(s, ["refs", "Example", {"ref_kind": "call", "all_kinds": True}]).call()

def test_inspect_code_strips_kind_prefix_from_target(tmp_path, monkeypatch):
    s = session(tmp_path)
    calls = []
    monkeypatch.setattr(CodeIndex, "available", lambda self: True)
    monkeypatch.setattr(csi, "search", lambda query, **kwargs: calls.append(query) or "ok")

    # "class Config" with kind "class" -> the redundant leading kind word is dropped.
    InspectCodeTool(s, ["find", "class Config", {"kind": "class"}]).call()
    assert calls[-1] == "Config"

    # Works when the kind option lists several kinds.
    InspectCodeTool(s, ["find", "function handoff", {"kind": "class,function"}]).call()
    assert calls[-1] == "handoff"

    # Only the declared kind is stripped: a bare language keyword is not, and still errors.
    with pytest.raises(ToolError):
        InspectCodeTool(s, ["find", "def foo", {"kind": "function"}]).call()
    # No kind provided -> nothing to key off, still rejected.
    with pytest.raises(ToolError):
        InspectCodeTool(s, ["find", "class Config"]).call()

def test_log_block_aligns_multiline_tool_arguments():
    block = LogBlock.hierarchy(
        LogLine("Bash", 'git commit -m "title\nbody"', LogRole.TOOL, syntax="bash"),
        [LogLine("done", role=LogRole.META, edge=LogEdge.END)],
    )
    expected = '  Bash  git commit -m "title\n        body"\n    └ done'

    assert str(block) == expected
    rendered = "".join(text for _style, text in UiPrinter(output_fn=lambda text: None).log_segments(block))
    assert rendered == expected + "\n"

def test_log_block_wraps_long_tool_arguments_with_hanging_indent(monkeypatch):
    command = 'git commit -m "system prompt: enhance with attitude, updates, review mode, and tooling rules"'
    block = LogBlock([LogLine("Bash", command, LogRole.TOOL, syntax="bash")])

    with monkeypatch.context() as patch:
        patch.setattr(shutil, "get_terminal_size", lambda fallback=(80, 24): os.terminal_size((40, 24)))
        rendered = "".join(text for _style, text in UiPrinter(output_fn=lambda text: None).log_segments(block))

    assert rendered.splitlines() == [
        '  Bash  git commit -m "system prompt:',
        "        enhance with attitude,",
        "        updates, review mode, and",
        '        tooling rules"',
    ]
    assert all(len(line) < 40 for line in rendered.splitlines())
