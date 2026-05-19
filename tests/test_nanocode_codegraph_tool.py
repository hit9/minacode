import json

import nanocode
import pytest

from nanocode import CodeGraphContextTool, CodeGraphSymbolTool, Session, ToolCallArgError, ToolCallError


def _init_codegraph_project(tmp_path):
    (tmp_path / ".codegraph").mkdir()
    return Session(cwd=str(tmp_path))


def test_codegraph_tool_requires_binary(tmp_path, monkeypatch):
    monkeypatch.setattr(nanocode.shutil, "which", lambda name: "")

    with pytest.raises(ToolCallError, match="codegraph not found"):
        CodeGraphContextTool.make(Session(cwd=str(tmp_path)), ["Tool class"])


def test_codegraph_tool_requires_initialized_project(tmp_path, monkeypatch):
    monkeypatch.setattr(nanocode.shutil, "which", lambda name: "/fake/codegraph" if name == "codegraph" else "")
    tool = CodeGraphContextTool.make(Session(cwd=str(tmp_path)), ["Tool class"])

    assert tool.requires_confirmation(Session(cwd=str(tmp_path))) is False
    with pytest.raises(ToolCallError, match="/codegraph init"):
        tool.call()


def test_codegraph_tool_context_numbers_code_blocks(tmp_path, monkeypatch):
    session = _init_codegraph_project(tmp_path)
    monkeypatch.setattr(nanocode.shutil, "which", lambda name: "/fake/codegraph" if name == "codegraph" else "")
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        return nanocode.subprocess.CompletedProcess(
            cmd,
            0,
            "\n".join(
                [
                    "## Code Context",
                    "",
                    "#### Tool (nanocode.py:1284)",
                    "",
                    "```python",
                    "class Tool:",
                    "    NAME: ClassVar[str]",
                    "```",
                ]
            ),
            "",
        )

    monkeypatch.setattr(nanocode.subprocess, "run", fake_run)

    result = CodeGraphContextTool.make(session, ["Tool class"]).call()

    assert seen["cmd"] == [
        "/fake/codegraph",
        "context",
        "Tool class",
        "--path",
        str(tmp_path),
        "--max-nodes",
        "40",
        "--max-code",
        "8",
        "--format",
        "markdown",
    ]
    assert "<CodeGraphContextToolResult>" in result
    assert "   1284 |class Tool:\n   1285 |    NAME: ClassVar[str]" in result


def test_codegraph_tool_rejects_extra_args(tmp_path, monkeypatch):
    monkeypatch.setattr(nanocode.shutil, "which", lambda name: "/fake/codegraph" if name == "codegraph" else "")

    with pytest.raises(ToolCallArgError, match="requires args: query"):
        CodeGraphContextTool.make(Session(cwd=str(tmp_path)), ["impact", ["nanocode.py"]])


def test_codegraph_symbol_tool_formats_locations(tmp_path, monkeypatch):
    session = _init_codegraph_project(tmp_path)
    monkeypatch.setattr(nanocode.shutil, "which", lambda name: "/fake/codegraph" if name == "codegraph" else "")
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        return nanocode.subprocess.CompletedProcess(
            cmd,
            0,
            json.dumps(
                [
                    {
                        "node": {
                            "kind": "class",
                            "name": "Tool",
                            "qualifiedName": "Tool",
                            "filePath": "nanocode.py",
                            "startLine": 1284,
                            "endLine": 1330,
                        },
                        "score": 90.005,
                    },
                    {
                        "node": {
                            "kind": "method",
                            "name": "tool_schema",
                            "qualifiedName": "Tool::tool_schema",
                            "filePath": "nanocode.py",
                            "startLine": 1316,
                            "endLine": 1327,
                            "signature": "(cls) -> Json",
                        },
                        "score": 32.99,
                    },
                ]
            ),
            "",
        )

    monkeypatch.setattr(nanocode.subprocess, "run", fake_run)

    result = CodeGraphSymbolTool.make(session, ["Tool"]).call()

    assert seen["cmd"] == ["/fake/codegraph", "query", "Tool", "--path", str(tmp_path), "--limit", "12", "-j"]
    assert "<CodeGraphSymbolToolResult>" in result
    assert "1. class Tool nanocode.py:1284-1330 score=90.0" in result
    assert "2. method Tool::tool_schema nanocode.py:1316-1327 (cls) -> Json score=33.0" in result
