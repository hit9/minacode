import json

import nanocode
import pytest

from nanocode import FindCodeSymbolTool, InspectCodeSymbolTool, OutlineCodeFileTool, Session, ToolCallArgError, ToolCallError


def test_inspect_code_requires_cymbal(tmp_path, monkeypatch):
    monkeypatch.setattr(nanocode.shutil, "which", lambda name: "")

    with pytest.raises(ToolCallError, match="cymbal not found"):
        InspectCodeSymbolTool.make(Session(cwd=str(tmp_path)), ["Tool"])


def test_inspect_code_schema_accepts_only_one_target_arg():
    for tool in (InspectCodeSymbolTool, OutlineCodeFileTool):
        args_schema = tool.tool_schema()["function"]["parameters"]["properties"]["args"]
        assert args_schema["minItems"] == 1
        assert args_schema["maxItems"] == 1
        assert args_schema["items"]["type"] == "string"
    args_schema = FindCodeSymbolTool.tool_schema()["function"]["parameters"]["properties"]["args"]
    assert args_schema["minItems"] == 1
    assert args_schema["maxItems"] == 2
    assert args_schema["items"]["type"] == ["string", "number"]


def test_inspect_code_rejects_natural_language(tmp_path, monkeypatch):
    monkeypatch.setattr(nanocode.shutil, "which", lambda name: "/fake/cymbal" if name == "cymbal" else "")

    with pytest.raises(ToolCallArgError, match="do not pass natural language"):
        InspectCodeSymbolTool.make(Session(cwd=str(tmp_path)), ["Tool class callers"])
    with pytest.raises(ToolCallArgError, match="do not pass natural language"):
        FindCodeSymbolTool.make(Session(cwd=str(tmp_path)), ["Tool class"])


def test_find_code_symbol_formats_symbol_results(tmp_path, monkeypatch):
    session = Session(cwd=str(tmp_path))
    monkeypatch.setattr(nanocode.shutil, "which", lambda name: "/fake/cymbal" if name == "cymbal" else "")
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["cwd"] = kwargs["cwd"]
        return nanocode.subprocess.CompletedProcess(
            cmd,
            0,
            json.dumps(
                {
                    "results": [
                        {
                            "name": "Tool",
                            "kind": "class",
                            "rel_path": "nanocode.py",
                            "start_line": 1292,
                            "end_line": 1338,
                            "signature": "class Tool:",
                        }
                    ]
                }
            ),
            "",
        )

    monkeypatch.setattr(nanocode.subprocess, "run", fake_run)

    result = FindCodeSymbolTool.make(session, ["Tool", 12]).call()

    assert seen == {"cmd": ["/fake/cymbal", "search", "Tool", "--limit", "12", "--json"], "cwd": str(tmp_path)}
    assert "<FindCodeSymbolToolResult>" in result
    assert "<symbols>" in result
    assert "class Tool nanocode.py:1291:1338 class Tool:" in result


def test_find_code_symbol_clamps_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(nanocode.shutil, "which", lambda name: "/fake/cymbal" if name == "cymbal" else "")
    assert FindCodeSymbolTool.make(Session(cwd=str(tmp_path)), ["Tool", 999]).limit == 80
    assert FindCodeSymbolTool.make(Session(cwd=str(tmp_path)), ["Tool", 0]).limit == 1
    with pytest.raises(ToolCallArgError, match="limit must be an integer"):
        FindCodeSymbolTool.make(Session(cwd=str(tmp_path)), ["Tool", "many"])


def test_inspect_code_symbol_rejects_files_directories_and_dotted_module_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(nanocode.shutil, "which", lambda name: "/fake/cymbal" if name == "cymbal" else "")
    (tmp_path / "orion" / "biz" / "handlers" / "syftpp").mkdir(parents=True)
    (tmp_path / "code.py").write_text("class Tool:\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))

    with pytest.raises(ToolCallArgError, match="file or directory"):
        InspectCodeSymbolTool.make(session, ["code.py"])
    with pytest.raises(ToolCallArgError, match="file or directory"):
        InspectCodeSymbolTool.make(session, ["orion.biz.handlers.syftpp"])
    with pytest.raises(ToolCallArgError, match="module path"):
        InspectCodeSymbolTool.make(session, ["pkg.module.symbol"])


def test_inspect_code_formats_investigate_result(tmp_path, monkeypatch):
    session = Session(cwd=str(tmp_path))
    monkeypatch.setattr(nanocode.shutil, "which", lambda name: "/fake/cymbal" if name == "cymbal" else "")
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["cwd"] = kwargs["cwd"]
        return nanocode.subprocess.CompletedProcess(
            cmd,
            0,
            json.dumps(
                {
                    "results": {
                        "result": {
                            "symbol": {
                                "name": "Tool",
                                "kind": "class",
                                "rel_path": "nanocode.py",
                                "start_line": 1284,
                                "end_line": 1285,
                                "signature": "class Tool:",
                            },
                            "source": "class Tool:\n    NAME: ClassVar[str]\n",
                            "members": [{"name": "tool_schema", "kind": "function", "rel_path": "nanocode.py", "start_line": 1315, "end_line": 1327}],
                            "refs": [{"name": "Tool", "rel_path": "nanocode.py", "line": 1742}],
                            "impact": [{"symbol": "Tool", "caller": "ReadTool", "rel_path": "nanocode.py", "line": 1742, "depth": 1}],
                            "implementors": [{"implementer": "ReadTool", "target": "Tool", "rel_path": "nanocode.py", "line": 1742, "resolved": True}],
                        }
                    }
                }
            ),
            "",
        )

    monkeypatch.setattr(nanocode.subprocess, "run", fake_run)

    result = InspectCodeSymbolTool.make(session, ["Tool"]).call()

    assert seen == {"cmd": ["/fake/cymbal", "investigate", "Tool", "--json"], "cwd": str(tmp_path)}
    assert "<InspectCodeSymbolToolResult>" in result
    assert '<note>Line numbers are 0-based and match Read/ReplaceRange ranges.</note>' in result
    assert "* symbol: class Tool nanocode.py:1283:1285 class Tool:" in result
    assert "   1283 |class Tool:" in result
    assert "<members>" in result
    assert "function tool_schema nanocode.py:1314:1327" in result
    assert "<references>" in result
    assert "Tool nanocode.py:1741" in result
    assert "<impact>" in result
    assert "ReadTool nanocode.py:1741 symbol=Tool" in result
    assert "<implementors>" in result
    assert "ReadTool nanocode.py:1741 target=Tool" in result


def test_outline_code_file_formats_file_outline(tmp_path, monkeypatch):
    session = Session(cwd=str(tmp_path))
    filepath = tmp_path / "code.py"
    filepath.write_text("class Tool:\n    pass\n", encoding="utf-8")
    monkeypatch.setattr(nanocode.shutil, "which", lambda name: "/fake/cymbal" if name == "cymbal" else "")
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["cwd"] = kwargs["cwd"]
        return nanocode.subprocess.CompletedProcess(
            cmd,
            0,
            json.dumps(
                {
                    "results": [
                        {
                            "name": "Tool",
                            "kind": "class",
                            "rel_path": "code.py",
                            "start_line": 1,
                            "end_line": 2,
                            "signature": "class Tool:",
                        }
                    ]
                }
            ),
            "",
        )

    monkeypatch.setattr(nanocode.subprocess, "run", fake_run)

    result = OutlineCodeFileTool.make(session, ["code.py"]).call()

    assert seen == {"cmd": ["/fake/cymbal", "outline", str(filepath), "--json"], "cwd": str(tmp_path)}
    assert "<OutlineCodeFileToolResult>" in result
    assert "<outline>" in result
    assert "class Tool code.py:0:2 class Tool:" in result


def test_outline_code_file_rejects_directories_and_symbols(tmp_path, monkeypatch):
    monkeypatch.setattr(nanocode.shutil, "which", lambda name: "/fake/cymbal" if name == "cymbal" else "")
    (tmp_path / "pkg").mkdir()
    session = Session(cwd=str(tmp_path))

    with pytest.raises(ToolCallArgError, match="existing file"):
        OutlineCodeFileTool.make(session, ["pkg"])
    with pytest.raises(ToolCallArgError, match="existing file"):
        OutlineCodeFileTool.make(session, ["Tool"])
