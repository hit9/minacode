import nanocode
import pytest

from nanocode import CodeGraphTool, Session, ToolCallError


def _init_codegraph_project(tmp_path):
    (tmp_path / ".codegraph").mkdir()
    return Session(cwd=str(tmp_path))


def test_codegraph_tool_requires_binary(tmp_path, monkeypatch):
    monkeypatch.setattr(nanocode.shutil, "which", lambda name: "")

    with pytest.raises(ToolCallError, match="codegraph not found"):
        CodeGraphTool.make(Session(cwd=str(tmp_path)), ["Tool class"])


def test_codegraph_tool_requires_initialized_project(tmp_path, monkeypatch):
    monkeypatch.setattr(nanocode.shutil, "which", lambda name: "/fake/codegraph" if name == "codegraph" else "")
    tool = CodeGraphTool.make(Session(cwd=str(tmp_path)), ["Tool class"])

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

    result = CodeGraphTool.make(session, ["Tool class"]).call()

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
    assert "<CodeGraphToolResult>" in result
    assert "* mode: context" in result
    assert "   1284 |class Tool:\n   1285 |    NAME: ClassVar[str]" in result


def test_codegraph_tool_impact_uses_paths(tmp_path, monkeypatch):
    session = _init_codegraph_project(tmp_path)
    (tmp_path / "nanocode.py").write_text("# sample\n", encoding="utf-8")
    monkeypatch.setattr(nanocode.shutil, "which", lambda name: "/fake/codegraph" if name == "codegraph" else "")
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        return nanocode.subprocess.CompletedProcess(cmd, 0, "affected tests\n", "")

    monkeypatch.setattr(nanocode.subprocess, "run", fake_run)

    result = CodeGraphTool.make(session, ["impact", ["nanocode.py"]]).call()

    assert seen["cmd"] == ["/fake/codegraph", "affected", "--path", str(tmp_path), "nanocode.py"]
    assert "* mode: impact" in result
    assert "affected tests" in result


def test_codegraph_tool_rejects_paths_outside_cwd(tmp_path, monkeypatch):
    other = tmp_path.parent / "other.py"
    other.write_text("# outside\n", encoding="utf-8")
    monkeypatch.setattr(nanocode.shutil, "which", lambda name: "/fake/codegraph" if name == "codegraph" else "")

    with pytest.raises(ToolCallError, match="path outside cwd"):
        CodeGraphTool.make(Session(cwd=str(tmp_path)), ["impact", [str(other)]])
