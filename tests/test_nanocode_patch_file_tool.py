import pytest

from nanocode import Agent, PatchFileTool, Session, ToolCallError


def test_patch_file_tool_applies_single_hunk(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))

    tool = PatchFileTool.make(session, ["sample.txt", "@@\n alpha\n-beta\n+BETA\n gamma\n"])
    display = tool.preview()
    result = tool.call()

    assert tool.requires_confirmation(session) is True
    assert "-beta\n" in display
    assert "+BETA\n" in display
    assert path.read_text(encoding="utf-8") == "alpha\nBETA\ngamma\n"
    assert result == "\n".join(
        [
            "<PatchFileToolResult>",
            "* path: sample.txt",
            "* hunks: 1",
            "</PatchFileToolResult>",
        ]
    )


def test_patch_file_tool_accepts_common_diff_headers(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    patch = """diff --git a/sample.txt b/sample.txt
index 1111111..2222222 100644
--- a/sample.txt
+++ b/sample.txt
@@ -1,3 +1,3 @@
 alpha
-beta
+BETA
 gamma
"""

    PatchFileTool.make(session, ["sample.txt", patch]).call()

    assert path.read_text(encoding="utf-8") == "alpha\nBETA\ngamma\n"


def test_patch_file_tool_applies_multiple_hunks_atomically(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\nbeta\ngamma\ndelta\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))

    PatchFileTool.make(
        session,
        [
            "sample.txt",
            "@@\n alpha\n-beta\n+BETA\n gamma\n@@\n gamma\n-delta\n+DELTA\n",
        ],
    ).call()

    assert path.read_text(encoding="utf-8") == "alpha\nBETA\ngamma\nDELTA\n"


def test_patch_file_tool_rejects_context_mismatch_without_writing(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))

    tool = PatchFileTool.make(session, ["sample.txt", "@@\n alpha\n-missing\n+MISSING\n gamma\n"])

    assert "hunk 1 context did not match" in tool.preview()
    assert "first old line: 'alpha'" in tool.preview()
    with pytest.raises(ToolCallError, match="hunk 1 context did not match"):
        tool.call()
    assert path.read_text(encoding="utf-8") == "alpha\nbeta\ngamma\n"


def test_patch_file_tool_rejects_ambiguous_context_without_writing(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\nbeta\nalpha\nbeta\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))

    tool = PatchFileTool.make(session, ["sample.txt", "@@\n alpha\n-beta\n+BETA\n"])

    with pytest.raises(ToolCallError, match="matched multiple locations"):
        tool.call()
    assert path.read_text(encoding="utf-8") == "alpha\nbeta\nalpha\nbeta\n"


def test_patch_file_tool_rejects_overlapping_hunks_without_writing(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))

    tool = PatchFileTool.make(
        session,
        [
            "sample.txt",
            "@@\n alpha\n-beta\n+BETA\n@@\n-beta\n-gamma\n+GAMMA\n",
        ],
    )

    with pytest.raises(ToolCallError, match="overlap"):
        tool.call()
    assert path.read_text(encoding="utf-8") == "alpha\nbeta\ngamma\n"


def test_patch_file_tool_rejects_malformed_patch(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))

    with pytest.raises(ToolCallError, match="patch content before first hunk"):
        PatchFileTool.make(session, ["sample.txt", "alpha\n"]).call()


def test_agent_executes_patch_file_and_requires_verification(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)

    latest = agent.execute_tool_calls(
        [
            {
                "name": "PatchFile",
                "intention": "patch sample",
                "args": ["sample.txt", "@@\n alpha\n-beta\n+BETA\n gamma\n"],
            }
        ],
        confirm=lambda call, tool: True,
    )

    assert path.read_text(encoding="utf-8") == "alpha\nBETA\ngamma\n"
    assert "<PatchFileToolResult>" in latest
    assert agent.blackboard.verification_required is True
