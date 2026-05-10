import pytest

from nanocode import ApplyPatchTool, Session, ToolCallError


def test_apply_patch_tool_applies_single_file_unified_diff(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    patch = "@@ -1,3 +1,3 @@\n alpha\n-beta\n+BETA\n gamma\n"

    tool = ApplyPatchTool.make(session, ["sample.txt", patch])
    display = tool.preview()
    result = tool.call()

    assert tool.requires_confirmation(session) is True
    assert "-beta\n" in display
    assert "+BETA\n" in display
    assert path.read_text(encoding="utf-8") == "alpha\nBETA\ngamma\n"
    assert result == "\n".join(
        [
            "<ApplyPatchToolResult>",
            "* path: sample.txt",
            "* hunks: 1",
            "</ApplyPatchToolResult>",
        ]
    )


def test_apply_patch_tool_creates_missing_file_from_empty_unified_diff(tmp_path):
    path = tmp_path / "created.txt"
    session = Session(cwd=str(tmp_path))
    patch = "--- /dev/null\n+++ b/created.txt\n@@ -0,0 +1,2 @@\n+alpha\n+beta\n"

    tool = ApplyPatchTool.make(session, ["created.txt", patch])
    display = tool.preview()
    result = tool.call()

    assert "+alpha\n" in display
    assert "+beta\n" in display
    assert path.read_text(encoding="utf-8") == "alpha\nbeta\n"
    assert result == "\n".join(
        [
            "<ApplyPatchToolResult>",
            "* path: created.txt",
            "* hunks: 1",
            "* created: true",
            "</ApplyPatchToolResult>",
        ]
    )


def test_apply_patch_tool_creates_missing_file_from_codex_add_file_patch(tmp_path):
    path = tmp_path / "created.txt"
    session = Session(cwd=str(tmp_path))
    patch = (
        "*** Begin Patch\n"
        "*** Add File: created.txt\n"
        "+alpha\n"
        "+beta\n"
        "*** End Patch\n"
    )

    tool = ApplyPatchTool.make(session, ["created.txt", patch])
    result = tool.call()

    assert path.read_text(encoding="utf-8") == "alpha\nbeta\n"
    assert "* created: true" in result


def test_apply_patch_tool_rejects_codex_add_file_when_target_exists(tmp_path):
    path = tmp_path / "created.txt"
    path.write_text("existing\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    patch = (
        "*** Begin Patch\n"
        "*** Add File: created.txt\n"
        "+alpha\n"
        "*** End Patch\n"
    )

    tool = ApplyPatchTool.make(session, ["created.txt", patch])

    with pytest.raises(ToolCallError, match="Add File patch target already exists"):
        tool.call()
    assert path.read_text(encoding="utf-8") == "existing\n"


def test_apply_patch_tool_rejects_mismatched_context(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\nbeta\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    patch = "@@ -1,2 +1,2 @@\n alpha\n-missing\n+MISSING\n"

    tool = ApplyPatchTool.make(session, ["sample.txt", patch])

    with pytest.raises(ToolCallError, match="hunk context did not match"):
        tool.call()
    assert path.read_text(encoding="utf-8") == "alpha\nbeta\n"


def test_apply_patch_tool_finds_unique_context_when_hunk_line_number_is_stale(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("prefix\nalpha\nbeta\ngamma\nsuffix\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    patch = "@@ -99,3 +99,3 @@\n alpha\n-beta\n+BETA\n gamma\n"

    tool = ApplyPatchTool.make(session, ["sample.txt", patch])
    display = tool.preview()
    result = tool.call()

    assert "# preview unavailable" not in display
    assert "-beta\n" in display
    assert "+BETA\n" in display
    assert path.read_text(encoding="utf-8") == "prefix\nalpha\nBETA\ngamma\nsuffix\n"
    assert "* hunks: 1" in result


def test_apply_patch_tool_applies_bare_fuzzy_hunk_and_previews_diff(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    patch = "@@\n-beta\n+BETA\n"

    tool = ApplyPatchTool.make(session, ["sample.txt", patch])
    display = tool.preview()
    result = tool.call()

    assert "# preview unavailable" not in display
    assert "-beta\n" in display
    assert "+BETA\n" in display
    assert path.read_text(encoding="utf-8") == "alpha\nBETA\ngamma\n"
    assert "* hunks: 1" in result


def test_apply_patch_tool_accepts_codex_style_update_file_patch(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    patch = (
        "*** Begin Patch\n"
        "*** Update File: sample.txt\n"
        "@@ around beta\n"
        " alpha\n"
        "-beta\n"
        "+BETA\n"
        " gamma\n"
        "*** End Patch\n"
    )

    tool = ApplyPatchTool.make(session, ["sample.txt", patch])
    display = tool.preview()
    result = tool.call()

    assert "# preview unavailable" not in display
    assert "-beta\n" in display
    assert "+BETA\n" in display
    assert path.read_text(encoding="utf-8") == "alpha\nBETA\ngamma\n"
    assert "* hunks: 1" in result


def test_apply_patch_tool_treats_unique_applied_codex_hunk_as_noop(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\nBETA\ngamma\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    patch = (
        "*** Begin Patch\n"
        "*** Update File: sample.txt\n"
        "@@\n"
        " alpha\n"
        "-beta\n"
        "+BETA\n"
        " gamma\n"
        "@@\n"
        " gamma\n"
        "+delta\n"
        "*** End Patch\n"
    )

    tool = ApplyPatchTool.make(session, ["sample.txt", patch])
    display = tool.preview()
    result = tool.call()

    assert "# preview unavailable" not in display
    assert "+delta\n" in display
    assert path.read_text(encoding="utf-8") == "alpha\nBETA\ngamma\ndelta\n"
    assert "* hunks: 2" in result


def test_apply_patch_tool_reports_hunk_number_and_context_for_mismatch(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\nbeta\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    patch = (
        "*** Begin Patch\n"
        "*** Update File: sample.txt\n"
        "@@\n"
        "-missing\n"
        "+MISSING\n"
        "*** End Patch\n"
    )

    tool = ApplyPatchTool.make(session, ["sample.txt", patch])

    with pytest.raises(ToolCallError) as error:
        tool.call()
    message = str(error.value)
    assert "hunk 1: hunk context did not match" in message
    assert "expected:\n-missing" in message
    assert "replacement:\n+MISSING" in message
    assert "hunk 1: hunk context did not match" in tool.preview()
    assert path.read_text(encoding="utf-8") == "alpha\nbeta\n"


def test_apply_patch_tool_rejects_codex_style_patch_for_different_file(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\nbeta\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    patch = (
        "*** Begin Patch\n"
        "*** Update File: other.txt\n"
        "@@\n"
        "-beta\n"
        "+BETA\n"
        "*** End Patch\n"
    )

    tool = ApplyPatchTool.make(session, ["sample.txt", patch])

    with pytest.raises(ToolCallError, match="patch target does not match filepath: other.txt"):
        tool.call()
    assert "patch target does not match filepath: other.txt" in tool.preview()
    assert path.read_text(encoding="utf-8") == "alpha\nbeta\n"


def test_apply_patch_tool_rejects_ambiguous_context_when_hunk_line_number_is_stale(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\nbeta\nalpha\nbeta\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    patch = "@@ -99,2 +99,2 @@\n alpha\n-beta\n+BETA\n"

    tool = ApplyPatchTool.make(session, ["sample.txt", patch])

    with pytest.raises(ToolCallError, match="matched multiple locations"):
        tool.call()
    assert path.read_text(encoding="utf-8") == "alpha\nbeta\nalpha\nbeta\n"


def test_apply_patch_tool_rejects_empty_patch(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))

    with pytest.raises(ToolCallError, match="unified_diff cannot be empty"):
        ApplyPatchTool.make(session, ["sample.txt", ""])


def test_apply_patch_tool_rejects_patch_without_hunks(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))

    tool = ApplyPatchTool.make(session, ["sample.txt", "--- a/sample.txt\n+++ b/sample.txt\n"])

    with pytest.raises(ToolCallError, match="patch has no hunks"):
        tool.call()


def test_apply_patch_tool_rejects_invalid_hunk_header(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))

    tool = ApplyPatchTool.make(session, ["sample.txt", "@@ bad @@\n-alpha\n+beta\n"])

    with pytest.raises(ToolCallError, match="invalid hunk header"):
        tool.call()


def test_apply_patch_tool_display_reports_unavailable_preview_for_invalid_patch(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))

    tool = ApplyPatchTool.make(session, ["sample.txt", "@@bad\n-alpha\n+beta\n"])

    display = tool.preview()

    assert display.startswith("ApplyPatch(")
    assert "unified_diff=..." in display
    assert "# preview unavailable: invalid hunk header" in display
    assert "-alpha" not in display
