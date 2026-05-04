import nanocode
import pytest

from nanocode import SearchTool, Session, ToolCallError


def test_search_tool_python_backend_finds_or_patterns_and_applies_glob(tmp_path, monkeypatch):
    (tmp_path / ".gitignore").write_text("ignored.txt\nignored_dir/\n", encoding="utf-8")
    (tmp_path / "keep.txt").write_text("alpha needle\nsecond hit\n", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("needle ignored by gitignore\n", encoding="utf-8")
    (tmp_path / ".hidden.txt").write_text("needle hidden file\n", encoding="utf-8")
    (tmp_path / "skip.py").write_text("needle in python\n", encoding="utf-8")
    ignored_dir = tmp_path / "ignored_dir"
    ignored_dir.mkdir()
    (ignored_dir / "nested.txt").write_text("needle ignored dir\n", encoding="utf-8")
    hidden_dir = tmp_path / ".hidden_dir"
    hidden_dir.mkdir()
    (hidden_dir / "nested.txt").write_text("needle hidden dir\n", encoding="utf-8")
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "hidden.txt").write_text("needle hidden\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    monkeypatch.setattr(nanocode.shutil, "which", lambda name: "")

    tool = SearchTool.make(session, ["needle|second", ".", "*.txt"])
    result = tool.call()

    assert "* engine: python" in result
    assert "* keep.txt:1: alpha needle" in result
    assert "* keep.txt:2: second hit" in result
    assert "skip.py" not in result
    assert "ignored.txt" not in result
    assert "ignored_dir" not in result
    assert ".hidden.txt" not in result
    assert ".hidden_dir" not in result
    assert "hidden.txt" not in result


def test_search_tool_prefers_rg_backend(tmp_path, monkeypatch):
    path = tmp_path / "sample.txt"
    path.write_text("needle\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    tool = SearchTool.make(session, ["needle", "sample.txt"])

    monkeypatch.setattr(nanocode.shutil, "which", lambda name: "/fake/rg" if name == "rg" else "")
    monkeypatch.setattr(SearchTool, "_call_rg", lambda self, rg: f"rg:{rg}")

    assert tool.call() == "rg:/fake/rg"


def test_search_tool_uses_python_when_rg_is_missing(tmp_path, monkeypatch):
    path = tmp_path / "sample.txt"
    path.write_text("needle\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    tool = SearchTool.make(session, ["needle", "sample.txt"])

    monkeypatch.setattr(nanocode.shutil, "which", lambda name: "")

    result = tool.call()

    assert "* engine: python" in result
    assert "* sample.txt:1: needle" in result
    assert "  > 1: needle" in result


def test_search_tool_python_backend_includes_two_context_lines(tmp_path, monkeypatch):
    path = tmp_path / "sample.txt"
    path.write_text("one\ntwo\nthree\nneedle\nfive\nsix\nseven\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    monkeypatch.setattr(nanocode.shutil, "which", lambda name: "")

    result = SearchTool.make(session, ["needle", "sample.txt"]).call()

    assert "* sample.txt:4: needle" in result
    assert "    1: one" not in result
    assert "    2: two" in result
    assert "    3: three" in result
    assert "  > 4: needle" in result
    assert "    5: five" in result
    assert "    6: six" in result
    assert "    7: seven" not in result


def test_search_tool_python_backend_supports_regex(tmp_path, monkeypatch):
    path = tmp_path / "sample.py"
    path.write_text(
        "class One:\n"
        "    def __init__(self):\n"
        "        pass\n"
        "class Two:\n"
        "    def __init__(self, name):\n"
        "        pass\n",
        encoding="utf-8",
    )
    session = Session(cwd=str(tmp_path))
    monkeypatch.setattr(nanocode.shutil, "which", lambda name: "")

    result = SearchTool.make(session, [r"re:def __init__\([^)]*,[^)]*\)", "sample.py"]).call()

    assert "* engine: python" in result
    assert "* sample.py:5:     def __init__(self, name):" in result
    assert "* sample.py:2:     def __init__(self):" not in result


def test_search_tool_supports_context_option_without_glob(tmp_path, monkeypatch):
    path = tmp_path / "sample.txt"
    path.write_text("one\ntwo\nthree\nneedle\nfive\nsix\nseven\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    monkeypatch.setattr(nanocode.shutil, "which", lambda name: "")

    result = SearchTool.make(session, ["needle", "sample.txt", "context=3"]).call()

    assert "    1: one" in result
    assert "    2: two" in result
    assert "    3: three" in result
    assert "  > 4: needle" in result
    assert "    5: five" in result
    assert "    6: six" in result
    assert "    7: seven" in result


def test_search_tool_supports_glob_and_context_option(tmp_path, monkeypatch):
    (tmp_path / "keep.txt").write_text("before\nneedle\nafter\n", encoding="utf-8")
    (tmp_path / "skip.py").write_text("before\nneedle\nafter\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    monkeypatch.setattr(nanocode.shutil, "which", lambda name: "")

    result = SearchTool.make(session, ["needle", ".", "*.txt", "context=1"]).call()

    assert "* keep.txt:2: needle" in result
    assert "  > 2: needle" in result
    assert "skip.py" not in result


def test_search_tool_rejects_empty_pattern(tmp_path):
    session = Session(cwd=str(tmp_path))

    with pytest.raises(ToolCallError, match="pattern cannot be empty"):
        SearchTool.make(session, ["", "."])


def test_search_tool_rejects_invalid_regex(tmp_path):
    session = Session(cwd=str(tmp_path))

    with pytest.raises(ToolCallError, match="invalid regex"):
        SearchTool.make(session, ["re:[", "."])


def test_search_tool_rejects_invalid_context(tmp_path):
    session = Session(cwd=str(tmp_path))

    with pytest.raises(ToolCallError, match="context must be an integer"):
        SearchTool.make(session, ["needle", ".", "context=bad"])


def test_search_tool_rejects_missing_target(tmp_path):
    session = Session(cwd=str(tmp_path))
    tool = SearchTool.make(session, ["needle", "missing.txt"])

    with pytest.raises(ToolCallError, match="not a file or directory"):
        tool.call()


def test_search_tool_returns_no_matches_for_glob_mismatch(tmp_path, monkeypatch):
    (tmp_path / "sample.py").write_text("needle\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    monkeypatch.setattr(nanocode.shutil, "which", lambda name: "")

    result = SearchTool.make(session, ["needle", "sample.py", "*.txt"]).call()

    assert result == "\n".join(
        [
            "<SearchToolResult>",
            "* engine: python",
            "No matches.",
            "</SearchToolResult>",
        ]
    )


def test_search_tool_truncates_python_results(tmp_path, monkeypatch):
    path = tmp_path / "sample.txt"
    path.write_text("needle 1\nneedle 2\nneedle 3\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    monkeypatch.setattr(nanocode.shutil, "which", lambda name: "")
    monkeypatch.setattr(SearchTool, "MAX_MATCHES", 2)

    result = SearchTool.make(session, ["needle", "sample.txt"]).call()

    assert "* sample.txt:1: needle 1" in result
    assert "* sample.txt:2: needle 2" in result
    assert "* sample.txt:3: needle 3" not in result
    assert "* truncated: true" in result


def test_search_tool_python_backend_honors_gitignore_glob(tmp_path, monkeypatch):
    (tmp_path / ".gitignore").write_text("*.log\n", encoding="utf-8")
    (tmp_path / "keep.txt").write_text("needle\n", encoding="utf-8")
    (tmp_path / "skip.log").write_text("needle hidden by gitignore glob\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    monkeypatch.setattr(nanocode.shutil, "which", lambda name: "")

    result = SearchTool.make(session, ["needle", "."]).call()

    assert "keep.txt" in result
    assert "skip.log" not in result
