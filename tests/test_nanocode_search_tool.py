import json
import re

import nanocode
import pytest

from nanocode import EditTool, SearchTool, Session, ToolCallError


def _search(pattern: str, *, path: str | None = None, glob: str | None = None, context: int | object | None = None, **extra: object):
    spec: dict[str, object] = {"pattern": pattern, **extra}
    if path is not None:
        spec["path"] = path
    if glob is not None:
        spec["glob"] = glob
    if context is not None:
        spec["context"] = context
    return [spec]


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

    tool = SearchTool.make(session, _search("needle|second", path=".", glob="*.txt"))
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


def test_search_tool_rejects_positional_args(tmp_path):
    session = Session(cwd=str(tmp_path))

    with pytest.raises(ToolCallError, match="Search args error: expected one object or multiple search objects"):
        SearchTool.make(session, ["class Edit", "class Bash", "class Search", "class Read", "class CreateFile"])


def test_search_tool_uses_structured_path(tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("class EditTool:\nclass BashTool:\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))

    tool = SearchTool.make(session, _search("class Edit|class Bash", path="sample.py"))

    assert tool.pattern == "class Edit|class Bash"
    assert tool.target_path == str(path)


def test_search_tool_reads_multiple_search_objects_as_args(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("version = \"1\"\n", encoding="utf-8")
    (tmp_path / "setup.cfg").write_text("version = 1\n", encoding="utf-8")
    (tmp_path / "tox.ini").write_text("version = 1\n", encoding="utf-8")
    (tmp_path / "skip.py").write_text("version = 1\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    monkeypatch.setattr(nanocode.shutil, "which", lambda name: "")

    tool = SearchTool.make(
        session,
        [
            {"pattern": "version", "glob": "*.toml"},
            {"pattern": "version", "glob": "*.cfg"},
            {"pattern": "version", "glob": "*.ini"},
        ],
    )
    result = tool.call()

    assert [request.glob_pattern for request in tool.requests] == ["*.toml", "*.cfg", "*.ini"]
    assert "* query_count: 3" in result
    assert "* pyproject.toml:1: version = \"1\"" in result
    assert "* setup.cfg:1: version = 1" in result
    assert "* tox.ini:1: version = 1" in result
    assert "skip.py" not in result


def test_search_tool_reads_stringified_search_objects_as_args(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("version = \"1\"\n", encoding="utf-8")
    (tmp_path / "setup.cfg").write_text("version = 1\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    monkeypatch.setattr(nanocode.shutil, "which", lambda name: "")

    tool = SearchTool.make(
        session,
        [
            json.dumps({"pattern": "version", "glob": "*.toml"}),
            json.dumps({"pattern": "version", "glob": "*.cfg"}),
        ],
    )
    result = tool.call()

    assert [request.glob_pattern for request in tool.requests] == ["*.toml", "*.cfg"]
    assert "* query_count: 2" in result
    assert "* pyproject.toml:1: version = \"1\"" in result
    assert "* setup.cfg:1: version = 1" in result


def test_search_tool_formats_stringified_objects_as_readable_cli_args():
    args = [
        json.dumps({"pattern": "version", "glob": "*.toml"}),
        json.dumps({"pattern": "version", "glob": "*.cfg"}),
        json.dumps({"pattern": "version", "glob": "*.ini"}),
    ]

    assert SearchTool.cli_args(args) == ["version", "glob=*.toml", "|", "version", "glob=*.cfg", "|", "version", "glob=*.ini"]


def test_search_tool_accepts_structured_path_with_regex_and_context(tmp_path, monkeypatch):
    path = tmp_path / "nanocode.py"
    path.write_text("class EditTool:\nclass BashTool:\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    monkeypatch.setattr(nanocode.shutil, "which", lambda name: "")

    tool = SearchTool.make(session, _search("class .*Tool", path="nanocode.py", context=0))
    result = tool.call()

    assert tool.target_path == str(path)
    assert tool.context_lines == 0
    assert "* nanocode.py:1: class EditTool:" in result
    assert "* nanocode.py:2: class BashTool:" in result


def test_search_tool_uses_default_context_when_omitted(tmp_path):
    path = tmp_path / "nanocode.py"
    path.write_text("class EditTool:\nclass BashTool:\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))

    tool = SearchTool.make(session, _search("class Edit", path="nanocode.py"))

    assert tool.target_path == str(path)
    assert tool.context_lines == SearchTool.CONTEXT_LINES


def test_search_tool_accepts_regex_alternatives(tmp_path):
    path = tmp_path / "nanocode.py"
    path.write_text("class EditTool:\nclass BashTool:\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))

    tool = SearchTool.make(session, _search("class Edit|class Bash", path="nanocode.py"))

    assert tool.pattern == "class Edit|class Bash"
    assert tool.target_path == str(path)


def test_search_tool_rejects_ignore_case_option(tmp_path):
    (tmp_path / "sample.py").write_text("Needle\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))

    with pytest.raises(ToolCallError, match="unexpected search option: ignore_case"):
        SearchTool.make(session, _search("needle", ignore_case=True))
    with pytest.raises(ToolCallError, match="unexpected search option: ignore_case"):
        SearchTool.make(session, _search("needle", path="sample.py", ignore_case=True))


def test_search_tool_uses_pipe_as_regex_or(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\nbeta\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))

    tool = SearchTool.make(session, _search("alpha|beta", path="sample.txt"))
    result = tool.call()

    assert "* sample.txt:1: alpha" in result
    assert "* sample.txt:2: beta" in result


def test_search_tool_prefers_rg_backend(tmp_path, monkeypatch):
    path = tmp_path / "sample.txt"
    path.write_text("needle\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    tool = SearchTool.make(session, _search("needle", path="sample.txt"))

    monkeypatch.setattr(nanocode.shutil, "which", lambda name: "/fake/rg" if name == "rg" else "")
    monkeypatch.setattr(SearchTool, "_call_rg", lambda self, rg: f"rg:{rg}")

    assert tool.call() == "rg:/fake/rg"


def test_search_tool_retries_rg_with_pcre2_for_lookaround(tmp_path, monkeypatch):
    path = tmp_path / "sample.py"
    path.write_text("Session()\nPromptSession()\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    calls = []

    def fake_run(cmd, text, capture_output, timeout):
        calls.append(cmd)
        if "--pcre2" not in cmd:
            return nanocode.subprocess.CompletedProcess(
                cmd,
                2,
                "",
                "regex parse error: look-around, including look-ahead and look-behind, is not supported; enable PCRE2 with --pcre2",
            )
        output = nanocode.json.dumps(
            {
                "type": "match",
                "data": {
                    "path": {"text": str(path)},
                    "lines": {"text": "Session()\n"},
                    "line_number": 1,
                },
            }
        )
        return nanocode.subprocess.CompletedProcess(cmd, 0, output + "\n", "")

    monkeypatch.setattr(nanocode.shutil, "which", lambda name: "/fake/rg" if name == "rg" else "")
    monkeypatch.setattr(nanocode.subprocess, "run", fake_run)

    result = SearchTool.make(session, _search(r"(?<!Prompt)Session\(", path="sample.py")).call()

    assert "--pcre2" not in calls[0]
    assert "--pcre2" in calls[1]
    assert "* engine: rg-pcre2" in result
    assert "* sample.py:1: Session()" in result


def test_search_tool_uses_python_when_rg_is_missing(tmp_path, monkeypatch):
    path = tmp_path / "sample.txt"
    path.write_text("needle\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    tool = SearchTool.make(session, _search("needle", path="sample.txt"))

    monkeypatch.setattr(nanocode.shutil, "which", lambda name: "")

    result = tool.call()

    assert "* engine: python" in result
    assert "* sample.txt:1: needle" in result
    assert "  > 0:" in result and "|needle" in result


def test_search_tool_context_anchor_can_drive_edit_file(tmp_path, monkeypatch):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    monkeypatch.setattr(nanocode.shutil, "which", lambda name: "")

    result = SearchTool.make(session, _search("beta", path="sample.txt", context=0)).call()
    anchor = re.search(r">\s+(\d+:[0-9a-f]{6})\|beta", result).group(1)

    EditTool.make(session, ["sample.txt", [{"op": "replace", "start": anchor, "end": anchor, "content": "BETA\n"}]]).call()

    assert path.read_text(encoding="utf-8") == "alpha\nBETA\ngamma\n"


def test_search_tool_python_backend_includes_default_context_lines(tmp_path, monkeypatch):
    path = tmp_path / "sample.txt"
    path.write_text("one\ntwo\nthree\nneedle\nfive\nsix\nseven\neight\nnine\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    monkeypatch.setattr(nanocode.shutil, "which", lambda name: "")

    result = SearchTool.make(session, _search("needle", path="sample.txt")).call()

    assert "* sample.txt:4: needle" in result
    assert "  > 3:" in result and "|needle" in result
    assert "|three" not in result
    assert "|five" not in result
    assert "|one" not in result
    assert "|two" not in result
    assert "|six" not in result
    assert "|nine" not in result


def test_search_tool_python_backend_supports_regex(tmp_path, monkeypatch):
    path = tmp_path / "sample.py"
    path.write_text(
        "class One:\n    def __init__(self):\n        pass\nclass Two:\n    def __init__(self, name):\n        pass\n",
        encoding="utf-8",
    )
    session = Session(cwd=str(tmp_path))
    monkeypatch.setattr(nanocode.shutil, "which", lambda name: "")

    result = SearchTool.make(session, _search(r"def __init__\([^)]*,[^)]*\)", path="sample.py")).call()

    assert "* engine: python" in result
    assert "* sample.py:5:     def __init__(self, name):" in result
    assert "* sample.py:2:     def __init__(self):" not in result


def test_search_tool_supports_context_option_without_glob(tmp_path, monkeypatch):
    path = tmp_path / "sample.txt"
    path.write_text("one\ntwo\nthree\nneedle\nfive\nsix\nseven\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    monkeypatch.setattr(nanocode.shutil, "which", lambda name: "")

    result = SearchTool.make(session, _search("needle", path="sample.txt", context=3)).call()

    assert "    0:" in result and "|one" in result
    assert "    1:" in result and "|two" in result
    assert "    2:" in result and "|three" in result
    assert "  > 3:" in result and "|needle" in result
    assert "    4:" in result and "|five" in result
    assert "    5:" in result and "|six" in result
    assert "    6:" in result and "|seven" in result


def test_search_tool_omits_context_before_outer_excerpt(tmp_path, monkeypatch):
    path = tmp_path / "sample.txt"
    path.write_text(("before " + "x" * 300 + "\nneedle\n") * 4, encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    monkeypatch.setattr(nanocode.shutil, "which", lambda name: "")
    monkeypatch.setattr(SearchTool, "OUTPUT_CHARS", 700)

    result = SearchTool.make(session, _search("needle", path="sample.txt", context=1)).call()

    assert "* context_omitted:" in result
    assert "* sample.txt:2: needle" in result
    assert "|before " not in result
    assert "[tool result excerpt]" not in result


def test_search_tool_accepts_context_30(tmp_path):
    session = Session(cwd=str(tmp_path))

    tool = SearchTool.make(session, _search("needle", path=".", context=30))

    assert tool.context_lines == 30


def test_search_tool_supports_numeric_context_option_with_glob(tmp_path, monkeypatch):
    (tmp_path / "keep.txt").write_text("zero\none\nneedle\nthree\nfour\n", encoding="utf-8")
    (tmp_path / "skip.py").write_text("zero\none\nneedle\nthree\nfour\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    monkeypatch.setattr(nanocode.shutil, "which", lambda name: "")

    result = SearchTool.make(session, _search("needle", path=".", glob="*.txt", context=2)).call()

    assert "* keep.txt:3: needle" in result
    assert "    0:" in result and "|zero" in result
    assert "    1:" in result and "|one" in result
    assert "  > 2:" in result and "|needle" in result
    assert "    3:" in result and "|three" in result
    assert "    4:" in result and "|four" in result
    assert "skip.py" not in result


def test_search_tool_supports_glob_and_context_option(tmp_path, monkeypatch):
    (tmp_path / "keep.txt").write_text("before\nneedle\nafter\n", encoding="utf-8")
    (tmp_path / "skip.py").write_text("before\nneedle\nafter\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    monkeypatch.setattr(nanocode.shutil, "which", lambda name: "")

    result = SearchTool.make(session, _search("needle", path=".", glob="*.txt", context=1)).call()

    assert "* keep.txt:2: needle" in result
    assert "  > 1:" in result and "|needle" in result
    assert "skip.py" not in result


def test_search_tool_accepts_named_glob_option(tmp_path, monkeypatch):
    (tmp_path / "keep.py").write_text("needle\n", encoding="utf-8")
    (tmp_path / "skip.txt").write_text("needle\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    monkeypatch.setattr(nanocode.shutil, "which", lambda name: "")

    result = SearchTool.make(session, _search("needle", path=".", glob="*.py")).call()

    assert "* keep.py:1: needle" in result
    assert "skip.txt" not in result


def test_search_tool_defaults_path_to_cwd_when_omitted(tmp_path):
    (tmp_path / "sample.txt").write_text("needle\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))

    tool = SearchTool.make(session, _search("needle"))

    assert tool.target_path == str(tmp_path)


def test_search_tool_accepts_context_option_without_path(tmp_path, monkeypatch):
    (tmp_path / "sample.txt").write_text("needle\nnearby\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    monkeypatch.setattr(nanocode.shutil, "which", lambda name: "")

    tool = SearchTool.make(session, _search("needle", context=0))
    result = tool.call()

    assert tool.target_path == str(tmp_path)
    assert tool.context_lines == 0
    assert "* sample.txt:1: needle" in result
    assert "nearby" not in result


def test_search_tool_accepts_glob_option_without_path(tmp_path, monkeypatch):
    (tmp_path / "keep.py").write_text("needle\n", encoding="utf-8")
    (tmp_path / "skip.txt").write_text("needle\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    monkeypatch.setattr(nanocode.shutil, "which", lambda name: "")

    tool = SearchTool.make(session, _search("needle", glob="*.py"))
    result = tool.call()

    assert tool.target_path == str(tmp_path)
    assert tool.glob_pattern == "*.py"
    assert "* keep.py:1: needle" in result
    assert "skip.txt" not in result


def test_search_tool_rejects_empty_pattern(tmp_path):
    session = Session(cwd=str(tmp_path))

    with pytest.raises(ToolCallError, match="pattern cannot be empty"):
        SearchTool.make(session, _search("", path="."))


def test_search_tool_treats_empty_path_as_cwd(tmp_path):
    (tmp_path / "sample.txt").write_text("needle\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))

    tool = SearchTool.make(session, _search("needle", path=""))

    assert tool.target_path == str(tmp_path)


def test_search_tool_rejects_invalid_regex(tmp_path):
    session = Session(cwd=str(tmp_path))

    with pytest.raises(ToolCallError, match="invalid regex"):
        SearchTool.make(session, _search("[", path="."))


def test_search_tool_defaults_to_regex(tmp_path):
    (tmp_path / "sample.py").write_text("class SearchTool:\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))

    tool = SearchTool.make(session, _search("class.*Tool", path="sample.py"))
    result = tool.call()

    assert "* sample.py:1: class SearchTool:" in result


def test_search_tool_supports_multiline_regex(tmp_path, monkeypatch):
    (tmp_path / "sample.py").write_text("@dataclass\nclass State:\n    pass\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    monkeypatch.setattr(nanocode.shutil, "which", lambda name: "")

    tool = SearchTool.make(session, _search(r"@dataclass.*\nclass.*State", path="sample.py", context=1))
    result = tool.call()

    assert tool.pattern == "@dataclass.*\nclass.*State"
    assert "* engine: python-multiline" in result
    assert "* sample.py:1: @dataclass class State" in result
    assert "  > 0:" in result and "|@dataclass" in result
    assert "    1:" in result and "|class State:" in result


def test_search_tool_rejects_invalid_context(tmp_path):
    session = Session(cwd=str(tmp_path))

    with pytest.raises(ToolCallError, match="context must be an integer"):
        SearchTool.make(session, _search("needle", path=".", context="bad"))


def test_search_tool_rejects_missing_target(tmp_path):
    session = Session(cwd=str(tmp_path))
    tool = SearchTool.make(session, _search("needle", path="missing.txt"))

    with pytest.raises(ToolCallError, match="not a file or directory"):
        tool.call()


def test_search_tool_uses_structured_path_for_plain_names(tmp_path):
    session = Session(cwd=str(tmp_path))
    tool = SearchTool.make(session, _search("needle", path="TOOLS"))

    assert tool.pattern == "needle"
    assert tool.target_path == str(tmp_path / "TOOLS")


def test_search_tool_rejects_placeholder_path_with_guidance(tmp_path):
    session = Session(cwd=str(tmp_path))
    tool = SearchTool.make(session, _search("needle", path="path", glob="*.py"))

    with pytest.raises(ToolCallError, match='"path" is a placeholder'):
        tool.call()


def test_search_tool_returns_no_matches_for_glob_mismatch(tmp_path, monkeypatch):
    (tmp_path / "sample.py").write_text("needle\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    monkeypatch.setattr(nanocode.shutil, "which", lambda name: "")

    result = SearchTool.make(session, _search("needle", path="sample.py", glob="*.txt")).call()

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

    result = SearchTool.make(session, _search("needle", path="sample.txt")).call()

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

    result = SearchTool.make(session, _search("needle", path=".")).call()

    assert "keep.txt" in result
    assert "skip.log" not in result


def test_search_tool_python_fallback_case_insensitive_normal(tmp_path):
    session = Session(cwd=str(tmp_path))
    (tmp_path / "test.txt").write_text("Hello World\n", encoding="utf-8")

    tool = SearchTool.make(session, _search("hello", path="."))
    assert tool._line_matches("Hello World") is True
    assert tool._line_matches("hello world") is True
    assert tool._line_matches("HELLO WORLD") is True


def test_search_tool_python_fallback_case_insensitive_regex(tmp_path):
    session = Session(cwd=str(tmp_path))
    (tmp_path / "test.txt").write_text("Hello World\n", encoding="utf-8")

    tool = SearchTool.make(session, _search("[h]ello", path="."))
    assert tool._line_matches("Hello World") is True
    assert tool._line_matches("hello world") is True
    assert tool._line_matches("HELLO WORLD") is True


def test_search_tool_rg_backend_is_case_insensitive(tmp_path):
    session = Session(cwd=str(tmp_path))

    tool = SearchTool.make(session, _search("hello", path="."))

    assert "-i" in tool._rg_command("rg")
