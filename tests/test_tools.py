import shutil
import subprocess

import pytest

import nanocode as n


def session(tmp_path):
    return n.Session(cwd=str(tmp_path))


def anchor(index, line):
    return f"{index}:{n.ReadTool.line_hash(line)}"


def test_read_linecount_list_search_success_paths(tmp_path):
    (tmp_path / "sample.py").write_text("alpha\nNeedle\nomega\n", encoding="utf-8")
    (tmp_path / "blob.bin").write_bytes(b"a\0b")
    s = session(tmp_path)

    read = n.ReadTool(s, [{"path": "sample.py", "ranges": [[0, 2], [2, 0]]}]).call()
    alpha_hash = n.ReadTool.line_hash("alpha\n")
    needle_hash = n.ReadTool.line_hash("Needle\n")
    omega_hash = n.ReadTool.line_hash("omega\n")
    assert f"0:{alpha_hash}|alpha" in read
    assert f"1:{needle_hash}|Needle" in read
    assert f"2:{omega_hash}|omega" in read

    counts = n.LineCountTool(s, ["sample.py", "missing.py"]).call()
    assert "<total>3</total>" in counts
    assert "missing.py" in counts

    listed = n.ListTool(s, ["."]).call()
    assert "file text: sample.py" in listed
    assert "file binary: blob.bin" in listed

    found = n.SearchTool(s, [{"pattern": "needle", "path": "."}]).call()
    assert f"sample.py:1:{needle_hash}|Needle" in found

    multiline = n.SearchTool(s, [{"pattern": "alpha\\nNeedle", "path": "sample.py"}]).call()
    assert "sample.py:0:" in multiline


def test_search_ignores_hidden_and_gitignored_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(n.shutil, "which", lambda name: None)
    (tmp_path / ".gitignore").write_text("ignored.txt\nignored_dir/\n", encoding="utf-8")
    (tmp_path / "visible.txt").write_text("needle\n", encoding="utf-8")
    (tmp_path / ".hidden.txt").write_text("needle\n", encoding="utf-8")
    (tmp_path / ".hidden_dir").mkdir()
    (tmp_path / ".hidden_dir" / "inside.txt").write_text("needle\n", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("needle\n", encoding="utf-8")
    (tmp_path / "ignored_dir").mkdir()
    (tmp_path / "ignored_dir" / "inside.txt").write_text("needle\n", encoding="utf-8")
    s = session(tmp_path)

    found = n.SearchTool(s, [{"pattern": "needle", "path": "."}]).call()
    direct_hidden = n.SearchTool(s, [{"pattern": "needle", "path": ".hidden.txt"}]).call()
    direct_ignored = n.SearchTool(s, [{"pattern": "needle", "path": "ignored_dir/inside.txt"}]).call()

    assert "visible.txt:0:" in found
    assert ".hidden" not in found
    assert "ignored" not in found
    assert ".hidden.txt:0:" not in direct_hidden
    assert "ignored_dir/inside.txt:0:" not in direct_ignored


def test_tool_validation_rejects_bad_shapes_without_side_effects(tmp_path):
    s = session(tmp_path)
    (tmp_path / "sample.py").write_text("alpha\n", encoding="utf-8")

    with pytest.raises(n.ToolError):
        n.ReadTool(s, [{"path": "sample.py"}]).call()
    with pytest.raises(n.ToolError):
        n.CreateFileTool(s, [["a.txt", "a"], ["b.txt", "b"]]).call()
    with pytest.raises(n.ToolError):
        n.BashTool(s, []).call()
    with pytest.raises(n.ToolError):
        n.SearchTool(s, [{"pattern": "["}]).call()
    with pytest.raises(n.ToolError):
        n.GitTool(s, ["cwd=..", "status"]).call()
    with pytest.raises(n.ToolError):
        n.InspectCodeTool(s, ["inspect", "two words"]).call()

    assert not (tmp_path / "a.txt").exists()
    assert not (tmp_path / "b.txt").exists()


def test_createfile_and_edit_operations(tmp_path):
    s = session(tmp_path)
    n.CreateFileTool(s, ["nested/note.txt", "one\ntwo\nthree\n"]).call()
    path = tmp_path / "nested" / "note.txt"
    assert path.read_text(encoding="utf-8") == "one\ntwo\nthree\n"

    with pytest.raises(n.ToolError):
        n.CreateFileTool(s, ["nested/note.txt", "again\n"]).call()

    n.EditTool(
        s,
        [
            "nested/note.txt",
            [
                {"op": "replace", "start": anchor(0, "one\n"), "end": anchor(0, "one\n"), "content": "ONE\n"},
                {"op": "insert_after", "start": anchor(1, "two\n"), "content": "TWO-AND-HALF\n"},
                {"op": "delete", "start": anchor(2, "three\n"), "end": anchor(2, "three\n")},
            ],
        ],
    ).call()
    assert path.read_text(encoding="utf-8") == "ONE\ntwo\nTWO-AND-HALF\n"

    n.EditTool(s, ["nested/note.txt", [{"op": "replace_all", "old": "TWO", "new": "two"}]]).call()
    assert path.read_text(encoding="utf-8") == "ONE\ntwo\ntwo-AND-HALF\n"

    with pytest.raises(n.ToolError):
        n.EditTool(s, ["nested/note.txt", [{"op": "replace", "start": anchor(0, "one\n"), "end": anchor(0, "one\n"), "content": "bad\n"}]]).call()


def test_bash_and_git_behaviors(tmp_path):
    s = session(tmp_path)
    bash = n.BashTool(s, ["printf out; printf err >&2; exit 3"]).call()
    assert "* exit_code: 3" in bash
    assert "<stdout>\nout\n</stdout>" in bash
    assert "<stderr>\nerr\n</stderr>" in bash

    if not shutil.which("git"):
        pytest.skip("git unavailable")
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True, text=True)
    (tmp_path / "sub").mkdir()
    git = n.GitTool(s, ["cwd=sub", "rev-parse", "--show-toplevel"]).call()
    assert str(tmp_path) in git
    assert not n.GitTool(s, ["status"]).needs_confirmation()
    assert n.GitTool(s, ["commit"]).needs_confirmation()


def test_inspect_code_modes_call_symbol_index_api(tmp_path, monkeypatch):
    s = session(tmp_path)
    (tmp_path / "sample.py").write_text("class Example:\n    pass\n", encoding="utf-8")
    calls = []

    monkeypatch.setattr(n.CodeIndex, "available", lambda self: True)
    monkeypatch.setattr(n.csi, "search", lambda query, **kwargs: calls.append(("search", query, kwargs)) or "search ok")
    monkeypatch.setattr(n.csi, "inspect", lambda query, **kwargs: calls.append(("inspect", query, kwargs)) or "inspect ok")
    monkeypatch.setattr(n.csi, "outline", lambda path, **kwargs: calls.append(("outline", path, kwargs)) or "outline ok")

    assert "search ok" in n.InspectCodeTool(s, ["find", "Example", {"kind": "class,function", "limit": 10, "exact_only": True}]).call()
    assert "inspect ok" in n.InspectCodeTool(s, ["inspect", "Example", {"path": "sample.py"}]).call()
    assert "outline ok" in n.InspectCodeTool(s, ["outline", "sample.py"]).call()

    assert calls[0] == (
        "search",
        "Example",
        {"root": str(tmp_path), "kind": "class,function", "path": None, "exact_only": True, "format": "text", "limit": 10},
    )
    assert calls[1] == (
        "inspect",
        "Example",
        {"root": str(tmp_path), "kind": None, "path": "sample.py", "exact_only": False, "format": "text", "limit": n.csi.DEFAULT_PAGE_LIMIT, "anchors": True},
    )
    assert calls[2] == (
        "outline",
        "sample.py",
        {"root": str(tmp_path), "symbol": None, "max_symbols": n.csi.DEFAULT_MAX_OUTLINE_SYMBOLS, "format": "text"},
    )

    with pytest.raises(n.ToolError):
        n.InspectCodeTool(s, ["outline", "missing.py"]).call()
    with pytest.raises(n.ToolError):
        n.InspectCodeTool(s, ["inspect", "sample.py"]).call()


def test_inspect_code_api_errors_return_failed_result(tmp_path, monkeypatch):
    s = session(tmp_path)
    monkeypatch.setattr(n.CodeIndex, "available", lambda self: True)
    monkeypatch.setattr(n.csi, "search", lambda *args, **kwargs: (_ for _ in ()).throw(n.csi.CodeSymbolIndexError("bad query")))

    result = n.InspectCodeTool(s, ["find", "Missing"]).call()

    assert "* exit_code: 1" in result
    assert "bad query" in result


def test_recall_and_forget_behaviors(tmp_path):
    s = session(tmp_path)
    first = s.store_tool_result("Read", ["a.txt"], "read", "a0\na1\na2\n")
    second = s.store_tool_result("Search", [{"pattern": "b"}], "search", "b0\nb1\n")

    sliced = n.RecallTool(s, [{"keys": [first, second], "ranges": [[1, 2]]}]).call()
    assert "a1" in sliced and "a0" not in sliced
    assert "b1" in sliced and "b0" not in sliced

    common_range = n.RecallTool(s, [first, "0:1"]).call()
    assert "a0" in common_range and "a1" not in common_range

    with pytest.raises(n.ToolError):
        n.RecallTool(s, [{"key": first, "ranges": [[2, "bad"]]}]).call()

    n.ForgetTool(s, [first, first, "tr.999"]).call()
    assert first not in s.tool_results
    assert second in s.tool_results


def test_tool_runner_short_call_formats_search_and_recall(tmp_path):
    s = session(tmp_path)
    runner = n.ToolRunner(s, n.ContextManager(s), output_fn=lambda text: None)

    search = runner.short_call(
        n.ToolCall(
            "s",
            "Search",
            [
                {"pattern": "done in", "glob": "*.py"},
                {"pattern": "elapsed.*s]", "path": "tests", "context": 2},
            ],
            "",
        )
    )
    assert search == 'Search "done in" glob=*.py; "elapsed.*s]" path=tests C=2'

    recall = runner.short_call(n.ToolCall("r", "Recall", [{"keys": ["tr.4", "tr.5"], "ranges": [[0, 80]]}], ""))
    assert recall == "Recall tr.4 0:80; tr.5 0:80"


def test_tool_schemas_are_strict_for_high_risk_tools():
    bash_args = n.BashTool.schema()["function"]["parameters"]["properties"]["args"]
    assert bash_args["minItems"] == 1
    assert bash_args["maxItems"] == 1
    assert bash_args["items"]["pattern"] == r"\S"

    create_args = n.CreateFileTool.schema()["function"]["parameters"]["properties"]["args"]
    assert create_args["minItems"] == 2
    assert create_args["maxItems"] == 2
    assert create_args["items"] == {"type": "string"}
    assert "prefixItems" not in create_args

    recall_item = n.RecallTool.schema()["function"]["parameters"]["properties"]["args"]["items"]
    assert recall_item["type"] == "object"
    assert recall_item["additionalProperties"] is False
    assert recall_item["properties"]["keys"]["items"]["pattern"] == r"^tr\.\d+$"

    forget_args = n.ForgetTool.schema()["function"]["parameters"]["properties"]["args"]
    assert forget_args["items"]["pattern"] == r"^tr\.\d+$"

    def walk(value):
        if isinstance(value, dict):
            assert "anyOf" not in value
            assert "prefixItems" not in value
            if "items" in value:
                assert isinstance(value["items"], dict)
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    for tool in n.TOOLS:
        walk(tool.schema())


def test_edit_rejects_overlaps_and_mixed_modes(tmp_path):
    s = session(tmp_path)
    path = tmp_path / "code.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")

    with pytest.raises(n.ToolError):
        n.EditTool(
            s,
            [
                "code.txt",
                [
                    {"op": "replace", "start": anchor(0, "a\n"), "end": anchor(1, "b\n"), "content": "x\n"},
                    {"op": "delete", "start": anchor(1, "b\n"), "end": anchor(1, "b\n")},
                ],
            ],
        ).call()
    assert path.read_text(encoding="utf-8") == "a\nb\nc\n"

    with pytest.raises(n.ToolError):
        n.EditTool(
            s,
            [
                "code.txt",
                [
                    {"op": "replace_all", "old": "a", "new": "A"},
                    {"op": "insert_before", "start": anchor(1, "b\n"), "content": "x\n"},
                ],
            ],
        ).call()
    assert path.read_text(encoding="utf-8") == "a\nb\nc\n"


def test_edit_inserts_before_existing_line_with_needed_newline(tmp_path):
    s = session(tmp_path)
    path = tmp_path / "code.txt"
    path.write_text("a\nb\n", encoding="utf-8")

    n.EditTool(s, ["code.txt", [{"op": "insert_before", "start": anchor(1, "b\n"), "content": "inserted"}]]).call()
    assert path.read_text(encoding="utf-8") == "a\ninserted\nb\n"


def test_bash_timeout_and_live_output(tmp_path):
    s = session(tmp_path)
    s.settings.shell_timeout = 1
    events = []
    tool = n.BashTool(s, ["printf live; sleep 5"])
    tool.live_output = lambda stream, text: events.append((stream, text))

    output = tool.call()

    assert "* exit_code: -1" in output
    assert "live" in output
    assert "timeout" in output
    assert ("stdout", "live") in events
    assert events[-1] == ("", "")


def test_code_index_updates_after_file_mutation_tools(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    updated = []
    monkeypatch.setattr(n.CodeIndex, "update", lambda self, paths: updated.extend(paths) or "")
    runner = n.ToolRunner(s, n.ContextManager(s), input_fn=lambda prompt: (_ for _ in ()).throw(AssertionError("unexpected prompt")), output_fn=lambda text: None)

    runner.run([n.ToolCall("create", "CreateFile", ["made.py", "print(1)\n"], "create file")])
    runner.run([n.ToolCall("edit", "Edit", ["made.py", [{"op": "replace_all", "old": "1", "new": "2"}]], "edit file")])

    assert (tmp_path / "made.py").read_text(encoding="utf-8") == "print(2)\n"
    assert updated == ["made.py", "made.py"]


def test_yolo_approves_mutating_tools_without_prompt(tmp_path):
    s = session(tmp_path)
    s.settings.yolo = True
    runner = n.ToolRunner(s, n.ContextManager(s), input_fn=lambda prompt: (_ for _ in ()).throw(AssertionError("unexpected prompt")), output_fn=lambda text: None)

    runner.run([n.ToolCall("create", "CreateFile", ["auto.txt", "ok\n"], "create file")])

    assert (tmp_path / "auto.txt").read_text(encoding="utf-8") == "ok\n"
    assert len(s.tool_records) == 1
    assert s.tool_errors == []
