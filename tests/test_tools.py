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
    single_range = n.ReadTool(s, [{"path": "sample.py", "ranges": [0, 2]}]).call()
    full_default = n.ReadTool(s, [{"path": "sample.py"}]).call()
    alpha_hash = n.ReadTool.line_hash("alpha\n")
    needle_hash = n.ReadTool.line_hash("Needle\n")
    omega_hash = n.ReadTool.line_hash("omega\n")
    assert f"0:{alpha_hash}|alpha" in read
    assert f"1:{needle_hash}|Needle" in read
    assert f"2:{omega_hash}|omega" in read
    assert f"0:{alpha_hash}|alpha" in single_range
    assert f"1:{needle_hash}|Needle" in single_range
    assert f"2:{omega_hash}|omega" in full_default

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


def test_line_hash_is_short_lowercase_base36(tmp_path):
    line_hash = n.ReadTool.line_hash("alpha\n")
    assert len(line_hash) == 5
    assert line_hash == line_hash.lower()
    assert set(line_hash) <= set("0123456789abcdefghijklmnopqrstuvwxyz")


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


def test_find_files_dirs_limits_and_ignores(tmp_path):
    (tmp_path / ".gitignore").write_text("ignored.py\nignored_dir/\n", encoding="utf-8")
    (tmp_path / "app.py").write_text("", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_app.py").write_text("", encoding="utf-8")
    (tmp_path / ".hidden.py").write_text("", encoding="utf-8")
    (tmp_path / "ignored.py").write_text("", encoding="utf-8")
    (tmp_path / "ignored_dir").mkdir()
    (tmp_path / "ignored_dir" / "keep.py").write_text("", encoding="utf-8")
    s = session(tmp_path)

    found = n.FindTool(s, [{"name": "*.py", "path": ".", "limit": 10}]).call()
    limited = n.FindTool(s, [{"name": "*.py", "path": ".", "limit": 1}]).call()
    dirs = n.FindTool(s, [{"name": "test*", "path": ".", "type": "dir"}]).call()

    assert "* file: app.py" in found
    assert "* file: tests/test_app.py" in found
    assert ".hidden" not in found
    assert "ignored" not in found
    assert 'matches=2' in limited
    assert "* omitted: 1" in limited
    assert "* dir: tests/" in dirs
    assert n.FindTool(s, [{"name": "*", "path": str(tmp_path.parent)}]).needs_confirmation()


def test_tool_validation_rejects_bad_shapes_without_side_effects(tmp_path):
    s = session(tmp_path)
    (tmp_path / "sample.py").write_text("alpha\n", encoding="utf-8")

    with pytest.raises(n.ToolError):
        n.ReadTool(s, [{"path": "sample.py", "ranges": []}]).call()
    with pytest.raises(n.ToolError):
        n.EditTool(s, ["a.txt", [{"op": "replace_all", "old": "", "new": "a\n"}]]).call()
    with pytest.raises(n.ToolError):
        n.BashTool(s, []).call()
    with pytest.raises(n.ToolError):
        n.SearchTool(s, [{"pattern": "["}]).call()
    with pytest.raises(n.ToolError):
        n.FindTool(s, [{"name": "*.py", "type": "bad"}]).call()
    with pytest.raises(n.ToolError):
        n.GitTool(s, ["cwd=..", "status"]).call()
    with pytest.raises(n.ToolError):
        n.InspectCodeTool(s, ["inspect", "two words"]).call()

    assert not (tmp_path / "a.txt").exists()
    assert not (tmp_path / "b.txt").exists()


def test_edit_creates_and_patches_file(tmp_path):
    s = session(tmp_path)
    n.TouchTool(s, ["empty/keep.txt"]).call()
    assert (tmp_path / "empty" / "keep.txt").read_text(encoding="utf-8") == ""
    n.TouchTool(s, ["empty/keep.txt"]).call()
    n.EditTool(s, ["empty/keep.txt", [{"op": "replace_all", "old": "", "new": "kept\n"}]]).call()
    assert (tmp_path / "empty" / "keep.txt").read_text(encoding="utf-8") == "kept\n"

    n.EditTool(s, ["nested/note.txt", [{"op": "replace_all", "old": "", "new": "one\ntwo\nthree\n"}], True]).call()
    path = tmp_path / "nested" / "note.txt"
    assert path.read_text(encoding="utf-8") == "one\ntwo\nthree\n"

    with pytest.raises(n.ToolError):
        n.EditTool(s, ["missing.txt", [{"op": "replace_all", "old": "", "new": "again\n"}]]).call()

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
        n.EditTool(s, ["nested/note.txt", [{"op": "replace_all", "old": "", "new": "bad\n"}]]).call()
    with pytest.raises(n.ToolError):
        n.EditTool(s, ["nested/note.txt", [{"op": "replace", "start": anchor(0, "one\n"), "end": anchor(0, "one\n"), "content": "bad\n"}]]).call()


def test_edit_stale_anchor_reports_current_line(tmp_path):
    s = session(tmp_path)
    path = tmp_path / "note.txt"
    path.write_text("old\n", encoding="utf-8")

    with pytest.raises(n.ToolError) as error:
        n.EditTool(s, ["note.txt", [{"op": "replace", "start": anchor(0, "wrong\n"), "end": anchor(0, "wrong\n"), "content": "new\n"}]]).call()

    assert "stale anchor" in str(error.value)
    assert "current is 0:" + n.ReadTool.line_hash("old\n") + "|old" in str(error.value)


def test_edit_create_file_decodes_escaped_newlines_for_preview_and_write(tmp_path):
    s = session(tmp_path)
    tool = n.EditTool(s, ["script.py", [{"op": "replace_all", "old": "", "new": "print(1)\\nprint(2)\\n"}], True])

    preview = tool.preview()
    output = tool.call()

    assert "+print(1)\n+print(2)\n" in preview
    assert (tmp_path / "script.py").read_text(encoding="utf-8") == "print(1)\nprint(2)\n"
    assert "<Edit path=" in output


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
    assert "outline ok" in n.InspectCodeTool(s, ["outline", "sample.py", {"limit": 300}]).call()

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
    assert calls[3] == (
        "outline",
        "sample.py",
        {"root": str(tmp_path), "symbol": None, "max_symbols": 300, "format": "text"},
    )

    with pytest.raises(n.ToolError):
        n.InspectCodeTool(s, ["outline", "missing.py"]).call()
    with pytest.raises(n.ToolError):
        n.InspectCodeTool(s, ["inspect", "sample.py"]).call()
    with pytest.raises(n.ToolError):
        n.InspectCodeTool(s, ["outline", "sample.py", {"limit": 1001}]).call()


def test_inspect_code_api_errors_return_failed_result(tmp_path, monkeypatch):
    s = session(tmp_path)
    monkeypatch.setattr(n.CodeIndex, "available", lambda self: True)
    monkeypatch.setattr(n.csi, "search", lambda *args, **kwargs: (_ for _ in ()).throw(n.csi.CodeSymbolIndexError("bad query")))

    result = n.InspectCodeTool(s, ["find", "Missing"]).call()

    assert "* exit_code: 1" in result
    assert "bad query" in result


def test_recall_behaviors(tmp_path):
    s = session(tmp_path)
    first = s.store_tool_result("Read", ["a.txt"], "a0\na1\na2\n")
    second = s.store_tool_result("Search", [{"pattern": "b"}], "b0\nb1\n")

    sliced = n.RecallTool(s, [{"keys": [first, second], "ranges": [[1, 2]]}]).call()
    assert "a1" in sliced and "a0" not in sliced
    assert "b1" in sliced and "b0" not in sliced

    common_range = n.RecallTool(s, [{"keys": [first], "ranges": [[0, 1]]}]).call()
    assert "a0" in common_range and "a1" not in common_range

    with pytest.raises(n.ToolError):
        n.RecallTool(s, [{"key": first, "ranges": [[2, "bad"]]}]).call()


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
        )
    )
    assert search == 'Search "done in" glob=*.py; "elapsed.*s]" path=tests C=2'

    recall = runner.short_call(n.ToolCall("r", "Recall", [{"keys": ["tr.4", "tr.5"], "ranges": [[0, 80]]}]))
    assert recall == "Recall tr.4 0:80; tr.5 0:80"

    s.state.known = ["existing"]
    note = runner.short_call(n.ToolCall("m", "Note", [{"goal": "ship", "plan": ["inspect", "patch"], "known": ["existing", "new fact"]}]))
    assert note == "Note goal -> ship\nplan:\n  - [~] inspect\n  - [ ] patch\nknown:\n  + new fact"


def test_tool_schemas_are_strict_for_high_risk_tools():
    bash_params = n.BashTool.schema()["function"]["parameters"]
    assert bash_params["required"] == ["command"]
    assert bash_params["properties"]["command"]["pattern"] == r"^.*\S.*$"

    edit_params = n.EditTool.schema()["function"]["parameters"]
    assert edit_params["required"] == ["path", "edits"]
    assert set(edit_params["properties"]) == {"path", "edits", "create_file"}
    assert "start/end anchors are inclusive" in n.EditTool.schema()["function"]["description"]

    recall_keys = n.RecallTool.schema()["function"]["parameters"]["properties"]["keys"]
    assert recall_keys["items"]["pattern"] == r"^tr\.\d+$"

    read_params = n.ReadTool.schema()["function"]["parameters"]
    assert {"path", "ranges", "files"} <= set(read_params["properties"])

    find_params = n.FindTool.schema()["function"]["parameters"]
    assert {"name", "queries"} <= set(find_params["properties"])
    find_item = find_params["properties"]["queries"]["items"]
    assert find_item["properties"]["type"]["enum"] == ["file", "dir", "any"]

    search_params = n.SearchTool.schema()["function"]["parameters"]
    assert {"pattern", "queries"} <= set(search_params["properties"])
    assert n.TouchTool.schema()["function"]["parameters"]["required"] == ["path"]

    def walk(value):
        if isinstance(value, dict):
            assert "anyOf" not in value
            assert "prefixItems" not in value
            if isinstance(value.get("pattern"), str):
                assert value["pattern"].startswith("^")
                assert value["pattern"].endswith("$")
            if "items" in value:
                assert isinstance(value["items"], dict)
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    for tool in n.TOOLS:
        params = tool.schema()["function"]["parameters"]
        assert "args" not in params.get("properties", {})
        walk(tool.schema())


def test_single_and_batch_payload_shapes_are_supported():
    assert n.ModelClient.tool_payload("Read", {"path": "a.py"}) == [{"path": "a.py", "ranges": [[0, 0]]}]
    assert n.ModelClient.tool_payload("Read", {"path": "a.py", "ranges": [0, 2]}) == [{"path": "a.py", "ranges": [[0, 2]]}]
    assert n.ModelClient.tool_payload("Read", {"files": [{"path": "a.py", "ranges": [[0, 1]]}]}) == [{"path": "a.py", "ranges": [[0, 1]]}]
    assert n.ReadTool(n.Session(cwd="."), [{"path": "nanocode.py"}]).targets()[0][1] == [(0, 0)]
    assert n.ModelClient.tool_payload("Find", {"name": "*.py"}) == [{"name": "*.py"}]
    assert n.ModelClient.tool_payload("Find", {"queries": [{"name": "*.py"}]}) == [{"name": "*.py"}]
    assert n.ModelClient.tool_payload("Search", {"pattern": "TODO"}) == [{"pattern": "TODO"}]
    assert n.ModelClient.tool_payload("Search", {"queries": [{"pattern": "TODO"}]}) == [{"pattern": "TODO"}]
    assert n.ModelClient.tool_payload("Note", {"goal": "ship"}) == [{"goal": "ship"}]


def test_note_tool_updates_durable_memory_without_result_key(tmp_path):
    s = session(tmp_path)
    s.state.known = ["existing"]
    runner = n.ToolRunner(s, n.ContextManager(s), output_fn=lambda text: None)

    output = []
    runner.output_fn = output.append
    runner.run([n.ToolCall("note", "Note", [{"goal": "ship", "plan": ["inspect", "patch"], "known": ["existing", "pytest"]}])])

    assert s.state.goal == "ship"
    assert s.state.plan == ["inspect", "patch"]
    assert s.state.known == ["existing", "pytest"]
    assert s.tool_records == []
    assert output == ["goal -> ship\nplan:\n  - [~] inspect\n  - [ ] patch\nknown:\n  + pytest"]


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


def test_tool_runner_batch_edit_accepts_drifted_anchor(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(n.CodeIndex, "update", lambda self, paths: "")
    path = tmp_path / "code.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")
    runner = n.ToolRunner(s, n.ContextManager(s), output_fn=lambda text: None)

    runner.run(
        [
            n.ToolCall("insert", "Edit", ["code.txt", [{"op": "insert_before", "start": anchor(1, "b\n"), "content": "x\n"}]]),
            n.ToolCall("replace", "Edit", ["code.txt", [{"op": "replace", "start": anchor(3, "c\n"), "end": anchor(3, "c\n"), "content": "C\n"}]]),
        ]
    )

    assert path.read_text(encoding="utf-8") == "a\nx\nb\nC\n"
    assert s.tool_errors == []


def test_tool_runner_batch_edit_maps_original_anchor_after_insert(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(n.CodeIndex, "update", lambda self, paths: "")
    path = tmp_path / "code.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")
    runner = n.ToolRunner(s, n.ContextManager(s), output_fn=lambda text: None)

    runner.run(
        [
            n.ToolCall("insert", "Edit", ["code.txt", [{"op": "insert_before", "start": anchor(1, "b\n"), "content": "x\n"}]]),
            n.ToolCall("replace", "Edit", ["code.txt", [{"op": "replace", "start": anchor(2, "c\n"), "end": anchor(2, "c\n"), "content": "C\n"}]]),
        ]
    )

    assert path.read_text(encoding="utf-8") == "a\nx\nb\nC\n"
    assert s.tool_errors == []


def test_tool_runner_batch_edit_maps_original_anchor_after_delete(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(n.CodeIndex, "update", lambda self, paths: "")
    path = tmp_path / "code.txt"
    path.write_text("a\nb\nc\nd\n", encoding="utf-8")
    runner = n.ToolRunner(s, n.ContextManager(s), output_fn=lambda text: None)

    runner.run(
        [
            n.ToolCall("delete", "Edit", ["code.txt", [{"op": "delete", "start": anchor(1, "b\n"), "end": anchor(1, "b\n")}]]),
            n.ToolCall("replace", "Edit", ["code.txt", [{"op": "replace", "start": anchor(3, "d\n"), "end": anchor(3, "d\n"), "content": "D\n"}]]),
        ]
    )

    assert path.read_text(encoding="utf-8") == "a\nc\nD\n"
    assert s.tool_errors == []


def test_tool_runner_batch_edit_can_create_then_patch_same_file(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(n.CodeIndex, "update", lambda self, paths: "")
    runner = n.ToolRunner(s, n.ContextManager(s), output_fn=lambda text: None)

    runner.run(
        [
            n.ToolCall("create", "Edit", ["new.txt", [{"op": "replace_all", "old": "", "new": "a\nb\n"}], True]),
            n.ToolCall("patch", "Edit", ["new.txt", [{"op": "replace", "start": anchor(1, "b\n"), "end": anchor(1, "b\n"), "content": "B\n"}]]),
        ]
    )

    assert (tmp_path / "new.txt").read_text(encoding="utf-8") == "a\nB\n"
    assert len([record for record in s.tool_records if record.name == "Edit"]) == 2
    assert s.tool_errors == []


def test_tool_runner_batch_edit_plans_files_independently(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(n.CodeIndex, "update", lambda self, paths: "")
    (tmp_path / "a.txt").write_text("a\nb\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("x\ny\n", encoding="utf-8")
    runner = n.ToolRunner(s, n.ContextManager(s), output_fn=lambda text: None)

    runner.run(
        [
            n.ToolCall("edit-a", "Edit", ["a.txt", [{"op": "insert_after", "start": anchor(0, "a\n"), "content": "A\n"}]]),
            n.ToolCall("edit-b", "Edit", ["b.txt", [{"op": "replace", "start": anchor(1, "y\n"), "end": anchor(1, "y\n"), "content": "Y\n"}]]),
        ]
    )

    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "a\nA\nb\n"
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "x\nY\n"
    assert s.tool_errors == []


def test_tool_runner_batch_edit_read_between_edits_sees_intermediate_file(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(n.CodeIndex, "update", lambda self, paths: "")
    path = tmp_path / "code.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")
    runner = n.ToolRunner(s, n.ContextManager(s), output_fn=lambda text: None)

    runner.run(
        [
            n.ToolCall("insert", "Edit", ["code.txt", [{"op": "insert_before", "start": anchor(1, "b\n"), "content": "x\n"}]]),
            n.ToolCall("read", "Read", [{"path": "code.txt", "ranges": [[0, 0]]}]),
            n.ToolCall("replace", "Edit", ["code.txt", [{"op": "replace", "start": anchor(3, "c\n"), "end": anchor(3, "c\n"), "content": "C\n"}]]),
        ]
    )

    read_record = next(record for record in s.tool_records if record.name == "Read")
    assert "|x" in read_record.output
    assert "|c" in read_record.output
    assert "|C" not in read_record.output
    assert path.read_text(encoding="utf-8") == "a\nx\nb\nC\n"


def test_tool_runner_batch_edit_rejects_anchor_for_line_changed_in_batch(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(n.CodeIndex, "update", lambda self, paths: "")
    path = tmp_path / "code.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")
    runner = n.ToolRunner(s, n.ContextManager(s), output_fn=lambda text: None)

    runner.run(
        [
            n.ToolCall("first", "Edit", ["code.txt", [{"op": "replace", "start": anchor(2, "c\n"), "end": anchor(2, "c\n"), "content": "C\n"}]]),
            n.ToolCall("second", "Edit", ["code.txt", [{"op": "replace", "start": anchor(2, "c\n"), "end": anchor(2, "c\n"), "content": "D\n"}]]),
        ]
    )

    assert path.read_text(encoding="utf-8") == "a\nb\nC\n"
    assert len([record for record in s.tool_records if record.name == "Edit"]) == 1
    assert s.tool_errors


def test_batch_edit_stale_anchor_reports_current_line(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(n.CodeIndex, "update", lambda self, paths: "")
    path = tmp_path / "code.txt"
    path.write_text("a\nb\n", encoding="utf-8")
    runner = n.ToolRunner(s, n.ContextManager(s), output_fn=lambda text: None)

    runner.run([n.ToolCall("bad", "Edit", ["code.txt", [{"op": "replace", "start": anchor(1, "wrong\n"), "end": anchor(1, "wrong\n"), "content": "B\n"}]])])

    assert s.tool_errors
    assert "current is 1:" + n.ReadTool.line_hash("b\n") + "|b" in s.tool_errors[0].error
    assert path.read_text(encoding="utf-8") == "a\nb\n"


def test_tool_runner_batch_edit_barrier_stops_original_anchor_mapping(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(n.CodeIndex, "update", lambda self, paths: "")
    path = tmp_path / "code.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")
    runner = n.ToolRunner(s, n.ContextManager(s), output_fn=lambda text: None)

    runner.run(
        [
            n.ToolCall("insert", "Edit", ["code.txt", [{"op": "insert_before", "start": anchor(1, "b\n"), "content": "x\n"}]]),
            n.ToolCall("barrier", "Bash", [":"]),
            n.ToolCall("replace", "Edit", ["code.txt", [{"op": "replace", "start": anchor(2, "c\n"), "end": anchor(2, "c\n"), "content": "C\n"}]]),
        ]
    )

    assert path.read_text(encoding="utf-8") == "a\nx\nb\nc\n"
    assert len([record for record in s.tool_records if record.name == "Edit"]) == 1
    assert s.tool_errors


def test_planned_edit_refuses_to_overwrite_external_change(tmp_path):
    s = session(tmp_path)
    path = tmp_path / "code.txt"
    path.write_text("a\nb\n", encoding="utf-8")
    call = n.ToolCall("edit", "Edit", ["code.txt", [{"op": "replace", "start": anchor(1, "b\n"), "end": anchor(1, "b\n"), "content": "B\n"}]])
    plan = n.EditBatchPlan(s).build([call])
    path.write_text("external\n", encoding="utf-8")

    with pytest.raises(n.ToolError, match="planned edit is stale"):
        plan.planned[call.id].call(n.EditTool(s, call.args))

    assert path.read_text(encoding="utf-8") == "external\n"


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


def test_tool_runner_starts_bash_live_preview_before_output(tmp_path):
    s = session(tmp_path)
    s.settings.yolo = True
    events = []
    runner = n.ToolRunner(s, n.ContextManager(s), input_fn=lambda prompt: (_ for _ in ()).throw(AssertionError("unexpected prompt")), output_fn=lambda text: None)
    runner.live_start = lambda: events.append(("start", ""))
    runner.live_output = lambda stream, text: events.append((stream, text))

    runner.run([n.ToolCall("bash", "Bash", ["printf live"])])

    assert events[0] == ("start", "")
    assert ("stdout", "live") in events
    assert events[-1] == ("", "")


def test_code_index_updates_after_file_mutation_tools(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    updated = []
    monkeypatch.setattr(n.CodeIndex, "update", lambda self, paths: updated.extend(paths) or "")
    runner = n.ToolRunner(s, n.ContextManager(s), input_fn=lambda prompt: (_ for _ in ()).throw(AssertionError("unexpected prompt")), output_fn=lambda text: None)

    runner.run([n.ToolCall("touch", "Touch", ["empty.py"])])
    runner.run([n.ToolCall("create", "Edit", ["made.py", [{"op": "replace_all", "old": "", "new": "print(1)\n"}], True])])
    runner.run([n.ToolCall("edit", "Edit", ["made.py", [{"op": "replace_all", "old": "1", "new": "2"}]])])

    assert (tmp_path / "made.py").read_text(encoding="utf-8") == "print(2)\n"
    assert updated == ["empty.py", "made.py", "made.py"]


def test_edit_index_update_uses_call_path_when_output_path_is_unparseable(tmp_path, monkeypatch):
    s = session(tmp_path)
    updated = []
    monkeypatch.setattr(n.CodeIndex, "update", lambda self, paths: updated.extend(paths) or "")

    n.ToolRunner(s, n.ContextManager(s), output_fn=lambda text: None).update_code_index(
        n.ToolCall("edit", "Edit", ["made.py", [{"op": "replace_all", "old": "", "new": "x\n"}], True]),
        "<Edit path=bad />",
    )

    assert updated == ["made.py"]


def test_yolo_approves_mutating_tools_without_prompt(tmp_path):
    s = session(tmp_path)
    s.settings.yolo = True
    runner = n.ToolRunner(s, n.ContextManager(s), input_fn=lambda prompt: (_ for _ in ()).throw(AssertionError("unexpected prompt")), output_fn=lambda text: None)

    runner.run([n.ToolCall("create", "Edit", ["auto.txt", [{"op": "replace_all", "old": "", "new": "ok\n"}], True])])

    assert (tmp_path / "auto.txt").read_text(encoding="utf-8") == "ok\n"
    assert len(s.tool_records) == 1
    assert s.tool_errors == []
