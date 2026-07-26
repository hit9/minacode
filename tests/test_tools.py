import shlex
import sys
import threading
import time

import pytest

import minacode as n


def session(tmp_path):
    return n.Session(cwd=str(tmp_path))


def anchor(index, line):
    return f"{index}:{n.ReadTool.line_hash(line)}"


def test_validate_edit_target_branches(tmp_path):
    s = session(tmp_path)
    tool = n.EditTool(s, [])
    (tmp_path / "a.py").write_text("x", encoding="utf-8")
    (tmp_path / "sub").mkdir()

    # Existing file, editing -> True (caller should read it).
    assert tool._validate_target(str(tmp_path / "a.py"), creating=False) is True
    # Missing file, creating inside the workspace -> False (create fresh).
    assert tool._validate_target(str(tmp_path / "new.py"), creating=True) is False
    # Each invalid state raises the same ToolError both edit paths relied on.
    with pytest.raises(n.ToolError, match="file already exists"):
        tool._validate_target(str(tmp_path / "a.py"), creating=True)
    with pytest.raises(n.ToolError, match="path is a directory"):
        tool._validate_target(str(tmp_path / "sub"), creating=False)
    with pytest.raises(n.ToolError, match="does not exist"):
        tool._validate_target(str(tmp_path / "missing.py"), creating=False)
    with pytest.raises(n.ToolError, match="outside workspace"):
        tool._validate_target("/etc/minacode_should_not_create.py", creating=True)


def test_read_and_search_success_paths(tmp_path):
    (tmp_path / "sample.py").write_text("alpha\nNeedle\nomega\n", encoding="utf-8")
    (tmp_path / "blob.bin").write_bytes(b"a\0b")
    s = session(tmp_path)

    read = n.ReadTool(s, [{"path": "sample.py", "ranges": [[0, 2], [2, 0]]}]).call()
    single_range = n.ReadTool(s, [{"path": "sample.py", "ranges": [0, 2]}]).call()
    full_default = n.ReadTool(s, [{"path": "sample.py"}]).call()
    alpha_hash = n.ReadTool.line_hash("alpha\n")
    needle_hash = n.ReadTool.line_hash("Needle\n")
    omega_hash = n.ReadTool.line_hash("omega\n")
    assert f"anchor=0:{alpha_hash} | alpha" in read
    assert f"anchor=1:{needle_hash} | Needle" in read
    assert f"anchor=2:{omega_hash} | omega" in read
    assert f"anchor=0:{alpha_hash} | alpha" in single_range
    assert f"anchor=1:{needle_hash} | Needle" in single_range
    assert f"anchor=2:{omega_hash} | omega" in full_default
    assert "<total_lines>3</total_lines>" in full_default  # Read reports the line count (replaces LineCount)

    found = n.SearchTool(s, [{"pattern": "needle", "path": "."}]).call()
    assert f"sample.py anchor=1:{needle_hash} | Needle" in found

    multiline = n.SearchTool(s, [{"pattern": "alpha\\nNeedle", "path": "sample.py"}]).call()
    assert "sample.py anchor=0:" in multiline


def test_line_hash_is_short_lowercase_base36(tmp_path):
    line_hash = n.ReadTool.line_hash("alpha\n")
    assert len(line_hash) == 5
    assert line_hash == line_hash.lower()
    assert set(line_hash) <= set("0123456789abcdefghijklmnopqrstuvwxyz")


def test_line_hash_ignores_trailing_newline():
    # An anchor must depend only on the visible content, so a line's anchor stays stable when only
    # the trailing newline changes (e.g. the last line gaining/losing the final "\n"). It must also
    # agree with the newline-stripping indexed hash the anchor matcher accepts.
    assert n.ReadTool.line_hash("code") == n.ReadTool.line_hash("code\n") == n.ReadTool.line_hash("code\n\n")
    assert n.ReadTool.anchor_matches("code\n", n.ReadTool.line_hash("code"))
    assert n.ReadTool.anchor_matches("code", n.ReadTool.line_hash("code\n"))


def test_split_lines_matches_readlines_only_on_newline():
    # Edit's line model must number lines exactly like Read (file.readlines), i.e. split on "\n"
    # only. str.splitlines(True) also breaks on \x0c and friends, which would desync anchors.
    assert n.ReadTool.split_lines("a\nb\x0cc\nd\n") == ["a\n", "b\x0cc\n", "d\n"]
    assert n.ReadTool.split_lines("a\nb") == ["a\n", "b"]
    assert n.ReadTool.split_lines("") == []
    assert n.ReadTool.split_lines("a\nb\x0cc\nd\n") != "a\nb\x0cc\nd\n".splitlines(True)


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

    assert "visible.txt anchor=0:" in found
    assert ".hidden" not in found
    assert "ignored" not in found
    assert ".hidden.txt anchor=0:" not in direct_hidden
    assert "ignored_dir/inside.txt anchor=0:" not in direct_ignored


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
        n.InspectCodeTool(s, ["inspect", "two words"]).call()

    assert not (tmp_path / "a.txt").exists()
    assert not (tmp_path / "b.txt").exists()


def test_edit_creates_and_patches_file(tmp_path):
    s = session(tmp_path)
    n.EditTool(s, ["empty/keep.txt", [{"op": "create", "content": ""}]]).call()
    assert (tmp_path / "empty" / "keep.txt").read_text(encoding="utf-8") == ""
    with pytest.raises(n.ToolError):
        n.EditTool(s, ["empty/keep.txt", [{"op": "create", "content": ""}]]).call()
    n.EditTool(s, ["empty/keep.txt", [{"op": "replace_all", "old": "", "new": "kept\n"}]]).call()
    assert (tmp_path / "empty" / "keep.txt").read_text(encoding="utf-8") == "kept\n"

    n.EditTool(s, ["nested/note.txt", [{"op": "create", "content": "one\ntwo\nthree\n"}]]).call()
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


def test_diff_segments_syntax_highlights_python(tmp_path):
    ui = n.UiPrinter()
    diff = "--- foo.py\n+++ foo.py\n@@ -1,2 +1,2 @@\n def hello():\n-    pass\n+    return 42\n"
    segments = ui.diff_segments(diff)

    assert any(t == "+" and s == "ansigreen bg:#003b00" for s, t in segments)
    assert any(t == "return" and s == "fg:#ff7b72 bg:#003b00" for s, t in segments)

    assert any(t == "-" and s == "ansired bg:#520000" for s, t in segments)
    assert any("pass" in t and s == "fg:default bg:#520000" for s, t in segments)

    # Line-number gutters and context code remain unfilled.
    assert any("1" in text and "|" in text and "bg:" not in style for style, text in segments)
    assert any(text == "def" and "bg:" not in style for style, text in segments)
    changed = [line for line in ui.segment_lines(segments) if any("bg:" in style for style, _text in line)]
    widths = [sum(n.get_cwidth(text.rstrip("\n")) for style, text in line if "bg:" in style) for line in changed]
    assert len(set(widths)) == 1


def test_diff_segments_gracefully_degrades_without_lexer(tmp_path):
    ui = n.UiPrinter()
    diff = "--- foo.unknownxyz\n+++ foo.unknownxyz\n@@ -1,1 +1,1 @@\n- old\n+ new\n"
    segments = ui.diff_segments(diff)

    assert any(t == "-" and s == "ansired bg:#520000" for s, t in segments)
    assert any("old" in t and s == "fg:default bg:#520000" for s, t in segments)
    assert any(t == "+" and s == "ansigreen bg:#003b00" for s, t in segments)


def test_diff_segments_gracefully_degrades_without_header_path(tmp_path):
    ui = n.UiPrinter()
    # No +++ line, so pygments cannot pick a lexer.
    diff = "@@ -1,1 +1,1 @@\n- old\n+ new\n"
    segments = ui.diff_segments(diff)

    assert any(t == "-" and s == "ansired bg:#520000" for s, t in segments)
    assert any(t == "+" and s == "ansigreen bg:#003b00" for s, t in segments)


def test_approval_segments_highlight_inline_edit_preview():
    preview = "--- foo.py\n+++ foo.py\n@@ -1,2 +1,2 @@\n def hello():\n-    pass\n+    return 42"
    block = n.LogBlock.hierarchy(
        n.LogLine("Edit", "foo.py", n.LogRole.TOOL),
        [
            n.LogLine("preview", role=n.LogRole.META, edge=n.LogEdge.BRANCH),
            *(n.LogLine("", line, n.LogRole.DIFF, n.LogEdge.CONTINUE) for line in preview.splitlines()),
        ],
    )
    segments = n.UiPrinter().log_segments(block)
    rendered = "".join(text for _style, text in segments)

    assert ("ansigreen", "Edit") in segments
    assert any(style == "fg:#ff7b72 bg:#003b00" and "return" in text for style, text in segments)
    assert any(style == "ansigreen bg:#003b00" and text == "+" for style, text in segments)
    assert any(style == "fg:default bg:#520000" and "pass" in text for style, text in segments)
    assert "\n\n" not in rendered


def test_edit_stale_anchor_reports_current_line(tmp_path):
    s = session(tmp_path)
    path = tmp_path / "note.txt"
    path.write_text("old\n", encoding="utf-8")

    with pytest.raises(n.ToolError) as error:
        n.EditTool(s, ["note.txt", [{"op": "replace", "start": anchor(0, "wrong\n"), "end": anchor(0, "wrong\n"), "content": "new\n"}]]).call()

    assert "stale anchor" in str(error.value)
    assert "current is anchor=0:" + n.ReadTool.line_hash("old\n") + " | old" in str(error.value)


def test_edit_anchor_survives_trailing_newline_change(tmp_path):
    # Regression: line_hash used to fold the trailing newline into the hash, so an anchor captured
    # for a last line without a newline went stale once an edit gave the file a trailing newline,
    # even though the line's visible text never changed.
    s = session(tmp_path)
    path = tmp_path / "note.txt"
    path.write_text("a\nb\n", encoding="utf-8")
    # Anchor built from the newline-less form of the line (as captured when "b" was the last line
    # before a trailing newline was added). It must still resolve against the current "b\n".
    anc = anchor(1, "b")
    n.EditTool(s, ["note.txt", [{"op": "replace", "start": anc, "end": anc, "content": "B\n"}]]).call()
    assert path.read_text(encoding="utf-8") == "a\nB\n"


def test_edit_anchor_consistent_with_read_on_exotic_line_boundary(tmp_path):
    # Regression: Edit split lines with str.splitlines(True) while Read uses readlines, so a file
    # containing a form-feed numbered lines differently and a valid Read anchor went stale in Edit.
    s = session(tmp_path)
    path = tmp_path / "ff.txt"
    path.write_text("a\nb\x0cc\nd\n", encoding="utf-8")  # form-feed inside the middle line
    read = n.ReadTool(s, [{"path": "ff.txt"}]).call()
    assert f"anchor=2:{n.ReadTool.line_hash('d')} | d" in read  # Read numbers "d" as line 2
    n.EditTool(s, ["ff.txt", [{"op": "replace", "start": anchor(2, "d\n"), "end": anchor(2, "d\n"), "content": "D\n"}]]).call()
    assert path.read_text(encoding="utf-8") == "a\nb\x0cc\nD\n"


def test_edit_accepts_inspect_code_anchor(tmp_path):
    s = session(tmp_path)
    path = tmp_path / "note.txt"
    path.write_text("old\n", encoding="utf-8")
    inspect_anchor = "anchor=0:" + n.ReadTool.indexed_line_hash("old\n")

    result = n.EditTool(s, ["note.txt", [{"op": "replace", "start": inspect_anchor, "end": inspect_anchor, "content": "new\n"}]]).call()

    assert '<Edit path="note.txt">' in result
    assert path.read_text(encoding="utf-8") == "new\n"


def test_edit_no_change_reports_current_target_range(tmp_path):
    s = session(tmp_path)
    path = tmp_path / "note.txt"
    path.write_text("old\n", encoding="utf-8")

    with pytest.raises(n.ToolError) as error:
        n.EditTool(s, ["note.txt", [{"op": "replace", "start": anchor(0, "old\n"), "end": anchor(0, "old\n"), "content": "old\n"}]]).call()

    message = str(error.value)
    assert "edit produced no changes; requested content already matches target range" in message
    assert "<current-target-ranges hashline-numbered>" in message
    assert "anchor=0:" + n.ReadTool.line_hash("old\n") + " | old" in message


def test_edit_no_change_replace_all_reports_identical_file(tmp_path):
    s = session(tmp_path)
    path = tmp_path / "note.txt"
    path.write_text("old\n", encoding="utf-8")

    with pytest.raises(n.ToolError) as error:
        n.EditTool(s, ["note.txt", [{"op": "replace_all", "old": "old", "new": "old"}]]).call()

    assert str(error.value) == "edit produced no changes; replace_all result is identical to current file"


def test_edit_create_decodes_escaped_newlines_for_preview_and_write(tmp_path):
    s = session(tmp_path)
    tool = n.EditTool(s, ["script.py", [{"op": "create", "content": "print(1)\\nprint(2)\\n"}]])

    preview = tool.preview()
    output = tool.call()

    assert "+print(1)\n+print(2)\n" in preview
    assert (tmp_path / "script.py").read_text(encoding="utf-8") == "print(1)\nprint(2)\n"
    assert "<Edit path=" in output


def test_bash_behaviors(tmp_path):
    s = session(tmp_path)
    bash = n.BashTool(s, ["printf out; printf err >&2; exit 3"]).call()
    assert "* exit_code: 3" in bash
    assert "<stdout>\nout\n</stdout>" in bash
    assert "<stderr>\nerr\n</stderr>" in bash

    # Multibyte UTF-8 output large enough to span 4096-byte read boundaries must decode cleanly
    # (regression: per-chunk decoding mangled split characters into replacement chars).
    wide = n.BashTool(s, ['python3 -c "print(chr(0x4e2d)*3000)"']).call()
    assert "�" not in wide
    assert wide.count(chr(0x4E2D)) == 3000


def test_bash_cancel_kills_active_process(tmp_path):
    tool = n.BashTool(session(tmp_path), ["sleep 30"])
    finished = threading.Event()

    def run():
        tool.call()
        finished.set()

    thread = threading.Thread(target=run)
    thread.start()
    deadline = time.monotonic() + 1
    while tool._process is None and time.monotonic() < deadline:
        time.sleep(0.01)

    tool.cancel()

    assert finished.wait(timeout=1)
    thread.join(timeout=1)


def test_bash_readonly_auto_approval_classification(tmp_path):
    s = session(tmp_path)

    def readonly(command):
        return not n.BashTool(s, [command]).needs_confirmation()

    # Safe read-only commands auto-run (no confirmation prompt in non-yolo mode).
    assert readonly("ls -la")
    assert readonly("cat file.txt")
    assert readonly("wc -l minacode.py")
    assert readonly("find . -name '*.py'")
    assert readonly("rg needle src")
    assert readonly("git status --short")
    assert readonly("git --no-pager status --short")
    assert readonly("git diff HEAD~1")
    assert readonly("cat a | grep foo | wc -l")  # pipeline of safe commands
    assert readonly("ls && cat README.md")  # sequence of safe commands
    assert readonly("cd /Users/x/proj && git log --oneline -10")  # cd prefix is a benign builtin
    assert readonly("cd a; ls")
    assert readonly("ls -la && find . -maxdepth 2 -type f | grep -v .git | sort | head -80")
    assert readonly("cat f | sort -u | uniq -c")  # sort/uniq are read-only in pipelines
    assert readonly("grep foo f 2>/dev/null")  # discarding stderr is not a file write
    assert readonly("ls -la >/dev/null 2>&1")  # /dev/null + stderr-merge
    assert readonly("cat f | sed -n '1,20p'")  # sed for read-only filtering
    assert readonly("tree -L 2 src")

    # Anything that writes, executes code, mutates git, or hides execution still asks.
    assert not readonly("rm -rf build")
    # Every stage of a chain is validated — a safe first command must not whitelist a mutating one.
    assert not readonly("git log && rm -rf x")
    assert not readonly("ls ; rm x")
    assert not readonly("cat f && python3 evil.py")
    assert not readonly("git log & rm x")  # backgrounding
    assert not readonly("git commit -m x")
    assert not readonly("git checkout main")
    assert not readonly("echo hi > out.txt")  # redirection
    assert not readonly("cat >/dev/nullx")  # /dev/null is only a prefix; writes real file /dev/nullx
    assert not readonly("echo x >/dev/null.bak")  # /dev/null prefix of a real file
    assert not readonly("cat 2>/dev/nullish")  # /dev/null prefix on a stderr redirect
    assert not readonly("cat >/dev/null2>&1")  # writes /dev/null2, not the null device
    assert not readonly("cat $(cmd)")  # command substitution
    assert not readonly("python3 script.py")  # arbitrary code
    assert not readonly("find . -delete")  # destructive flag
    assert not readonly("find . -name x -fprint0 out")  # file-writing flag
    assert not readonly("cat f > g")  # redirection to a real file
    assert not readonly("sed -i s/a/b/ f")  # in-place edit
    assert not readonly("sort -o out.txt f")  # sort output file
    assert not readonly("tree -o out.txt")  # tree output file
    assert not readonly("sed -i s/a/b/ f")  # in-place edit
    assert not readonly("git diff --output=patch.txt")  # file-writing git option
    assert not readonly("git grep -O needle")  # opens files via pager/editor
    assert not readonly("git --paginate log")  # can invoke configured pager
    assert not readonly("ls & rm x")  # backgrounding
    assert not readonly("ls; rm x")  # unsafe stage in a sequence
    assert not readonly("FOO=1 env")  # env assignment / wrapper


def test_inspect_code_modes_call_symbol_index_api(tmp_path, monkeypatch):
    s = session(tmp_path)
    (tmp_path / "sample.py").write_text("class Example:\n    pass\n", encoding="utf-8")
    calls = []

    monkeypatch.setattr(n.CodeIndex, "available", lambda self: True)
    monkeypatch.setattr(n.csi, "search", lambda query, **kwargs: calls.append(("search", query, kwargs)) or "search ok")
    monkeypatch.setattr(n.csi, "inspect", lambda query, **kwargs: calls.append(("inspect", query, kwargs)) or "inspect ok")
    monkeypatch.setattr(n.csi, "outline", lambda path, **kwargs: calls.append(("outline", path, kwargs)) or "outline ok")
    monkeypatch.setattr(n.csi, "refs", lambda query, **kwargs: calls.append(("refs", query, kwargs)) or "refs ok")
    monkeypatch.setattr(n.csi, "impls", lambda query, **kwargs: calls.append(("impls", query, kwargs)) or "impls ok")
    monkeypatch.setattr(n.csi, "callers", lambda query, **kwargs: calls.append(("callers", query, kwargs)) or "callers ok")
    monkeypatch.setattr(n.csi, "callees", lambda query, **kwargs: calls.append(("callees", query, kwargs)) or "callees ok")

    assert "search ok" in n.InspectCodeTool(s, ["find", "Example", {"kind": "class,function", "limit": 10, "exact_only": True}]).call()
    assert "inspect ok" in n.InspectCodeTool(s, ["inspect", "Example", {"path": "sample.py"}]).call()
    assert "outline ok" in n.InspectCodeTool(s, ["outline", "sample.py"]).call()
    assert "outline ok" in n.InspectCodeTool(s, ["outline", "sample.py", {"limit": 300}]).call()
    assert "refs ok" in n.InspectCodeTool(s, ["refs", "Example", {"all_kinds": True, "offset": 5}]).call()
    assert "impls ok" in n.InspectCodeTool(s, ["impls", "Example", {"kind": "class"}]).call()
    assert "callers ok" in n.InspectCodeTool(s, ["callers", "Example", {"depth": 2}]).call()
    assert "callees ok" in n.InspectCodeTool(s, ["callees", "Example"]).call()

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
            "limit": n.csi.DEFAULT_PAGE_LIMIT,
            "anchors": True,
            "anchor_format": "explicit",
        },
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
    assert calls[4] == (
        "refs",
        "Example",
        {
            "root": str(tmp_path),
            "kind": None,
            "path": None,
            "exact_only": False,
            "format": "text",
            "limit": n.csi.DEFAULT_MAX_REFERENCES,
            "offset": 5,
            "ref_kinds": "all",
        },
    )
    assert calls[5] == (
        "impls",
        "Example",
        {"root": str(tmp_path), "kind": "class", "path": None, "exact_only": False, "format": "text", "limit": n.csi.DEFAULT_MAX_IMPLEMENTORS, "offset": 0},
    )
    assert calls[6] == (
        "callers",
        "Example",
        {"root": str(tmp_path), "kind": None, "path": None, "exact_only": False, "format": "text", "limit": n.csi.DEFAULT_MAX_CALLERS, "depth": 2},
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
            "limit": n.csi.DEFAULT_MAX_CALLEES,
            "depth": 3,
            "loose": False,
        },
    )

    assert "refs ok" in n.InspectCodeTool(s, ["refs", "Example", {"ref_kind": "call,write"}]).call()
    assert calls[8][2]["ref_kinds"] == "call,write"
    assert "callees ok" in n.InspectCodeTool(s, ["callees", "Example", {"loose": True}]).call()
    assert calls[9][2]["loose"] is True

    with pytest.raises(n.ToolError):
        n.InspectCodeTool(s, ["outline", "missing.py"]).call()
    with pytest.raises(n.ToolError):
        n.InspectCodeTool(s, ["inspect", "sample.py"]).call()
    with pytest.raises(n.ToolError):
        n.InspectCodeTool(s, ["outline", "sample.py", {"limit": 1001}]).call()
    with pytest.raises(n.ToolError):
        n.InspectCodeTool(s, ["refs", "sample.py"]).call()
    with pytest.raises(n.ToolError):
        n.InspectCodeTool(s, ["callers", "Example", {"depth": 9}]).call()
    with pytest.raises(n.ToolError):
        n.InspectCodeTool(s, ["refs", "Example", {"ref_kind": "bogus"}]).call()
    with pytest.raises(n.ToolError):
        n.InspectCodeTool(s, ["refs", "Example", {"ref_kind": "call", "all_kinds": True}]).call()


def test_inspect_code_strips_kind_prefix_from_target(tmp_path, monkeypatch):
    s = session(tmp_path)
    calls = []
    monkeypatch.setattr(n.CodeIndex, "available", lambda self: True)
    monkeypatch.setattr(n.csi, "search", lambda query, **kwargs: calls.append(query) or "ok")

    # "class Config" with kind "class" -> the redundant leading kind word is dropped.
    n.InspectCodeTool(s, ["find", "class Config", {"kind": "class"}]).call()
    assert calls[-1] == "Config"

    # Works when the kind option lists several kinds.
    n.InspectCodeTool(s, ["find", "function handoff", {"kind": "class,function"}]).call()
    assert calls[-1] == "handoff"

    # Only the declared kind is stripped: a bare language keyword is not, and still errors.
    with pytest.raises(n.ToolError):
        n.InspectCodeTool(s, ["find", "def foo", {"kind": "function"}]).call()
    # No kind provided -> nothing to key off, still rejected.
    with pytest.raises(n.ToolError):
        n.InspectCodeTool(s, ["find", "class Config"]).call()


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


def test_recall_history_returns_segment_text(tmp_path):
    s = session(tmp_path)
    s.history.append(n.HistorySegment(key="seg.1", title="task", text="user:\nfind bug"))

    result = n.RecallContextTool(s, [{"keys": ["seg.1"]}]).call()

    assert "<RecallContextResult>" in result
    assert 'key="seg.1"' in result
    assert "find bug" in result


def test_recall_history_reports_missing_segment(tmp_path):
    s = session(tmp_path)

    assert "seg.9: missing" in n.RecallContextTool(s, [{"keys": ["seg.9"]}]).call()


def test_recall_history_rejects_bad_key_format(tmp_path):
    s = session(tmp_path)

    with pytest.raises(n.ToolError):
        n.RecallContextTool(s, [{"keys": ["tr.1"]}]).call()


def test_recall_history_regex_searches_titles_and_text(tmp_path):
    s = session(tmp_path)
    s.history.extend(
        [
            n.HistorySegment(key="seg.1", title="cache work", text="user:\nStable prefix design"),
            n.HistorySegment(key="seg.2", title="notes", text="assistant:\nTask Memory placement"),
            n.HistorySegment(key="seg.3", title="unrelated", text="assistant:\nNothing relevant"),
        ]
    )

    result = n.RecallContextTool(s, [{"query": "stable prefix|task memory"}]).call()

    assert '<RecallContextSearchResult query="stable prefix|task memory" matches=2>' in result
    assert "seg.1 2" in result
    assert "Stable prefix design" in result
    assert "seg.2 2" in result
    assert "Task Memory placement" in result
    assert "seg.3" not in result


def test_recall_history_regex_supports_key_scope_case_and_limit(tmp_path):
    s = session(tmp_path)
    s.history.extend(
        [
            n.HistorySegment(key="seg.1", title="one", text="Needle first"),
            n.HistorySegment(key="seg.2", title="two", text="needle second\nneedle third"),
            n.HistorySegment(key="seg.3", title="three", text="needle fourth"),
        ]
    )

    result = n.RecallContextTool(
        s,
        [{"keys": ["seg.1", "seg.2"], "query": "needle", "case_sensitive": True, "limit": 1}],
    ).call()

    assert "matches=1" in result
    assert "seg.1" not in result
    assert "seg.2 1" in result
    assert "needle second" in result
    assert "needle third" not in result
    assert "seg.3" not in result


def test_recall_history_regex_validates_search_arguments(tmp_path):
    s = session(tmp_path)

    for payload in ({}, {"query": "["}, {"query": "x", "limit": 0}, {"keys": ["seg.1"], "case_sensitive": True}):
        with pytest.raises(n.ToolError):
            n.RecallContextTool(s, [payload]).call()


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
    note = runner.short_call(
        n.ToolCall(
            "m",
            "Note",
            [
                {
                    "set_goal": "ship",
                    "replace_plan": [{"status": "doing", "text": "inspect"}, {"status": "todo", "text": "patch"}],
                    "append_known": ["existing", "new fact"],
                }
            ],
        )
    )
    assert note == "Note goal: ship\nplan:\n  - [~] inspect\n  - [ ] patch\nknown:\n  + new fact"


def test_reject_collapses_display(tmp_path):
    s = session(tmp_path)
    out = []
    runner = n.ToolRunner(s, n.ContextManager(s), output_fn=lambda text: out.append(str(text)))

    msg = runner.reject(n.ToolCall("c", "Read", [{"path": "x"}]), "ToolError: Read requires non-empty ranges")

    # display collapses to one quiet line, no full [failed]/error block
    assert any("· rejected: Read requires non-empty ranges" in t for t in out)
    assert not any("[failed]" in t or t.startswith("  error ") for t in out)
    # model still receives the full error
    assert "Read requires non-empty ranges" in msg


def test_uiprinter_renders_rejected_line_dim():
    ui = n.UiPrinter(output_fn=lambda text: None)
    segs = ui.log_segments(n.LogBlock([n.LogLine("Read", "· rejected: needs ranges", n.LogRole.MUTED)]))

    assert any(style == "ansibrightblack" and "rejected" in text for style, text in segs)
    assert not any(style in ("ansired", "ansigreen") for style, text in segs)


def test_tool_runner_reject_records_error_and_returns_failed_message(tmp_path):
    s = n.Session(cwd=str(tmp_path))
    runner = n.ToolRunner(s, n.ContextManager(s))
    call = n.ToolCall("e1", "Bash", ["bad cmd"])
    out = []
    runner.output_fn = out.append
    result = runner.reject(call, "ToolError: command not found")
    assert len(out) == 1
    assert isinstance(out[0], n.LogBlock)
    assert "command not found" in str(out[0])
    # Should record the error
    assert len(s.tool_errors) == 1
    assert s.tool_errors[0].name == "Bash"
    assert "command not found" in s.tool_errors[0].error
    # reject returns a plain-text tool-message representation
    assert "failed" in result.lower()
    assert "command not found" in result


def test_uiprinter_renders_tool_root_without_generic_prefix():
    block = n.LogBlock([n.LogLine("Read", "minacode.py 0:100 → tr.6 [auto]", n.LogRole.TOOL)])
    segments = n.UiPrinter(output_fn=lambda text: None).log_segments(block)
    text = "".join(value for _style, value in segments)

    assert text == "  Read  minacode.py 0:100 → tr.6 [auto]\n"
    assert any(style == "fg:default" and "minacode.py 0:100 → tr.6 [auto]" in value for style, value in segments)


def test_uiprinter_syntax_highlights_bash_arguments(tmp_path):
    s = session(tmp_path)
    line = n.ToolRunner(s, n.ContextManager(s)).log_root("Bash cd /tmp && printf '%s\\n' value")

    assert line.syntax == "bash"
    segments = n.UiPrinter(output_fn=lambda text: None).log_segments(n.LogBlock([line]))
    assert ("fg:#79c0ff", "cd") in segments
    assert ("fg:#79c0ff", "printf") in segments
    assert ("fg:#a5d6ff", "'%s\\n'") in segments
    assert not any("bg:" in style for style, _text in segments)


def test_uiprinter_highlights_generic_tool_arguments(tmp_path):
    s = session(tmp_path)
    runner = n.ToolRunner(s, n.ContextManager(s))
    line = runner.log_root('Search "done in" glob=*.py C=2')

    assert line.syntax == "tool-args"
    segments = n.UiPrinter(output_fn=lambda text: None).log_segments(n.LogBlock([line]))
    assert ("fg:#a5d6ff", '"done in"') in segments
    assert ("fg:#79c0ff", "glob=") in segments
    assert ("fg:#d2a8ff", "2") in segments


def test_job_start_uses_bash_highlighting(tmp_path):
    s = session(tmp_path)
    runner = n.ToolRunner(s, n.ContextManager(s))
    start = n.ToolCall("j1", "Job", [{"action": "start", "command": "pytest -q"}])
    wait = n.ToolCall("j2", "Job", [{"action": "wait", "job": "job.1"}])

    start_line = runner.log_root(runner.short_call(start), call=start)
    wait_line = runner.log_root(runner.short_call(wait), call=wait)

    assert start_line.syntax == "bash"
    assert wait_line.syntax == "tool-args"
    wait_segments = n.UiPrinter(output_fn=lambda text: None).log_segments(n.LogBlock([wait_line]))
    assert ("fg:#d2a8ff", "job.1") in wait_segments


def test_job_start_reclaims_finished_capacity(tmp_path, monkeypatch):
    s = session(tmp_path)
    monkeypatch.setattr(n.JobTool, "MAX_JOBS", 1)
    n.JobTool(s, [{"action": "start", "command": "true"}]).call()
    s.jobs["job.1"].process.wait(timeout=2)

    result = n.JobTool(s, [{"action": "start", "command": "true"}]).call()

    assert result.startswith("Started job.2")
    s.jobs["job.2"].process.wait(timeout=2)


def test_ps_hides_jobs_that_finished_without_polling(tmp_path):
    s = session(tmp_path)
    n.JobTool(s, [{"action": "start", "command": "true"}]).call()
    s.jobs["job.1"].process.wait(timeout=2)
    command_loop = n.CommandLoop(n.Agent(s), input_fn=lambda prompt="": "", output_fn=lambda text: None)

    assert command_loop.ps_command("") == "No active jobs (1 total)."


def test_kill_finished_job_does_not_signal_stale_process(tmp_path):
    s = session(tmp_path)
    n.JobTool(s, [{"action": "start", "command": "true"}]).call()
    s.jobs["job.1"].process.wait(timeout=2)

    result = n.JobTool(s, [{"action": "kill", "job": "job.1"}]).call()

    assert "status=done" in result
    assert "exit_code=0" in result


def test_job_status_accepts_bare_numeric_id(tmp_path):
    s = session(tmp_path)
    n.JobTool(s, [{"action": "start", "command": "true"}]).call()
    s.jobs["job.1"].process.wait(timeout=2)

    result = n.JobTool(s, [{"action": "status", "job": "1"}]).call()

    assert "Status: done" in result
    assert "Exit code: 0" in result


def test_job_captures_large_output_via_log_file(tmp_path):
    s = session(tmp_path)
    code = 'import sys; sys.stdout.write("x" * 1000000)'
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"
    n.JobTool(s, [{"action": "start", "command": command}]).call()
    job = s.jobs["job.1"]

    try:
        job.process.wait(timeout=2)
    finally:
        if job.process.poll() is None:
            job.kill(grace=0.1)

    job.update_status()
    assert job.status == "done"
    assert job.exit_code == 0
    assert job.tail(100) == "..." + "x" * 97


def test_job_start_runs_shell_builtins_and_compound_commands(tmp_path):
    """`Job(start)` must run commands through the shell rather than `exec` the first word, or
    builtins like `cd` and compound commands like `cd dir && cmd` fail with `exec: cd: not found`."""
    s = session(tmp_path)
    sub = tmp_path / "sub"
    sub.mkdir()
    n.JobTool(s, [{"action": "start", "command": f"cd {shlex.quote(str(sub))} && printf marker"}]).call()
    job = s.jobs["job.1"]

    try:
        job.process.wait(timeout=2)
    finally:
        if job.process.poll() is None:
            job.kill(grace=0.1)

    job.update_status()
    assert job.status == "done"
    assert job.exit_code == 0
    assert "marker" in job.tail(100)


def test_job_start_captures_every_stage_of_a_compound_command(tmp_path):
    """The whole command is grouped before redirection, so output from early stages (not just the
    last) lands in the job log instead of leaking to the inherited stdout."""
    s = session(tmp_path)
    n.JobTool(s, [{"action": "start", "command": "printf first; printf second && printf third"}]).call()
    job = s.jobs["job.1"]

    try:
        job.process.wait(timeout=2)
    finally:
        if job.process.poll() is None:
            job.kill(grace=0.1)

    job.update_status()
    assert job.status == "done"
    log = job.tail(1000)
    assert "first" in log and "second" in log and "third" in log


def test_job_tail_respects_limits_smaller_than_ellipsis(tmp_path):
    s = session(tmp_path)
    n.JobTool(s, [{"action": "start", "command": "printf abcdef"}]).call()
    job = s.jobs["job.1"]
    job.process.wait(timeout=2)

    assert job.tail(1) == "."
    assert job.tail(2) == ".."
    assert job.tail(3) == "..."


def test_log_block_aligns_multiline_tool_arguments():
    block = n.LogBlock.hierarchy(
        n.LogLine("Bash", 'git commit -m "title\nbody"', n.LogRole.TOOL, syntax="bash"),
        [n.LogLine("done", role=n.LogRole.META, edge=n.LogEdge.END)],
    )
    expected = '  Bash  git commit -m "title\n        body"\n    └ done'

    assert str(block) == expected
    rendered = "".join(text for _style, text in n.UiPrinter(output_fn=lambda text: None).log_segments(block))
    assert rendered == expected + "\n"


def test_log_block_wraps_long_tool_arguments_with_hanging_indent(monkeypatch):
    monkeypatch.setattr(n.shutil, "get_terminal_size", lambda fallback: n.os.terminal_size((40, 24)))
    command = 'git commit -m "system prompt: enhance with attitude, updates, review mode, and tooling rules"'
    block = n.LogBlock([n.LogLine("Bash", command, n.LogRole.TOOL, syntax="bash")])

    rendered = "".join(text for _style, text in n.UiPrinter(output_fn=lambda text: None).log_segments(block))

    assert rendered.splitlines() == [
        '  Bash  git commit -m "system prompt:',
        "        enhance with attitude,",
        "        updates, review mode, and",
        '        tooling rules"',
    ]
    assert all(len(line) < 40 for line in rendered.splitlines())


def test_uiprinter_renders_note_memory_status_colors():
    ui = n.UiPrinter(output_fn=lambda text: None)
    segs = ui.segments("goal: ship\ncheck: passed\nplan:\n  - [~] inspect\n  - [x] patch\nknown:\n  + pytest")

    assert ("ansimagenta", "goal: ship") in segs
    assert ("ansimagenta", "check: passed") in segs
    assert ("ansicyan", "plan:") in segs
    assert ("ansiyellow", "  - [~] inspect") in segs
    assert ("ansigreen", "  - [x] patch") in segs
    assert ("ansigreen", "  + pytest") in segs


def test_uiprinter_renders_bash_preview_like_live_output():
    ui = n.UiPrinter(output_fn=lambda text: None)
    block = n.LogBlock.hierarchy(
        n.LogLine("Bash", "cmd", n.LogRole.TOOL),
        [
            n.LogLine("", "stderr:", n.LogRole.OUTPUT, n.LogEdge.CONTINUE),
            n.LogLine("", "  Traceback", n.LogRole.OUTPUT, n.LogEdge.CONTINUE),
            n.LogLine("", "    File x", n.LogRole.OUTPUT, n.LogEdge.CONTINUE),
            n.LogLine("", "  AttributeError", n.LogRole.OUTPUT, n.LogEdge.CONTINUE),
        ],
    )
    segs = ui.log_segments(block)

    assert ("ansibrightblack", "stderr:") in segs
    assert ("ansibrightblack", "  Traceback") in segs
    assert ("ansibrightblack", "    File x") in segs
    assert ("ansibrightblack", "  AttributeError") in segs


def test_uiprinter_renders_stored_result_dim():
    ui = n.UiPrinter(output_fn=lambda text: None)
    block = n.LogBlock.hierarchy(None, [n.LogLine("stored", "tr.50 [approved]", n.LogRole.META, n.LogEdge.END)])

    assert ui.log_segments(block) == [
        ("", "    "),
        ("ansibrightblack", "└ "),
        ("ansibrightblack", "stored"),
        ("ansibrightblack", " tr.50 [approved]"),
        ("", "\n"),
    ]


def test_tool_schemas_are_strict_for_high_risk_tools():
    bash_params = n.BashTool.schema()["function"]["parameters"]
    assert bash_params["required"] == ["command"]
    assert bash_params["properties"]["command"]["pattern"] == r"^.*\S.*$"

    edit_params = n.EditTool.schema()["function"]["parameters"]
    assert edit_params["required"] == ["path", "edits"]
    assert set(edit_params["properties"]) == {"path", "edits"}
    assert "start/end anchors are inclusive" in n.EditTool.schema()["function"]["description"]

    recall_keys = n.RecallTool.schema()["function"]["parameters"]["properties"]["keys"]
    assert recall_keys["items"]["pattern"] == r"^tr\.\d+$"

    read_params = n.ReadTool.schema()["function"]["parameters"]
    assert {"path", "ranges", "files"} <= set(read_params["properties"])

    note_params = n.NoteTool.schema()["function"]["parameters"]
    assert "minItems" not in note_params["properties"]["replace_plan"]
    assert note_params["properties"]["replace_plan"]["items"]["properties"]["status"]["enum"] == ["todo", "doing", "done", "blocked"]
    assert "minItems" not in note_params["properties"]["replace_known"]

    search_params = n.SearchTool.schema()["function"]["parameters"]
    assert {"pattern", "queries"} <= set(search_params["properties"])
    assert search_params["properties"]["queries"]["items"]["properties"]["context"]["type"] == "integer"

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
    assert n.ReadTool(n.Session(cwd="."), [{"path": "minacode.py"}]).targets()[0][1] == [(0, 0)]
    assert n.ModelClient.tool_payload("Search", {"pattern": "TODO"}) == [{"pattern": "TODO"}]
    assert n.ModelClient.tool_payload("Search", {"queries": [{"pattern": "TODO"}]}) == [{"pattern": "TODO"}]
    assert n.ModelClient.tool_payload("Note", {"set_goal": "ship"}) == [{"set_goal": "ship"}]


def test_note_tool_updates_durable_memory_without_result_key(tmp_path):
    s = session(tmp_path)
    s.state.known = ["existing"]
    runner = n.ToolRunner(s, n.ContextManager(s), output_fn=lambda text: None)

    output = []
    runner.output_fn = output.append
    runner.run(
        [
            n.ToolCall(
                "note",
                "Note",
                [
                    {
                        "set_goal": "ship",
                        "replace_plan": [{"status": "doing", "text": "inspect"}, {"status": "todo", "text": "patch"}],
                        "append_known": ["existing", "pytest"],
                    }
                ],
            )
        ]
    )

    assert s.state.goal == "ship"
    assert [vars(item) for item in s.state.plan] == [{"status": "doing", "text": "inspect"}, {"status": "todo", "text": "patch"}]
    assert s.state.known == ["existing", "pytest"]
    assert s.tool_records == []
    assert output == ["goal: ship\nplan:\n  - [~] inspect\n  - [ ] patch\nknown:\n  + pytest"]


def test_note_tool_validates_before_mutating_state(tmp_path):
    s = session(tmp_path)
    s.state.goal = "old goal"
    s.state.plan = ["old plan"]
    s.state.known = ["old fact"]

    with pytest.raises(n.ToolError) as error:
        n.NoteTool(s, [{"set_goal": "new goal", "replace_plan": "inspect"}]).call()

    assert str(error.value) == 'Note replace_plan must be an array of plan items, e.g. {"replace_plan":[{"status":"doing","text":"inspect"}]}'
    assert s.state.goal == "old goal"
    assert s.state.plan == ["old plan"]
    assert s.state.known == ["old fact"]

    with pytest.raises(n.ToolError, match="Note replace_plan status must be one of"):
        n.NoteTool(s, [{"replace_plan": [{"status": "started", "text": "inspect"}]}]).call()


def test_note_tool_replace_known(tmp_path):
    s = session(tmp_path)
    s.state.known = ["old fact"]
    runner = n.ToolRunner(s, n.ContextManager(s), output_fn=lambda text: None)

    short = runner.short_call(n.ToolCall("n", "Note", [{"replace_known": ["new fact a", "new fact b"]}]))
    assert short == "Note known:\n  new fact a\n  new fact b"

    output = []
    runner.output_fn = output.append
    runner.run([n.ToolCall("n", "Note", [{"replace_known": ["new fact a", "new fact b"]}])])
    assert s.state.known == ["new fact a", "new fact b"]
    assert output == ["known:\n  new fact a\n  new fact b"]

    runner.run([n.ToolCall("n", "Note", [{"replace_known": []}])])
    assert s.state.known == []


def test_note_tool_set_check(tmp_path):
    s = session(tmp_path)
    runner = n.ToolRunner(s, n.ContextManager(s), output_fn=lambda text: None)

    short = runner.short_call(n.ToolCall("n", "Note", [{"set_check": "pytest -q passed"}]))
    assert short == "Note check: pytest -q passed"

    output = []
    runner.output_fn = output.append
    runner.run([n.ToolCall("n", "Note", [{"set_check": "pytest -q passed"}])])
    assert s.state.check == "pytest -q passed"
    assert output == ["check: pytest -q passed"]


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


@pytest.mark.parametrize(
    ("original", "raw_edits"),
    [
        ("", [{"op": "create", "content": "a\nb"}]),
        ("aba\n", [{"op": "replace_all", "old": "a", "new": "A"}]),
        (
            "a\nb\nc\n",
            [
                {"op": "replace", "start": anchor(0, "a\n"), "end": anchor(0, "a\n"), "content": "A\n"},
                {"op": "insert_after", "start": anchor(1, "b\n"), "content": "x\n"},
                {"op": "delete", "start": anchor(2, "c\n"), "end": anchor(2, "c\n")},
            ],
        ),
        ("a\nb\n", [{"op": "insert_before", "start": anchor(1, "b\n"), "content": "inserted"}]),
        ("a\nb", [{"op": "delete", "start": anchor(1, "b"), "end": anchor(1, "b")}]),
    ],
)
def test_single_and_batch_edit_application_are_equivalent(tmp_path, original, raw_edits):
    tool = n.EditTool(session(tmp_path), ["code.txt", raw_edits])
    _path, edits = tool.parse()
    single = tool.apply(original, edits)
    original_lines = n.ReadTool.split_lines(original)
    plan = n.EditBatchPlan(tool.session)
    state = plan.FileState(
        "code.txt",
        [plan.Line(line, index) for index, line in enumerate(original_lines)],
        original_lines,
        edits[0].op != "create",
    )

    batch = plan.apply(tool, state, edits)

    assert "".join(line.text for line in batch.lines) == single.content
    assert batch.changes == single.changes
    assert batch.replacements == single.replacements
    assert batch.replace_all == single.replace_all


@pytest.mark.parametrize(
    ("original", "raw_edits"),
    [
        (
            "a\nb\nc\n",
            [
                {"op": "replace", "start": anchor(0, "a\n"), "end": anchor(1, "b\n"), "content": "x\n"},
                {"op": "delete", "start": anchor(1, "b\n"), "end": anchor(1, "b\n")},
            ],
        ),
        (
            "a\nb\n",
            [
                {"op": "replace_all", "old": "a", "new": "A"},
                {"op": "insert_before", "start": anchor(1, "b\n"), "content": "x\n"},
            ],
        ),
        ("a\n", [{"op": "replace_all", "old": "", "new": "x"}]),
        ("a\nb\n", [{"op": "delete", "start": anchor(1, "b\n"), "end": anchor(0, "a\n")}]),
    ],
)
def test_single_and_batch_edit_application_raise_the_same_error(tmp_path, original, raw_edits):
    tool = n.EditTool(session(tmp_path), ["code.txt", raw_edits])
    _path, edits = tool.parse()
    original_lines = n.ReadTool.split_lines(original)
    plan = n.EditBatchPlan(tool.session)
    state = plan.FileState("code.txt", [plan.Line(line, index) for index, line in enumerate(original_lines)], original_lines, True)

    with pytest.raises(n.ToolError) as single_error:
        tool.apply(original, edits)
    with pytest.raises(n.ToolError) as batch_error:
        plan.apply(tool, state, edits)

    assert str(batch_error.value) == str(single_error.value)


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
            n.ToolCall("create", "Edit", ["new.txt", [{"op": "create", "content": "a\nb\n"}]]),
            n.ToolCall("patch", "Edit", ["new.txt", [{"op": "replace", "start": anchor(1, "b\n"), "end": anchor(1, "b\n"), "content": "B\n"}]]),
        ]
    )

    assert (tmp_path / "new.txt").read_text(encoding="utf-8") == "a\nB\n"
    assert len([record for record in s.tool_records if record.name == "Edit"]) == 2
    assert s.tool_errors == []


def test_tool_runner_batch_edit_can_create_empty_then_patch_same_file(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(n.CodeIndex, "update", lambda self, paths: "")
    runner = n.ToolRunner(s, n.ContextManager(s), output_fn=lambda text: None)

    runner.run(
        [
            n.ToolCall("create", "Edit", ["empty.txt", [{"op": "create", "content": ""}]]),
            n.ToolCall("patch", "Edit", ["empty.txt", [{"op": "replace_all", "old": "", "new": "filled\n"}]]),
        ]
    )

    assert (tmp_path / "empty.txt").read_text(encoding="utf-8") == "filled\n"
    assert len([record for record in s.tool_records if record.name == "Edit"]) == 2
    assert s.tool_errors == []


def test_tool_runner_batch_edit_rejects_duplicate_create_same_file(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(n.CodeIndex, "update", lambda self, paths: "")
    runner = n.ToolRunner(s, n.ContextManager(s), output_fn=lambda text: None)

    runner.run(
        [
            n.ToolCall("create", "Edit", ["dup.txt", [{"op": "create", "content": "one\n"}]]),
            n.ToolCall("again", "Edit", ["dup.txt", [{"op": "create", "content": "two\n"}]]),
        ]
    )

    assert (tmp_path / "dup.txt").read_text(encoding="utf-8") == "one\n"
    assert len([record for record in s.tool_records if record.name == "Edit"]) == 1
    assert s.tool_errors and "file already exists" in s.tool_errors[0].error


def test_tool_runner_batch_edit_rejects_create_mixed_with_patch_ops(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(n.CodeIndex, "update", lambda self, paths: "")
    runner = n.ToolRunner(s, n.ContextManager(s), output_fn=lambda text: None)

    runner.run(
        [
            n.ToolCall(
                "bad",
                "Edit",
                ["bad.txt", [{"op": "create", "content": "one\n"}, {"op": "replace_all", "old": "one", "new": "two"}]],
            )
        ]
    )

    assert not (tmp_path / "bad.txt").exists()
    assert s.tool_records == []
    assert s.tool_errors and "create cannot be mixed" in s.tool_errors[0].error


def test_tool_runner_batch_edit_rejects_patch_missing_file_without_create(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(n.CodeIndex, "update", lambda self, paths: "")
    runner = n.ToolRunner(s, n.ContextManager(s), output_fn=lambda text: None)

    runner.run([n.ToolCall("patch", "Edit", ["missing.txt", [{"op": "replace_all", "old": "", "new": "x\n"}]])])

    assert not (tmp_path / "missing.txt").exists()
    assert s.tool_records == []
    assert s.tool_errors and "use op=create" in s.tool_errors[0].error


def test_edit_rejects_directory_target(tmp_path):
    s = session(tmp_path)
    (tmp_path / "pkg").mkdir()

    with pytest.raises(n.ToolError, match="path is a directory"):
        n.EditTool(s, ["pkg", [{"op": "replace_all", "old": "", "new": "x\n"}]]).call()


def test_tool_runner_batch_edit_rejects_directory_target(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(n.CodeIndex, "update", lambda self, paths: "")
    (tmp_path / "pkg").mkdir()
    runner = n.ToolRunner(s, n.ContextManager(s), output_fn=lambda text: None)

    runner.run([n.ToolCall("patch", "Edit", ["pkg", [{"op": "replace_all", "old": "", "new": "x\n"}]])])

    assert s.tool_records == []
    assert s.tool_errors and "path is a directory" in s.tool_errors[0].error


def test_tool_runner_batch_edit_create_and_existing_file_edit_are_independent(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(n.CodeIndex, "update", lambda self, paths: "")
    (tmp_path / "old.txt").write_text("a\nb\n", encoding="utf-8")
    runner = n.ToolRunner(s, n.ContextManager(s), output_fn=lambda text: None)

    runner.run(
        [
            n.ToolCall("create", "Edit", ["new.txt", [{"op": "create", "content": "n\n"}]]),
            n.ToolCall("edit", "Edit", ["old.txt", [{"op": "replace", "start": anchor(1, "b\n"), "end": anchor(1, "b\n"), "content": "B\n"}]]),
        ]
    )

    assert (tmp_path / "new.txt").read_text(encoding="utf-8") == "n\n"
    assert (tmp_path / "old.txt").read_text(encoding="utf-8") == "a\nB\n"
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
    assert "| x" in read_record.output
    assert "| c" in read_record.output
    assert "| C" not in read_record.output
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
    assert "current is anchor=1:" + n.ReadTool.line_hash("b\n") + " | b" in s.tool_errors[0].error
    assert path.read_text(encoding="utf-8") == "a\nb\n"


def test_batch_edit_no_change_reports_current_target_range(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(n.CodeIndex, "update", lambda self, paths: "")
    path = tmp_path / "code.txt"
    path.write_text("a\nb\n", encoding="utf-8")
    runner = n.ToolRunner(s, n.ContextManager(s), output_fn=lambda text: None)

    runner.run([n.ToolCall("noop", "Edit", ["code.txt", [{"op": "replace", "start": anchor(1, "b\n"), "end": anchor(1, "b\n"), "content": "b\n"}]])])

    assert s.tool_errors
    message = s.tool_errors[0].error
    assert "edit produced no changes; requested content already matches target range" in message
    assert "anchor=1:" + n.ReadTool.line_hash("b\n") + " | b" in message
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


def test_bash_timeout_applies_after_output_streams_close(tmp_path):
    s = session(tmp_path)
    s.settings.shell_timeout = 0.05

    output = n.BashTool(s, ["exec 1>&- 2>&-; sleep 1"]).call()

    assert "* exit_code: -1" in output
    assert "timeout" in output


def test_bash_fast_command_does_not_promote(tmp_path):
    s = session(tmp_path)
    s.settings.bash_wait_timeout = 5
    s.settings.shell_timeout = 30

    output = n.BashTool(s, ["printf hi"]).call()

    assert "* exit_code: 0" in output
    assert "hi" in output
    assert "backgrounded" not in output
    assert not s.jobs


def test_bash_slow_command_promotes_to_job(tmp_path):
    s = session(tmp_path)
    s.settings.bash_wait_timeout = 1
    s.settings.shell_timeout = 30

    output = n.BashTool(s, ["printf early; sleep 3; printf late"]).call()

    assert "* exit_code: -1" in output
    assert "early" in output
    assert "backgrounded after 1s" in output
    assert "job.1" in output
    assert "job.1" in s.jobs
    job = s.jobs["job.1"]
    assert job.stream_buffer is not None
    # `early` was consumed by the foreground streaming loop before promotion, so it lives in the
    # Bash result payload (asserted above). `late` was produced after the drainer took over, so it
    # lives in the promoted job's tail buffer.
    job.process.wait(timeout=10)
    for _ in range(50):
        if "late" in job.tail(4096):
            break
        import time as _t

        _t.sleep(0.05)
    assert "late" in job.tail(4096)
    job.update_status()
    assert job.status == "done"
    assert job.exit_code == 0


def test_bash_promotion_disabled_when_wait_timeout_zero(tmp_path):
    s = session(tmp_path)
    s.settings.bash_wait_timeout = 0
    s.settings.shell_timeout = 1

    output = n.BashTool(s, ["sleep 5"]).call()

    assert "* exit_code: -1" in output
    assert "timeout" in output
    assert "backgrounded" not in output
    assert not s.jobs


def test_bash_promoted_job_is_killable(tmp_path):
    s = session(tmp_path)
    s.settings.bash_wait_timeout = 1
    s.settings.shell_timeout = 30

    n.BashTool(s, ["sleep 60"]).call()
    assert "job.1" in s.jobs
    job = s.jobs["job.1"]
    job.kill()
    assert job.status in {"done", "killed"}
    assert job.process.poll() is not None


def test_tool_runner_starts_bash_live_preview_before_output(tmp_path):
    s = session(tmp_path)
    s.settings.yolo = True
    events = []
    runner = n.ToolRunner(
        s, n.ContextManager(s), input_fn=lambda prompt: (_ for _ in ()).throw(AssertionError("unexpected prompt")), output_fn=lambda text: None
    )
    runner.live_start = lambda: events.append(("start", ""))
    runner.live_output = lambda stream, text: events.append((stream, text))

    runner.run([n.ToolCall("bash", "Bash", ["printf live"])])

    assert events[0] == ("start", "")
    assert ("stdout", "live") in events
    assert events[-1] == ("", "")


def test_tool_runner_prints_bash_header_before_live_output(tmp_path):
    s = session(tmp_path)
    events = []
    runner = n.ToolRunner(
        s,
        n.ContextManager(s),
        input_fn=lambda prompt: (_ for _ in ()).throw(AssertionError("unexpected prompt")),
        output_fn=lambda text: events.append(("display", str(text))),
    )
    runner.live_start = lambda: events.append(("start", ""))
    runner.live_output = lambda stream, text: events.append((stream, text))

    runner.run([n.ToolCall("bash", "Bash", ["printf live"])])

    assert events[0] == ("display", "  Bash  printf live")
    assert events[1] == ("start", "")
    assert ("stdout", "live") in events
    assert events[-1][0] == "display"
    assert "    ├ output" in events[-1][1]
    assert "Ctrl-O for more" in events[-1][1]
    assert "live" in events[-1][1]
    assert "    └ stored tr." in events[-1][1]
    assert sum("printf live" in text for kind, text in events if kind == "display") == 1
    assert sum("Bash" in text for kind, text in events if kind == "display") == 1
    assert "live" in s.tool_records[-1].output


def test_tool_runner_approved_live_bash_does_not_repeat_command(tmp_path):
    s = session(tmp_path)
    events = []
    runner = n.ToolRunner(s, n.ContextManager(s), input_fn=lambda prompt: "", output_fn=lambda text: events.append(("display", str(text))))
    runner.live_start = lambda: events.append(("start", ""))
    runner.live_output = lambda stream, text: events.append((stream, text))

    runner.run([n.ToolCall("bash", "Bash", ["bash -lc 'printf approved'"])])

    display = [text for kind, text in events if kind == "display"]
    assert display[0].startswith("  Bash  ")
    assert "approval required" not in display[0]
    assert display[-1].startswith("    ├ output")
    assert "Ctrl-O for more" in display[-1]
    assert "    └ stored tr." in display[-1]
    assert display[-1].endswith("[approved]")
    assert sum(text.startswith("  Bash  ") for text in display) == 1
    assert sum("printf approved" in text for text in display) == 1


def test_tool_runner_finish_display_bounds_bash_output(tmp_path):
    s = session(tmp_path)
    runner = n.ToolRunner(s, n.ContextManager(s), output_fn=lambda text: None)
    stdout = "\n".join(f"out {index}" for index in range(20))
    output = n.Tool.process_result("BashToolResult", 0, stdout, "err")

    display = str(runner.finish_display(n.ToolCall("bash", "Bash", ["printf lots"]), "tr.1", output, failed=False))

    assert display.startswith("  Bash  printf lots\n")
    assert "    ├ output Ctrl-O for more" in display
    assert "out 0" in display
    assert "... 17 lines omitted ..." in display
    assert "out 18" in display and "out 19" in display
    assert "err" in display
    assert display.endswith("    └ stored tr.1")


def test_tool_runner_finish_display_keeps_bounded_bash_output_after_live_preview(tmp_path):
    s = session(tmp_path)
    runner = n.ToolRunner(s, n.ContextManager(s), output_fn=lambda text: None)
    output = n.Tool.process_result("BashToolResult", 0, "live output", "")

    display = str(runner.finish_display(n.ToolCall("bash", "Bash", ["printf live"]), "tr.1", output, failed=False))

    assert "    ├ output Ctrl-O for more" in display
    assert "live output" in display
    assert display.endswith("    └ stored tr.1")


def test_tool_runner_compact_bash_result_keeps_bounded_output_without_live_frame(tmp_path):
    s = session(tmp_path)
    runner = n.ToolRunner(s, n.ContextManager(s), output_fn=lambda text: None)
    output = n.Tool.process_result("BashToolResult", 0, "visible output", "")

    display = str(
        runner.finish_display(
            n.ToolCall("bash", "Bash", ["printf visible"]),
            "tr.1",
            output,
            failed=False,
            d=n.ToolDisplay(nested_display=True),
        )
    )

    assert display.startswith("    ├ output Ctrl-O for more")
    assert "visible output" in display


def test_tool_runner_failed_live_bash_does_not_repeat_command(tmp_path, monkeypatch):
    s = session(tmp_path)
    output = []
    runner = n.ToolRunner(s, n.ContextManager(s), output_fn=lambda text: output.append(str(text)))
    runner.live_start = lambda: None
    runner.live_output = lambda _stream, _text: None
    monkeypatch.setattr(n.subprocess, "Popen", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("spawn failed")))

    runner.run([n.ToolCall("bash", "Bash", ["printf duplicate"])])

    assert output[0] == "  Bash  printf duplicate"
    assert output[1].startswith("    └ error ")
    assert "printf duplicate" not in output[1]
    assert "spawn failed" in output[1]


def test_tool_runner_bash_preview_keeps_literal_closing_tags(tmp_path):
    s = session(tmp_path)
    runner = n.ToolRunner(s, n.ContextManager(s), output_fn=lambda text: None)
    output = n.Tool.process_result("BashToolResult", 0, "before </stdout> after", "before </stderr> after")

    preview = runner.bash_result_preview(output)

    assert "before </stdout> after" in preview
    assert "before </stderr> after" in preview


def test_tool_runner_bash_preview_omits_past_limit(tmp_path):
    s = session(tmp_path)
    runner = n.ToolRunner(s, n.ContextManager(s), output_fn=lambda text: None)
    lines = [f"line {index}" for index in range(n.ToolRunner.BASH_PREVIEW_LINES + 1)]

    preview = runner.preview_lines("\n".join(lines))

    assert len(preview) == n.ToolRunner.BASH_PREVIEW_LINES + 1
    assert preview[0] == "line 0"
    assert preview[n.ToolRunner.BASH_PREVIEW_LINES // 2] == "... 1 line omitted ..."
    assert preview[-1] == lines[-1]


def test_bash_live_preview_skips_unchanged_redraws(monkeypatch):
    printed = []
    now = [100.4]
    monkeypatch.setattr(n.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(n.render, "print_formatted_text", lambda ft, **kw: printed.append("".join(t for _, t in ft)))

    class FakeOut:
        def write_raw(self, s=""):
            pass

        def erase_end_of_line(self):
            pass

        def flush(self):
            pass

    p = n.BashLivePreview()
    p.output = FakeOut()
    p.active = True
    p.started_at = 100.0

    p.render()
    first = len(printed)
    p.render()
    assert len(printed) == first

    now[0] = 101.1
    p.render()
    assert len(printed) > first
    # BashLivePreview uses sub-second precision (`1.1s`) so the ticker feels live.
    assert any("  ├ running… 1.1s" in line for line in printed[first:])


def test_code_index_updates_after_file_mutation_tools(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.settings.yolo = True
    updated = []
    monkeypatch.setattr(n.CodeIndex, "update", lambda self, paths: updated.extend(paths) or "")
    runner = n.ToolRunner(
        s, n.ContextManager(s), input_fn=lambda prompt: (_ for _ in ()).throw(AssertionError("unexpected prompt")), output_fn=lambda text: None
    )

    runner.run([n.ToolCall("empty", "Edit", ["empty.py", [{"op": "create", "content": ""}]])])
    runner.run([n.ToolCall("create", "Edit", ["made.py", [{"op": "create", "content": "print(1)\n"}]])])
    runner.run([n.ToolCall("edit", "Edit", ["made.py", [{"op": "replace_all", "old": "1", "new": "2"}]])])

    assert (tmp_path / "made.py").read_text(encoding="utf-8") == "print(2)\n"
    assert updated == ["empty.py", "made.py", "made.py"]


def test_edit_index_update_uses_call_path_when_output_path_is_unparseable(tmp_path, monkeypatch):
    s = session(tmp_path)
    updated = []
    monkeypatch.setattr(n.CodeIndex, "update", lambda self, paths: updated.extend(paths) or "")

    n.ToolRunner(s, n.ContextManager(s), output_fn=lambda text: None).update_code_index(
        n.ToolCall("edit", "Edit", ["made.py", [{"op": "create", "content": "x\n"}]]),
        "<Edit path=bad />",
    )

    assert updated == ["made.py"]


def test_yolo_approves_mutating_tools_without_prompt(tmp_path):
    s = session(tmp_path)
    s.settings.yolo = True
    runner = n.ToolRunner(
        s, n.ContextManager(s), input_fn=lambda prompt: (_ for _ in ()).throw(AssertionError("unexpected prompt")), output_fn=lambda text: None
    )

    runner.run([n.ToolCall("create", "Edit", ["auto.txt", [{"op": "create", "content": "ok\n"}]])])

    assert (tmp_path / "auto.txt").read_text(encoding="utf-8") == "ok\n"
    assert len(s.tool_records) == 1


def test_gitignore_cache_populated_and_reused(tmp_path):
    """Cache stores parsed patterns and reuses them on subsequent calls."""
    (tmp_path / ".gitignore").write_text("ignored.txt\nbuild/\n", encoding="utf-8")
    s = session(tmp_path)
    tool = n.SearchTool(s, [{"pattern": "x"}])

    # First call populates the cache
    patterns1 = tool.gitignore_patterns(str(tmp_path))
    assert "ignored.txt" in patterns1
    assert "build/" in patterns1

    # Cache should exist for the workspace .gitignore
    ws_gitignore = str(tmp_path / ".gitignore")
    assert ws_gitignore in s._gitignore_cache
    cached_mtime, cached_patterns = s._gitignore_cache[ws_gitignore]
    assert cached_patterns == patterns1

    # Second call reuses cache (mtime unchanged)
    patterns2 = tool.gitignore_patterns(str(tmp_path))
    assert patterns2 == patterns1
    # Cache entry unchanged
    assert s._gitignore_cache[ws_gitignore][0] == cached_mtime


def test_gitignore_cache_invalidates_on_file_change(tmp_path):
    """Cache re-reads .gitignore when mtime changes."""
    gitignore = tmp_path / ".gitignore"
    gitignore.write_text("old.txt\n", encoding="utf-8")
    s = session(tmp_path)
    tool = n.SearchTool(s, [{"pattern": "x"}])

    patterns1 = tool.gitignore_patterns(str(tmp_path))
    assert patterns1 == ["old.txt"]

    ws_gitignore = str(gitignore)
    old_mtime = s._gitignore_cache[ws_gitignore][0]

    # Modify the .gitignore file
    gitignore.write_text("new.txt\n", encoding="utf-8")
    patterns2 = tool.gitignore_patterns(str(tmp_path))
    assert patterns2 == ["new.txt"]

    new_mtime = s._gitignore_cache[ws_gitignore][0]
    assert new_mtime != old_mtime


def test_gitignore_cache_cleanup_on_file_delete(tmp_path):
    """Cache entry is removed when .gitignore is deleted."""
    gitignore = tmp_path / ".gitignore"
    gitignore.write_text("delete_me.txt\n", encoding="utf-8")
    s = session(tmp_path)
    tool = n.SearchTool(s, [{"pattern": "x"}])

    tool.gitignore_patterns(str(tmp_path))
    ws_gitignore = str(gitignore)
    assert ws_gitignore in s._gitignore_cache

    # Delete the .gitignore file
    gitignore.unlink()
    patterns = tool.gitignore_patterns(str(tmp_path))
    assert patterns == []
    assert ws_gitignore not in s._gitignore_cache


def test_gitignore_cache_shared_across_tools(tmp_path):
    """SearchTool instances share the same gitignore cache via Session."""
    (tmp_path / ".gitignore").write_text("secret.log\n", encoding="utf-8")
    s = session(tmp_path)

    find = n.SearchTool(s, [{"pattern": "x"}])
    search = n.SearchTool(s, [{"pattern": "needle", "path": "."}])

    # Find populates the cache
    find_patterns = find.gitignore_patterns(str(tmp_path))
    assert find_patterns == ["secret.log"]

    # Search reuses the same cache entry
    search_patterns = search.gitignore_patterns(str(tmp_path))
    assert search_patterns == find_patterns

    ws_key = str(tmp_path / ".gitignore")
    assert ws_key in s._gitignore_cache
    # Only one cache entry, not duplicated
    assert len(s._gitignore_cache) == 1


def test_gitignore_cache_keyed_by_root(tmp_path):
    """Different root directories cache independently."""
    sub = tmp_path / "sub"
    sub.mkdir()
    (tmp_path / ".gitignore").write_text("root_ignored.txt\n", encoding="utf-8")
    (sub / ".gitignore").write_text("sub_ignored.txt\n", encoding="utf-8")

    s = session(tmp_path)
    tool = n.SearchTool(s, [{"pattern": "x"}])

    # Root patterns include only workspace .gitignore
    root_patterns = tool.gitignore_patterns(str(tmp_path))
    assert "root_ignored.txt" in root_patterns
    assert "sub_ignored.txt" not in root_patterns

    # Sub patterns include workspace + sub .gitignore
    sub_patterns = tool.gitignore_patterns(str(sub))
    assert "root_ignored.txt" in sub_patterns  # workspace always included
    assert "sub_ignored.txt" in sub_patterns

    # Two separate cache entries
    assert len(s._gitignore_cache) == 2


def test_gitignore_cache_noop_when_no_gitignore(tmp_path):
    """When no .gitignore exists, returns empty list and cache stays empty."""
    s = session(tmp_path)
    tool = n.SearchTool(s, [{"pattern": "x"}])

    patterns = tool.gitignore_patterns(str(tmp_path))
    assert patterns == []
    assert len(s._gitignore_cache) == 0


def test_gitignore_cache_preserves_order(tmp_path):
    """After a no-op stat (no change), patterns come from cache unchanged."""
    (tmp_path / ".gitignore").write_text("a.txt\nb.txt\n", encoding="utf-8")
    s = session(tmp_path)
    tool = n.SearchTool(s, [{"pattern": "x"}])

    p1 = tool.gitignore_patterns(str(tmp_path))
    p2 = tool.gitignore_patterns(str(tmp_path))

    # Same object identity isn't required, but content must match
    assert p1 == p2 == ["a.txt", "b.txt"]


def test_gitignore_line_filtering_unchanged(tmp_path):
    """Cache still filters blank lines, comments, and negation patterns."""
    (tmp_path / ".gitignore").write_text("keep.txt\n\n  # comment\n!negated.txt\n  \n", encoding="utf-8")
    s = session(tmp_path)
    tool = n.SearchTool(s, [{"pattern": "x"}])

    patterns = tool.gitignore_patterns(str(tmp_path))
    assert patterns == ["keep.txt"]

    assert s.tool_errors == []


# ---------------------------------------------------------------------------
# AskTool
# ---------------------------------------------------------------------------


def _q(*items):
    """Wrap question item dicts into the Ask tool payload args."""
    return [{"questions": list(items)}]


def test_ask_tool_schema():
    """params_schema requires a questions array of question objects, strict."""
    schema = n.AskTool.params_schema()
    assert schema["type"] == "object"
    assert schema["required"] == ["questions"]
    assert schema["additionalProperties"] is False
    questions = schema["properties"]["questions"]
    assert questions["type"] == "array"
    assert questions["minItems"] == 1
    item = questions["items"]
    assert item["required"] == ["question"]
    assert item["additionalProperties"] is False
    props = item["properties"]
    assert props["question"]["type"] == "string"
    assert props["choices"]["items"]["type"] == "string"
    assert props["previews"]["items"]["type"] == "string"
    assert props["recommended"]["type"] == "integer"


def test_ask_tool_registered():
    """AskTool is in TOOLS and TOOL_REGISTRY."""
    assert n.AskTool.NAME == "Ask"
    assert n.AskTool in n.TOOLS
    assert n.TOOL_REGISTRY["Ask"] is n.AskTool
    assert "Question" not in n.TOOL_REGISTRY
    assert not hasattr(n, "QuestionTool")


def test_ask_tool_call_basic(tmp_path):
    """call() returns question text when question_fn is None."""
    s = session(tmp_path)
    assert n.AskTool(s, _q({"question": "Which approach?"})).call() == "Which approach?"


def test_ask_tool_call_with_choices(tmp_path):
    """call() accepts choices and returns fallback question text."""
    s = session(tmp_path)
    assert n.AskTool(s, _q({"question": "Which?", "choices": ["A", "B"]})).call() == "Which?"


def test_ask_tool_call_with_choices_and_previews(tmp_path):
    """call() accepts choices + previews."""
    s = session(tmp_path)
    tool = n.AskTool(
        s,
        _q(
            {
                "question": "Which?",
                "choices": ["A", "B"],
                "previews": ["Preview A", "Preview B"],
            }
        ),
    )
    assert tool.call() == "Which?"


def test_ask_tool_call_invokes_callback(tmp_path):
    """call() invokes question_fn with question/choices/previews/recommended."""
    s = session(tmp_path)
    calls = []

    def fake_fn(spec, position):
        calls.append((spec, position))
        return "user chose B"

    tool = n.AskTool(s, _q({"question": "A or B?", "choices": ["A", "B"], "previews": ["PA", "PB"], "recommended": 1}))
    tool.question_fn = fake_fn
    result = tool.call()
    assert result == "user chose B"
    (spec, position) = calls[0]
    assert (spec.question, spec.choices, spec.previews, spec.recommended) == ("A or B?", ["A", "B"], ["PA", "PB"], 1)
    assert position == ""  # a single question carries no position indicator


def test_ask_tool_call_multiple_questions(tmp_path):
    """call() asks each question in sequence and labels the combined answers."""
    s = session(tmp_path)
    asked = []

    def fake_fn(spec, position):
        asked.append((spec.question, position))
        return {"Runtime?": "Node", "Name?": "core"}[spec.question]

    tool = n.AskTool(
        s,
        _q(
            {"question": "Runtime?", "choices": ["Node", "Deno"]},
            {"question": "Name?"},
        ),
    )
    tool.question_fn = fake_fn
    result = tool.call()
    assert asked == [("Runtime?", "1/2"), ("Name?", "2/2")]  # sequential, with position
    assert result == "Q: Runtime?\nA: Node\n\nQ: Name?\nA: core"


def test_tool_runner_finish_display_keeps_ask_answer(tmp_path):
    s = session(tmp_path)
    runner = n.ToolRunner(s, n.ContextManager(s), output_fn=lambda text: None)

    display = str(runner.finish_display(n.ToolCall("ask", "Ask", _q({"question": "Which?"})), "tr.1", "typed answer", failed=False))

    assert display.startswith("  Ask  Which? → tr.1\n")
    assert display.endswith("    └ answer typed answer")


def test_ask_tool_validates_batch_before_asking(tmp_path):
    """A malformed later question raises before any question is asked."""
    s = session(tmp_path)
    asked = []

    def fake_fn(spec, position):
        asked.append(spec.question)
        return "x"

    tool = n.AskTool(
        s,
        _q(
            {"question": "First?", "choices": ["A"]},
            {"question": "Second?", "choices": ["A", "B"], "recommended": 5},  # out of range
        ),
    )
    tool.question_fn = fake_fn
    with pytest.raises(n.ToolError, match="valid 0-based choice index"):
        tool.call()
    assert asked == []  # validation happens up front, so nothing was asked


def test_ask_tool_call_callback_passthrough_choices_none(tmp_path):
    """call() passes choices/previews/recommended as None when not provided."""
    s = session(tmp_path)
    calls = []

    def fake_fn(spec, position):
        calls.append((spec, position))
        return "free text answer"

    tool = n.AskTool(s, _q({"question": "Name?"}))
    tool.question_fn = fake_fn
    assert tool.call() == "free text answer"
    (spec, position) = calls[0]
    assert spec.choices is None
    assert spec.previews is None
    assert spec.recommended is None
    assert position == ""


def test_ask_tool_call_empty_question_raises(tmp_path):
    """call() raises ToolError for empty/missing question text."""
    s = session(tmp_path)
    with pytest.raises(n.ToolError, match="each question requires a 'question' field"):
        n.AskTool(s, _q({"question": ""})).call()
    with pytest.raises(n.ToolError, match="each question requires a 'question' field"):
        n.AskTool(s, _q({})).call()


def test_ask_tool_call_empty_list_raises(tmp_path):
    """call() raises ToolError when questions list is missing or empty."""
    s = session(tmp_path)
    with pytest.raises(n.ToolError, match="non-empty 'questions' list"):
        n.AskTool(s, [{"questions": []}]).call()
    with pytest.raises(n.ToolError, match="non-empty 'questions' list"):
        n.AskTool(s, [{}]).call()


def test_ask_tool_call_invalid_args_raises(tmp_path):
    """call() raises ToolError for malformed top-level args."""
    s = session(tmp_path)
    with pytest.raises(n.ToolError, match="Ask requires named fields"):
        n.AskTool(s, ["just a string"]).call()
    with pytest.raises(n.ToolError, match="Ask requires named fields"):
        n.AskTool(s, []).call()


def test_ask_tool_call_invalid_choices_raises(tmp_path):
    """call() validates choices type."""
    s = session(tmp_path)
    with pytest.raises(n.ToolError, match="Ask choices must be a list of strings"):
        n.AskTool(s, _q({"question": "Q", "choices": "not-a-list"})).call()
    with pytest.raises(n.ToolError, match="Ask choices must be a list of strings"):
        n.AskTool(s, _q({"question": "Q", "choices": [1, 2, 3]})).call()


def test_ask_tool_call_invalid_previews_raises(tmp_path):
    """call() validates previews type and length."""
    s = session(tmp_path)
    with pytest.raises(n.ToolError, match="Ask previews must be a list of strings"):
        n.AskTool(s, _q({"question": "Q", "choices": ["A"], "previews": [1]})).call()
    with pytest.raises(n.ToolError, match="Ask previews must match choices length"):
        n.AskTool(s, _q({"question": "Q", "choices": ["A", "B"], "previews": ["only one"]})).call()


def test_ask_tool_call_no_previews_with_choices(tmp_path):
    """call() allows choices without previews."""
    s = session(tmp_path)
    assert n.AskTool(s, _q({"question": "Q", "choices": ["A", "B"]})).call() == "Q"


def test_ask_tool_call_invalid_recommended_raises(tmp_path):
    """call() validates recommended is an in-range choice index."""
    s = session(tmp_path)
    with pytest.raises(n.ToolError, match="valid 0-based choice index"):
        n.AskTool(s, _q({"question": "Q", "choices": ["A", "B"], "recommended": 2})).call()
    with pytest.raises(n.ToolError, match="valid 0-based choice index"):
        n.AskTool(s, _q({"question": "Q", "recommended": 0})).call()  # no choices
    with pytest.raises(n.ToolError, match="valid 0-based choice index"):
        n.AskTool(s, _q({"question": "Q", "choices": ["A"], "recommended": True})).call()  # bool not int


def test_ask_tool_short_args(tmp_path):
    """short_args() shows the first question and a count of the rest."""
    s = session(tmp_path)
    tool = n.AskTool(s, _q({"question": "Which approach should I use?"}))
    args = tool.short_args()
    assert len(args) == 1
    assert "Which approach" in args[0]
    assert "more" not in args[0]
    multi = n.AskTool(s, _q({"question": "First?"}, {"question": "Second?"}))
    assert "(+1 more)" in multi.short_args()[0]
    assert len(n.AskTool(s, []).short_args()) == 1


def test_ask_tool_wired_in_tool_runner(tmp_path):
    """ToolRunner injects question_fn into AskTool instances."""
    s = session(tmp_path)
    ctx = n.ContextManager(s)
    captured = []

    def fake_question_fn(spec, position):
        captured.append((spec, position))
        return "test answer"

    runner = n.ToolRunner(s, ctx, output_fn=lambda text: None)
    runner.question_fn = fake_question_fn
    results = runner.run([n.ToolCall("q", "Ask", [{"questions": [{"question": "A or B?", "choices": ["A", "B"], "recommended": 0}]}])])
    assert len(results) == 1
    assert results[0]["tool_call_id"] == "q"
    assert results[0]["role"] == "tool"
    assert "test answer" in results[0]["content"]
    (spec, position) = captured[0]
    assert (spec.question, spec.choices, spec.recommended, position) == ("A or B?", ["A", "B"], 0, "")


def test_ask_tool_schema_strict(tmp_path):
    """schema() enforces additionalProperties=False at both levels."""
    schema = n.AskTool.schema()
    params = schema["function"]["parameters"]
    assert params["additionalProperties"] is False
    assert "questions" in params["properties"]
    item = params["properties"]["questions"]["items"]
    assert item["additionalProperties"] is False
    assert "question" in item["properties"]
    assert "choices" in item["properties"]
    assert "previews" in item["properties"]


def test_auto_approved_tool_prints_single_line_with_tag(tmp_path):
    # In yolo mode a confirmation-requiring tool without a preview (Bash) should print only the
    # result line tagged [auto], not a redundant "auto …" pre-line that duplicates the header.
    s = session(tmp_path)
    s.settings.yolo = True
    out = []
    runner = n.ToolRunner(s, n.ContextManager(s), output_fn=lambda text: out.append(str(text)))
    runner.run([n.ToolCall("b0", "Bash", [":"])])
    assert len(out) == 1
    assert out[0].startswith("  Bash  ")
    assert out[0].rstrip().endswith("[auto]")
    assert sum(line.startswith("  Bash  ") for line in out) == 1


def test_auto_approved_edit_keeps_preview_pre_line(tmp_path, monkeypatch):
    # Edit's "auto …" pre-line carries the approval preview; the result line is tagged [auto].
    s = session(tmp_path)
    s.settings.yolo = True
    monkeypatch.setattr(n.CodeIndex, "update", lambda self, paths: "")
    (tmp_path / "a.txt").write_text("hello\nworld\n", encoding="utf-8")
    out = []
    runner = n.ToolRunner(s, n.ContextManager(s), output_fn=out.append)
    runner.run([n.ToolCall("e0", "Edit", ["a.txt", [{"op": "insert_after", "start": anchor(0, "hello\n"), "content": "NEW\n"}]])])
    assert len(out) == 2
    assert isinstance(out[0], n.LogBlock)
    root, _level = next(out[0].walk())
    assert root.role is n.LogRole.AUTO
    assert "preview" in str(out[0])
    assert str(out[1]).rstrip().endswith("[auto]")
