"""@file: mentions: grammar (T1), completion matching/ranking (T3), the path source (T4),
the FILE MENTIONS resolver (T5), and the 50k-path performance guard (T7)."""

import os
import shutil
import subprocess
import tempfile
import threading
import time

from agent_harness import session
from prompt_toolkit.document import Document

from minacode.cli.view import CommandCompleter
from minacode.mentions import FileMentions, FzfPicker, active_mention, encode_file_mention, scan_mentions


def completions(completer, text):
    return [c.text for c in completer.get_completions(Document(text), None)]


# --- T1: grammar ---


def test_file_mention_parses_and_email_does_not(tmp_path):
    s = session(tmp_path)
    (tmp_path / "a.py").write_text("print(1)\n", encoding="utf-8")
    assert "[a.py] 1 lines" in s.mentions.resolve_mentions("see @file:a.py")
    assert s.mentions.resolve_mentions("mail hit9@icloud.com") == ""  # @ follows a word char
    assert s.mentions.resolve_mentions("see @file:a.py and @file:a.py")  # both captured, deduped


def test_mcp_and_skill_forms_parse(tmp_path):
    s = session(tmp_path)
    assert [(span.kind, span.payload) for span in scan_mentions("@github @mcp:github @mcp:github.search @file:a.py @skill:release")] == [
        ("bare", "github"),
        ("mcp", "github"),
        ("mcp", "github.search"),
        ("file", "a.py"),
        ("skill", "release"),
    ]
    assert s.skills.resolve_mentions("price $30") == ""  # a price is not a skill mention
    assert not [span for span in scan_mentions("mail hit9@icloud.com") if span.kind == "bare"]


def test_file_mention_quoted_round_trip_and_incomplete_span():
    for path in ("src/app.py", "docs/design notes.txt", "文档/设计.txt", "odd\tname.txt", "odd\nname.txt"):
        mention = encode_file_mention(path)
        spans = scan_mentions("see " + mention)
        assert spans[-1].payload == path
        assert spans[-1].complete
    span = active_mention('see @file:"unfinished path')
    assert span is not None and span.kind == "file" and not span.complete and span.payload == "unfinished path"


# --- T3: matching and ranking ---


FILES = (
    ("minacode/cli/view.py", "minacode/cli/view.py"),
    ("minacode/tui.py", "minacode/tui.py"),
    ("minacode/hints.py", "minacode/hints.py"),
)


def test_matching_substring_and_case_insensitive():
    c = CommandCompleter(files=lambda: FILES)
    assert completions(c, "@file:view") == ["@file:minacode/cli/view.py"]
    assert completions(c, "@file:cli/view") == ["@file:minacode/cli/view.py"]  # whole-path substring
    assert completions(c, "@file:VIEW") == ["@file:minacode/cli/view.py"]  # case-insensitive
    assert completions(c, "@file:minacode/tui.py") == ["@file:minacode/tui.py"]


def test_matching_ranks_basename_prefix_substring_then_path():
    files = (
        ("z/notes/deep.txt", "z/notes/deep.txt"),  # path substring only
        ("a/notes.txt", "a/notes.txt"),  # basename prefix, shortest
        ("b/notes.txt", "b/notes.txt"),  # basename prefix, tie with a/ by length, alphabetical
        ("b2/notes2.txt", "b2/notes2.txt"),  # basename prefix, longer
        ("c/xnotes.txt", "c/xnotes.txt"),  # basename substring
    )
    c = CommandCompleter(files=lambda: files)
    assert completions(c, "@file:notes") == [
        "@file:a/notes.txt",
        "@file:b/notes.txt",
        "@file:b2/notes2.txt",
        "@file:c/xnotes.txt",
        "@file:z/notes/deep.txt",
    ]
    # Deterministic: the same query answers the same way twice.
    assert completions(c, "@file:notes") == completions(c, "@file:notes")


def test_matching_caps_at_50_rows():
    paths = tuple((f"dir/f{i}.py", f"dir/f{i}.py") for i in range(60))
    c = CommandCompleter(files=lambda: paths)
    assert len(completions(c, "@file:f")) == 50


def test_bare_menu_does_not_scan_or_merge_repository_files():
    c = CommandCompleter(mcp_servers=lambda: ("viewer",), skills=lambda: (), files=lambda: FILES)
    assert completions(c, "@view") == ["@mcp:viewer"]
    assert completions(c, "@cli/view") == []


def test_kind_completion_keeps_canonical_at_prefix():
    c = CommandCompleter(mcp_servers=lambda: ("github", "gitlab"), skills=lambda: ("release",), files=lambda: FILES)
    assert completions(c, "use @") == ["@file:", "@mcp:", "@skill:"]
    assert completions(c, "use @file:vie") == ["@file:minacode/cli/view.py"]
    assert completions(c, "use @mcp:git") == ["@mcp:github", "@mcp:gitlab"]
    assert completions(c, "use @skill:rel") == ["@skill:release"]


# --- T4: source ---


def test_path_source_non_git_walk(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "x.js").write_text("", encoding="utf-8")
    (tmp_path / ".hidden.py").write_text("", encoding="utf-8")
    s = session(tmp_path)
    rels = {rel for _lower, rel in s.mentions.paths()}
    assert "src/a.py" in rels
    assert "node_modules/x.js" in rels  # only ignore rules and .git exclude paths
    assert ".hidden.py" in rels


def test_path_source_git_repo_and_untracked_appears(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=tmp_path, check=True)
    (tmp_path / "tracked.py").write_text("", encoding="utf-8")
    (tmp_path / "tracked.tmp").write_text("", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=tmp_path, check=True)
    (tmp_path / "untracked.py").write_text("", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("*.tmp\n", encoding="utf-8")
    (tmp_path / "ignored.tmp").write_text("", encoding="utf-8")
    s = session(tmp_path)
    rels = {rel for _lower, rel in s.mentions.paths()}
    assert "tracked.py" in rels
    assert "untracked.py" in rels  # git ls-files -o --exclude-standard
    assert "ignored.tmp" not in rels
    assert "tracked.tmp" not in rels  # product rule also excludes tracked files matching ignore
    assert not any(".git" in rel.split("/") for rel in rels)


def test_path_source_git_nested_ignore_negation_and_unusual_names(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / ".gitignore").write_text("*.tmp\n!keep.tmp\n", encoding="utf-8")
    (nested / "drop.tmp").write_text("", encoding="utf-8")
    (nested / "keep.tmp").write_text("", encoding="utf-8")
    (nested / "中文 name.txt").write_text("", encoding="utf-8")
    (nested / "line\nbreak.txt").write_text("", encoding="utf-8")

    rels = {rel for _lower, rel in session(tmp_path).mentions.paths()}
    assert "nested/drop.tmp" not in rels
    assert "nested/keep.tmp" in rels
    assert "nested/中文 name.txt" in rels
    assert "nested/line\nbreak.txt" in rels


def test_git_candidate_queries_run_concurrently(monkeypatch, tmp_path):
    mentions = session(tmp_path).mentions
    started = threading.Barrier(2)

    def git_ls_files(flags):
        started.wait(timeout=5)
        return ["a.py"] if "--others" in flags else []

    monkeypatch.setattr(mentions, "_git_ls_files", git_ls_files)

    assert mentions._git_paths() == ["a.py"]


def test_python_fallback_honors_nested_gitignore(monkeypatch, tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / ".gitignore").write_text("*.tmp\n!keep.tmp\n", encoding="utf-8")
    (nested / "drop.tmp").write_text("", encoding="utf-8")
    (nested / "keep.tmp").write_text("", encoding="utf-8")
    monkeypatch.setattr("minacode.mentions.shutil.which", lambda _name: None)

    rels = {rel for _lower, rel in session(tmp_path).mentions.paths()}
    assert "nested/drop.tmp" not in rels
    assert "nested/keep.tmp" in rels


def test_path_cache_refreshes_after_window(tmp_path):
    (tmp_path / "a.py").write_text("", encoding="utf-8")
    s = session(tmp_path)
    s.mentions._paths_cache = (time.monotonic() - 10, (("a.py", "a.py"),))  # stale: only a.py
    (tmp_path / "b.py").write_text("", encoding="utf-8")
    assert {rel for _lower, rel in s.mentions.paths()} == {"a.py", "b.py"}
    s.mentions._paths_cache = (time.monotonic(), (("a.py", "a.py"),))  # fresh: kept as is
    assert {rel for _lower, rel in s.mentions.paths()} == {"a.py"}


# --- T5: resolver ---


def test_resolver_inlines_small_file(tmp_path):
    s = session(tmp_path)
    (tmp_path / "small.py").write_text("line1\nline2\n", encoding="utf-8")
    block = s.mentions.resolve_mentions("fix @file:small.py")
    assert block.startswith("--- FILE MENTIONS ---")
    assert "[small.py] 2 lines" in block
    assert "line1" in block and "line2" in block
    assert block.count("small.py") == 1  # one block line; the header never names the file


def test_resolver_large_file_becomes_pointer_with_size(tmp_path):
    s = session(tmp_path)
    long_file = tmp_path / "long.py"
    long_file.write_text("x\n" * (FileMentions.MAX_INLINE_LINES + 1), encoding="utf-8")
    block = s.mentions.resolve_mentions("see @file:long.py")
    assert "too large to inline; Read the part you need" in block
    assert "lines" in block
    assert "x\n" not in block  # content never inlined

    huge = tmp_path / "huge.bin"
    huge.write_bytes(b"y" * (FileMentions.MAX_INLINE_BYTES + 1))
    block = s.mentions.resolve_mentions("see @file:huge.bin")
    assert "KB" in block and "too large to inline" in block


def test_resolver_outside_workspace_never_inlined(tmp_path):
    outside_dir = tempfile.mkdtemp()
    try:
        outside = os.path.join(outside_dir, "secret.txt")
        with open(outside, "w", encoding="utf-8") as handle:
            handle.write("classified content\n")
        s = session(tmp_path)
        block = s.mentions.resolve_mentions(f"see @file:{outside}")
        assert "outside the workspace; Read it" in block
        assert "classified content" not in block
    finally:
        shutil.rmtree(outside_dir, ignore_errors=True)


def test_resolver_missing_path_reports_itself(tmp_path):
    s = session(tmp_path)
    assert "[missing.py] not found" in s.mentions.resolve_mentions("see @file:missing.py")


def test_resolver_handles_quoted_paths_binary_and_unterminated_last_line(tmp_path):
    s = session(tmp_path)
    (tmp_path / "中文 notes.txt").write_text("one line", encoding="utf-8")
    (tmp_path / "binary.dat").write_bytes(b"abc\0def")
    text = f"see {encode_file_mention('中文 notes.txt')} and @file:binary.dat"
    block = s.mentions.resolve_mentions(text)
    assert "[中文 notes.txt] 1 lines" in block
    assert "one line" in block
    assert "[binary.dat] binary file" in block


def test_resolver_ten_file_cap_holds(tmp_path):
    s = session(tmp_path)
    for index in range(12):
        (tmp_path / f"f{index}.py").write_text(f"content {index}\n", encoding="utf-8")
    text = " ".join(f"@file:f{index}.py" for index in range(12))
    block = s.mentions.resolve_mentions(text)
    assert block.count("content ") == 10  # first ten inlined
    assert "[f10.py] not inlined - the 10-file inline cap is reached" in block
    assert "[f11.py] not inlined - the 10-file inline cap is reached" in block


def test_resolver_deduplicates_mentions(tmp_path):
    s = session(tmp_path)
    (tmp_path / "a.py").write_text("one\n", encoding="utf-8")
    block = s.mentions.resolve_mentions("@file:a.py and @file:./a.py and " + encode_file_mention(str(tmp_path / "a.py")))
    assert "[a.py] 1 lines" in block
    assert block.count("[a.py]") == 1


# --- fzf adapter ---


def fake_fzf(tmp_path, body):
    path = tmp_path / "fake-fzf"
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import os, sys\n" + body,
        encoding="utf-8",
    )
    path.chmod(0o755)
    return str(path)


def test_fzf_picker_uses_nul_path_scheme_query_and_isolated_environment(monkeypatch, tmp_path):
    selected = "docs/中文 notes.txt"
    (tmp_path / "docs").mkdir()
    (tmp_path / selected).write_text("", encoding="utf-8")
    executable = fake_fzf(
        tmp_path,
        "assert not any(name in os.environ for name in ('FZF_DEFAULT_COMMAND', 'FZF_DEFAULT_OPTS', 'FZF_DEFAULT_OPTS_FILE'))\n"
        "assert '--read0' in sys.argv and '--print0' in sys.argv and '--scheme=path' in sys.argv\n"
        "assert '--header=Ctrl-N/P or ↑/↓ move · Enter select · Esc close' in sys.argv\n"
        "assert '--bind=ctrl-n:down,ctrl-p:up' in sys.argv\n"
        "assert sys.argv[sys.argv.index('--query') + 1] == 'notes'\n"
        "items = sys.stdin.buffer.read().split(b'\\0')\n"
        f"assert {selected!r}.encode() in items\n"
        f"sys.stdout.buffer.write({selected!r}.encode() + b'\\0')\n",
    )
    for name in ("FZF_DEFAULT_COMMAND", "FZF_DEFAULT_OPTS", "FZF_DEFAULT_OPTS_FILE"):
        monkeypatch.setenv(name, "unsafe")
    mentions = session(tmp_path).mentions
    mentions._paths_cache = (time.monotonic(), ((selected.lower(), selected),))

    result = FzfPicker(mentions, executable).pick("notes")

    assert result.selection == selected
    assert not result.unavailable


def test_fzf_picker_cancel_and_protocol_failure_are_bounded(tmp_path):
    (tmp_path / "a.py").write_text("", encoding="utf-8")
    mentions = session(tmp_path).mentions
    mentions._paths_cache = (time.monotonic(), (("a.py", "a.py"),))
    cancelled = fake_fzf(tmp_path, "sys.stdin.buffer.read()\nraise SystemExit(130)\n")
    assert FzfPicker(mentions, cancelled).pick("").selection is None

    broken = tmp_path / "broken-fzf"
    broken.write_text("#!/bin/sh\nexit 2\n", encoding="utf-8")
    broken.chmod(0o755)
    result = FzfPicker(mentions, str(broken)).pick("")
    assert result.unavailable


def test_fzf_picker_caches_missing_executable(monkeypatch, tmp_path):
    mentions = session(tmp_path).mentions
    calls = []
    monkeypatch.setattr("minacode.mentions.shutil.which", lambda name: calls.append(name))
    picker = FzfPicker(mentions)

    assert not picker.available()
    assert not picker.available()
    assert calls == ["fzf"]


def test_fzf_picker_candidate_failure_routes_to_fallback(monkeypatch, tmp_path):
    executable = fake_fzf(tmp_path, "sys.stdin.buffer.read()\nraise SystemExit(1)\n")
    mentions = session(tmp_path).mentions

    def fail():
        raise RuntimeError("candidate source failed")

    monkeypatch.setattr(mentions, "_collect", fail)

    result = FzfPicker(mentions, executable).pick("")

    assert result.unavailable


def test_fzf_cancel_during_cold_scan_coalesces_background_refresh(monkeypatch, tmp_path):
    executable = fake_fzf(tmp_path, "import time\ntime.sleep(0.1)\nraise SystemExit(130)\n")
    mentions = session(tmp_path).mentions
    started = threading.Event()
    release = threading.Event()
    calls = []

    def collect():
        calls.append(None)
        started.set()
        release.wait(2)
        return ()

    monkeypatch.setattr(mentions, "_collect", collect)
    try:
        assert FzfPicker(mentions, executable).pick("").selection is None
        assert started.is_set()
        assert FzfPicker(mentions, executable).pick("").selection is None
        assert len(calls) == 1
    finally:
        release.set()


def test_fzf_picker_uses_stale_snapshot_while_refreshing(monkeypatch, tmp_path):
    selected = "a.py"
    (tmp_path / selected).write_text("", encoding="utf-8")
    mentions = session(tmp_path).mentions
    mentions._paths_cache = (0, ((selected, selected),))
    refresh_callbacks = []
    monkeypatch.setattr(mentions, "refresh_async", lambda callback=None: refresh_callbacks.append(callback))
    executable = fake_fzf(
        tmp_path,
        "items = sys.stdin.buffer.read().split(b'\\0')\n"
        f"assert {selected!r}.encode() in items\n"
        f"sys.stdout.buffer.write({selected!r}.encode() + b'\\0')\n",
    )

    result = FzfPicker(mentions, executable).pick("")

    assert result.selection == selected
    assert refresh_callbacks == [None]


# --- T7: performance ---


def test_filter_50k_paths_has_no_gross_algorithmic_regression():
    """Catch an accidental O(n²) implementation without treating scheduler jitter as a benchmark."""
    paths = tuple((f"module{i // 100}/file{i}.py", f"module{i // 100}/file{i}.py") for i in range(50_000))
    c = CommandCompleter(files=lambda: paths)
    start = time.perf_counter()
    matches = completions(c, "@file:py")  # matches every candidate
    elapsed = time.perf_counter() - start
    assert len(matches) == 50
    assert elapsed < 0.050
