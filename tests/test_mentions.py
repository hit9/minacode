"""@file: mentions: grammar (T1), completion matching/ranking (T3), the path source (T4),
the FILE MENTIONS resolver (T5), and the 50k-path performance guard (T7)."""

import asyncio
import os
import shutil
import tempfile
import time

import pytest
from agent_harness import session
from prompt_toolkit.document import Document

from wizolt.cli import TuiRuntime
from wizolt.cli.view import CommandCompleter
from wizolt.mentions import FileMentions, FzfPicker, active_mention, encode_file_mention, scan_mentions


def completions(completer, text):
    return [c.text for c in completer.get_completions(Document(text), None)]


async def run_git(cwd, *args):
    process = await asyncio.create_subprocess_exec(
        "git",
        *args,
        cwd=cwd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    assert await process.wait() == 0


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
    ("wizolt/cli/view.py", "wizolt/cli/view.py"),
    ("wizolt/tui.py", "wizolt/tui.py"),
    ("wizolt/hints.py", "wizolt/hints.py"),
)


def test_matching_substring_and_case_insensitive():
    c = CommandCompleter(files=lambda: FILES)
    assert completions(c, "@file:view") == ["@file:wizolt/cli/view.py"]
    assert completions(c, "@file:cli/view") == ["@file:wizolt/cli/view.py"]  # whole-path substring
    assert completions(c, "@file:VIEW") == ["@file:wizolt/cli/view.py"]  # case-insensitive
    assert completions(c, "@file:wizolt/tui.py") == ["@file:wizolt/tui.py"]


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


def test_merged_menu_counts_kind_row_toward_50_row_cap():
    servers = tuple(f"f{index:02}" for index in range(60))
    c = CommandCompleter(mcp_servers=lambda: servers)

    rows = completions(c, "@f")

    assert len(rows) == 50
    assert rows[0] == "@file:"
    assert rows[-1] == "@mcp:f48"


def test_bare_menu_does_not_scan_or_merge_repository_files():
    c = CommandCompleter(mcp_servers=lambda: ("viewer",), skills=lambda: (), files=lambda: FILES)
    assert completions(c, "@view") == ["@mcp:viewer"]
    assert completions(c, "@cli/view") == []


def test_kind_completion_keeps_canonical_at_prefix():
    c = CommandCompleter(mcp_servers=lambda: ("github", "gitlab"), skills=lambda: ("release",), files=lambda: FILES)
    assert completions(c, "use @") == ["@file:", "@mcp:", "@skill:"]
    assert completions(c, "use @file:vie") == ["@file:wizolt/cli/view.py"]
    assert completions(c, "use @mcp:git") == ["@mcp:github", "@mcp:gitlab"]
    assert completions(c, "use @skill:rel") == ["@skill:release"]


# --- T4: source ---


async def test_path_source_non_git_walk(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "x.js").write_text("", encoding="utf-8")
    (tmp_path / ".hidden.py").write_text("", encoding="utf-8")
    s = session(tmp_path)
    rels = {rel for _, rel in await s.mentions.refresh()}
    assert "src/a.py" in rels
    assert "node_modules/x.js" in rels  # only ignore rules and .git exclude paths
    assert ".hidden.py" in rels


async def test_path_source_git_repo_and_untracked_appears(tmp_path):
    await run_git(tmp_path, "init", "-q")
    await run_git(tmp_path, "config", "user.email", "test@example.com")
    await run_git(tmp_path, "config", "user.name", "test")
    (tmp_path / "tracked.py").write_text("", encoding="utf-8")
    (tmp_path / "tracked.tmp").write_text("", encoding="utf-8")
    await run_git(tmp_path, "add", ".")
    await run_git(tmp_path, "commit", "-qm", "init")
    (tmp_path / "untracked.py").write_text("", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("*.tmp\n", encoding="utf-8")
    (tmp_path / "ignored.tmp").write_text("", encoding="utf-8")
    s = session(tmp_path)
    rels = {rel for _, rel in await s.mentions.refresh()}
    assert "tracked.py" in rels
    assert "untracked.py" in rels  # git ls-files -o --exclude-standard
    assert "ignored.tmp" not in rels
    assert "tracked.tmp" not in rels  # product rule also excludes tracked files matching ignore
    assert not any(".git" in rel.split("/") for rel in rels)


async def test_path_source_git_nested_ignore_negation_and_unusual_names(tmp_path):
    await run_git(tmp_path, "init", "-q")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / ".gitignore").write_text("*.tmp\n!keep.tmp\n", encoding="utf-8")
    (nested / "drop.tmp").write_text("", encoding="utf-8")
    (nested / "keep.tmp").write_text("", encoding="utf-8")
    (nested / "中文 name.txt").write_text("", encoding="utf-8")
    (nested / "line\nbreak.txt").write_text("", encoding="utf-8")

    rels = {rel for _, rel in await session(tmp_path).mentions.refresh()}
    assert "nested/drop.tmp" not in rels
    assert "nested/keep.tmp" in rels
    assert "nested/中文 name.txt" in rels
    assert "nested/line\nbreak.txt" in rels


async def test_git_candidate_queries_overlap(monkeypatch, tmp_path):
    """Both queries only read Git's index, so they run together: one gate neither can pass alone."""
    mentions = session(tmp_path).mentions
    both_started = asyncio.Event()
    started = 0

    async def git_ls_files(flags):
        nonlocal started
        started += 1
        if started == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), 5)
        return ["a.py"] if "--others" in flags else []

    monkeypatch.setattr(mentions, "_git_ls_files", git_ls_files)

    assert await mentions._git_paths() == ["a.py"]


async def test_python_fallback_honors_nested_gitignore(monkeypatch, tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / ".gitignore").write_text("*.tmp\n!keep.tmp\n", encoding="utf-8")
    (nested / "drop.tmp").write_text("", encoding="utf-8")
    (nested / "keep.tmp").write_text("", encoding="utf-8")
    monkeypatch.setattr("wizolt.mentions.shutil.which", lambda _name: None)

    rels = {rel for _, rel in await session(tmp_path).mentions.refresh()}
    assert "nested/drop.tmp" not in rels
    assert "nested/keep.tmp" in rels


async def test_a_stale_snapshot_is_served_now_and_refreshed_behind_the_caller(tmp_path):
    """A stale list still opens the picker immediately -- the selection is revalidated anyway --
    and a fresh one is served as is, without touching the filesystem at all."""
    (tmp_path / "a.py").write_text("", encoding="utf-8")
    s = session(tmp_path)
    (tmp_path / "b.py").write_text("", encoding="utf-8")

    s.mentions._paths_cache = (time.monotonic() - 10, (("a.py", "a.py"),))  # stale: only a.py
    scans = []
    s.mentions.refresh_owner = lambda: scans.append(True)
    assert {rel for _, rel in await s.mentions.candidates()} == {"a.py"}
    assert scans == [True]  # served stale, and refreshed for the next caller

    s.mentions._paths_cache = (time.monotonic(), (("a.py", "a.py"),))  # fresh: kept as is
    assert {rel for _, rel in await s.mentions.candidates()} == {"a.py"}
    assert scans == [True]

    # A cold cache is the one case that waits, and it sees both files.
    s.mentions._paths_cache = None
    s.mentions.refresh_owner = None
    assert {rel for _, rel in await s.mentions.candidates()} == {"a.py", "b.py"}


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
        "#!/usr/bin/env python3\nimport os, sys\n" + body,
        encoding="utf-8",
    )
    path.chmod(0o755)
    return str(path)


async def test_fzf_picker_uses_nul_path_scheme_query_and_isolated_environment(monkeypatch, tmp_path):
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

    result = await FzfPicker(mentions, executable).pick("notes")

    assert result.selection == selected
    assert not result.unavailable


async def test_fzf_picker_cancel_and_protocol_failure_are_bounded(tmp_path):
    (tmp_path / "a.py").write_text("", encoding="utf-8")
    mentions = session(tmp_path).mentions
    mentions._paths_cache = (time.monotonic(), (("a.py", "a.py"),))
    cancelled = fake_fzf(tmp_path, "sys.stdin.buffer.read()\nraise SystemExit(130)\n")
    assert (await FzfPicker(mentions, cancelled).pick("")).selection is None

    broken = tmp_path / "broken-fzf"
    broken.write_text("#!/bin/sh\nexit 2\n", encoding="utf-8")
    broken.chmod(0o755)
    result = await FzfPicker(mentions, str(broken)).pick("")
    assert result.unavailable


def test_fzf_picker_caches_missing_executable(monkeypatch, tmp_path):
    mentions = session(tmp_path).mentions
    calls = []
    monkeypatch.setattr("wizolt.mentions.shutil.which", lambda name: calls.append(name))
    picker = FzfPicker(mentions)

    assert not picker.available()
    assert not picker.available()
    assert calls == ["fzf"]


async def test_fzf_picker_candidate_failure_routes_to_fallback(monkeypatch, tmp_path):
    executable = fake_fzf(tmp_path, "sys.stdin.buffer.read()\nraise SystemExit(1)\n")
    mentions = session(tmp_path).mentions

    async def fail():
        raise RuntimeError("candidate source failed")

    monkeypatch.setattr(mentions, "candidates", fail)

    result = await FzfPicker(mentions, executable).pick("")

    assert result.unavailable


async def test_escape_closes_fzf_while_a_cold_scan_is_still_running(tmp_path, monkeypatch):
    """A cold cache races the scan against fzf's own exit. Escape must close the picker rather than
    hold the terminal until a large worktree has finished being walked."""
    executable = fake_fzf(tmp_path, "import time\ntime.sleep(0.05)\nraise SystemExit(130)\n")
    mentions = session(tmp_path).mentions
    scanning = asyncio.Event()

    async def never_finishes():
        scanning.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(mentions, "candidates", never_finishes)

    result = await FzfPicker(mentions, executable).pick("")

    assert result.selection is None
    assert scanning.is_set()  # the scan really had started; the picker did not wait for it


async def test_concurrent_mention_refreshes_coalesce_onto_one_scan(tmp_path):
    """Every caller joins the scan already running: one traversal, and all waiters see its result."""
    from tui_harness import loop as command_loop_for

    command_loop = command_loop_for(tmp_path)
    (tmp_path / "a.py").write_text("", encoding="utf-8")
    mentions = command_loop.session.mentions
    assert mentions is not None
    scans = 0
    real_collect = mentions._collect

    def collect(rels):
        nonlocal scans
        scans += 1
        return real_collect(rels)

    mentions._collect = collect
    command_loop.open_background()
    try:
        results = await asyncio.gather(*(mentions.candidates() for _ in range(5)))
    finally:
        await command_loop.close_background()

    assert scans == 1
    assert all(result == results[0] for result in results)
    assert {rel for _, rel in results[0]} == {"a.py"}


async def test_cancelling_one_candidate_waiter_keeps_the_shared_scan_alive(tmp_path, monkeypatch):
    """A picker owns only its wait: the CommandLoop-owned scan may have other consumers."""
    from tui_harness import loop as command_loop_for

    command_loop = command_loop_for(tmp_path)
    mentions = command_loop.session.mentions
    assert mentions is not None
    started = asyncio.Event()
    release = asyncio.Event()

    async def refresh():
        started.set()
        await release.wait()
        return (("a.py", "a.py"),)

    monkeypatch.setattr(mentions, "refresh", refresh)
    command_loop.open_background()
    try:
        shared = command_loop.refresh_mentions()
        assert shared is not None
        waiter = asyncio.create_task(mentions.candidates())
        await started.wait()
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert not shared.done()
        release.set()
        assert await shared == (("a.py", "a.py"),)
    finally:
        await command_loop.close_background()


async def test_command_loop_owns_and_settles_the_file_picker(tmp_path, monkeypatch):
    from tui_harness import loop as command_loop_for

    command_loop = command_loop_for(tmp_path)
    mentions = command_loop.session.mentions
    assert mentions is not None
    started = asyncio.Event()
    settled = asyncio.Event()

    async def pick(_query):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            settled.set()

    monkeypatch.setattr(mentions.picker, "pick", pick)
    command_loop.open_background()
    waiting = asyncio.create_task(command_loop.pick_file(""))
    await started.wait()
    await command_loop.close_background()

    with pytest.raises(asyncio.CancelledError):
        await waiting
    assert settled.is_set()


async def test_fzf_picker_uses_stale_snapshot_while_refreshing(monkeypatch, tmp_path):
    selected = "a.py"
    (tmp_path / selected).write_text("", encoding="utf-8")
    mentions = session(tmp_path).mentions
    mentions._paths_cache = (0, ((selected, selected),))
    scheduled = []
    monkeypatch.setattr(mentions, "refresh_owner", lambda: scheduled.append(True))
    executable = fake_fzf(
        tmp_path,
        f"items = sys.stdin.buffer.read().split(b'\\0')\nassert {selected!r}.encode() in items\nsys.stdout.buffer.write({selected!r}.encode() + b'\\0')\n",
    )

    result = await FzfPicker(mentions, executable).pick("")

    assert result.selection == selected
    assert scheduled == [True]  # opened on the stale list, refreshed for the next invocation


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


async def test_a_newer_completion_query_supersedes_an_older_one(tmp_path, monkeypatch):
    """The menu must show what is being typed now. A query overtaken while it ranked publishes
    nothing, so a slow early keystroke cannot overwrite the newest cache."""
    mentions = session(tmp_path).mentions
    mentions._paths_cache = (time.monotonic(), (("notes.txt", "notes.txt"), ("other.txt", "other.txt")))
    ready = []
    ranked = []
    release_first = asyncio.Event()
    real_matches = FileMentions._literal_matches

    def slow_first(paths, query):
        ranked.append(query)
        if query == "n":
            while not release_first.is_set():
                time.sleep(0.005)
        return real_matches(paths, query)

    monkeypatch.setattr(FileMentions, "_literal_matches", staticmethod(slow_first))

    first = asyncio.ensure_future(mentions.complete("n", lambda: ready.append("n")))
    await asyncio.sleep(0.02)
    second = asyncio.ensure_future(mentions.complete("other", lambda: ready.append("other")))
    await second
    assert ranked == ["n"]  # the newer request replaced queued work; it did not start a second worker
    release_first.set()
    await first

    assert ranked == ["n", "other"]
    assert ready == ["other"]  # the superseded query never notified
    assert mentions.cached_matches("other") == ("other.txt",)
    assert mentions.cached_matches("n") == ()


async def test_no_completion_callback_fires_after_the_owner_closes(tmp_path):
    """Shutdown means the input widget is gone; a completion that lands afterwards must not call
    back into it."""
    from tui_harness import loop as command_loop_for

    command_loop = command_loop_for(tmp_path)
    runtime = TuiRuntime(command_loop)
    mentions = command_loop.session.mentions
    assert mentions is not None
    mentions._paths_cache = (time.monotonic(), (("a.py", "a.py"),))
    fired = []

    command_loop.open_background()
    runtime.complete_mentions("a", lambda: fired.append(True))
    await command_loop.close_background()
    fired.clear()

    runtime.complete_mentions("a", lambda: fired.append(True))
    await asyncio.sleep(0.05)

    assert fired == []


async def test_cancelling_a_refresh_reaps_its_discovery_subprocesses(tmp_path, monkeypatch):
    """`git ls-files` on a large worktree is not something to leave running behind a cancelled
    refresh: both children are killed and waited for before cancellation is reported."""
    mentions = session(tmp_path).mentions
    processes = []
    both_started = asyncio.Event()
    real_exec = asyncio.create_subprocess_exec

    async def slow_git(*argv, **kwargs):
        process = await real_exec("sleep", "30", **kwargs)
        processes.append(process)
        if len(processes) == 2:
            both_started.set()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", slow_git)

    scan = asyncio.ensure_future(mentions.refresh())
    await asyncio.wait_for(both_started.wait(), 5)
    scan.cancel()
    with pytest.raises(asyncio.CancelledError):
        await scan

    assert processes
    assert all(process.returncode is not None for process in processes)


async def test_cancelling_the_picker_kills_and_reaps_fzf(tmp_path):
    """A shutdown while the picker is up must not leave fzf holding the terminal."""
    executable = fake_fzf(tmp_path, "import time\nsys.stdin.buffer.read()\ntime.sleep(30)\n")
    mentions = session(tmp_path).mentions
    mentions._paths_cache = (time.monotonic(), (("a.py", "a.py"),))
    picker = FzfPicker(mentions, executable)
    processes = []
    real_exec = asyncio.create_subprocess_exec

    async def record(*argv, **kwargs):
        process = await real_exec(*argv, **kwargs)
        processes.append(process)
        return process

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(asyncio, "create_subprocess_exec", record)
        pick = asyncio.ensure_future(picker.pick(""))
        while not processes:
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)
        pick.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pick

    assert processes[0].returncode is not None


async def test_fzf_output_past_the_bound_is_refused_and_the_process_reaped(tmp_path):
    """A single-selection picker cannot answer with a megabyte, so an answer that long is not one."""
    executable = fake_fzf(
        tmp_path,
        "sys.stdin.buffer.read()\nsys.stdout.buffer.write(b'x' * ((1 << 20) + 16))\nsys.stdout.flush()\nimport time\ntime.sleep(30)\n",
    )
    mentions = session(tmp_path).mentions
    mentions._paths_cache = (time.monotonic(), (("a.py", "a.py"),))

    result = await FzfPicker(mentions, executable).pick("")

    assert result.selection is None
    assert not result.unavailable


async def test_fzf_that_closes_its_input_early_still_returns_the_selection(tmp_path):
    """fzf reads what it needs and exits; the writer meets a broken pipe and that is not an error."""
    selected = "a.py"
    (tmp_path / selected).write_text("", encoding="utf-8")
    mentions = session(tmp_path).mentions
    mentions._paths_cache = (time.monotonic(), tuple((f"f{index}.py", f"f{index}.py") for index in range(5000)) + ((selected, selected),))
    executable = fake_fzf(
        tmp_path,
        "sys.stdin.buffer.close()\n" + f"sys.stdout.buffer.write({selected!r}.encode() + b'\\0')\n",
    )

    result = await FzfPicker(mentions, executable).pick("")

    assert not result.unavailable
