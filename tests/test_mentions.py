"""@file: mentions: grammar (T1), completion matching/ranking (T3), the path source (T4),
the FILE MENTIONS resolver (T5), and the 50k-path performance guard (T7)."""

import os
import shutil
import subprocess
import tempfile
import time

from agent_harness import session
from prompt_toolkit.document import Document

from minacode.cli.view import CommandCompleter
from minacode.mentions import FileMentions


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
    # Legacy and namespaced forms resolve through the same patterns; unknown names stay ignored.
    assert s.mcp.MENTION_PATTERN.findall("@github @mcp:github @mcp:github.search") == [
        ("github", ""),
        ("github", ""),
        ("github", "search"),
    ]
    assert s.skills.MENTION_PATTERN.findall("$release @skill:release") == ["release", "release"]
    assert s.skills.resolve_mentions("price $30") == ""  # a price is not a skill mention


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


def test_merged_menu_bare_word_prefers_servers_slash_prefers_files():
    c = CommandCompleter(mcp_servers=lambda: ("viewer",), skills=lambda: (), files=lambda: FILES)
    assert completions(c, "@view") == ["@mcp:viewer", "@file:minacode/cli/view.py"]
    assert completions(c, "@cli/view") == ["@file:minacode/cli/view.py"]  # "/" ranks files first


def test_bare_kind_form_completes_after_kind_insertion():
    """SPEC 4.4: accepting a kind replaces the "@" with "kind:", so the bare kind prefix must
    keep completing that kind's source (the trigger reopens the menu on the bare form)."""
    c = CommandCompleter(mcp_servers=lambda: ("github", "gitlab"), skills=lambda: ("release",), files=lambda: FILES)
    assert completions(c, "use file:vie") == ["@file:minacode/cli/view.py"]
    assert completions(c, "use mcp:git") == ["@mcp:github", "@mcp:gitlab"]
    assert completions(c, "use skill:rel") == ["@skill:release"]


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
    assert "node_modules/x.js" not in rels  # SKIP_DIRS, same walk as Search
    assert ".hidden.py" not in rels


def test_path_source_git_repo_and_untracked_appears(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=tmp_path, check=True)
    (tmp_path / "tracked.py").write_text("", encoding="utf-8")
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


def test_path_cache_refreshes_after_window(tmp_path):
    (tmp_path / "a.py").write_text("", encoding="utf-8")
    s = session(tmp_path)
    s.mentions._paths_cache = (time.monotonic() - 10, [("a.py", "a.py")])  # stale: only a.py
    (tmp_path / "b.py").write_text("", encoding="utf-8")
    assert {rel for _lower, rel in s.mentions.paths()} == {"a.py", "b.py"}
    s.mentions._paths_cache = (time.monotonic(), [("a.py", "a.py")])  # fresh: kept as is
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


def test_resolver_ten_file_cap_holds(tmp_path):
    s = session(tmp_path)
    for index in range(12):
        (tmp_path / f"f{index}.py").write_text(f"content {index}\n", encoding="utf-8")
    text = " ".join(f"@file:f{index}.py" for index in range(12))
    block = s.mentions.resolve_mentions(text)
    assert block.count("content ") == 10  # first ten inlined
    assert "[f10.py] not included - the 10-file cap is reached" in block
    assert "[f11.py] not included - the 10-file cap is reached" in block


def test_resolver_deduplicates_mentions(tmp_path):
    s = session(tmp_path)
    (tmp_path / "a.py").write_text("one\n", encoding="utf-8")
    block = s.mentions.resolve_mentions("@file:a.py and @file:a.py")
    assert "[a.py] 1 lines" in block
    assert block.count("[a.py]") == 1


# --- T7: performance ---


def test_filter_50k_paths_stays_under_10ms():
    """Guards G3 against an accidental O(n^2): substring + ranking over a synthetic 50k list."""
    paths = tuple((f"module{i // 100}/file{i}.py", f"module{i // 100}/file{i}.py") for i in range(50_000))
    c = CommandCompleter(files=lambda: paths)
    start = time.perf_counter()
    matches = completions(c, "@file:py")  # matches every candidate
    elapsed = time.perf_counter() - start
    assert len(matches) == 50
    assert elapsed < 0.010
