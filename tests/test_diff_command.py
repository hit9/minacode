import subprocess

import nanocode as n


def session(tmp_path):
    config = n.Config()
    config.data_dir = str(tmp_path / "data")
    return n.Session(cwd=str(tmp_path), config=config)


def loop(session):
    return n.CommandLoop(n.Agent(session, output_fn=lambda text: None), output_fn=lambda text: None)


def git_init(path):
    subprocess.run(["git", "-C", str(path), "init"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "test@test"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Test"], check=True, capture_output=True)


def test_diff_is_in_completer_commands():
    assert "/diff" in n.CommandLoop.COMMANDS


def test_diff_appears_in_help():
    assert "/diff" in n.CommandLoop.HELP


def test_diff_is_allowed_while_agent_works():
    assert "/diff" in n.CommandLoop.QUEUE_RUN_COMMANDS


def test_diff_rejects_args(tmp_path):
    lp = loop(session(tmp_path))
    assert lp.diff_command("extra") == "Usage: /diff"


def test_diff_outside_git_repo(tmp_path):
    lp = loop(session(tmp_path))
    assert lp.diff_command("") == "No changes"


def test_diff_clean_session(tmp_path):
    lp = loop(session(tmp_path))
    assert lp.diff_command("") == "No changes"


def test_diff_round_with_no_net_changes_is_empty(tmp_path):
    s = session(tmp_path)
    s.store_turn_diff("tr.1", 1, "a.py", "-old\n+mid\n", before="old\n", after="mid\n", round=1)
    s.store_turn_diff("tr.2", 2, "a.py", "-mid\n+old\n", before="mid\n", after="old\n", round=1)

    assert loop(s).diff_command("") == "No changes"


def test_diff_shows_latest_round(tmp_path):
    s = session(tmp_path)
    s.store_turn_diff("tr.1", 1, "old.py", "-old\n+older\n", round=1)
    s.store_turn_diff("tr.2", 2, "new.py", "-old\n+new\n", round=2)

    lp = loop(s)
    result = lp.diff_command("")

    assert "### Latest · Round 2" in result
    assert "#### new.py" in result
    assert "+new" in result
    latest, _, session_section = result.partition("### Session")
    assert "old.py" not in latest
    assert "#### old.py" in session_section


def test_diff_shows_latest_round_outside_git_repo(tmp_path):
    s = session(tmp_path)
    s.store_turn_diff("tr.1", 3, "x.py", "-a\n+b\n")

    lp = loop(s)
    result = lp.diff_command("")

    assert "### Latest · Round 3" in result
    assert "#### x.py" in result
    assert "+b" in result
    assert result != "Not in a git repository"


def test_diff_ignores_git_worktree_changes(tmp_path):
    git_init(tmp_path)
    (tmp_path / "a.py").write_text("old\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "a.py"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-m", "init"], check=True, capture_output=True)
    (tmp_path / "a.py").write_text("new\n", encoding="utf-8")
    (tmp_path / "untracked.py").write_text("hello\n", encoding="utf-8")

    lp = loop(session(tmp_path))
    assert lp.diff_command("") == "No changes"


def test_diff_bounds_large_session_output(tmp_path):
    s = session(tmp_path)
    large = "\n".join(f"+line {index}" for index in range(2_000))
    s.store_turn_diff("tr.1", 1, "a.py", large, round=1)

    lp = loop(s)
    result = lp.diff_command("")
    assert "truncated" in result.lower()


def test_ui_segment_lines_keeps_styled_diff_lines_together():
    ui = n.UiPrinter(output_fn=lambda text: None)
    diff = "--- a.py\n+++ a.py\n@@ -1 +1 @@\n-old\n+return 42\n"

    segments = ui.diff_segments(diff)
    lines = ui.segment_lines(segments)

    assert len(lines) == len(diff.splitlines())
    assert "".join(text for line in lines for _style, text in line) == "".join(text for _style, text in segments)
    assert any("+return 42" in "".join(text for _style, text in line) for line in lines)


def test_diff_counts_only_hunk_changes():
    diff = "--- a.py\n+++ a.py\n@@ -1 +1,2 @@\n-old\n+++heading\n+new\n"

    assert n.CommandLoop.diff_counts(diff) == (2, 1)


def test_diff_viewer_list_shows_change_counts_without_status_prefix(tmp_path):
    s = session(tmp_path)
    s.store_turn_diff("tr.1", 1, "a.py", "unused", before="old\n", after="new\nextra\n", round=1)
    lp = loop(s)
    rendered = []

    class Modal:
        def show_modal(self, fragments_fn, _key_fn, **_kwargs):
            rendered.extend(fragments_fn())

    lp.tui = Modal()

    lp.diff_viewer()

    text = "".join(fragment for _style, fragment in rendered)
    assert "+2 -1 a.py" in text
    assert "Edit" not in text


def test_tool_runner_captures_edit_turn_diff(tmp_path):
    git_init(tmp_path)
    (tmp_path / "a.py").write_text("old\n", encoding="utf-8")
    s = session(tmp_path)
    s.state.turn_step = 1
    s.state.round_count = 1
    s.settings.yolo = True

    runner = n.ToolRunner(s, n.ContextManager(s), input_fn=lambda prompt: "", output_fn=lambda text: None)
    call = n.ToolCall("edit-1", "Edit", ["a.py", [{"op": "replace_all", "old": "old\n", "new": "new\n"}]])
    status, message = runner.run_one(call)

    assert status == "ok"
    assert len(s.turn_diffs) == 1
    td = s.turn_diffs[0]
    assert td.path == "a.py"
    assert td.turn == 1
    assert td.round == 1
    assert td.key.startswith("tr.")
    assert "-old" in td.diff
    assert "+new" in td.diff
    assert td.before == "old\n"
    assert td.after == "new\n"


def test_session_diff_sections_show_overall_file_effect(tmp_path):
    s = session(tmp_path)
    s.store_turn_diff("tr.1", 1, "a.py", "-old\n+mid\n", before="old\n", after="mid\n")
    s.store_turn_diff("tr.2", 2, "a.py", "-mid\n+new\n", before="mid\n", after="new\n")

    sections = s.session_diff_sections()

    assert len(sections) == 1
    status, path, diff = sections[0]
    assert status == "overall"
    assert path == "a.py"
    assert "-old" in diff
    assert "+new" in diff
    assert "mid" not in diff


def test_diff_sections_follow_file_across_unambiguous_moves(tmp_path):
    s = session(tmp_path)
    created = "one\ntwo\nthree\n"
    trimmed = "one\ntwo\n"
    final = "one\nchanged\nextra\n"
    s.store_turn_diff("tr.1", 1, "draft.md", "unused", before="", after=created, round=1)
    s.store_turn_diff("tr.2", 2, "SKILL.md", "unused", before=created, after=trimmed, round=2)
    s.store_turn_diff("tr.3", 3, "skill/SKILL.md", "unused", before=trimmed, after=final, round=2)

    latest = s.latest_round_diff_sections()
    session_sections = s.session_diff_sections()

    assert latest is not None
    assert [path for _status, path, _diff in latest[1]] == ["skill/SKILL.md"]
    assert n.CommandLoop.diff_counts(latest[1][0][2]) == (2, 2)
    assert [path for _status, path, _diff in session_sections] == ["skill/SKILL.md"]
    assert n.CommandLoop.diff_counts(session_sections[0][2]) == (3, 0)


def test_diff_sections_do_not_guess_ambiguous_moves(tmp_path):
    s = session(tmp_path)
    s.store_turn_diff("tr.1", 1, "source.md", "unused", before="", after="same\n")
    s.store_turn_diff("tr.2", 2, "first.md", "unused", before="same\n", after="first\n")
    s.store_turn_diff("tr.3", 3, "second.md", "unused", before="same\n", after="second\n")

    assert [path for _status, path, _diff in s.session_diff_sections()] == ["source.md", "first.md", "second.md"]


def test_session_diff_sections_fall_back_to_legacy_diffs(tmp_path):
    s = session(tmp_path)
    s.store_turn_diff("tr.1", 1, "a.py", "-old\n+new\n")

    assert s.session_diff_sections() == [("overall", "a.py", "-old\n+new\n")]


def test_store_turn_diff_drops_large_net_snapshots(tmp_path):
    s = session(tmp_path)
    large = "x" * (n.TurnDiff.SNAPSHOT_CHAR_LIMIT + 1)
    s.store_turn_diff("tr.1", 1, "large.py", "-old\n+new\n", before=large, after="new\n", round=1)

    diff = s.turn_diffs[0]
    assert diff.diff == "-old\n+new\n"
    assert diff.before == ""
    assert diff.after == ""
    latest = s.latest_round_diff_sections()
    assert latest is not None
    assert latest[1] == [("edit", "large.py", "-old\n+new\n")]
    assert s.session_diff_sections() == [("overall", "large.py", "-old\n+new\n")]


def test_legacy_diff_reconstructed_from_disk(tmp_path):
    import difflib

    fpath = tmp_path / "big.py"
    original = "line1\nline2\nline3\nline4\nline5\n"
    v1 = "line1\nlineTWO\nline3\nline4\nline5\n"
    v2 = "line1\nlineTWO\nline3\nlineFOUR\nline5\n"
    v3 = "lineONE\nlineTWO\nline3\nlineFOUR\nline5\n"
    fpath.write_text(v3)  # disk matches the last tracked edit

    s = session(tmp_path)
    s.cwd = str(tmp_path)
    def unified(before: str, after: str) -> str:
        return "".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile="big.py", tofile="big.py"))
    s.store_turn_diff("t1", 1, "big.py", unified(original, v1), round=1)
    s.store_turn_diff("t2", 2, "big.py", unified(v1, v2), round=1)
    s.store_turn_diff("t3", 3, "big.py", unified(v2, v3), round=1)

    sections = s.session_diff_sections()
    assert len(sections) == 1
    _, path, diff = sections[0]
    assert path == "big.py"
    # Reconstruction folds the three per-Edit hunks into one clean unified diff — one header,
    # aggregate net changes only (no intermediate lineTWO/lineFOUR churn from earlier steps).
    assert diff.count("--- big.py") == 1
    assert "+lineONE" in diff and "-line1" in diff
    assert "+lineTWO" in diff and "-line2" in diff
    assert "+lineFOUR" in diff and "-line4" in diff


def test_legacy_diff_falls_back_when_disk_drifted(tmp_path):
    import difflib

    fpath = tmp_path / "a.py"
    original = "one\ntwo\nthree\n"
    edited = "one\nTWO\nthree\n"
    fpath.write_text("something else entirely\n")  # disk doesn't match — reconstruction must bail

    s = session(tmp_path)
    s.cwd = str(tmp_path)
    diff_text = "".join(difflib.unified_diff(original.splitlines(True), edited.splitlines(True), fromfile="a.py", tofile="a.py"))
    s.store_turn_diff("t1", 1, "a.py", diff_text, round=1)

    sections = s.session_diff_sections()
    assert len(sections) == 1
    _, _, diff = sections[0]
    # Reconstruction bailed; concatenated raw hunks still show so nothing is silently dropped.
    assert "-two" in diff
    assert "+TWO" in diff


def test_latest_round_coalesces_legacy_diffs_for_same_path(tmp_path):
    s = session(tmp_path)
    large = "x" * (n.TurnDiff.SNAPSHOT_CHAR_LIMIT + 1)
    first = "--- a.py\n+++ a.py\n@@ -1 +1 @@\n-old\n+large\n"
    second = "--- a.py\n+++ a.py\n@@ -1 +1 @@\n-large\n+new\n"
    s.store_turn_diff("tr.1", 1, "a.py", first, before="old\n", after=large, round=1)
    s.store_turn_diff("tr.2", 2, "a.py", second, before=large, after="new\n", round=1)

    latest = s.latest_round_diff_sections()

    assert latest is not None
    assert len(latest[1]) == 1
    assert latest[1][0][:2] == ("edit", "a.py")
    assert first in latest[1][0][2]
    assert second in latest[1][0][2]
    assert n.CommandLoop.diff_counts(latest[1][0][2]) == (2, 2)


def test_latest_round_diffs_include_all_steps_in_round(tmp_path):
    s = session(tmp_path)
    s.store_turn_diff("tr.1", 1, "a.py", "-old\n+mid\n", before="old\n", after="mid\n", round=1)
    s.store_turn_diff("tr.2", 2, "b.py", "-one\n+two\n", before="one\n", after="two\n", round=1)
    s.store_turn_diff("tr.3", 2, "a.py", "-mid\n+new\n", before="mid\n", after="new\n", round=1)
    s.store_turn_diff("tr.4", 0, "older.py", "-x\n+y\n", before="x\n", after="y\n", round=0)

    latest = s.latest_round_diff_sections()

    assert latest is not None
    round, sections = latest
    assert round == 1
    assert [path for _status, path, _diff in sections] == ["a.py", "b.py"]
    a_diff = sections[0][2]
    assert "-old" in a_diff
    assert "+new" in a_diff
    assert "mid" not in a_diff


def test_session_snapshot_turn_diff_roundtrip(tmp_path):
    s = session(tmp_path)
    s.messages.append({"role": "user", "content": "seed"})
    s.store_turn_diff("tr.1", 2, "x.py", "-a\n+b\n", round=1)

    store = n.SessionSnapshotStore(s)
    uid = store.save()
    loaded = n.SessionSnapshotStore.load(uid, s.config, s.settings)

    assert len(loaded.turn_diffs) == 1
    assert loaded.turn_diffs[0].path == "x.py"
    assert loaded.turn_diffs[0].diff == "-a\n+b\n"
    assert loaded.turn_diffs[0].turn == 2
    assert loaded.turn_diffs[0].round == 1


def test_resume_renders_turn_diffs_from_the_snapshot(tmp_path):
    """Edit diffs survive a resume: `/diff` rebuilds both tabs from the persisted turn_diffs,
    file snapshots included."""
    s = session(tmp_path)
    (tmp_path / "x.py").write_text("new\n")
    s.messages.append({"role": "user", "content": "edit it"})
    s.store_turn_diff("tr.1", 1, "x.py", "--- x.py\n+++ x.py\n@@ -1 +1 @@\n-old\n+new\n", before="old\n", after="new\n", round=1)
    s.save_snapshot()

    loaded = n.Session.load_snapshot(s.uid, config=s.config, settings=s.settings, cwd=str(tmp_path))
    result = loop(loaded).diff_command("")

    assert [(diff.key, diff.path, diff.before, diff.after) for diff in loaded.turn_diffs] == [("tr.1", "x.py", "old\n", "new\n")]
    assert "### Latest" in result
    assert "### Session" in result
    assert result.count("-old") == 2
    assert result.count("+new") == 2


def _diff(key, turn, path, before, after, text):
    return n.TurnDiff(key, turn, path, text, before=before, after=after, round=turn)


def test_net_diff_emits_one_description_per_path_when_snapshots_stop(tmp_path):
    """A file that grows past the snapshot size limit partway through leaves some edits with
    snapshots and some without. Both describe the same file, so only one may be emitted."""
    (tmp_path / "x.py").write_text("c\n")
    kept = _diff("tr.1", 1, "x.py", "a\n", "b\n", "--- x.py\n+++ x.py\n@@ -1 +1 @@\n-a\n+b\n")
    dropped = _diff("tr.2", 2, "x.py", "", "", "--- x.py\n+++ x.py\n@@ -1 +1 @@\n-b\n+c\n")

    sections = n.Session.net_diff_sections([kept, dropped], "overall", cwd=str(tmp_path))

    assert len(sections) == 1
    text = sections[0][2]
    assert text.count("--- ") == 1  # one diff for the file, not one per description
    assert [line for line in text.splitlines() if line[:1] in "+-" and not line.startswith(("---", "+++"))] == ["-a", "+c"]


def test_net_diff_prefers_snapshots_when_the_last_edit_has_them(tmp_path):
    """A snapshot-less edit in the middle is already reflected in the next snapshot's `after`."""
    (tmp_path / "x.py").write_text("unused\n")
    first = _diff("tr.1", 1, "x.py", "a\n", "b\n", "--- x.py\n+++ x.py\n@@ -1 +1 @@\n-a\n+b\n")
    dropped = _diff("tr.2", 2, "x.py", "", "", "--- x.py\n+++ x.py\n@@ -1 +1 @@\n-b\n+c\n")
    last = _diff("tr.3", 3, "x.py", "b\n", "c\n", "--- x.py\n+++ x.py\n@@ -1 +1 @@\n-b\n+c\n")

    sections = n.Session.net_diff_sections([first, dropped, last], "overall", cwd=str(tmp_path))

    # The on-disk content is ignored here: the recorded snapshots already cover the whole history.
    assert len(sections) == 1
    text = sections[0][2]
    assert text.count("--- ") == 1
    assert [line for line in text.splitlines() if line[:1] in "+-" and not line.startswith(("---", "+++"))] == ["-a", "+c"]


def test_net_diff_falls_back_to_hunks_when_the_file_is_gone(tmp_path):
    """With snapshots stopping and no file on disk, the recorded hunks are all that is left."""
    kept = _diff("tr.1", 1, "gone.py", "a\n", "b\n", "--- gone.py\n+++ gone.py\n@@ -1 +1 @@\n-a\n+b\n")
    dropped = _diff("tr.2", 2, "gone.py", "", "", "--- gone.py\n+++ gone.py\n@@ -1 +1 @@\n-b\n+c\n")

    sections = n.Session.net_diff_sections([kept, dropped], "overall", cwd=str(tmp_path))

    assert len(sections) == 1
    assert sections[0][2].count("--- ") == 1


def test_turn_diff_bounded_snapshots_under_limit():
    assert n.TurnDiff.bounded_snapshots("a", "b") == ("a", "b")
    assert n.TurnDiff.bounded_snapshots("", "") == ("", "")


def test_turn_diff_bounded_snapshots_over_limit_drops_both():
    limit = n.TurnDiff.SNAPSHOT_CHAR_LIMIT
    large = "x" * (limit // 2 + 1)
    assert n.TurnDiff.bounded_snapshots(large, large) == ("", "")


def test_turn_diff_bounded_snapshots_at_limit_keeps_both():
    limit = n.TurnDiff.SNAPSHOT_CHAR_LIMIT
    before = "x" * (limit // 2)
    after = "y" * (limit // 2)
    result = n.TurnDiff.bounded_snapshots(before, after)
    assert result == (before, after)


def test_diff_view_state_open_and_close_file():
    state = n.DiffViewState(view=n.TabbedViewState(titles=("Session",)))
    state.open_file(3)
    assert state.mode is n.DiffViewState.Mode.FILE
    assert state.file == 0
    state.file = 1
    state.close_file()
    assert state.mode is n.DiffViewState.Mode.LIST
    assert state.view.scroll == 0


def test_diff_view_state_move_and_clamp_file():
    state = n.DiffViewState(view=n.TabbedViewState(titles=("Session",)))
    state.move_file(1, 3)
    assert state.file == 1
    state.move_file(2, 3)
    assert state.file == 0
    state.move_file(-1, 3)
    assert state.file == 2
    state.move_file(1, 0)  # no-op when count is zero
    assert state.file == 2


def test_diff_view_state_reset_clears_mode_and_scroll():
    state = n.DiffViewState(view=n.TabbedViewState(titles=("Session",)))
    state.open_file(2)
    state.file = 1
    state.view.scroll = 10
    state.reset()
    assert state.mode is n.DiffViewState.Mode.LIST
    assert state.file == 0
    assert state.view.scroll == 0


def test_diff_view_state_switch_tab_calls_reset():
    state = n.DiffViewState(view=n.TabbedViewState(titles=("Session", "Latest")))
    state.open_file(2)
    state.file = 1
    state.switch_tab(1)
    assert state.view.tab == 1
    assert state.mode is n.DiffViewState.Mode.LIST
    assert state.file == 0


def test_net_diff_for_path_returns_none_when_unchanged():
    assert n.Session.net_diff_for_path("edit", "a.py", "same\n", "same\n") is None


def test_net_diff_for_path_returns_unified_diff():
    result = n.Session.net_diff_for_path("edit", "a.py", "old\n", "new\n")
    assert result is not None
    status, path, diff = result
    assert status == "edit"
    assert path == "a.py"
    assert "@@" in diff or "-old" in diff


def test_net_diff_for_path_uses_dev_null_for_created_files():
    result = n.Session.net_diff_for_path("edit", "new.py", "", "new content\n")
    assert result is not None
    _, _, diff = result
    assert "/dev/null" in diff


def test_find_unambiguous_move_returns_none_without_states():
    assert n.Session._find_unambiguous_move({}, {}) is None


def test_find_unambiguous_move_detects_single_match():
    states = {"old.py": ("content", "moved_content"), "new.py": ("moved_content", "final")}
    assert n.Session._find_unambiguous_move(states, {}) == ("old.py", "new.py")


def test_find_unambiguous_move_ignores_ambiguous():
    states = {"a.py": ("c", "x"), "b.py": ("c", "x")}
    assert n.Session._find_unambiguous_move(states, {}) is None


def test_find_unambiguous_move_skips_self_loop():
    states = {"a.py": ("c", "c")}
    assert n.Session._find_unambiguous_move(states, {}) is None


def test_find_unambiguous_move_skips_legacy_paths():
    states = {"old.py": ("c", "x"), "new.py": ("x", "c")}
    legacy = {"old.py": ["-- a\n++ b"]}
    assert n.Session._find_unambiguous_move(states, legacy) is None
