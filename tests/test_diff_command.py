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
    assert "/diff" in n.CommandCompleter.COMMANDS


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
    assert "old.py" not in result


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

    assert n.diff_counts(diff) == (2, 1)


def test_diff_viewer_list_shows_change_counts_without_status_prefix(tmp_path):
    s = session(tmp_path)
    s.store_turn_diff("tr.1", 1, "a.py", "unused", before="old\n", after="new\nextra\n", round=1)
    lp = loop(s)
    rendered = []

    def render(app):
        assert app.full_screen is True
        rendered.extend(app.layout.current_control.text())

    lp.run_input_app = render

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
    assert n.diff_counts(latest[1][0][2]) == (2, 2)
    assert [path for _status, path, _diff in session_sections] == ["skill/SKILL.md"]
    assert n.diff_counts(session_sections[0][2]) == (3, 0)


def test_diff_sections_do_not_guess_ambiguous_moves(tmp_path):
    s = session(tmp_path)
    s.store_turn_diff("tr.1", 1, "source.md", "unused", before="", after="same\n")
    s.store_turn_diff("tr.2", 2, "first.md", "unused", before="same\n", after="first\n")
    s.store_turn_diff("tr.3", 3, "second.md", "unused", before="same\n", after="second\n")

    assert [path for _status, path, _diff in s.session_diff_sections()] == ["source.md", "first.md", "second.md"]


def test_session_diff_sections_ignore_legacy_diffs_without_before_after(tmp_path):
    s = session(tmp_path)
    s.store_turn_diff("tr.1", 1, "a.py", "-old\n+new\n")

    assert s.session_diff_sections() == []


def test_store_turn_diff_drops_large_net_snapshots(tmp_path):
    s = session(tmp_path)
    large = "x" * (n.TURN_DIFF_SNAPSHOT_CHAR_LIMIT + 1)
    s.store_turn_diff("tr.1", 1, "large.py", "-old\n+new\n", before=large, after="new\n", round=1)

    diff = s.turn_diffs[0]
    assert diff.diff == "-old\n+new\n"
    assert diff.before == ""
    assert diff.after == ""
    latest = s.latest_round_diff_sections()
    assert latest is not None
    assert latest[1] == [("edit", "large.py", "-old\n+new\n")]
    assert s.session_diff_sections() == []


def test_latest_round_coalesces_legacy_diffs_for_same_path(tmp_path):
    s = session(tmp_path)
    large = "x" * (n.TURN_DIFF_SNAPSHOT_CHAR_LIMIT + 1)
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
    assert n.diff_counts(latest[1][0][2]) == (2, 2)


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


def test_resume_recovers_latest_round_diff_from_old_edit_records(tmp_path):
    s = session(tmp_path)
    path = tmp_path / "data" / "sessions" / f"{s.uid}.jsonl"
    path.parent.mkdir(parents=True)
    output = "\n".join(
        [
            '<Edit path="x.py">',
            '<file_stat mtime_ns="1" size="2"/>',
            "--- x.py",
            "+++ x.py",
            "@@ -1 +1 @@",
            "-old",
            "+new",
            "<invalidate>0:1</invalidate>",
            "</Edit>",
        ]
    )
    n.SessionSnapshotStore.write_jsonl(
        str(path),
        {
            "uid": s.uid,
            "cwd": str(tmp_path),
            "messages": [{"role": "user", "content": "seed"}],
            "tool_records": [{"key": "tr.1", "name": "Edit", "args": ["x.py"], "output": output, "note": ""}],
            "tool_counter": 1,
        },
        mode="w",
    )

    loaded = n.Session.load_snapshot(s.uid, config=s.config, settings=s.settings)
    result = loop(loaded).diff_command("")

    assert len(loaded.turn_diffs) == 1
    assert loaded.turn_diffs[0].key == "tr.1"
    assert loaded.turn_diffs[0].path == "x.py"
    assert "### Latest" in result
    assert "-old" in result
    assert "+new" in result


def test_edit_diff_recovery_ignores_malformed_output():
    assert n.SessionSnapshotCodec.edit_diff_from_output("<Edit path=bad />") == ("", "")
