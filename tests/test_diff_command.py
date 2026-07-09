import subprocess

import pytest

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
    assert lp.diff_command("") == "Not in a git repository"


def test_diff_clean_repo(tmp_path):
    git_init(tmp_path)
    lp = loop(session(tmp_path))
    assert lp.diff_command("") == "No changes"


def test_diff_unstaged_tracked_file(tmp_path):
    git_init(tmp_path)
    (tmp_path / "a.py").write_text("old\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "a.py"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-m", "init"], check=True, capture_output=True)
    (tmp_path / "a.py").write_text("new\n", encoding="utf-8")

    lp = loop(session(tmp_path))
    result = lp.diff_command("")
    assert "### Unstaged" in result
    assert "-old" in result
    assert "+new" in result


def test_diff_staged_section(tmp_path):
    git_init(tmp_path)
    (tmp_path / "a.py").write_text("old\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "a.py"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-m", "init"], check=True, capture_output=True)
    (tmp_path / "a.py").write_text("staged\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "a.py"], check=True, capture_output=True)

    lp = loop(session(tmp_path))
    result = lp.diff_command("")
    assert "### Staged" in result
    assert "-old" in result
    assert "+staged" in result
    assert "### Unstaged" not in result


def test_diff_untracked_file_synthesized(tmp_path):
    git_init(tmp_path)
    (tmp_path / "new.py").write_text("hello\n", encoding="utf-8")

    lp = loop(session(tmp_path))
    result = lp.diff_command("")
    assert "### Untracked files" in result
    assert "+hello" in result


def test_diff_bounds_large_output(tmp_path):
    git_init(tmp_path)
    (tmp_path / "a.py").write_text("line\n" * 10_000, encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "a.py"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-m", "init"], check=True, capture_output=True)
    (tmp_path / "a.py").write_text("changed\n" * 10_000, encoding="utf-8")

    lp = loop(session(tmp_path))
    result = lp.diff_command("")
    assert "### Unstaged" in result
    assert "truncated" in result.lower()


def test_git_diff_service_split_files():
    diff = "--- a.py\n+++ b.py\n@@ -1 +1 @@\n-old\n+new\n--- c.py\n+++ d.py\n@@ -1 +1 @@\n-1\n+2\n"
    sections = n.GitDiffService.split_files(diff)
    assert len(sections) == 2
    assert sections[0][0] == "b.py"
    assert "old" in sections[0][1]
    assert sections[1][0] == "d.py"


def test_tool_runner_captures_edit_turn_diff(tmp_path):
    git_init(tmp_path)
    (tmp_path / "a.py").write_text("old\n", encoding="utf-8")
    s = session(tmp_path)
    s.state.turn_step = 1
    s.settings.yolo = True

    runner = n.ToolRunner(s, n.ContextManager(s), input_fn=lambda prompt: "", output_fn=lambda text: None)
    call = n.ToolCall("edit-1", "Edit", ["a.py", [{"op": "replace_all", "old": "old\n", "new": "new\n"}]])
    status, message = runner.run_one(call)

    assert status == "ok"
    assert len(s.turn_diffs) == 1
    td = s.turn_diffs[0]
    assert td.path == "a.py"
    assert td.turn == 1
    assert td.key.startswith("tr.")
    assert "-old" in td.diff
    assert "+new" in td.diff
    assert td.accepted is True


def test_session_snapshot_turn_diff_roundtrip(tmp_path):
    s = session(tmp_path)
    s.messages.append({"role": "user", "content": "seed"})
    s.store_turn_diff("tr.1", 2, "x.py", "-a\n+b\n")

    store = n.SessionSnapshotStore(s)
    uid = store.save()
    loaded = n.SessionSnapshotStore.load(uid, s.config, s.settings)

    assert len(loaded.turn_diffs) == 1
    assert loaded.turn_diffs[0].path == "x.py"
    assert loaded.turn_diffs[0].diff == "-a\n+b\n"
    assert loaded.turn_diffs[0].turn == 2
    assert loaded.turn_diffs[0].accepted is True
