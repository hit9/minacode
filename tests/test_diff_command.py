import shlex
import shutil
import subprocess
import sys
import time

import wizolt.cli.commands as commands_mod
from wizolt.base import ToolCall
from wizolt.cli import QUEUE_SAFE_COMMANDS, CommandLoop
from wizolt.cli.commands import diff_command
from wizolt.cli.modals import diff_viewer
from wizolt.config import (
    Config,
)
from wizolt.context import ContextManager
from wizolt.engine import Agent
from wizolt.render import UiPrinter
from wizolt.runner import ToolRunner
from wizolt.session import Session, SessionSnapshotStore, TurnDiff
from wizolt.session.diffs import _find_unambiguous_move, net_diff_for_path, net_diff_sections
from wizolt.tools import ReadTool
from wizolt.tui import DiffViewState, TabbedViewState


def session(tmp_path):
    config = Config()
    config.data_dir = str(tmp_path / "data")
    return Session(cwd=str(tmp_path), config=config)


def loop(session):
    return CommandLoop(Agent(session, output_fn=lambda text: None), output_fn=lambda text: None)


def git_init(path):
    subprocess.run(["git", "-C", str(path), "init"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "test@test"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Test"], check=True, capture_output=True)


async def test_diff_is_in_completer_commands():
    assert "/diff" in CommandLoop.COMMANDS


async def test_diff_appears_in_help():
    assert "/diff" in CommandLoop.HELP


async def test_diff_is_allowed_while_agent_works():
    assert "/diff" in QUEUE_SAFE_COMMANDS


async def test_diff_preserves_cli_history_when_tmux_alternate_screen_is_off(tmp_path):
    executable = shutil.which("tmux")
    if executable is None:
        return
    socket = "wizolt-test-" + tmp_path.name
    command = [executable, "-L", socket]
    probe = tmp_path / "diff_tmux_probe.py"
    probe.write_text(
        """import asyncio
import tempfile
import threading
import time

from wizolt.config import (
    Config,
)

from wizolt.engine import Agent
from wizolt.cli import CommandLoop
from wizolt.cli.commands import diff_command
from wizolt.session import Session
from wizolt.tui import TuiApp

session = Session(cwd="/tmp", config=Config(data_dir=tempfile.mkdtemp()))
session.store_turn_diff("tr.1", 1, "a.py", "-old\\n+new\\n", round=1)
loop = CommandLoop(Agent(session))
app = TuiApp()
loop.tui = app


def drive():
    while app.app is None or not app.app.is_running:
        time.sleep(0.005)
    print("HISTORY MARKER", flush=True)
    result = asyncio.run_coroutine_threadsafe(diff_command(loop, ""), app.app.loop)
    print(result.result(), flush=True)


threading.Thread(target=drive, daemon=True).start()
asyncio.run(app.run())
"""
    )
    try:
        subprocess.run([*command, "new-session", "-d", "-s", "probe", "sleep 30"], check=True)
        subprocess.run([*command, "set-option", "-g", "remain-on-exit", "on"], check=True)
        subprocess.run([*command, "set-option", "-t", "probe", "alternate-screen", "off"], check=True)
        pane_command = f"{shlex.quote(sys.executable)} {shlex.quote(str(probe))}"
        subprocess.run([*command, "respawn-pane", "-k", "-t", "probe", pane_command], check=True)
        deadline = time.monotonic() + 2
        screen = ""
        while "### Latest · Round 1" not in screen and time.monotonic() < deadline:
            time.sleep(0.01)
            screen = subprocess.run([*command, "capture-pane", "-p", "-t", "probe", "-S", "-100"], check=True, capture_output=True, text=True).stdout
        assert "HISTORY MARKER" in screen
        assert "### Latest · Round 1" in screen
    finally:
        subprocess.run([*command, "kill-server"], check=False, capture_output=True)


async def test_alternate_screen_probe_reads_the_resolved_window_option(tmp_path):
    """alternate-screen is a window option, so `show-options` reports it only where a window
    overrides it and stays silent for the usual global `set -wg` form in a tmux.conf. The probe
    has to answer for both, or /diff takes over the primary screen and eats the transcript."""
    executable = shutil.which("tmux")
    if executable is None:
        return
    socket = "wizolt-test-probe-" + tmp_path.name
    command = [executable, "-L", socket]
    probe = tmp_path / "alternate_screen_probe.py"
    probe.write_text(
        f"""import asyncio
import subprocess

from wizolt.tui import TuiApp

print(asyncio.run(TuiApp.alternate_screen_available()))
subprocess.run([{executable!r}, "set-option", "-wg", "alternate-screen", "off"], check=True)
print(asyncio.run(TuiApp.alternate_screen_available()))
subprocess.run([{executable!r}, "set-option", "-wg", "alternate-screen", "on"], check=True)
print(asyncio.run(TuiApp.alternate_screen_available()))
"""
    )
    run = f"{shlex.quote(sys.executable)} {shlex.quote(str(probe))} > {shlex.quote(str(tmp_path / 'out'))} 2>&1"

    def probe_values():
        subprocess.run([*command, "new-window", "-d", "-t", "holder", run], check=True, capture_output=True)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            values = (tmp_path / "out").read_text().splitlines() if (tmp_path / "out").exists() else []
            if len(values) == 3:
                return values
            time.sleep(0.01)
        return []

    try:
        subprocess.run([*command, "new-session", "-d", "-s", "holder", "sleep 60"], check=True, capture_output=True)
        # The global window-option form is invisible to `show-options` without -g; all three
        # observations happen in one tmux client process so process startup does not dominate.
        assert probe_values() == ["True", "False", "True"]
    finally:
        subprocess.run([*command, "kill-server"], check=False, capture_output=True)


async def test_diff_falls_back_to_inline_output_without_alternate_screen(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.store_turn_diff("tr.1", 1, "a.py", "-old\n+new\n", round=1)
    lp = loop(s)
    lp.interactive_input = True
    lp.ui.color = True

    async def unavailable():
        return False

    lp.tui = type("Tui", (), {"alternate_screen_available": staticmethod(unavailable)})()
    opened = []
    monkeypatch.setattr(commands_mod, "diff_viewer", lambda _loop: opened.append(True))

    result = await diff_command(lp, "")

    assert opened == []
    assert "### Latest · Round 1" in result
    assert "+new" in result


async def test_diff_rejects_args(tmp_path):
    lp = loop(session(tmp_path))
    assert await diff_command(lp, "extra") == "Usage: /diff"


async def test_diff_outside_git_repo(tmp_path):
    lp = loop(session(tmp_path))
    assert await diff_command(lp, "") == "No changes"


async def test_diff_clean_session(tmp_path):
    lp = loop(session(tmp_path))
    assert await diff_command(lp, "") == "No changes"


async def test_diff_round_with_no_net_changes_is_empty(tmp_path):
    s = session(tmp_path)
    s.store_turn_diff("tr.1", 1, "a.py", "-old\n+mid\n", before="old\n", after="mid\n", round=1)
    s.store_turn_diff("tr.2", 2, "a.py", "-mid\n+old\n", before="mid\n", after="old\n", round=1)

    assert await diff_command(loop(s), "") == "No changes"


async def test_diff_shows_latest_round(tmp_path):
    s = session(tmp_path)
    s.store_turn_diff("tr.1", 1, "old.py", "-old\n+older\n", round=1)
    s.store_turn_diff("tr.2", 2, "new.py", "-old\n+new\n", round=2)

    lp = loop(s)
    result = await diff_command(lp, "")

    assert "### Latest · Round 2" in result
    assert "#### new.py" in result
    assert "+new" in result
    latest, _, session_section = result.partition("### Session")
    assert "old.py" not in latest
    assert "#### old.py" in session_section


async def test_diff_shows_latest_round_outside_git_repo(tmp_path):
    s = session(tmp_path)
    s.store_turn_diff("tr.1", 3, "x.py", "-a\n+b\n")

    lp = loop(s)
    result = await diff_command(lp, "")

    assert "### Latest · Round 3" in result
    assert "#### x.py" in result
    assert "+b" in result
    assert result != "Not in a git repository"


async def test_diff_ignores_git_worktree_changes(tmp_path):
    git_init(tmp_path)
    (tmp_path / "a.py").write_text("old\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "a.py"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-m", "init"], check=True, capture_output=True)
    (tmp_path / "a.py").write_text("new\n", encoding="utf-8")
    (tmp_path / "untracked.py").write_text("hello\n", encoding="utf-8")

    lp = loop(session(tmp_path))
    assert await diff_command(lp, "") == "No changes"


async def test_diff_bounds_large_session_output(tmp_path):
    s = session(tmp_path)
    large = "\n".join(f"+line {index}" for index in range(2_000))
    s.store_turn_diff("tr.1", 1, "a.py", large, round=1)

    lp = loop(s)
    result = await diff_command(lp, "")
    assert "truncated" in result.lower()


async def test_ui_segment_lines_keeps_styled_diff_lines_together():
    ui = UiPrinter(output_fn=lambda text: None)
    diff = "--- a.py\n+++ a.py\n@@ -1 +1 @@\n-old\n+return 42\n"

    segments = ui.diff_segments(diff)
    lines = ui.segment_lines(segments)

    assert len(lines) == len(diff.splitlines())
    assert "".join(text for line in lines for _, text in line) == "".join(text for _, text in segments)
    assert any("+return 42" in "".join(text for _, text in line) for line in lines)


async def test_diff_counts_only_hunk_changes():
    diff = "--- a.py\n+++ a.py\n@@ -1 +1,2 @@\n-old\n+++heading\n+new\n"

    assert CommandLoop.diff_counts(diff) == (2, 1)


async def test_diff_viewer_list_shows_change_counts_without_status_prefix(tmp_path):
    s = session(tmp_path)
    s.store_turn_diff("tr.1", 1, "a.py", "unused", before="old\n", after="new\nextra\n", round=1)
    lp = loop(s)
    rendered = []

    class Modal:
        async def show_modal(self, fragments_fn, _key_fn, **_kwargs):
            rendered.extend(fragments_fn())

    lp.tui = Modal()

    await diff_viewer(lp)

    text = "".join(fragment for _, fragment in rendered)
    assert "+2 -1 a.py" in text
    assert "Edit" not in text


async def test_tool_runner_captures_edit_turn_diff(tmp_path):
    git_init(tmp_path)
    (tmp_path / "a.py").write_text("old\n", encoding="utf-8")
    s = session(tmp_path)
    s.state.turn_step = 1
    s.state.round_count = 1
    s.settings.yolo = True

    runner = ToolRunner(s, ContextManager(s), input_fn=lambda prompt: "", output_fn=lambda text: None)
    read = ReadTool(s, [{"path": "a.py"}]).call()
    key = s.register_source_drafts(list(read.drafts))[0]
    call = ToolCall("edit-1", "Edit", ["a.py", key, [{"op": "replace", "start": 1, "end": 1, "content": "new\n"}]])
    status, _, observation = await runner.run_one(call)

    assert status == "ok"
    assert observation is None
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


async def test_session_diff_sections_show_overall_file_effect(tmp_path):
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


async def test_diff_sections_follow_file_across_unambiguous_moves(tmp_path):
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
    assert [path for _, path, _diff in latest[1]] == ["skill/SKILL.md"]
    assert CommandLoop.diff_counts(latest[1][0][2]) == (2, 2)
    assert [path for _, path, _diff in session_sections] == ["skill/SKILL.md"]
    assert CommandLoop.diff_counts(session_sections[0][2]) == (3, 0)


async def test_diff_sections_do_not_guess_ambiguous_moves(tmp_path):
    s = session(tmp_path)
    s.store_turn_diff("tr.1", 1, "source.md", "unused", before="", after="same\n")
    s.store_turn_diff("tr.2", 2, "first.md", "unused", before="same\n", after="first\n")
    s.store_turn_diff("tr.3", 3, "second.md", "unused", before="same\n", after="second\n")

    assert [path for _, path, _diff in s.session_diff_sections()] == ["source.md", "first.md", "second.md"]


async def test_session_diff_sections_fall_back_to_legacy_diffs(tmp_path):
    s = session(tmp_path)
    s.store_turn_diff("tr.1", 1, "a.py", "-old\n+new\n")

    assert s.session_diff_sections() == [("overall", "a.py", "-old\n+new\n")]


async def test_store_turn_diff_drops_large_net_snapshots(tmp_path):
    s = session(tmp_path)
    large = "x" * (TurnDiff.SNAPSHOT_CHAR_LIMIT + 1)
    s.store_turn_diff("tr.1", 1, "large.py", "-old\n+new\n", before=large, after="new\n", round=1)

    diff = s.turn_diffs[0]
    assert diff.diff == "-old\n+new\n"
    assert diff.before == ""
    assert diff.after == ""
    latest = s.latest_round_diff_sections()
    assert latest is not None
    assert latest[1] == [("edit", "large.py", "-old\n+new\n")]
    assert s.session_diff_sections() == [("overall", "large.py", "-old\n+new\n")]


async def test_legacy_diff_reconstructed_from_disk(tmp_path):
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


async def test_legacy_diff_falls_back_when_disk_drifted(tmp_path):
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


async def test_latest_round_coalesces_legacy_diffs_for_same_path(tmp_path):
    s = session(tmp_path)
    large = "x" * (TurnDiff.SNAPSHOT_CHAR_LIMIT + 1)
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
    assert CommandLoop.diff_counts(latest[1][0][2]) == (2, 2)


async def test_latest_round_diffs_include_all_steps_in_round(tmp_path):
    s = session(tmp_path)
    s.store_turn_diff("tr.1", 1, "a.py", "-old\n+mid\n", before="old\n", after="mid\n", round=1)
    s.store_turn_diff("tr.2", 2, "b.py", "-one\n+two\n", before="one\n", after="two\n", round=1)
    s.store_turn_diff("tr.3", 2, "a.py", "-mid\n+new\n", before="mid\n", after="new\n", round=1)
    s.store_turn_diff("tr.4", 0, "older.py", "-x\n+y\n", before="x\n", after="y\n", round=0)

    latest = s.latest_round_diff_sections()

    assert latest is not None
    round, sections = latest
    assert round == 1
    assert [path for _, path, _diff in sections] == ["a.py", "b.py"]
    a_diff = sections[0][2]
    assert "-old" in a_diff
    assert "+new" in a_diff
    assert "mid" not in a_diff


async def test_session_snapshot_turn_diff_roundtrip(tmp_path):
    s = session(tmp_path)
    s.messages.append({"role": "user", "content": "seed"})
    s.store_turn_diff("tr.1", 2, "x.py", "-a\n+b\n", round=1)

    uid = await s.save_snapshot()
    loaded = SessionSnapshotStore.load(uid, s.config, s.settings)

    assert len(loaded.turn_diffs) == 1
    assert loaded.turn_diffs[0].path == "x.py"
    assert loaded.turn_diffs[0].diff == "-a\n+b\n"
    assert loaded.turn_diffs[0].turn == 2
    assert loaded.turn_diffs[0].round == 1


async def test_resume_renders_turn_diffs_from_the_snapshot(tmp_path):
    """Edit diffs survive a resume: `/diff` rebuilds both tabs from the persisted turn_diffs,
    file snapshots included."""
    s = session(tmp_path)
    (tmp_path / "x.py").write_text("new\n")
    s.messages.append({"role": "user", "content": "edit it"})
    s.store_turn_diff("tr.1", 1, "x.py", "--- x.py\n+++ x.py\n@@ -1 +1 @@\n-old\n+new\n", before="old\n", after="new\n", round=1)
    await s.save_snapshot()

    loaded = Session.load_snapshot(s.uid, config=s.config, settings=s.settings, cwd=str(tmp_path))
    result = await diff_command(loop(loaded), "")

    assert [(diff.key, diff.path, diff.before, diff.after) for diff in loaded.turn_diffs] == [("tr.1", "x.py", "old\n", "new\n")]
    assert "### Latest" in result
    assert "### Session" in result
    assert result.count("-old") == 2
    assert result.count("+new") == 2


def _diff(key, turn, path, before, after, text):
    return TurnDiff(key, turn, path, text, before=before, after=after, round=turn)


async def test_net_diff_emits_one_description_per_path_when_snapshots_stop(tmp_path):
    """A file that grows past the snapshot size limit partway through leaves some edits with
    snapshots and some without. Both describe the same file, so only one may be emitted."""
    (tmp_path / "x.py").write_text("c\n")
    kept = _diff("tr.1", 1, "x.py", "a\n", "b\n", "--- x.py\n+++ x.py\n@@ -1 +1 @@\n-a\n+b\n")
    dropped = _diff("tr.2", 2, "x.py", "", "", "--- x.py\n+++ x.py\n@@ -1 +1 @@\n-b\n+c\n")

    sections = net_diff_sections([kept, dropped], "overall", cwd=str(tmp_path))

    assert len(sections) == 1
    text = sections[0][2]
    assert text.count("--- ") == 1  # one diff for the file, not one per description
    assert [line for line in text.splitlines() if line[:1] in "+-" and not line.startswith(("---", "+++"))] == ["-a", "+c"]


async def test_net_diff_prefers_snapshots_when_the_last_edit_has_them(tmp_path):
    """A snapshot-less edit in the middle is already reflected in the next snapshot's `after`."""
    (tmp_path / "x.py").write_text("unused\n")
    first = _diff("tr.1", 1, "x.py", "a\n", "b\n", "--- x.py\n+++ x.py\n@@ -1 +1 @@\n-a\n+b\n")
    dropped = _diff("tr.2", 2, "x.py", "", "", "--- x.py\n+++ x.py\n@@ -1 +1 @@\n-b\n+c\n")
    last = _diff("tr.3", 3, "x.py", "b\n", "c\n", "--- x.py\n+++ x.py\n@@ -1 +1 @@\n-b\n+c\n")

    sections = net_diff_sections([first, dropped, last], "overall", cwd=str(tmp_path))

    # The on-disk content is ignored here: the recorded snapshots already cover the whole history.
    assert len(sections) == 1
    text = sections[0][2]
    assert text.count("--- ") == 1
    assert [line for line in text.splitlines() if line[:1] in "+-" and not line.startswith(("---", "+++"))] == ["-a", "+c"]


async def test_net_diff_recovers_legacy_prefix_when_the_file_shrinks(tmp_path):
    """A file that starts above the snapshot limit records its early edits without snapshots, then
    keeps snapshots once it shrinks below it. The snapshot span starts at the first kept edit, so the
    snapshot-less prefix must be recovered from its hunks or the early changes vanish from the diff."""
    (tmp_path / "x.py").write_text("d\n")
    prefix = _diff("tr.1", 1, "x.py", "", "", "--- x.py\n+++ x.py\n@@ -1 +1 @@\n-a\n+b\n")
    shrink = _diff("tr.2", 2, "x.py", "", "", "--- x.py\n+++ x.py\n@@ -1 +1 @@\n-b\n+c\n")
    last = _diff("tr.3", 3, "x.py", "c\n", "d\n", "--- x.py\n+++ x.py\n@@ -1 +1 @@\n-c\n+d\n")

    sections = net_diff_sections([prefix, shrink, last], "overall", cwd=str(tmp_path))

    assert len(sections) == 1
    text = sections[0][2]
    assert text.count("--- ") == 1
    assert [line for line in text.splitlines() if line[:1] in "+-" and not line.startswith(("---", "+++"))] == ["-a", "+d"]


async def test_net_diff_recovers_snapshot_history_when_the_file_is_gone(tmp_path):
    """With snapshots stopping and no file on disk, the trailing hunks forward-apply onto the last
    snapshot's `after` to recover the final content, so the whole history survives the missing file."""
    kept = _diff("tr.1", 1, "gone.py", "a\n", "b\n", "--- gone.py\n+++ gone.py\n@@ -1 +1 @@\n-a\n+b\n")
    dropped = _diff("tr.2", 2, "gone.py", "", "", "--- gone.py\n+++ gone.py\n@@ -1 +1 @@\n-b\n+c\n")

    sections = net_diff_sections([kept, dropped], "overall", cwd=str(tmp_path))

    assert len(sections) == 1
    text = sections[0][2]
    assert text.count("--- ") == 1
    assert [line for line in text.splitlines() if line[:1] in "+-" and not line.startswith(("---", "+++"))] == ["-a", "+c"]


async def test_turn_diff_bounded_snapshots_under_limit():
    assert TurnDiff.bounded_snapshots("a", "b") == ("a", "b")
    assert TurnDiff.bounded_snapshots("", "") == ("", "")


async def test_turn_diff_bounded_snapshots_over_limit_drops_both():
    """Either snapshot exceeding the cap drops the pair — one alone would read as a whole-file
    creation or deletion."""
    limit = TurnDiff.SNAPSHOT_CHAR_LIMIT
    large = "x" * (limit + 1)
    assert TurnDiff.bounded_snapshots(large, large) == ("", "")
    assert TurnDiff.bounded_snapshots(large, "small\n") == ("", "")
    assert TurnDiff.bounded_snapshots("small\n", large) == ("", "")


async def test_turn_diff_bounded_snapshots_at_limit_keeps_both():
    """The cap applies per snapshot, not to the two summed: a file just under it is still tracked
    even though the pair is twice the cap."""
    limit = TurnDiff.SNAPSHOT_CHAR_LIMIT
    before = "x" * limit
    after = "y" * limit
    result = TurnDiff.bounded_snapshots(before, after)
    assert result == (before, after)


async def test_diff_view_state_open_and_close_file():
    state = DiffViewState(view=TabbedViewState(titles=("Session",)))
    state.open_file(3)
    assert state.mode is DiffViewState.Mode.FILE
    assert state.file == 0
    state.file = 1
    state.close_file()
    assert state.mode is DiffViewState.Mode.LIST
    assert state.view.scroll == 0


async def test_diff_view_state_move_and_clamp_file():
    state = DiffViewState(view=TabbedViewState(titles=("Session",)))
    state.move_file(1, 3)
    assert state.file == 1
    state.move_file(2, 3)
    assert state.file == 0
    state.move_file(-1, 3)
    assert state.file == 2
    state.move_file(1, 0)  # no-op when count is zero
    assert state.file == 2


async def test_diff_view_state_reset_clears_mode_and_scroll():
    state = DiffViewState(view=TabbedViewState(titles=("Session",)))
    state.open_file(2)
    state.file = 1
    state.view.scroll = 10
    state.reset()
    assert state.mode is DiffViewState.Mode.LIST
    assert state.file == 0
    assert state.view.scroll == 0


async def test_diff_view_state_switch_tab_calls_reset():
    state = DiffViewState(view=TabbedViewState(titles=("Session", "Latest")))
    state.open_file(2)
    state.file = 1
    state.switch_tab(1)
    assert state.view.tab == 1
    assert state.mode is DiffViewState.Mode.LIST
    assert state.file == 0


async def test_net_diff_for_path_returns_none_when_unchanged():
    assert net_diff_for_path("edit", "a.py", "same\n", "same\n") is None


async def test_net_diff_for_path_returns_unified_diff():
    result = net_diff_for_path("edit", "a.py", "old\n", "new\n")
    assert result is not None
    status, path, diff = result
    assert status == "edit"
    assert path == "a.py"
    assert "@@" in diff or "-old" in diff


async def test_net_diff_for_path_uses_dev_null_for_created_files():
    result = net_diff_for_path("edit", "new.py", "", "new content\n")
    assert result is not None
    _, _, diff = result
    assert "/dev/null" in diff


async def test_find_unambiguous_move_returns_none_without_states():
    assert _find_unambiguous_move({}, {}) is None


async def test_find_unambiguous_move_detects_single_match():
    states = {"old.py": ("content", "moved_content"), "new.py": ("moved_content", "final")}
    assert _find_unambiguous_move(states, {}) == ("old.py", "new.py")


async def test_find_unambiguous_move_ignores_ambiguous():
    states = {"a.py": ("c", "x"), "b.py": ("c", "x")}
    assert _find_unambiguous_move(states, {}) is None


async def test_find_unambiguous_move_skips_self_loop():
    states = {"a.py": ("c", "c")}
    assert _find_unambiguous_move(states, {}) is None


async def test_find_unambiguous_move_skips_legacy_paths():
    states = {"old.py": ("c", "x"), "new.py": ("x", "c")}
    legacy = {"old.py": ["-- a\n++ b"]}
    assert _find_unambiguous_move(states, legacy) is None
