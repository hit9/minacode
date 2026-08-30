"""loop sessions (split from tests/test_loop_commands.py)."""
import json
import os
import time

from agent_harness import session
from prompt_toolkit.utils import get_cwidth

import wizolt.cli.commands as commands_mod
from wizolt.base import (
    SESSION_EVENT_KEY,
)
from wizolt.cli import CommandLoop
from wizolt.cli.commands import (
    name_command,
    session_label_fn,
    session_preview,
    session_rows,
    session_table,
    sessions_command,
)
from wizolt.config import (
    Config,
)
from wizolt.engine import Agent
from wizolt.session import Session, SessionEntry, SessionSnapshotStore
from wizolt.tui import TuiApp


def test_exit_command_prints_resume_command(tmp_path):
    s = session(tmp_path)
    s.messages.append({"role": "user", "content": "hello"})
    output = []
    loop = CommandLoop(Agent(s, output_fn=output.append), output_fn=output.append)

    handled, exit_now = loop.command("/exit")

    assert (handled, exit_now) == (True, True)
    # The session took its name from the opening message; the pasted line still carries the uid.
    assert output[-1] == f"Resume 'hello' with:\nwizolt --resume {s.uid}"
    assert os.path.exists(SessionSnapshotStore.session_path(s.config.data_dir, s.cwd, s.uid))

def stored_session(tmp_path, text, *, name=""):
    """A saved session in the same project, so /sessions has something to list."""
    other = Session(cwd=str(tmp_path), config=Config(data_dir=str(tmp_path / "data")))
    other.messages.append({"role": "user", "content": text})
    if name:
        other.rename(name)
    other.save_snapshot()
    return other

def test_resume_is_an_alias_for_sessions(tmp_path):
    s = session(tmp_path)
    s.config.data_dir = str(tmp_path / "data")
    loop = CommandLoop(Agent(s, output_fn=lambda text: None), input_fn=lambda prompt: "", output_fn=lambda text: None)
    emitted = []
    loop.emit = lambda text="", indent=0: emitted.append(text)

    assert loop.command("/resume") == (True, False)

    # `--resume` is the flag people already know; the command answers to the same word.
    assert emitted == ["No saved sessions yet."]
    assert "/resume" in CommandLoop.COMMANDS

def test_sessions_command_lists_saved_sessions_without_a_tui(tmp_path):
    s = session(tmp_path)
    s.config.data_dir = str(tmp_path / "data")
    loop = CommandLoop(Agent(s, output_fn=lambda text: None), input_fn=lambda prompt: "", output_fn=lambda text: None)

    assert sessions_command(loop, "") == "No saved sessions yet."

    older = stored_session(tmp_path, "sort the picker by date")
    s.messages.append({"role": "user", "content": "current work"})
    s.save_snapshot()
    listed = sessions_command(loop, "")

    assert older.uid in listed and "sort the picker by date" in listed
    assert s.uid in listed and "current" in listed
    assert sessions_command(loop, "nonsense") == "Usage: /sessions [all]"
    assert loop.resume_request == ""

def test_sessions_command_hands_the_chosen_session_to_the_next_run(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.config.data_dir = str(tmp_path / "data")
    s.messages.append({"role": "user", "content": "current work"})
    s.save_snapshot()
    target = stored_session(tmp_path, "the one we want", name="picked")
    loop = CommandLoop(Agent(s, output_fn=lambda text: None), input_fn=lambda prompt: "", output_fn=lambda text: None)
    loop.tui = TuiApp()
    loop.interactive_input = True
    monkeypatch.setattr(commands_mod, "choice_application", lambda _loop, *args, **kwargs: target.uid)

    handled, exit_now = loop.command("/sessions")

    # Choosing a session ends this run the way /exit does; main() starts the next one on it.
    assert (handled, exit_now) == (True, True)
    assert loop.resume_request == target.uid

def test_sessions_command_choosing_the_current_session_changes_nothing(tmp_path):
    s = session(tmp_path)
    s.config.data_dir = str(tmp_path / "data")
    s.messages.append({"role": "user", "content": "current work"})
    s.save_snapshot()
    loop = CommandLoop(Agent(s, output_fn=lambda text: None), input_fn=lambda prompt: "", output_fn=lambda text: None)
    loop.tui = TuiApp()
    loop.interactive_input = True
    loop.choice_application = lambda *args, **kwargs: s.uid

    assert loop.command("/sessions") == (True, False)
    assert loop.resume_request == ""

    # Cancelling the picker is likewise not a request to go anywhere.
    loop.choice_application = lambda *args, **kwargs: None
    assert loop.command("/sessions") == (True, False)
    assert loop.resume_request == ""

def test_session_labels_carry_age_and_size(tmp_path):
    s = session(tmp_path)
    s.config.data_dir = str(tmp_path / "data")
    s.messages.append({"role": "user", "content": "current work"})
    s.state.round_count = 4
    s.save_snapshot()
    loop = CommandLoop(Agent(s, output_fn=lambda text: None), input_fn=lambda prompt: "", output_fn=lambda text: None)

    entry = SessionSnapshotStore.list_sessions(s.config.data_dir, s.cwd)[0]
    rows, widths = session_table(loop, [entry])
    row = session_rows(rows, widths)[0]

    assert row.startswith("current work")
    assert "just now" in row and "4 rounds" in row and "current" in row
    s.state.round_count = 1
    s.save_snapshot()
    entry = SessionSnapshotStore.list_sessions(s.config.data_dir, s.cwd)[0]
    rows, widths = session_table(loop, [entry])
    row = session_rows(rows, widths)[0]
    assert "1 round " in row + " "
    assert session_preview(entry) == []  # no summary, no preview

def test_sessions_rows_align_columns_in_display_cells(tmp_path, monkeypatch):
    """The picker's labels are table rows: each column padded to the widest value in it, so names
    of different lengths -- CJK included -- still line up their ages and round counts."""
    s = session(tmp_path)
    s.config.data_dir = str(tmp_path / "data")
    loop = CommandLoop(Agent(s, output_fn=lambda text: None), input_fn=lambda prompt: "", output_fn=lambda text: None)
    loop.tui = TuiApp()
    loop.interactive_input = True
    for text in ("a", "中文名", "quite a long session name"):
        other = stored_session(tmp_path, text)
        other.state.round_count = 3
        other.save_snapshot()

    captured: dict[str, dict[str, str]] = {}
    monkeypatch.setattr(commands_mod, "choice_application", lambda _loop, *args, **kwargs: captured.update(labels=args[2]) or None)

    assert loop.command("/sessions") == (True, False)
    labels = list(captured["labels"].values())
    assert len(labels) == 3
    # Once the name and age columns are padded, the round count starts at the same display
    # column in every row; it would drift with the label lengths without the padding. The
    # char-index find() differs across CJK rows, so compare padded display widths instead.
    assert len({get_cwidth(row[: row.find("3 rounds")]) for row in labels}) == 1

def test_sessions_picker_runs_full_screen_with_styled_rows_and_summaries(tmp_path, monkeypatch):
    """The picker is exclusive (alternate screen) with a viewport cap, its rows are styled per
    field, and the preview carries the session's recent messages."""
    s = session(tmp_path)
    s.config.data_dir = str(tmp_path / "data")
    s.messages.append({"role": "user", "content": "current work"})
    s.save_snapshot()
    target = stored_session(tmp_path, "the one we want", name="picked")
    target.messages.append({"role": "assistant", "content": "the latest answer"})
    target.save_snapshot()
    loop = CommandLoop(Agent(s, output_fn=lambda text: None), input_fn=lambda prompt: "", output_fn=lambda text: None)
    loop.tui = TuiApp()
    loop.interactive_input = True
    captured: dict[str, object] = {}
    monkeypatch.setattr(commands_mod, "choice_application", lambda _loop, *args, **kwargs: captured.update(args=args, kwargs=kwargs) or target.uid)

    assert loop.command("/sessions") == (True, True)
    assert captured["kwargs"]["exclusive"] is True
    assert captured["kwargs"]["max_rows"] > 0
    label_fn = captured["kwargs"]["label_fn"]
    assert label_fn is not None
    assert any(style == "class:choice.meta" for style, _ in label_fn(target.uid))
    assert any(style == "class:choice.live" for style, _ in label_fn(s.uid))
    preview = "".join(text for _, text in captured["kwargs"]["preview_fn"](target.uid))
    assert "the latest answer" in preview
    # The preview reads like the transcript: the user bullet takes the prompt colour and the
    # message the transcript's warm tone, newest exchange at the bottom.
    parts = captured["kwargs"]["preview_fn"](target.uid)
    assert ("class:prompt", "• ") in parts
    assert any(style == "class:choice.user" for style, _ in parts)
    assert parts[-1] == ("", "  the latest answer\n")

def test_session_summary_tails_the_recent_messages(tmp_path):
    other = stored_session(tmp_path, "opening")
    other.messages.append({"role": "user", "content": "one"})
    other.messages.append({"role": "assistant", "content": "two"})
    other.messages.append({"role": "assistant", "content": "three"})
    other.save_snapshot()
    entry = SessionSnapshotStore.list_sessions(other.config.data_dir, other.cwd)[0]
    assert SessionSnapshotStore.tail_summary(entry.path) == [("assistant", "three"), ("assistant", "two"), ("user", "one"), ("user", "opening")]
    assert SessionSnapshotStore.tail_summary(entry.path, limit=2) == [("assistant", "three"), ("assistant", "two")]

def test_session_summary_skips_internal_events(tmp_path):
    """Session-resume markers are stored as user-role messages; the preview must not show them as
    conversation, or the 'recent messages' read as a wall of <session_event ...> lines."""
    other = stored_session(tmp_path, "opening")
    other.messages.append({SESSION_EVENT_KEY: "resumed", "content": '<session_event type="resumed" at="2026-08-20" />'})
    other.messages.append({"role": "user", "content": "real question"})
    other.messages.append({"role": "assistant", "content": "real answer"})
    other.save_snapshot()
    entry = SessionSnapshotStore.list_sessions(other.config.data_dir, other.cwd)[0]
    summary = SessionSnapshotStore.tail_summary(entry.path)
    assert summary[:2] == [("assistant", "real answer"), ("user", "real question")]
    assert all("<session_event" not in text for _, text in summary)

def test_session_summary_shows_tool_calls_when_a_turn_has_no_text(tmp_path):
    """A tool-heavy session has almost no assistant text; the preview shows the tool names of
    textless turns so it still says something useful."""
    other = stored_session(tmp_path, "investigate")
    other.messages.append(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "Bash", "arguments": "{}"}},
                {"function": {"name": "Read", "arguments": "{}"}},
            ],
        }
    )
    other.messages.append({"role": "assistant", "content": "found it"})
    other.save_snapshot()
    entry = SessionSnapshotStore.list_sessions(other.config.data_dir, other.cwd)[0]
    assert SessionSnapshotStore.tail_summary(entry.path) == [("assistant", "found it"), ("user", "investigate"), ("tool", "→ Bash, Read")]

def test_session_summary_merges_tool_calls_and_prefers_text(tmp_path):
    """Tool-only turns collapse into one counted line at the end, and when the preview is already
    full of text no tool line is added at all."""
    other = stored_session(tmp_path, "q")
    for name in ("Bash", "Bash", "Read", "Bash"):
        other.messages.append({"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": "{}"}}]})
    other.messages.append({"role": "assistant", "content": "answer"})
    other.save_snapshot()
    entry = SessionSnapshotStore.list_sessions(other.config.data_dir, other.cwd)[0]
    assert SessionSnapshotStore.tail_summary(entry.path) == [("assistant", "answer"), ("user", "q"), ("tool", "→ Bash ×3, Read")]

    full = stored_session(tmp_path, "t0")
    for i in range(1, 6):
        full.messages.append({"role": "user", "content": f"q{i}"})
    full.messages.append({"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "Bash", "arguments": "{}"}}]})
    full.save_snapshot()
    full_entry = SessionSnapshotStore.list_sessions(full.config.data_dir, full.cwd)[0]
    summary = SessionSnapshotStore.tail_summary(full_entry.path)
    assert len(summary) == 5
    assert all(not text.startswith("→") for _, text in summary)

def test_session_summary_widens_the_window_to_reach_buried_text(tmp_path):
    """A tool result can bury the conversation under hundreds of kilobytes; the summary widens its
    tail window until it holds enough text, capped by the budget."""
    other = stored_session(tmp_path, "q0")
    for i in range(1, 6):
        other.messages.append({"role": "user", "content": f"q{i}"})
    other.messages.append({"role": "tool", "content": "x" * 200000})
    other.save_snapshot()
    entry = SessionSnapshotStore.list_sessions(other.config.data_dir, other.cwd)[0]
    assert SessionSnapshotStore.tail_summary(entry.path) == [("user", f"q{i}") for i in range(5, 0, -1)]

def test_session_summary_survives_a_seek_inside_a_cjk_character(tmp_path, monkeypatch):
    """The tail read must never decode from an arbitrary byte: when the seek point lands inside a
    multi-byte character, the old text-mode readline raised UnicodeDecodeError and took /sessions
    down with it. The binary line split skips the torn line instead. The tail budget is shrunk so
    the seek lands inside the character regardless of the default budget."""
    monkeypatch.setattr(SessionSnapshotStore, "TAIL_BUDGET", 65536)
    header = json.dumps({"v": 4})
    # The CJK character sits right at the start of a padding line whose tail pushes the seek point
    # (size - budget) onto the character's second byte.
    pad = '{"padding": "' + "中" + "a" * 65478 + '"}'
    snapshot = '{"messages": [{"role": "user", "content": "latest"}]}'
    line = header + "\n" + pad + "\n" + snapshot + "\n"
    data = line.encode("utf-8")
    cjk = data.find("中".encode())
    seek = len(data) - 65536
    assert seek - cjk in (1, 2)  # the seek point sits inside the multi-byte character
    path = tmp_path / "torn.jsonl"
    path.write_bytes(data)
    entry = SessionEntry(uid="torn", name="", opening="", rounds=0, cwd=str(tmp_path), updated_at=time.time(), path=str(path))
    assert SessionSnapshotStore.tail_summary(entry.path) == [("user", "latest")]

def test_session_label_fn_matches_the_text_layout(tmp_path):
    """The styled rows line up exactly like the plain ones: the styled text of every field is the
    padded table row, and the current marker takes the live colour. A second session whose round
    count is a different width than the current one's makes the last column not always the widest
    value in its own column -- the one spot where the styled and plain layouts can disagree."""
    s = session(tmp_path)
    s.config.data_dir = str(tmp_path / "data")
    s.messages.append({"role": "user", "content": "current work"})
    s.state.round_count = 100
    s.save_snapshot()
    other = stored_session(tmp_path, "a different session")
    other.state.round_count = 3
    other.save_snapshot()
    loop = CommandLoop(Agent(s, output_fn=lambda text: None), input_fn=lambda prompt: "", output_fn=lambda text: None)
    entries = SessionSnapshotStore.list_sessions(s.config.data_dir, s.cwd)
    rows, widths = session_table(loop, entries)
    label_fn = session_label_fn({entry.uid: row for entry, row in zip(entries, rows)}, widths)
    text_rows = session_rows(rows, widths)

    # Every styled row joins to exactly its plain counterpart, including the rows whose last column
    # is narrower than the widest value in that column.
    for entry in entries:
        index = entries.index(entry)
        parts = label_fn(entry.uid)
        assert "".join(text for _, text in parts) == text_rows[index]
    parts = label_fn(s.uid)
    current = next(entry for entry in entries if entry.uid == s.uid)
    index = entries.index(current)
    assert parts[0] == ("", rows[index][0] + " " * (widths[0] - get_cwidth(rows[index][0])) + "  ")  # name plain, padded to the column plus the gap
    assert parts[1][0] == "class:choice.meta"  # age dim
    assert parts[2][0] == "class:choice.meta"  # rounds dim
    assert parts[-1] == ("class:choice.live", "current")

def test_name_command_shows_and_sets_the_session_name(tmp_path):
    s = session(tmp_path)
    s.messages.append({"role": "user", "content": "make the divider smoother"})
    loop = CommandLoop(Agent(s, output_fn=lambda text: None), input_fn=lambda prompt: "", output_fn=lambda text: None)
    s.save_snapshot()

    assert name_command(loop, "") == "Session name: make the divider smoother (from the opening message)"

    assert name_command(loop, "divider polish").startswith("Session named: divider polish")
    assert name_command(loop, "") == "Session name: divider polish (set by you)"
    # The rename is durable on its own, without waiting for the next turn to save.
    assert Session.load_snapshot(s.uid, config=s.config).name == "divider polish"

def test_name_command_reports_an_unnamed_session(tmp_path):
    loop = CommandLoop(Agent(session(tmp_path), output_fn=lambda text: None), input_fn=lambda prompt: "", output_fn=lambda text: None)

    assert name_command(loop, "") == "Session name: (unnamed)"
