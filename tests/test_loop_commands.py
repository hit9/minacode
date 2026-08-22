"""CommandLoop surfaces around a turn: the input queue, slash commands, skills, transcript
rendering, and status output."""

import itertools
import json
import os
import time
import tomllib
from types import SimpleNamespace

import pytest
from agent_harness import call, queue, session
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.utils import get_cwidth

import minacode.cli as loop_module
import minacode.cli.commands as commands_mod
import minacode.cli.modals as modals_mod
from minacode.base import (
    DISMISSED,
    SELECTION_BACK,
    SESSION_EVENT_KEY,
    LogBlock,
    LogLine,
    MinacodeError,
    Text,
    ToolError,
    TurnBox,
    __version__,
)
from minacode.cli import CommandLoop
from minacode.cli.commands import (
    name_command,
    session_label_fn,
    session_preview,
    session_rows,
    session_table,
    sessions_command,
    skills_command,
    status,
)
from minacode.cli.modals import choice_application, question_interaction, select_choice
from minacode.config import (
    Config,
)
from minacode.context import ContextManager
from minacode.engine import Agent
from minacode.prompts import SYSTEM_PROMPT
from minacode.render import StatusBar, UiPrinter
from minacode.runner import ToolRunner
from minacode.session import Session, SessionEntry, SessionSnapshotStore, ToolResultRecord
from minacode.skill import SkillLibrary
from minacode.tools import AskSpec, CodeIndex, SkillTool, Tool
from minacode.tui import ASK_DONE, ASK_FREE_TEXT, TuiApp


def _write_skill(root, name, description, body, *, scripts=None):
    folder = os.path.join(root, ".minacode", "skills", name)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "SKILL.md"), "w", encoding="utf-8") as handle:
        handle.write(f"---\nname: {name}\ndescription: {description}\n---\n{body}\n")
    for script_name, script_body in (scripts or {}).items():
        script_dir = os.path.join(folder, "scripts")
        os.makedirs(script_dir, exist_ok=True)
        with open(os.path.join(script_dir, script_name), "w", encoding="utf-8") as handle:
            handle.write(script_body)
    return folder


def queued_texts(s):
    return [item.text for item in s.pending_user_inputs]


def test_ps_command_uses_markdown_renderer(tmp_path):
    s = session(tmp_path)
    s.jobs["job.1"] = SimpleNamespace(id="job.1", status="running", command="pytest -q", elapsed=lambda: 13.7, update_status=lambda: None)
    loop = CommandLoop(Agent(s, output_fn=lambda text: None), input_fn=lambda prompt: "", output_fn=lambda text: None)
    rendered = []
    plain = []
    loop.ui.emit_answer = lambda text, **kwargs: rendered.append(text)
    loop.emit = lambda text="", indent=0: plain.append(text)

    assert loop.command("/ps") == (True, False)

    assert plain == []
    assert len(rendered) == 1
    assert rendered[0].startswith("### Active jobs")
    assert "| id | status | elapsed | command |" in rendered[0]


def test_emit_indents_plain_text_without_losing_its_style(tmp_path):
    """`emit(text, indent)` moves a plain line into a column. The margin is applied after
    styling, because `segments` dispatches on what the text starts with -- prefixing spaces
    first would strip every line of its color. A blank line gets no margin, or an empty emit
    would print a row of trailing spaces into the scrollback."""
    del tmp_path
    ui = UiPrinter(lambda _text: None)
    ui.color = True
    margin = LogBlock.margin(TurnBox.CONTENT_LEVEL)

    error = ui.indent_segments(ui.segments("Error: provider is down"), margin)
    assert error[0] == ("ansired", margin)  # the margin carries the style of the line it opens
    assert "".join(text for _style, text in error) == f"{margin}Error: provider is down\n"

    two_lines = ui.indent_segments(ui.segments("first\nsecond"), margin)
    assert "".join(text for _style, text in two_lines) == f"{margin}first\n{margin}second\n"

    assert "".join(text for _style, text in ui.indent_segments(ui.segments(""), margin)) == "\n"


def test_turn_output_shares_one_column_and_session_chrome_does_not(tmp_path):
    """One left edge for the exchange: the user's line (its `• ` bullet hanging in the margin),
    the turn outcome, and a command's reply. The banner and the resume line frame it flush left."""
    s = session(tmp_path)
    output = []
    loop = CommandLoop(Agent(s, output_fn=output.append), input_fn=lambda prompt: "", output_fn=output.append)
    margin = LogBlock.margin(TurnBox.CONTENT_LEVEL)

    loop.emit_turn("Cancelled")
    loop.emit(f"minacode {__version__}. /help for commands.")

    assert output == [f"{margin}Cancelled", f"minacode {__version__}. /help for commands."]


def test_tui_completion_applies_single_match():
    class OneCompletion(Completer):
        def get_completions(self, document, _complete_event):
            yield Completion("hello", start_position=-len(document.text))

    buffer = Buffer(document=Document("he"), completer=OneCompletion())
    TuiApp.complete_input(buffer)
    assert buffer.text == "hello"


def test_tui_completion_starts_and_cycles_multiple_matches():
    class MultipleCompletions(Completer):
        def get_completions(self, document, _complete_event):
            yield Completion("alpha", start_position=-len(document.text))
            yield Completion("alpine", start_position=-len(document.text))

    completer = MultipleCompletions()
    buffer = Buffer(document=Document("al"), completer=completer)
    started = []
    buffer.start_completion = lambda **kwargs: started.append(kwargs)

    TuiApp.complete_input(buffer)
    assert started == [{"select_first": False}]

    completions = list(completer.get_completions(buffer.document, CompleteEvent()))
    buffer._set_completions(completions)
    TuiApp.complete_input(buffer)
    assert buffer.text == "alpha"
    TuiApp.complete_input(buffer, reverse=True)
    assert buffer.text == "al"
    TuiApp.complete_input(buffer, reverse=True)
    assert buffer.text == "alpine"


def test_queue_acknowledges_only_claimed_duplicate_messages(tmp_path):
    s = session(tmp_path)
    queue(s, "same", "same")
    claimed = s.claim_user_inputs()
    s.enqueue_user_input("same")

    s.acknowledge_user_inputs(claimed)

    assert queued_texts(s) == ["same"]
    assert not s.pending_user_inputs[0].inflight


def test_queue_release_restores_interrupted_inputs(tmp_path):
    s = session(tmp_path)
    s.enqueue_user_input("ready")
    queued = s.pending_user_inputs[0]

    assert s.claim_user_inputs() == [queued]
    s.release_user_inputs()

    assert not queued.inflight


def test_recall_pending_input_can_revise_latest_inflight_message(tmp_path):
    s = session(tmp_path)
    queue(s, "first", "second")
    s.claim_user_inputs()
    loop = CommandLoop(Agent(s, output_fn=lambda text: None), input_fn=lambda prompt: "", output_fn=lambda text: None)
    retried = []

    text = loop.recall_pending_input(lambda: retried.append(True))

    assert text == "second"
    assert queued_texts(s) == ["first"]
    assert s.pending_user_inputs[0].inflight is False
    assert retried == [True]


def test_clearing_recalled_message_leaves_it_deleted(tmp_path):
    s = session(tmp_path)
    queue(s, "first", "delete me")
    loop = CommandLoop(Agent(s, output_fn=lambda text: None), input_fn=lambda prompt: "", output_fn=lambda text: None)

    assert loop.recall_pending_input(lambda: None) == "delete me"

    assert queued_texts(s) == ["first"]
    restored = Session.load_snapshot(s.uid, config=s.config)
    assert queued_texts(restored) == ["first"]


def test_pending_user_inputs_auto_submit_at_round_end(tmp_path):
    """Unconsumed pending_user_inputs are auto-submitted as next input."""
    s = session(tmp_path)
    queue(s, "leftover instruction")

    class FakeModel:
        def request(self, messages, tools=None):
            return {"role": "assistant", "content": "done"}, [], "done"

    agent = Agent(s, output_fn=lambda text: None)
    agent.model = FakeModel()

    def fake_read(prompt="", **kw):
        raise EOFError()

    loop = CommandLoop(agent, input_fn=fake_read, output_fn=lambda text: None)

    loop.run()

    assert s.pending_user_inputs == []
    assert any("leftover instruction" in msg.get("content", "") for msg in s.messages)


def test_queue_live_region_shows_divider_and_pending(tmp_path):
    s = session(tmp_path)
    loop = CommandLoop(Agent(s, output_fn=lambda text: None), input_fn=lambda prompt: "", output_fn=lambda text: None)
    queue(s, "run tests", "then push")

    sent, waiting = loop.view.followup_fragments()
    text = "".join(t for _, t in [*sent, *waiting])
    assert "2 queued" in text and "working" in text
    assert "+ run tests" in text and "+ then push" in text

    claimed = s.claim_user_inputs()
    sent, waiting = loop.view.followup_fragments()
    sent_text = "".join(t for _, t in sent)
    waiting_text = "".join(t for _, t in waiting)
    assert "• run tests" in sent_text and "• then push" in sent_text
    assert "queued" not in waiting_text
    assert "run tests" not in waiting_text and "then push" not in waiting_text

    queue(s, "after claim")
    sent, waiting = loop.view.followup_fragments()
    assert "• run tests" in "".join(t for _, t in sent)
    assert "1 queued" in "".join(t for _, t in waiting)
    assert "+ after claim" in "".join(t for _, t in waiting)

    s.release_user_inputs()
    sent, waiting = loop.view.followup_fragments()
    assert sent == []
    released = "".join(t for _, t in waiting)
    assert "3 queued" in released
    assert "+ run tests" in released and "+ then push" in released and "+ after claim" in released

    s.claim_user_inputs()
    s.acknowledge_user_inputs(claimed)
    sent, waiting = loop.view.followup_fragments()
    assert "run tests" not in "".join(t for _, t in [*sent, *waiting])
    assert "then push" not in "".join(t for _, t in [*sent, *waiting])

    # The divider animates a comet head across the dashes while its label remains stable.
    with pytest.MonkeyPatch.context() as mp:
        seen_head = False
        for tick in range(200):
            mp.setattr(time, "monotonic", lambda tick=tick: tick * 0.1)
            fragments = loop.view.queue_divider_fragments()
            seen_head = seen_head or any(style == "class:divider.glow0" and text == "-" for style, text in fragments)
            assert any(style == "class:divider.working" and text.startswith("working") for style, text in fragments)
            assert all(not style.startswith("class:divider.glow") or text == "-" for style, text in fragments)
        assert seen_head

    s.pending_user_inputs = []
    sent, waiting = loop.view.followup_fragments()
    empty = "".join(t for _, t in [*sent, *waiting])
    assert "working" in empty and "queued" not in empty and "run tests" not in empty


def divider_glow_steps(fragments):
    """The comet's glow step per dash, None where the dash fell back to the plain rule."""
    return [int(style.removeprefix("class:divider.glow")) if style.startswith("class:divider.glow") else None for style, text in fragments if text == "-"]


def test_divider_comet_advances_one_cell_per_animation_frame(tmp_path):
    loop = CommandLoop(Agent(session(tmp_path), output_fn=lambda text: None), input_fn=lambda prompt: "", output_fn=lambda text: None)

    # A head that outruns its own glow between frames stops reading as motion, so the sweep speed
    # is tied to the frame rate rather than chosen independently of it.
    assert loop.view.QUEUE_SWEEP_CELLS_PER_SEC * TuiApp.ANIMATION_INTERVAL == pytest.approx(1.0)

    with pytest.MonkeyPatch.context() as mp:
        heads = []
        for frame in range(6):
            mp.setattr(time, "monotonic", lambda frame=frame: 1000.0 + frame * TuiApp.ANIMATION_INTERVAL)
            steps = divider_glow_steps(loop.view.queue_divider_fragments())
            heads.append(min(range(len(steps)), key=lambda index: (steps[index] is None, steps[index])))

    assert [second - first for first, second in itertools.pairwise(heads)] == [1, 1, 1, 1, 1]


def test_divider_glow_fades_between_cells_and_every_step_has_a_style(tmp_path):
    loop = CommandLoop(Agent(session(tmp_path), output_fn=lambda text: None), input_fn=lambda prompt: "", output_fn=lambda text: None)
    styled = {rule for rule, _style in loop.view.style().style_rules}

    with pytest.MonkeyPatch.context() as mp:
        seen = set()
        for tick in range(400):
            mp.setattr(time, "monotonic", lambda tick=tick: 1000.0 + tick * 0.017)
            seen.update(step for step in divider_glow_steps(loop.view.queue_divider_fragments()) if step is not None)

    # Every shade the comet can emit must exist in the style, or those cells render as plain text.
    assert seen and all(f"divider.glow{step}" in styled for step in seen)
    # A head resting between two cells lights both at the same reduced shade instead of snapping
    # onto the nearer one, which is what keeps the motion smooth when a frame arrives late.
    with pytest.MonkeyPatch.context() as mp:
        span = loop.view.GLOW_STEPS / loop.view.GLOW_REACH
        mp.setattr(time, "monotonic", lambda: (3 + 0.5) / loop.view.QUEUE_SWEEP_CELLS_PER_SEC)
        steps = divider_glow_steps(loop.view.queue_divider_fragments())

    assert steps[3] == steps[4] == int(0.5 * span)
    assert steps[3] > 0  # dimmer than a head sitting exactly on a cell
    assert steps[2] == steps[5] > steps[3]


def test_live_bash_output_stays_above_working_divider_and_queue(tmp_path):
    s = session(tmp_path)
    loop = CommandLoop(Agent(s, output_fn=lambda text: None), input_fn=lambda prompt: "", output_fn=lambda text: None)
    queue(s, "follow up")
    loop.live_preview.active = True
    loop.live_preview.text = "live output"
    loop.live_preview.started_at = time.monotonic()

    text = "".join(fragment for _, fragment in loop.view.tui_activity_fragments())

    assert text.index("live output") < text.index("working") < text.index("+ follow up")
    assert "live output\n\n---" in text


def test_queue_flush_moves_messages_into_log(tmp_path, monkeypatch):
    s = session(tmp_path)
    loop = CommandLoop(Agent(s, output_fn=lambda text: None), input_fn=lambda prompt: "", output_fn=lambda _text: None)
    # The agent's flush hook is wired to move queued messages up into the scrollback log.
    assert loop.agent.on_queue_flush == loop.flush_queued_to_log

    echoed = []
    monkeypatch.setattr(loop_module, "print_formatted_text", lambda value, **_kwargs: echoed.append("".join(text for _style, text in value)))

    loop.flush_queued_to_log(["do a thing", "then verify", "  "])

    assert echoed == ["\n• do a thing\n\n• then verify\n\n"]


def test_queue_command_runs_readonly(tmp_path):
    """A read-only slash command in the queue runs immediately and is not queued for the LLM."""
    s = session(tmp_path)
    out = []
    loop = CommandLoop(Agent(s, output_fn=lambda text: None), input_fn=lambda *a, **k: "", output_fn=out.append)

    loop.run_queued_command("/status")

    assert s.pending_user_inputs == []
    assert out and not any("unavailable" in t for t in out)


def test_queue_command_runs_yolo_toggle(tmp_path):
    """/yolo flips the runtime flag from the queue while the agent works."""
    s = session(tmp_path)
    out = []
    loop = CommandLoop(Agent(s, output_fn=lambda text: None), input_fn=lambda *a, **k: "", output_fn=out.append)

    before = s.settings.yolo
    loop.run_queued_command("/yolo")

    assert s.settings.yolo is (not before)
    assert s.pending_user_inputs == []


def test_queue_command_runs_hints_toggle(tmp_path):
    """/hints reports the state it toggled to, not an ambiguous initial-state reading."""
    s = session(tmp_path)
    out = []
    loop = CommandLoop(Agent(s, output_fn=lambda text: None), input_fn=lambda *a, **k: "", output_fn=out.append)

    assert s.settings.quick_hints is True
    loop.run_queued_command("/hints")
    assert s.settings.quick_hints is False
    assert out[-1].endswith("quick hints toggled off")

    loop.run_queued_command("/hints")
    assert s.settings.quick_hints is True
    assert out[-1].endswith("quick hints toggled on")
    assert s.pending_user_inputs == []


def test_queue_command_rejects_mutating(tmp_path):
    """A state-mutating slash command is refused while the agent works, not queued or run."""
    s = session(tmp_path)
    out = []
    loop = CommandLoop(Agent(s, output_fn=lambda text: None), input_fn=lambda *a, **k: "", output_fn=out.append)

    loop.run_queued_command("/model")

    assert s.pending_user_inputs == []
    assert any("unavailable while the agent is working" in t for t in out)


def test_queue_command_rejects_mutating_mcp_subcommand(tmp_path):
    """Read-only /mcp is allowed; mutating subcommands like connect are refused."""
    s = session(tmp_path)
    out = []
    loop = CommandLoop(Agent(s, output_fn=lambda text: None), input_fn=lambda *a, **k: "", output_fn=out.append)

    loop.run_queued_command("/mcp connect test")

    assert any("read-only /mcp" in t for t in out)


def test_tool_input_without_tui_uses_injected_input(tmp_path):
    s = session(tmp_path)
    calls = []
    loop = CommandLoop(
        Agent(s, output_fn=lambda text: None),
        input_fn=lambda prompt: calls.append(prompt) or "y",
        output_fn=lambda text: None,
    )

    assert loop.tool_input("[Y/n or reason] ") == "y"

    assert calls == ["[Y/n or reason] "]


def test_tool_runner_edit_approval_prints_full_inline_preview(tmp_path, monkeypatch):
    s = session(tmp_path)
    outputs = []
    monkeypatch.setattr(CodeIndex, "update", lambda self, paths: "")
    runner = ToolRunner(s, ContextManager(s), input_fn=lambda prompt: "y", output_fn=lambda text: outputs.append(str(text)))
    content = "".join(f"line {index}\n" for index in range(50))

    runner.run([call("Edit", ["new.txt", [{"op": "create", "content": content}]])])

    assert outputs[0].startswith("  Edit  new.txt\n    ├ preview")
    assert "+line 49" in outputs[0]
    assert "preview truncated" not in outputs[0]
    assert any("[approved]" in output for output in outputs)


def test_exit_command_prints_resume_command(tmp_path):
    s = session(tmp_path)
    s.messages.append({"role": "user", "content": "hello"})
    output = []
    loop = CommandLoop(Agent(s, output_fn=output.append), output_fn=output.append)

    handled, exit_now = loop.command("/exit")

    assert (handled, exit_now) == (True, True)
    # The session took its name from the opening message; the pasted line still carries the uid.
    assert output[-1] == f"Resume 'hello' with:\nminacode --resume {s.uid}"
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
    assert any(style == "class:choice.meta" for style, _text in label_fn(target.uid))
    assert any(style == "class:choice.live" for style, _text in label_fn(s.uid))
    preview = "".join(text for _style, text in captured["kwargs"]["preview_fn"](target.uid))
    assert "the latest answer" in preview
    # The preview reads like the transcript: the user bullet takes the prompt colour and the
    # message the transcript's warm tone, newest exchange at the bottom.
    parts = captured["kwargs"]["preview_fn"](target.uid)
    assert ("class:prompt", "• ") in parts
    assert any(style == "class:choice.user" for style, _text in parts)
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
    assert all("<session_event" not in text for _role, text in summary)


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
    assert all(not text.startswith("→") for _role, text in summary)


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
        assert "".join(text for _style, text in parts) == text_rows[index]
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


def test_empty_exit_does_not_print_resume_command(tmp_path):
    s = session(tmp_path)
    output = []
    loop = CommandLoop(Agent(s, output_fn=output.append), output_fn=output.append)

    handled, exit_now = loop.command("/exit")

    assert (handled, exit_now) == (True, True)
    assert output == []
    assert not os.path.exists(SessionSnapshotStore.session_path(s.config.data_dir, s.cwd, s.uid))


def test_resumed_session_does_not_render_tool_results(tmp_path):
    s = session(tmp_path)
    s.resumed = True
    arguments = json.dumps({"files": [{"path": "a.py", "ranges": [[0, 1]]}]})
    s.messages.extend(
        [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": "need tool",
                "tool_calls": [{"id": "tc.1", "type": "function", "function": {"name": "Read", "arguments": arguments}}],
            },
            {"role": "tool", "tool_call_id": "tc.1", "content": "raw tool result"},
            {"role": "system", "content": f"[Session resumed: uid={s.uid}]"},
        ]
    )
    s.tool_records.append(ToolResultRecord("tr.1", "Read", [{"path": "a.py", "ranges": [[0, 1]]}], "raw tool result", "a.py 0:1"))
    output = []
    loop = CommandLoop(Agent(s, output_fn=output.append), output_fn=output.append)

    loop.render_resumed_session()

    text = "\n".join(output)
    assert s.resumed is False
    assert f"Restored session: {s.uid}" in text
    assert "• hello" in text
    assert "  need tool" in text
    assert "user:" not in text and "assistant:" not in text
    assert "Read  a.py 0:1 → tr.1" in text
    assert "tool:" not in text
    assert "raw tool result" not in text


def test_resumed_session_matches_retried_tool_results_by_call_id(tmp_path):
    s = session(tmp_path)
    s.resumed = True
    arguments = json.dumps({"files": [{"path": "a.py", "ranges": [[0, 1]]}]})
    failed_call = {"id": "failed", "type": "function", "function": {"name": "Read", "arguments": arguments}}
    successful_call = {"id": "successful", "type": "function", "function": {"name": "Read", "arguments": arguments}}
    s.transcript_messages.extend(
        [
            {"role": "user", "content": "read it"},
            {"role": "assistant", "content": None, "tool_calls": [failed_call]},
            {"role": "tool", "tool_call_id": "failed", "result_key": "", "status": "failed"},
            {"role": "assistant", "content": None, "tool_calls": [successful_call]},
            {"role": "tool", "tool_call_id": "successful", "result_key": "tr.1", "status": "ok"},
        ]
    )
    s.tool_records.append(ToolResultRecord("tr.1", "Read", [{"path": "a.py", "ranges": [[0, 1]]}], "data", "a.py 0:1"))
    output = []
    loop = CommandLoop(Agent(s, output_fn=output.append), output_fn=output.append)

    loop.render_resumed_session()

    text = "\n".join(output)
    failed = text.index("[failed]")
    stored = text.index("tr.1")
    assert failed < stored
    assert text.count("tr.1") == 1


def test_resumed_session_warns_when_older_version_wrote_after_transcript(tmp_path):
    s = session(tmp_path)
    s.resumed = True
    s.transcript_incomplete = True
    s.transcript_messages.append({"role": "user", "content": "visible"})
    output = []
    loop = CommandLoop(Agent(s, output_fn=output.append), output_fn=output.append)

    loop.render_resumed_session()

    assert output[0] == f"Restored session: {s.uid}"
    assert output[1] == "Warning: this transcript may omit turns written by an older minacode version."


def test_resumed_session_hides_internal_checkpoint_and_resume_events(tmp_path):
    s = session(tmp_path)
    s.resumed = True
    s.messages.extend(
        [
            {"role": "user", "content": "visible request"},
            {
                "role": "user",
                "content": "hidden compaction checkpoint",
                SESSION_EVENT_KEY: "compaction_checkpoint",
            },
            {
                "role": "user",
                "content": "hidden working-state checkpoint",
                SESSION_EVENT_KEY: "state_checkpoint",
            },
            {
                "role": "user",
                "content": '<session_event type="resumed" at="2026-07-31T08:00:00+08:00" />',
                SESSION_EVENT_KEY: "resumed",
            },
        ]
    )
    output = []
    loop = CommandLoop(Agent(s, output_fn=output.append), output_fn=output.append)

    loop.render_resumed_session()

    text = "\n".join(output)
    assert f"Restored session: {s.uid}" in text
    assert "visible request" in text
    assert "hidden compaction checkpoint" not in text
    assert "hidden working-state checkpoint" not in text
    assert "<session_event" not in text


def test_resumed_session_with_only_internal_events_still_confirms_restore(tmp_path):
    s = session(tmp_path)
    s.resumed = True
    s.messages.extend(
        [
            {"role": "user", "content": "hidden checkpoint", SESSION_EVENT_KEY: "compaction_checkpoint"},
            {"role": "user", "content": "hidden lifecycle event", SESSION_EVENT_KEY: "resumed"},
        ]
    )
    output = []
    loop = CommandLoop(Agent(s, output_fn=output.append), output_fn=output.append)

    loop.render_resumed_session()

    assert output == [f"Restored session: {s.uid}"]


def test_resumed_session_renders_saved_tool_records_without_matching_tool_calls(tmp_path):
    s = session(tmp_path)
    s.resumed = True
    s.messages.extend(
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "compacted answer\nfinal detail"},
        ]
    )
    s.tool_records.append(ToolResultRecord("tr.1", "Bash", ["wc -l minacode.py"], "999 minacode.py", "wc -l minacode.py"))
    output = []
    loop = CommandLoop(Agent(s, output_fn=output.append), output_fn=output.append)

    loop.render_resumed_session()

    text = "\n".join(output)
    assert f"Restored session: {s.uid}" in text
    assert "  compacted answer\n  final detail" in text  # the answer sits in the content column
    assert "user:" not in text and "assistant:" not in text
    assert "  Bash  wc -l minacode.py\n    └ stored tr.1" in text
    assert "999 minacode.py" not in text


def test_resumed_session_separates_turn_boxes(tmp_path):
    s = session(tmp_path)
    s.resumed = True
    s.messages.extend(
        [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "one"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "two"},
        ]
    )
    output = []
    loop = CommandLoop(Agent(s, output_fn=output.append), output_fn=output.append)

    loop.render_resumed_session()

    # Both answers sit in the content column, where the live turn printed them; the user's `• `
    # bullet hangs in that same two-space margin, so every line of text starts at column 2.
    assert output[1:] == ["\n• first", "  one", "", "\n• second", "  two"]


def test_turn_box_groups_followup_users_until_final_assistant():
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "working", "tool_calls": [{"id": "one"}]},
        {"role": "user", "content": "follow-up"},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "next"},
    ]

    boxes = TurnBox.group(messages)

    assert [len(box.messages) for box in boxes] == [4, 1]


def test_turn_box_groups_tool_results_with_calling_assistant():
    # Tool results (role="tool") are kept in the same TurnBox as the
    # assistant that issued the tool_calls, not split prematurely.
    messages = [
        {"role": "user", "content": "read a.py"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "tr.1", "function": {"name": "Read"}}]},
        {"role": "tool", "tool_call_id": "tr.1", "content": "# file content"},
        {"role": "assistant", "content": "done"},
    ]
    boxes = TurnBox.group(messages)
    assert len(boxes) == 1
    assert len(boxes[0].messages) == 4
    roles = [m["role"] for m in boxes[0].messages]
    assert roles == ["user", "assistant", "tool", "assistant"]


def test_eof_exit_prints_resume_command(tmp_path):
    s = session(tmp_path)
    s.messages.append({"role": "user", "content": "hello"})
    output = []
    loop = CommandLoop(Agent(s, output_fn=output.append), input_fn=lambda prompt="": (_ for _ in ()).throw(EOFError()), output_fn=output.append)

    assert loop.run() == 0

    # The session took its name from the opening message; the pasted line still carries the uid.
    assert output[-1] == f"Resume 'hello' with:\nminacode --resume {s.uid}"
    assert os.path.exists(SessionSnapshotStore.session_path(s.config.data_dir, s.cwd, s.uid))


@pytest.mark.parametrize(("interrupt_phase", "expected_cancelled"), [("input", 0), ("request", 1)])
def test_simple_repl_ctrl_c_output_matches_interrupted_phase(tmp_path, monkeypatch, interrupt_phase, expected_cancelled):
    output = []
    reads = iter([KeyboardInterrupt(), EOFError()] if interrupt_phase == "input" else ["question", EOFError()])

    def read_input(_prompt=""):
        value = next(reads)
        if isinstance(value, BaseException):
            raise value
        return value

    agent = Agent(session(tmp_path), output_fn=output.append)
    if interrupt_phase == "request":
        agent.run = lambda _input: (_ for _ in ()).throw(KeyboardInterrupt())
    command_loop = CommandLoop(agent, input_fn=read_input, output_fn=output.append)
    monkeypatch.setattr(loop_module.UpdateChecker, "start", lambda _checker: None)
    monkeypatch.setattr(CodeIndex, "status", lambda _index: False)
    monkeypatch.setattr(CodeIndex, "update_pending_async", lambda _index: None)

    assert command_loop.run() == 0

    assert [line.strip() for line in output].count("Cancelled") == expected_cancelled


@pytest.mark.parametrize("raised", [None, "error"])
def test_simple_repl_publishes_the_final_answer_exactly_once(tmp_path, monkeypatch, raised):
    """The engine publishes a completed turn's answer through output_fn, so the REPL must not
    print the return value on top of it; an error raised before that publish still prints here,
    because nothing else put it in the scrollback."""
    output = []
    reads = iter(["question", EOFError()])

    def read_input(_prompt=""):
        value = next(reads)
        if isinstance(value, BaseException):
            raise value
        return value

    agent = Agent(session(tmp_path), output_fn=output.append)

    def run(_user_input):
        if raised:
            raise MinacodeError("provider is down")
        agent.output_fn("The answer.")  # what Agent.run does before it returns
        return "The answer."

    agent.run = run
    command_loop = CommandLoop(agent, input_fn=read_input, output_fn=output.append)
    monkeypatch.setattr(loop_module.UpdateChecker, "start", lambda _checker: None)
    monkeypatch.setattr(CodeIndex, "status", lambda _index: False)
    monkeypatch.setattr(CodeIndex, "update_pending_async", lambda _index: None)

    assert command_loop.run() == 0

    printed = [str(item) for item in output]
    expected = "Error: provider is down" if raised else "The answer."
    assert sum(expected in line for line in printed) == 1


def test_select_choice_noninteractive_does_not_prompt(tmp_path):
    output = []
    loop = CommandLoop(Agent(session(tmp_path), output_fn=output.append), input_fn=lambda prompt="": "1", output_fn=output.append)

    assert select_choice(loop, "Pick", ("a", "b"), labels={"a": "A"}, current="a") is None
    assert output == []


def test_choice_application_expands_escaped_preview_newlines(tmp_path):
    output = []
    loop = CommandLoop(Agent(session(tmp_path), output_fn=output.append), input_fn=lambda prompt="": "", output_fn=output.append)
    loop.interactive_input = True
    rendered = []

    class Modal:
        def show_modal(self, fragments_fn, key_fn, exclusive=False):
            rendered.extend(fragments_fn())
            return key_fn("enter", "")

    loop.tui = Modal()

    result = choice_application(
        loop,
        "Select:",
        ("A", "B"),
        {},
        "",
        set(),
        preview_fn=lambda choice: "one\\ntwo" if choice == "A" else "",
    )

    assert result == "A"
    previews = [text for style, text in rendered if style == "class:choice.preview"]
    assert previews == ["  │ one\n", "  │ two\n"]
    assert all("\\n" not in text for _, text in rendered)


def test_ask_free_text_prompt_has_no_control_newline(tmp_path):
    """A free-text page drops out of the modal to the shared input row; the answer flows into
    the batch and the modal reopens (ASK_DONE ends it)."""
    output = []
    loop = CommandLoop(Agent(session(tmp_path), output_fn=output.append), input_fn=lambda prompt="": "", output_fn=output.append)
    loop.interactive_input = True
    prompts = []
    results = iter([(ASK_FREE_TEXT, 0), ASK_DONE])
    loop.tui = SimpleNamespace(
        request_input=lambda prompt: prompts.append(prompt) or "typed answer",
        show_modal=lambda fragments_fn, key_fn: next(results),
    )

    assert question_interaction(loop, [AskSpec("Pick?", choices=["A"], previews=["preview"])]) == ["typed answer"]
    assert prompts == ["\nPick?"]  # one shared-input prompt, the question spelled out again


def test_ask_free_text_empty_answer_is_kept(tmp_path):
    """An explicitly empty free-text answer is a legal answer: the batch must return [""] and
    never fall back to the question text (which is only the placeholder for unanswered pages)."""
    output = []
    loop = CommandLoop(Agent(session(tmp_path), output_fn=output.append), input_fn=lambda prompt="": "", output_fn=output.append)
    loop.interactive_input = True
    results = iter([(ASK_FREE_TEXT, 0), ASK_DONE])
    loop.tui = SimpleNamespace(
        request_input=lambda prompt: "",
        show_modal=lambda fragments_fn, key_fn: next(results),
    )

    assert question_interaction(loop, [AskSpec("Pick?")]) == [""]


def test_ask_free_text_on_last_question_submits_without_reentering_modal(tmp_path):
    """A free-text answer to the final question completes the batch right after the shared input
    row; the modal must not reopen for it (a second show_modal would fail the call-count assert)."""
    loop = CommandLoop(Agent(session(tmp_path), output_fn=lambda text: None), input_fn=lambda prompt: "", output_fn=lambda text: None)
    loop.interactive_input = True
    calls = []

    def show_modal(fragments_fn, key_fn):
        calls.append(1)
        key_fn("enter")  # page 1: accept the preselected option, advance to page 2
        key_fn("2")  # page 2: move onto "Type freely..." (digits only move the cursor)
        return key_fn("enter")  # ...and select it -> drops to the shared input row

    loop.tui = SimpleNamespace(request_input=lambda prompt: "typed", show_modal=show_modal)

    assert question_interaction(loop, [AskSpec("One?", choices=["A"]), AskSpec("Two?", choices=["B"])]) == ["A", "typed"]
    assert len(calls) == 1


def test_ask_without_choices_uses_shared_tui_input(tmp_path):
    """A question without choices is a single Type-freely page; Enter drops to the shared row."""
    loop = CommandLoop(Agent(session(tmp_path), output_fn=lambda text: None), input_fn=lambda prompt: "fallback", output_fn=lambda text: None)
    loop.interactive_input = True
    prompts = []
    results = iter([(ASK_FREE_TEXT, 0), ASK_DONE])
    loop.tui = SimpleNamespace(
        request_input=lambda prompt: prompts.append(prompt) or "typed answer",
        show_modal=lambda fragments_fn, key_fn: next(results),
    )

    assert question_interaction(loop, [AskSpec("Explain the issue")]) == ["typed answer"]
    assert prompts == ["\nExplain the issue"]


def test_ask_headless_keeps_plain_per_question_prompts(tmp_path):
    """Without a TUI the batch falls back to one read_input per question, in order."""
    loop = CommandLoop(Agent(session(tmp_path), output_fn=lambda text: None), input_fn=lambda prompt: "fallback", output_fn=lambda text: None)
    prompts = []
    loop.read_input = lambda prompt: prompts.append(prompt) or "answer"

    assert question_interaction(loop, [AskSpec("One?"), AskSpec("Two?", choices=["A"])]) == ["answer", "answer"]
    assert prompts == ["\nOne?", "\nTwo?"]


def test_ask_choice_is_not_echoed_before_final_tool_log(tmp_path, monkeypatch):
    loop = CommandLoop(Agent(session(tmp_path), output_fn=lambda text: None), output_fn=lambda text: None)
    emitted = []
    loop.emit = lambda text="", indent=0: emitted.append(text)
    monkeypatch.setattr(modals_mod, "question_interaction", lambda _loop, specs: ["B"])

    assert modals_mod.question_interaction(loop, [AskSpec("Which?", choices=["A", "B"])]) == ["B"]
    assert emitted == []


def test_ask_notes_flow_into_the_answer(tmp_path):
    """A note entered on a page (`n`, text, Enter) is appended to that question's answer."""
    loop = CommandLoop(Agent(session(tmp_path), output_fn=lambda text: None), input_fn=lambda prompt: "fallback", output_fn=lambda text: None)
    loop.interactive_input = True

    def show_modal(fragments_fn, key_fn):
        key_fn("n")
        for ch in "keep the header":
            key_fn("any", ch)
        key_fn("enter")  # save the note
        return key_fn("enter")  # pick the recommended "A" and submit the batch

    loop.tui = SimpleNamespace(show_modal=show_modal)

    assert question_interaction(loop, [AskSpec("Q?", choices=["A"], recommended=0)]) == ["A\n\nUser notes: keep the header"]


def test_ask_escape_cancels_the_whole_batch(tmp_path):
    """Esc on any page cancels every question with the DISMISSED marker."""
    loop = CommandLoop(Agent(session(tmp_path), output_fn=lambda text: None), input_fn=lambda prompt: "fallback", output_fn=lambda text: None)
    loop.interactive_input = True
    loop.tui = SimpleNamespace(show_modal=lambda fragments_fn, key_fn: SELECTION_BACK)

    result = question_interaction(loop, [AskSpec("One?"), AskSpec("Two?")])
    assert result == [DISMISSED, DISMISSED]


def test_elapsed_since_uses_whole_seconds(monkeypatch):
    monkeypatch.setattr(time, "monotonic", lambda: 104.9)
    assert Text.elapsed_since(100.0) == "4s"

    monkeypatch.setattr(time, "monotonic", lambda: 162.9)
    assert Text.elapsed_since(100.0) == "1m02s"


def test_bash_live_start_pauses_standalone_status(tmp_path):
    loop = CommandLoop(Agent(session(tmp_path), output_fn=lambda text: None), output_fn=lambda text: None)
    loop.ui.color = True
    loop.live_preview.start = lambda: setattr(loop.live_preview, "active", True)
    loop.status_bar.thread = object()
    loop.status_bar.stop = lambda: setattr(loop.status_bar, "thread", None)
    loop.status_bar.start = lambda **_kwargs: setattr(loop.status_bar, "thread", object())

    loop.tool_live_start()
    assert loop.live_status_paused is True
    assert loop.status_bar.thread is None

    loop.tool_live_output("", "")
    assert loop.live_status_paused is False
    assert loop.status_bar.thread is not None


def test_command_loop_indents_intermediate_and_final_messages(tmp_path):
    output = []
    loop = CommandLoop(Agent(session(tmp_path), output_fn=output.append), output_fn=output.append)

    loop.emit_agent_output("First line.\nSecond line.")
    loop.ui.emit_answer("Done.\nFinal detail.")

    assert output == ["  First line.\n  Second line.", "Done.\nFinal detail."]


def test_colored_assistant_and_tool_blocks_each_start_with_one_blank_line(tmp_path):
    loop = CommandLoop(Agent(session(tmp_path), output_fn=lambda _text: None), output_fn=lambda _text: None)
    loop.ui.color = True
    events = []
    loop.emit = lambda text="", indent=0: events.append(text)
    loop.ui.emit_answer = lambda text, **_kwargs: events.append(text)
    first = LogBlock.hierarchy(LogLine("Bash", "first"), [])
    first_result = LogBlock.hierarchy(None, [LogLine("stored", "tr.1")])
    second = LogBlock.hierarchy(LogLine("Bash", "second"), [])

    loop.emit_agent_output("Working on it.")
    loop.tool_output(first)
    loop.tool_output(first_result)
    loop.tool_output(second)

    assert events == ["", "Working on it.", "", first, first_result, "", second]


def test_skill_library_index_and_lookup(tmp_path):
    _write_skill(tmp_path, "release-notes", "Draft a CHANGELOG entry.", "Do the thing.")
    s = session(tmp_path)

    index = s.skills.index()
    assert index.startswith("--- SKILLS ---")
    assert "- release-notes: Draft a CHANGELOG entry." in index
    assert s.skills.get("Release-Notes").name == "release-notes"  # case-insensitive
    assert s.skills.get("missing") is None


def test_builtin_minacode_help_uses_normal_skill_paths(tmp_path):
    s = session(tmp_path)

    skill = s.skills.get("minacode-help")
    assert skill is not None
    assert skill.source == "builtin"
    assert "troubleshoot minacode" in skill.description
    assert "- minacode-help:" in s.skills.index()
    body = SkillTool(s, ["minacode-help"]).call()
    assert "## Inspect the implementation" in body
    assert "### Provider-side tools and web search" in body
    assert all(term in body for term in ("builtin_tools", "$web_search", "pause_turn", "OpenRouter"))
    assert "## Configure providers" in s.skills.resolve_mentions("help with $minacode-help")


def test_every_builtin_skill_is_declared_as_package_data(tmp_path):
    """A builtin skill only exists for installed users if the wheel carries its SKILL.md.

    Running from a checkout hides an omission completely, so the packaging declaration is checked
    here rather than discovered as a missing skill after release."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    builtin_root = os.path.join(repo_root, "minacode", "builtin_skills")
    with open(os.path.join(repo_root, "pyproject.toml"), "rb") as handle:
        packaging = tomllib.load(handle)["tool"]["setuptools"]
    patterns = packaging["package-data"]["minacode"]

    assert "minacode.builtin_skills" in packaging["packages"]
    assert "builtin_skills/*/SKILL.md" in patterns
    for entry in sorted(os.listdir(builtin_root)):
        if os.path.isdir(os.path.join(builtin_root, entry)) and entry != "__pycache__":
            assert os.path.isfile(os.path.join(builtin_root, entry, "SKILL.md")), entry


def test_skill_project_overrides_user_and_user_overrides_builtin(tmp_path):
    user_skill = tmp_path / "data" / "skills" / "minacode-help"
    user_skill.mkdir(parents=True)
    (user_skill / "SKILL.md").write_text("---\nname: minacode-help\ndescription: user version\n---\nuser body\n", encoding="utf-8")

    user_session = session(tmp_path)
    skill = user_session.skills.get("minacode-help")
    assert skill.source == "user"
    assert skill.description == "user version"

    _write_skill(tmp_path, "minacode-help", "project version", "project body")

    project_session = session(tmp_path)
    skill = project_session.skills.get("minacode-help")
    assert skill.source == "project"
    assert skill.description == "project version"


def test_skill_tool_expands_skill_dir(tmp_path):
    folder = _write_skill(tmp_path, "build", "build it", 'Run python "{skill_dir}/scripts/go.py".', scripts={"go.py": "print(1)"})
    s = session(tmp_path)

    output = SkillTool(s, ["build"]).call()
    assert output.startswith('<Skill name="build">')
    assert f'python "{folder}/scripts/go.py"' in output
    assert "{skill_dir}" not in output


def test_skill_tool_unknown_lists_available(tmp_path):
    _write_skill(tmp_path, "known", "known skill", "body")
    s = session(tmp_path)
    with pytest.raises(ToolError) as excinfo:
        SkillTool(s, ["nope"]).call()
    assert "unknown skill 'nope'" in str(excinfo.value)
    assert "known" in str(excinfo.value)


def test_skill_mentions_inject_body(tmp_path):
    _write_skill(tmp_path, "triage", "triage a bug", "Reproduce first.")
    s = session(tmp_path)

    resolved = s.skills.resolve_mentions("please $triage this")
    assert "--- SKILL MENTIONS ---" in resolved
    assert "[triage] triage a bug" in resolved
    assert "Reproduce first." in resolved
    # a bare word without $ is not a mention; an unknown $token is ignored
    assert s.skills.resolve_mentions("triage this") == ""
    assert s.skills.resolve_mentions("$unknown") == ""


def test_skill_tool_absent_only_when_no_skills(tmp_path):
    _write_skill(tmp_path, "available", "available skill", "body")
    withskill = ContextManager(session(tmp_path))
    assert "--- SKILLS ---" in withskill.skills_context()
    assert any(t["function"]["name"] == "Skill" for t in Tool.resolved_schemas(withskill.session))
    messages = withskill.model_messages("system", [{"role": "user", "content": "hi"}])
    assert any(m["content"].startswith("--- SKILLS ---") for m in messages)

    # When truly no skills exist, the tool and section drop out and the prefix stays clean.
    bare = ContextManager(session(tmp_path))
    bare.session.skills = SkillLibrary({})
    assert bare.skills_context() == ""
    tools = Tool.resolved_schemas(bare.session)
    assert not any(t["function"]["name"] == "Skill" for t in tools)
    assert all("--- SKILLS ---" not in str(message.get("content", "")) for message in bare.model_messages(SYSTEM_PROMPT))


def test_skills_command_lists_installed(tmp_path):
    base = CommandLoop(Agent(session(tmp_path), output_fn=lambda t: None), output_fn=lambda t: None)
    assert "### Skills · 1" in skills_command(base, "")
    assert "| `minacode-help` | builtin |" in skills_command(base, "")

    _write_skill(tmp_path, "release-notes", "Draft a CHANGELOG entry.", "body")
    loop = CommandLoop(Agent(session(tmp_path), output_fn=lambda t: None), output_fn=lambda t: None)
    output = skills_command(loop, "")
    assert "| skill | source | description |" in output
    assert "| `release-notes` | project | Draft a CHANGELOG entry. |" in output


def test_skill_loads_dedup_on_repeat(tmp_path):
    _write_skill(tmp_path, "guide", "a guide", "FULL GUIDE INSTRUCTIONS")
    s = session(tmp_path)
    body = SkillTool(s, ["guide"]).call()
    messages = [{"role": "tool", "content": "tr.1 " + body}, {"role": "tool", "content": "tr.7 " + body}]

    deduped = ContextManager(s).dedup_skill_loads(messages)
    assert "FULL GUIDE INSTRUCTIONS" in deduped[0]["content"]  # first copy kept
    assert "FULL GUIDE INSTRUCTIONS" not in deduped[1]["content"]  # repeat collapsed
    assert "repeat load of skill guide" in deduped[1]["content"]
    assert "tr.1" in deduped[1]["content"]


def test_status_and_bar_show_skill_count(tmp_path):
    _write_skill(tmp_path, "one", "d1", "b")
    _write_skill(tmp_path, "two", "d2", "b")
    s = session(tmp_path)
    s.config.mcp = {
        "connected": {"url": "https://connected.example/mcp"},
        "disconnected": {"url": "https://disconnected.example/mcp"},
    }
    s.mcp.tools["connected"] = []
    s.mcp.resources["connected"] = []
    loop = CommandLoop(Agent(s, output_fn=lambda t: None), output_fn=lambda t: None)

    count = len(s.skills.skills)
    assert count == 3
    rendered = status(loop, "")
    assert "mcp `1`" in rendered
    assert f"skills `{count}`" in rendered
    assert f"/ {loop.agent.context.request_token_budget() / 1_000:.1f}K" in rendered
    assert "| cache | (no requests yet) |" in rendered
    assert "| field | value |" in rendered
    bar_text = " | ".join(text for text, _ in StatusBar(s).entries(show_elapsed=False))
    assert f"skills {count}" in bar_text


def test_status_shows_agents_md_state(tmp_path):
    # No candidate file in cwd: still on, but nothing loaded.
    s = session(tmp_path)
    loop = CommandLoop(Agent(s, output_fn=lambda text: None), output_fn=lambda text: None)
    assert "agents_md on (none)" in status(loop, "")

    # Loaded from the project's AGENTS.md.
    (tmp_path / "AGENTS.md").write_text("# Rules\n", encoding="utf-8")
    loaded = session(tmp_path)
    loaded_loop = CommandLoop(Agent(loaded, output_fn=lambda text: None), output_fn=lambda text: None)
    rendered = status(loaded_loop, "")
    assert "agents_md on (AGENTS.md)" in rendered

    # Disabled at runtime: reports off.
    loaded.settings.agents_md = False
    assert "agents_md off" in status(loaded_loop, "")


def test_status_keeps_active_turn_in_context_percentage(tmp_path):
    s = session(tmp_path)
    s.settings.max_context_tokens = 100_000
    s._active_turn_messages = [{"role": "user", "content": "active " + "x" * 200_000}]
    context = ContextManager(s)
    tools = Tool.resolved_schemas(s)
    # Persisted-only baseline (no active turn); status must reflect the active turn on top of this.
    persisted_percent = context.request_tokens(context.model_messages(SYSTEM_PROMPT), tools) * 100 // context.request_token_budget()
    loop = CommandLoop(Agent(s, output_fn=lambda text: None), output_fn=lambda text: None)

    rendered = status(loop, "")

    # status recomputes context_percent with the active turn included, so it exceeds persisted-only
    # and the rendered row shows that recomputed value (not a stale or persisted-only figure).
    assert s.state.context_percent > persisted_percent
    context_row = next(line for line in rendered.splitlines() if line.startswith("| context |"))
    assert f"`{s.state.context_percent}%`" in context_row


def test_status_context_row_uses_last_real_tokens_when_available(tmp_path):
    s = session(tmp_path)
    s.settings.max_context_tokens = 100_000
    estimate_percent = 61  # what the estimate would claim before the call recomputes it
    s.state.context_percent = estimate_percent
    s.usage.last_prompt_tokens = 20_000  # provider reported 20K for the last request
    s.usage.last_prompt_budget = 80_000  # the budget that request was prepared against
    loop = CommandLoop(Agent(s, output_fn=lambda text: None), output_fn=lambda text: None)

    def context_row() -> str:
        return next(line for line in status(loop, "").splitlines() if line.startswith("| context |"))

    assert "`~20.0K / 80.0K`" in context_row()
    assert "`25%`" in context_row()
    assert f"`{estimate_percent}%`" not in context_row()

    # The recorded budget, not today's configuration, stays the denominator.
    s.config.provider.max_tokens = 60_000
    assert "`~20.0K / 80.0K`" in context_row()


def test_status_cache_row_labels_last_and_session_token_counts(tmp_path):
    s = session(tmp_path)
    s.usage.last_cached_prompt_tokens = 76_000
    s.usage.last_cache_write_prompt_tokens = 1_200
    s.usage.last_prompt_tokens = 76_100
    s.usage.cached_prompt_tokens = 83_400
    s.usage.cache_write_prompt_tokens = 4_500
    s.usage.prompt_tokens = 100_000
    loop = CommandLoop(Agent(s, output_fn=lambda text: None), output_fn=lambda text: None)

    cache_row = next(line for line in status(loop, "").splitlines() if line.startswith("| cache |"))

    # Ratios, not the raw pairs: this was the one row long enough to wrap on a normal terminal.
    assert "last `99.9%` (w 1.2K); session `83.4%` (w 4.5K)" in cache_row
    assert "76.1K" not in cache_row


def test_status_command_uses_rich_table_without_outer_rule(tmp_path):
    loop = CommandLoop(Agent(session(tmp_path), output_fn=lambda _text: None), output_fn=lambda _text: None)
    plain = []
    rich = []
    loop.emit = lambda text="", indent=0: plain.append(text)
    loop.ui.emit_answer = lambda text, **kwargs: rich.append((text, kwargs))

    assert loop.command("/status") == (True, False)
    assert plain == []
    assert len(rich) == 1
    assert rich[0][0].startswith("| field | value |")  # one flat table, no section headings
    assert "###" not in rich[0][0]
    assert rich[0][0].count("| --- | --- |") == 1
    assert rich[0][1] == {"rule": False, "compact": True, "indent": TurnBox.CONTENT_LEVEL}


def test_session_from_config_file_theme_param(tmp_path):
    cfg = tmp_path / "minacode.toml"
    cfg.write_text('[runtime]\ntheme = "light"\n')
    s = Session.from_config_file(path=str(cfg), theme="dark")
    assert s.settings.theme == "dark"

    s2 = Session.from_config_file(path=str(cfg))
    assert s2.settings.theme == "light"

    s3 = Session.from_config_file(path=str(cfg), theme="")
    assert s3.settings.theme == "light"
