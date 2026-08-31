"""loop queue ui (split from tests/test_loop_commands.py)."""
import itertools
import time

import pytest
from agent_harness import call, queue, session

import wizolt.cli.loop as loop_module
from wizolt.cli import CommandLoop
from wizolt.context import ContextManager
from wizolt.engine import Agent
from wizolt.runner import ToolRunner
from wizolt.tools import CodeIndex
from wizolt.tui import TuiApp


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
    styled = {rule for rule, _ in loop.view.style().style_rules}

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
    monkeypatch.setattr(loop_module, "print_formatted_text", lambda value, **_kwargs: echoed.append("".join(text for _, text in value)))

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

def test_hints_command_is_removed(tmp_path):
    s = session(tmp_path)
    out = []
    loop = CommandLoop(Agent(s, output_fn=lambda text: None), input_fn=lambda *a, **k: "", output_fn=out.append)

    # Every /hints spelling follows the normal unknown-command path; there is no toggle left.
    for variant in ("/hints", "/hints on", "/hints off"):
        handled, _ = loop.command(variant)
        assert handled is True
        assert out[-1].endswith("Unknown command: /hints")
    assert "/hints" not in loop_module.COMMAND_LOOKUP
    assert "/hints" not in loop_module.CommandLoop.COMMANDS

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

    runner.run([call("Edit", ["new.txt", "", [{"op": "create", "content": content}]])])

    assert outputs[0].startswith("  Edit  new.txt\n    ├ preview")
    assert "+line 49" in outputs[0]
    assert "preview truncated" not in outputs[0]
    assert any("[approved]" in output for output in outputs)
