"""status bar (split from tests/test_ui_render.py)."""
import itertools
import os
import shutil
import sys
import time
import pytest
from prompt_toolkit.formatted_text import to_formatted_text
from prompt_toolkit.utils import get_cwidth
from rich.console import Console
from tui_harness import loop, session
import minacode.render as render_module
from minacode.base import (
    SELECTION_BACK,
    SELECTION_FREE_TEXT,
    LogBlock,
    LogEdge,
    LogLine,
    LogRole,
    Text,
)
from minacode.config import (
    request_budget_for,
)
from minacode.render import BashLivePreview, StatusBar, Theme, UiPrinter
from minacode.tools import AskSpec
from minacode.tui import ASK_DONE, ASK_FREE_TEXT, TUI_MODAL_PENDING, AskViewState, ChoiceViewState, TuiApp

def test_bash_live_preview_status_shows_wait_countdown_when_deadline_set(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    preview = BashLivePreview()
    preview.active = True
    preview.started_at = 100.0

    preview.deadline = 103.0
    status = "".join(text for _style, text in preview.frame_rows()[0])
    assert " · 3s left" in status

    preview.deadline = 99.0  # 已过期的预算:剩余显示 0s,不出现负数
    status = "".join(text for _style, text in preview.frame_rows()[0])
    assert " · 0s left" in status

    preview.deadline = None  # Bash 无预算:状态行与现状一致,不带倒计时
    status = "".join(text for _style, text in preview.frame_rows()[0])
    assert "s left" not in status

def test_status_bar_clips_wide_model_name_by_display_width(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.config.provider.model = "模型" * 20

    with monkeypatch.context() as patch:
        patch.setattr(shutil, "get_terminal_size", lambda fallback=(80, 24): os.terminal_size((20, 24)))
        fragments = StatusBar(s).fragments(sweep=False, show_elapsed=False)

    assert get_cwidth("".join(text for _style, text in fragments)) < 20

def test_status_bar_idle_clip_keeps_role_colors(tmp_path, monkeypatch):
    s = session(tmp_path)
    bar = StatusBar(s)

    with monkeypatch.context() as patch:
        patch.setattr(shutil, "get_terminal_size", lambda fallback=(80, 24): os.terminal_size((30, 24)))
        fragments = bar.fragments(sweep=False, show_elapsed=False)

    # A narrow idle bar clips but keeps its per-role colors instead of collapsing the whole line
    # to one status.base tone, which read as a colorless white bar in a tmux split.
    styles = {style for style, text in fragments if text.strip()}
    assert len(styles) > 1
    assert Theme.style("status.base") in styles
    assert Theme.style("status.reason") in styles
    assert get_cwidth("".join(text for _style, text in fragments)) < 30

def test_status_bar_clip_fragments_preserves_segment_styles():
    fragments = [("#aaaaaa", "alpha "), ("#bbbbbb", "beta "), ("#cccccc", "gamma")]

    clipped = StatusBar.clip_fragments(fragments, 12)

    # The clip cuts mid-second segment; each surviving segment keeps its own style and the
    # ellipsis inherits the style of the segment it interrupted.
    assert "".join(text for _style, text in clipped) == "alpha bet..."
    assert {style for style, _ in clipped} == {"#aaaaaa", "#bbbbbb"}

def test_status_bar_clip_fragments_mirrors_clip_width_ellipsis():
    fragments = [("#aaaaaa", "hello world")]

    assert StatusBar.clip_fragments(fragments, 0) == [("", "")]
    for width in (1, 2, 3, 4, 8):
        clipped = StatusBar.clip_fragments(fragments, width)
        assert "".join(text for _style, text in clipped) == Text.clip_width("hello world", width)
        assert get_cwidth("".join(text for _style, text in clipped)) <= width

def test_status_bar_sweep_shares_styles_between_neighbouring_cells(tmp_path, monkeypatch):
    s = session(tmp_path)
    bar = StatusBar(s)
    text = "dashscope/qwen3.7-plus | high | ctx 23% · cache 98% | index | step 160/200"
    now = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])

    runs = []
    seen = set()
    for frame in range(120):  # four seconds of frames
        now[0] = 1000.0 + frame / 30
        styles = [style for style, _text in bar.sweep_fragments(text)]
        assert len(styles) == len(text)
        seen.update(styles)
        runs.append(1 + sum(1 for left, right in itertools.pairwise(styles) if left != right))

    # A colour per cell costs an escape sequence per column on every frame, and mints a style string
    # that prompt-toolkit's renderer caches for the life of the process. Quantized, neighbours share
    # a style, so the runs collapse and the set of strings stays bounded however long a turn runs.
    assert max(runs) < len(text) / 2
    assert len(seen) <= bar.SWEEP_BANDS * bar.SWEEP_LEVELS

def test_status_bar_sweep_crest_travels_and_stays_within_the_palette(tmp_path, monkeypatch):
    s = session(tmp_path)
    bar = StatusBar(s)
    text = "x" * 80
    now = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])

    def crest_at(offset: float) -> int:
        now[0] = 1000.0 + offset
        styles = [style for style, _text in bar.sweep_fragments(text)]
        crest = Theme.style("status.sweep.crest")
        return min(range(len(styles)), key=lambda index: sum(abs(a - b) for a, b in zip(Theme.rgb(styles[index]), Theme.rgb(crest), strict=True)))

    # The crest crosses the line once per cycle and drifts by a cell or so per frame, which is what
    # keeps the band reading as a travelling light rather than a blink.
    positions = [crest_at(frame / 30) for frame in range(10)]
    assert positions == sorted(positions)
    assert 0 < positions[-1] - positions[0] <= 30

    quarter = crest_at(0.25 / bar.SWEEP_CYCLES_PER_SEC)
    assert abs(quarter - len(text) // 4) <= 2

def test_status_bar_does_not_treat_long_model_calls_as_pressure(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.config.provider.timeout = 120
    s.state.current_model_call_started_at = 1.0
    bar = StatusBar(s)
    now = [1.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])

    initial = bar.sweep_fragments("status")
    now[0] = 121.0  # Same sweep phase after a full configured timeout.

    assert bar.sweep_fragments("status") == initial
    assert all("resend" not in text for text, _role in bar.entries(show_elapsed=True))

def test_status_bar_shows_last_request_cache_hit_ratio(tmp_path):
    s = session(tmp_path)
    bar = StatusBar(s)

    def ctx_text() -> str:
        return next(text for text, role in bar.entries(show_elapsed=False) if role == "ctx")

    # No requests yet: the ctx segment carries no cache suffix.
    assert "cache" not in ctx_text()

    s.usage.last_prompt_tokens = 1000
    s.usage.last_cached_prompt_tokens = 870
    assert ctx_text().endswith("· cache 87%")
    # Rendering exercises the merged ctx/cache segment end-to-end.
    rendered = bar.fragments(sweep=False, show_elapsed=False)
    assert any("cache 87%" in text for _style, text in rendered)

    s.usage.last_cached_prompt_tokens = 0
    assert ctx_text().endswith("· cache 0%")

def test_status_bar_ctx_percent_uses_last_real_tokens_when_available(tmp_path):
    s = session(tmp_path)
    s.state.context_percent = 7  # estimate would claim 7%
    s.usage.last_prompt_tokens = 20_000  # provider reported 20K for the last request
    s.usage.last_prompt_budget = 80_000  # the budget that request was prepared against
    bar = StatusBar(s)

    ctx_text = next(text for text, role in bar.entries(show_elapsed=False) if role == "ctx")

    assert "ctx 25%" in ctx_text
    assert "ctx 7%" not in ctx_text

def test_status_bar_ctx_percent_keeps_the_request_time_budget(tmp_path):
    """Changing max_tokens or max_context_tokens after the request must not move the recorded fill:
    the denominator is the budget the last request was prepared against, not today's configuration."""
    s = session(tmp_path)
    s.settings.max_context_tokens = 100_000
    s.usage.last_prompt_tokens = 40_000
    s.usage.last_prompt_budget = request_budget_for(100_000, 10_000)  # 40K of an 85.9K budget
    bar = StatusBar(s)

    def ctx_percent() -> int:
        text = next(t for t, role in bar.entries(show_elapsed=False) if role == "ctx")
        return int(text.split("%")[0].split(" ")[1])

    recorded = 40_000 * 100 // request_budget_for(100_000, 10_000)
    assert ctx_percent() == recorded

    s.config.provider.max_tokens = 60_000  # today's budget would read as ~111% -> 100%
    assert ctx_percent() == recorded

def test_status_bar_ctx_percent_falls_back_to_estimate_without_requests(tmp_path):
    s = session(tmp_path)
    s.state.context_percent = 23
    bar = StatusBar(s)

    ctx_text = next(text for text, role in bar.entries(show_elapsed=False) if role == "ctx")

    assert f"ctx {s.state.context_percent}%" in ctx_text
    assert "cache" not in ctx_text

def test_status_bar_ctx_percent_falls_back_when_the_recorded_budget_is_missing(tmp_path):
    """A session resumed from a snapshot taken before last_prompt_budget existed has tokens but no
    budget; the estimate is the honest fallback rather than a division by zero."""
    s = session(tmp_path)
    s.state.context_percent = 31
    s.usage.last_prompt_tokens = 20_000
    s.usage.last_prompt_budget = 0
    bar = StatusBar(s)

    ctx_text = next(text for text, role in bar.entries(show_elapsed=False) if role == "ctx")

    assert f"ctx {s.state.context_percent}%" in ctx_text

def test_status_bar_shows_step_only_near_max_steps(tmp_path):
    s = session(tmp_path)
    bar = StatusBar(s)
    s.settings.max_steps = 200

    s.state.turn_step = 1
    assert all(not text.startswith("step ") for text, _role in bar.entries(show_elapsed=True))

    s.state.turn_step = 160
    assert ("step 160/200", "warn") in bar.entries(show_elapsed=True)

def test_status_clear_erases_rendered_line(tmp_path, recording_output):
    status = StatusBar(session(tmp_path))
    status.output = recording_output
    status.rendered = True

    status.clear()

    assert recording_output.events == [("write", "\r"), ("erase", ""), ("flush", "")]
    assert not status.rendered

def test_clip_width_returns_unchanged_text_when_within_width():
    assert Text.clip_width("hello", 10) == "hello"
    assert Text.clip_width("", 5) == ""
    assert Text.clip_width("hello", 5) == "hello"

def test_clip_width_clips_wide_text_with_ellipsis():
    assert Text.clip_width("hello world", 8) == "hello..."
    # When width is less than 3, the ellipsis shrinks to fit
    assert Text.clip_width("hello world", 1) == "."
    assert Text.clip_width("hello world", 2) == ".."
    assert Text.clip_width("hello world", 3) == "..."
    assert Text.clip_width("hello world", 4) == "h..."

def test_clip_width_clamps_negative_width_to_zero():
    assert Text.clip_width("hello", -1) == ""

def test_clip_width_handles_cjk_wide_characters():
    assert Text.clip_width("你好世界", 5) == "你..."
    assert Text.clip_width("a你好", 5) == "a你好"
