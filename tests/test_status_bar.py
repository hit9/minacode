"""status bar (split from tests/test_ui_render.py)."""

import os
import shutil
import sys
import time

from prompt_toolkit.utils import get_cwidth
from tui_harness import session

from wizolt.base import (
    Text,
)
from wizolt.config import (
    request_budget_for,
)
from wizolt.render import BashLivePreview, StatusBar, Theme


def test_bash_live_preview_status_shows_wait_countdown_when_deadline_set(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    preview = BashLivePreview()
    preview.active = True
    preview.started_at = 100.0

    preview.deadline = 103.0
    status = "".join(text for _, text in preview.frame_rows()[0])
    assert " · 3s left" in status

    preview.deadline = 99.0  # 已过期的预算:剩余显示 0s,不出现负数
    status = "".join(text for _, text in preview.frame_rows()[0])
    assert " · 0s left" in status

    preview.deadline = None  # Bash 无预算:状态行与现状一致,不带倒计时
    status = "".join(text for _, text in preview.frame_rows()[0])
    assert "s left" not in status


def status_text(bar: StatusBar) -> str:
    return "".join(text for _, text in bar.fragments())


def test_status_bar_has_fixed_order_and_no_working_only_fields(tmp_path):
    s = session(tmp_path)
    s.settings.yolo = True
    s.config.active_provider = "default"
    s.config.provider.model = "vendor/model"
    s.config.provider.reasoning = "high"
    s.state.context_percent = 23
    s.state.code_index_status = "stale"
    s.state.turn_step = s.settings.max_steps
    s.update.latest = "99.0.0"

    text = status_text(StatusBar(s))

    assert text == f"[yolo] default/model · high | mcp 0 · skills {len(s.skills.skills)} | ctx 23% · cache 0% | index*"
    assert all(word not in text for word in ("worker", "compaction", "jobs", "update", "step", "retry", "attempt"))


def test_status_bar_keeps_semantic_colors(tmp_path):
    s = session(tmp_path)
    s.settings.yolo = True
    s.config.provider.model = "vendor/model"
    fragments = StatusBar(s).fragments()
    by_text = {text: style for style, text in fragments}

    assert by_text["[yolo] "] == Theme.fg("status_yolo")
    assert by_text["[yolo] "] != Theme.fg("status_base")
    assert by_text["default/model"] == Theme.fg("status_provider")
    assert by_text[s.config.provider.reasoning] == Theme.fg("status_reason")
    assert by_text["mcp 0"] == Theme.fg("status_mcp")
    assert by_text["ctx 0%"] == Theme.fg("status_context")
    assert by_text["index"] == Theme.fg("status_index")


def test_status_bar_clips_wide_model_name_by_display_width(tmp_path, monkeypatch):
    s = session(tmp_path)
    s.config.provider.model = "模型" * 20
    with monkeypatch.context() as patch:
        patch.setattr(shutil, "get_terminal_size", lambda fallback=(80, 24): os.terminal_size((20, 24)))
        fragments = StatusBar(s).fragments()
    assert get_cwidth("".join(text for _, text in fragments)) < 20


def test_status_bar_clip_keeps_role_colors(tmp_path, monkeypatch):
    s = session(tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr(shutil, "get_terminal_size", lambda fallback=(80, 24): os.terminal_size((30, 24)))
        fragments = StatusBar(s).fragments()
    styles = {style for style, text in fragments if text.strip()}
    assert len(styles) > 1
    assert Theme.fg("status_provider") in styles
    assert Theme.fg("status_reason") in styles
    assert get_cwidth("".join(text for _, text in fragments)) < 30


def test_status_bar_clip_fragments_preserves_segment_styles():
    fragments = [("#aaaaaa", "alpha "), ("#bbbbbb", "beta "), ("#cccccc", "gamma")]
    clipped = StatusBar.clip_fragments(fragments, 12)
    assert "".join(text for _, text in clipped) == "alpha bet..."
    assert {style for style, _ in clipped} == {"#aaaaaa", "#bbbbbb"}


def test_status_bar_clip_fragments_mirrors_clip_width_ellipsis():
    fragments = [("#aaaaaa", "hello world")]
    assert StatusBar.clip_fragments(fragments, 0) == [("", "")]
    for width in (1, 2, 3, 4, 8):
        clipped = StatusBar.clip_fragments(fragments, width)
        assert "".join(text for _, text in clipped) == Text.clip_width("hello world", width)
        assert get_cwidth("".join(text for _, text in clipped)) <= width


def test_status_bar_refreshes_facts_without_changing_its_shape(tmp_path):
    s = session(tmp_path)
    bar = StatusBar(s)
    assert "ctx 0% · cache 0%" in status_text(bar)
    s.usage.last_prompt_tokens = 1_000
    s.usage.last_prompt_budget = 2_000
    s.usage.last_cached_prompt_tokens = 870
    assert "ctx 50% · cache 87%" in status_text(bar)


def test_status_bar_ctx_percent_keeps_the_request_time_budget(tmp_path):
    s = session(tmp_path)
    s.usage.last_prompt_tokens = 40_000
    s.usage.last_prompt_budget = request_budget_for(100_000, 10_000)
    bar = StatusBar(s)

    recorded = 40_000 * 100 // request_budget_for(100_000, 10_000)
    assert f"ctx {recorded}%" in status_text(bar)
    s.config.provider.max_tokens = 60_000
    assert f"ctx {recorded}%" in status_text(bar)


def test_status_bar_ctx_percent_falls_back_when_the_recorded_budget_is_missing(tmp_path):
    s = session(tmp_path)
    s.state.context_percent = 31
    s.usage.last_prompt_tokens = 20_000
    s.usage.last_prompt_budget = 0
    assert "ctx 31%" in status_text(StatusBar(s))


def test_status_start_draws_once_without_a_repaint_thread(tmp_path, monkeypatch, recording_output):
    bar = StatusBar(session(tmp_path))
    bar.output = recording_output
    draws = []
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True)
    monkeypatch.setattr("wizolt.render.print_formatted_text", lambda value, **kwargs: draws.append(value))

    bar.start()
    bar.start()

    assert bar.is_running()
    assert not hasattr(bar, "thread")
    assert len(draws) == 1
    assert recording_output.events == [("write", "\r"), ("erase", "")]

    bar.stop()
    assert not bar.is_running()
    assert recording_output.events[-3:] == [("write", "\r"), ("erase", ""), ("flush", "")]


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
