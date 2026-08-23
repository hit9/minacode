"""context environment (split from tests/test_context.py)."""
import platform
import shutil
from dataclasses import replace

import pytest
from agent_harness import session

from minacode.base import (
    MAX_AGENTS_MD_TOKENS,
)
from minacode.context import ContextManager
from minacode.prompts import (
    SYSTEM_PROMPT,
)
from minacode.session import Session
from minacode.skill import SkillLibrary


def test_model_messages_are_ordered_context_messages(tmp_path):
    s = session(tmp_path)
    s.skills = SkillLibrary({})  # no skills: assert the base frame ordering
    s.messages.extend([{"role": "user", "content": "old request"}, {"role": "assistant", "content": "old answer"}])
    turn = [
        {"role": "user", "content": "current request"},
        {"role": "user", "content": "extra one"},
        {"role": "user", "content": "extra two"},
    ]
    messages = ContextManager(s).model_messages(" system ", turn)

    assert [message["role"] for message in messages] == ["system", "user", "user", "assistant", "user", "user", "user"]
    assert messages[0]["content"] == "system"
    assert messages[1]["content"].startswith("--- Environment ---")
    assert "- cwd: " + str(tmp_path) in messages[1]["content"]
    assert [message["content"] for message in messages[2:4]] == ["old request", "old answer"]
    assert f"- session_started_at: {s.created_at}" in messages[1]["content"]
    assert [message["content"] for message in messages[4:]] == ["current request", "extra one", "extra two"]
    assert [message["content"] for message in turn] == ["current request", "extra one", "extra two"]
    assert not any("FILE STATE" in message["content"] for message in messages)

def test_language_auto_injects_nothing_byte_identical(tmp_path):
    s = session(tmp_path)
    s.skills = SkillLibrary({})  # no skills: assert the base frame
    turn = [{"role": "user", "content": "request"}]
    messages = ContextManager(s).model_messages(SYSTEM_PROMPT, turn)

    assert messages[0]["content"] == SYSTEM_PROMPT.strip()
    assert messages[1]["content"].startswith("--- Environment ---")
    assert [message["role"] for message in messages] == ["system", "user", "user"]

def test_language_directive_appends_stable_block_to_system_tail(tmp_path):
    s = session(tmp_path)
    turn = [{"role": "user", "content": "request"}]
    context = ContextManager(s)
    auto_messages = context.model_messages(SYSTEM_PROMPT, turn)

    s.settings.language = "Chinese"
    forced_messages = context.model_messages(SYSTEM_PROMPT, turn)

    assert forced_messages[0]["content"].startswith(SYSTEM_PROMPT.strip())
    assert forced_messages[0]["content"].endswith("commands verbatim.")
    assert "LANGUAGE OVERRIDE:" in forced_messages[0]["content"]
    assert "Chinese" in forced_messages[0]["content"]
    assert forced_messages[0]["content"].count("LANGUAGE OVERRIDE:") == 1
    # only the system tail changes: everything after it is byte-identical to the auto request
    assert forced_messages[1:] == auto_messages[1:]
    # the block is a pure function of the value: repeated projections are identical
    assert context.model_messages(SYSTEM_PROMPT, turn) == forced_messages

def test_environment_uses_cached_system_info(tmp_path, monkeypatch):
    calls = []

    def fake_which(name):
        calls.append(name)
        return "/bin/" + name if name in {"bash", "rg", "sed"} else None

    monkeypatch.setattr(platform, "system", lambda: "TestOS")
    monkeypatch.setattr(platform, "machine", lambda: "test-arch")
    monkeypatch.setattr(shutil, "which", fake_which)

    s = session(tmp_path)
    initial_calls = list(calls)
    context = ContextManager(s)
    first = context.environment()
    second = context.model_messages("sys", [{"role": "user", "content": "request"}])[1]["content"]

    assert calls == initial_calls
    assert "- cwd: " + str(tmp_path) in first
    assert "- os: TestOS" in first
    assert "- arch: test-arch" in first
    assert "- detected_commands (available via Bash): bash, rg, sed" in first
    assert "- detected_commands (available via Bash): bash, rg, sed" in second

def test_environment_injects_agents_md(tmp_path):
    (tmp_path / "AGENTS.md").write_text("# Rules\nAlways run pytest.\n", encoding="utf-8")
    s = session(tmp_path)
    context = ContextManager(s)
    env = context.environment()
    assert "--- Project instructions (AGENTS.md) ---" in env
    assert "# Rules\nAlways run pytest." in env
    messages = context.model_messages(SYSTEM_PROMPT, [{"role": "user", "content": "request"}])
    assert messages[1]["content"].startswith("--- Environment ---")
    assert messages[1]["content"] == "--- Environment ---\n" + env
    assert "--- Project instructions (AGENTS.md) ---" in messages[1]["content"]
    # still the same Environment user message: the injected section sits after the header
    assert messages[1]["content"].index("--- Project instructions (AGENTS.md) ---") > messages[1]["content"].index("--- Environment ---")
    assert messages[1]["content"].count("--- Project instructions") == 1

def test_environment_falls_back_to_claude_md(tmp_path):
    (tmp_path / "CLAUDE.md").write_text("# Claude rules\nAlways run pytest.\n", encoding="utf-8")
    s = session(tmp_path)
    env = ContextManager(s).environment()
    assert "--- Project instructions (CLAUDE.md) ---" in env
    assert "# Claude rules\nAlways run pytest." in env

def test_environment_agents_md_precedence(tmp_path):
    (tmp_path / "AGENTS.md").write_text("agents content\n", encoding="utf-8")
    (tmp_path / "CLAUDE.md").write_text("claude content\n", encoding="utf-8")
    s = session(tmp_path)
    env = ContextManager(s).environment()
    assert "--- Project instructions (AGENTS.md) ---" in env
    assert "agents content" in env
    assert "claude content" not in env

def test_environment_agents_md_disabled(tmp_path):
    (tmp_path / "AGENTS.md").write_text("# Rules\nAlways run pytest.\n", encoding="utf-8")
    s = session(tmp_path)
    s.settings.agents_md = False
    env = ContextManager(s).environment()
    assert "Project instructions" not in env
    assert "# Rules" not in env

def test_environment_agents_md_bounded(tmp_path):
    (tmp_path / "AGENTS.md").write_text("\n".join(f"line {i}" for i in range(20000)), encoding="utf-8")
    s = session(tmp_path)
    context = ContextManager(s)
    env = context.environment()
    assert "truncated to fit the prefix" in env
    injected = env.split("--- Project instructions (AGENTS.md) ---", 1)[1].lstrip("\n")
    assert context.estimated_text_tokens(injected) <= MAX_AGENTS_MD_TOKENS

@pytest.mark.parametrize(
    ("label", "text"),
    [
        ("line oriented", "\n".join(f"rule {index} " + "x" * 60 for index in range(6000))),
        ("one very long line", "x" * 200_000),  # excerpts cannot snap to a line boundary here
        ("wide characters", "中文规则一二三四五六七八九十" * 4000),
        ("barely over the cap", "z" * (MAX_AGENTS_MD_TOKENS * 4 + 10)),
    ],
)
def test_environment_agents_md_bounding_spends_the_budget_it_is_given(tmp_path, label, text):
    """Bounding has a cap to respect and a budget to spend. Reserving the marker up front instead of
    shrinking until it fits keeps a large file near its whole allowance -- overshooting the shrink
    used to leave a quarter of the cap unused, which is a quarter of the project's instructions."""
    (tmp_path / "AGENTS.md").write_text(text, encoding="utf-8")
    context = ContextManager(session(tmp_path))
    injected = context.environment().split("--- Project instructions (AGENTS.md) ---", 1)[1].lstrip("\n")
    tokens = context.estimated_text_tokens(injected)

    assert "truncated to fit the prefix" in injected
    assert tokens <= MAX_AGENTS_MD_TOKENS, f"{label} exceeded the cap"
    assert tokens >= MAX_AGENTS_MD_TOKENS * 0.95, f"{label} wasted the cap"

def test_environment_agents_md_absent(tmp_path):
    s = session(tmp_path)
    info = s.system_info
    assert info is not None
    baseline = "\n".join(
        [
            f"- cwd: {info.cwd}",
            f"- session_started_at: {s.created_at}",
            "- detected_commands (available via Bash): " + (", ".join(info.commands) or "(none)"),
            f"- os: {info.os}",
            f"- arch: {info.arch}",
            f"- shell_timeout: {s.settings.shell_timeout}s",
        ]
    )
    env = ContextManager(s).environment()
    assert "Project instructions" not in env
    assert env == baseline  # byte-identical to the pre-injection Environment, no extra blank rows

def test_environment_agents_md_cache_stable(tmp_path):
    (tmp_path / "AGENTS.md").write_text("# Rules\nAlways run pytest.\n", encoding="utf-8")
    s = session(tmp_path)
    context = ContextManager(s)
    turn = [{"role": "user", "content": "request"}]
    assert context.model_messages(SYSTEM_PROMPT, turn) == context.model_messages(SYSTEM_PROMPT, turn)

def test_environment_agents_md_cache_stable_across_worker(tmp_path):
    """A worker inherits the parent's shared system_info and settings, so its Environment is
    byte-identical — the spawn invariant DelegateTool._spawn_worker relies on (cache-critical)."""
    (tmp_path / "AGENTS.md").write_text("# Rules\nAlways run pytest.\n", encoding="utf-8")
    parent = session(tmp_path)
    worker = Session(
        cwd=parent.cwd,
        system_info=parent.system_info,  # shared: skip a SystemInfo.detect
        settings=replace(parent.settings),
        created_at=parent.created_at,  # the Environment layer includes session_started_at
    )
    assert ContextManager(worker).environment() == ContextManager(parent).environment()
