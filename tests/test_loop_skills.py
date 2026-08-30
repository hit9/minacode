"""loop skills (split from tests/test_loop_commands.py)."""

import os
import tomllib

import pytest
from agent_harness import session
from test_loop_commands import _write_skill

from wizolt.base import (
    ToolError,
    TurnBox,
)
from wizolt.cli import CommandLoop
from wizolt.cli.commands import (
    skills_command,
    status,
)
from wizolt.context import ContextManager
from wizolt.engine import Agent
from wizolt.prompts import SYSTEM_PROMPT
from wizolt.render import StatusBar
from wizolt.session import Session
from wizolt.skill import SkillLibrary
from wizolt.tools import SkillTool, Tool


def test_skill_library_index_and_lookup(tmp_path):
    _write_skill(tmp_path, "release-notes", "Draft a CHANGELOG entry.", "Do the thing.")
    s = session(tmp_path)

    index = s.skills.index()
    assert index.startswith("--- SKILLS ---")
    assert "- release-notes: Draft a CHANGELOG entry." in index
    assert s.skills.get("Release-Notes").name == "release-notes"  # case-insensitive
    assert s.skills.get("missing") is None


def test_builtin_wizolt_help_uses_normal_skill_paths(tmp_path):
    s = session(tmp_path)

    skill = s.skills.get("wizolt-help")
    assert skill is not None
    assert skill.source == "builtin"
    assert "troubleshoot wizolt" in skill.description
    assert "- wizolt-help:" in s.skills.index()
    body = SkillTool(s, ["wizolt-help"]).call()
    assert "## Inspect the implementation" in body
    assert "### Provider-side tools and web search" in body
    assert all(term in body for term in ("builtin_tools", "$web_search", "pause_turn", "OpenRouter"))
    assert "## Configure providers" in s.skills.resolve_mentions("help with $wizolt-help")


def test_project_skills_prefer_wizolt_and_fall_back_to_minacode(tmp_path):
    legacy = tmp_path / ".minacode" / "skills" / "guide"
    legacy.mkdir(parents=True)
    (legacy / "SKILL.md").write_text("---\nname: guide\ndescription: legacy\n---\nlegacy body\n", encoding="utf-8")

    skill = session(tmp_path).skills.get("guide")
    assert skill is not None
    assert skill.description == "legacy"

    _write_skill(tmp_path, "guide", "current", "current body")
    skill = session(tmp_path).skills.get("guide")
    assert skill is not None
    assert skill.description == "current"


def test_every_builtin_skill_is_declared_as_package_data(tmp_path):
    """A builtin skill only exists for installed users if the wheel carries its SKILL.md.

    Running from a checkout hides an omission completely, so the packaging declaration is checked
    here rather than discovered as a missing skill after release."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    builtin_root = os.path.join(repo_root, "wizolt", "builtin_skills")
    with open(os.path.join(repo_root, "pyproject.toml"), "rb") as handle:
        packaging = tomllib.load(handle)["tool"]["setuptools"]
    patterns = packaging["package-data"]["wizolt"]

    assert "wizolt.builtin_skills" in packaging["packages"]
    assert "builtin_skills/*/SKILL.md" in patterns
    for entry in sorted(os.listdir(builtin_root)):
        if os.path.isdir(os.path.join(builtin_root, entry)) and entry != "__pycache__":
            assert os.path.isfile(os.path.join(builtin_root, entry, "SKILL.md")), entry


def test_skill_project_overrides_user_and_user_overrides_builtin(tmp_path):
    user_skill = tmp_path / "data" / "skills" / "wizolt-help"
    user_skill.mkdir(parents=True)
    (user_skill / "SKILL.md").write_text("---\nname: wizolt-help\ndescription: user version\n---\nuser body\n", encoding="utf-8")

    user_session = session(tmp_path)
    skill = user_session.skills.get("wizolt-help")
    assert skill.source == "user"
    assert skill.description == "user version"

    _write_skill(tmp_path, "wizolt-help", "project version", "project body")

    project_session = session(tmp_path)
    skill = project_session.skills.get("wizolt-help")
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
    assert "| `wizolt-help` | builtin |" in skills_command(base, "")

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
    cfg = tmp_path / "wizolt.toml"
    cfg.write_text('[runtime]\ntheme = "light"\n')
    s = Session.from_config_file(path=str(cfg), theme="dark")
    assert s.settings.theme == "dark"

    s2 = Session.from_config_file(path=str(cfg))
    assert s2.settings.theme == "light"

    s3 = Session.from_config_file(path=str(cfg), theme="")
    assert s3.settings.theme == "light"
