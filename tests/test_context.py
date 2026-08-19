"""Context projection and compaction: what a request carries, what compaction keeps, and the
history index it leaves behind."""

import os
import platform
import re
import shutil
from dataclasses import replace
from types import SimpleNamespace

import pytest
from agent_harness import call, session

import minacode.context as context_module
from minacode.base import (
    MAX_AGENTS_MD_TOKENS,
    MAX_TOOL_OUTPUT_TOKENS,
    SESSION_EVENT_KEY,
    ModelError,
    Text,
)
from minacode.cli import CommandLoop
from minacode.cli.commands import compact
from minacode.config import (
    DEFAULT_OUTPUT_RESERVE_TOKENS,
    MIN_CONTEXT_SAFETY_TOKENS,
)
from minacode.context import ContextManager
from minacode.engine import Agent
from minacode.model import ModelClient
from minacode.prompts import COMPACTION_SUMMARY_TITLE, CURRENT_TURN_CONTEXT_TRIMMED, SYSTEM_PROMPT
from minacode.runner import ToolRunner
from minacode.session import HistorySegment, Session
from minacode.skill import SkillLibrary
from minacode.tools import EditTool, ReadTool


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


# --- AGENTS.md / CLAUDE.md injection (runtime.agents_md, default on) ---


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


def test_session_tool_result_store_prunes_old_records(tmp_path):
    s = session(tmp_path)
    for index in range(405):
        s.store_tool_result("Bash", [str(index)], f"output {index}")

    assert len(s.tool_results) == 400
    assert len(s.tool_records) == 400
    assert "tr.1" not in s.tool_results
    assert s.tool_records[0].key == "tr.6"
    assert "tr.405" in s.tool_results


def test_bounded_output_marks_recall_key(tmp_path):
    s = session(tmp_path)
    context = ContextManager(s)
    large = "head\n" + "\n".join(f"line {index}" for index in range(20000)) + "\ntail\n"
    bounded = context.bound_output(large, "tr.large")

    assert "head" in bounded
    assert "tail" in bounded
    assert "<bounded_output" in bounded
    assert 'recall="tr.large"' in bounded


def test_bounded_output_keeps_head_and_tail_of_a_single_line_payload(tmp_path):
    """The reported failure: an MCP server returns one long line of compact JSON, and snapping the
    excerpts to a line boundary threw away everything but the wrapper tags -- the model saw a marker
    claiming ~9.5k omitted tokens out of ~9.5k, with no content on either side of it."""
    s = session(tmp_path)
    context = ContextManager(s)
    payload = '{"id": 1, "system_prompt": "' + "x" * 40000 + '", "tail_field": "last"}'
    large = f'<MCPCall server="orion" tool="get_agent">\n{payload}\n</MCPCall>'

    bounded = context.bound_output(large, "tr.1")

    assert '"system_prompt"' in bounded  # real head, not just the opening tag
    assert '"tail_field": "last"' in bounded  # real tail, not just the closing tag
    estimated = context.estimated_text_tokens(large)
    omitted = int(re.findall(r'omitted_tokens="(\d+)"', bounded)[0])
    # Roughly the whole budget is spent on content, rather than snapped away with it.
    assert omitted <= estimated - MAX_TOOL_OUTPUT_TOKENS // 2


def test_bounded_output_still_snaps_excerpts_to_line_boundaries(tmp_path):
    """Line-oriented output keeps whole lines: snapping is only abandoned when it would cost more
    than half the budget, so ordinary logs never gain a half-line at the cut."""
    s = session(tmp_path)
    context = ContextManager(s)
    large = "\n".join(f"line {index} " + "y" * 40 for index in range(20000))

    bounded = context.bound_output(large, "tr.1")
    head, _, rest = bounded.partition("<bounded_output")
    _, _, tail = rest.partition("/>")

    assert all(line in large.split("\n") for line in head.strip().split("\n"))
    assert all(line in large.split("\n") for line in tail.strip().split("\n"))


def test_bounded_output_materializes_full_output_to_asset_file(tmp_path):
    s = session(tmp_path)
    context = ContextManager(s)
    large = "head\n" + "\n".join(f"line {index}" for index in range(20000)) + "\ntail\n"
    bounded = context.bound_output(large, "tr.1")

    path = os.path.join(s.images.assets_dir(), "tr.1.txt")
    assert 'recall="tr.1"' in bounded
    assert f'file="{path}"' in bounded
    with open(path, encoding="utf-8") as file:
        assert file.read() == large


def test_bounded_output_marker_names_the_cheaper_way_to_read_the_rest(tmp_path):
    """`recall` and `file` say where the omitted middle went, not what to do about it, and the cheap
    move is the non-obvious one. The marker points at whichever one the result actually has."""
    s = session(tmp_path)
    context = ContextManager(s)
    large = "head\n" + "\n".join(f"line {index}" for index in range(20000)) + "\ntail\n"

    bounded = context.bound_output(large, "tr.1")
    assert f'hint="{ContextManager.OMITTED_OUTPUT_HINT}"' in bounded

    # No file to point at: the marker falls back to naming the Recall form that pages.
    (tmp_path / "blocked.txt").write_text("x", encoding="utf-8")
    s.images.assets_dir = lambda: str(tmp_path / "blocked.txt" / "sub")
    fileless = context.bound_output(large, "tr.2")
    assert 'file="' not in fileless
    assert f'hint="{ContextManager.OMITTED_OUTPUT_RECALL_HINT}"' in fileless

    # The compaction summary is bounded with no key at all: nothing to recall, nothing to advise.
    assert "hint=" not in context.bound_output(large, "")


def test_bounded_output_small_or_keyless_never_writes_asset_file(tmp_path):
    s = session(tmp_path)
    context = ContextManager(s)

    small = context.bound_output("small output", "tr.1")
    assert 'file="' not in small

    large = "head\n" + "\n".join(f"line {index}" for index in range(20000)) + "\ntail\n"
    keyless = context.bound_output(large, "")
    assert 'recall="' not in keyless
    assert 'file="' not in keyless


def test_bounded_output_survives_asset_write_failure(tmp_path, monkeypatch):
    s = session(tmp_path)
    context = ContextManager(s)
    large = "head\n" + "\n".join(f"line {index}" for index in range(20000)) + "\ntail\n"
    # A regular file where the assets dir would go: makedirs raises OSError.
    (tmp_path / "blocked.txt").write_text("x", encoding="utf-8")
    monkeypatch.setattr(s.images, "assets_dir", lambda: str(tmp_path / "blocked.txt" / "sub"))

    bounded = context.bound_output(large, "tr.1")
    assert 'recall="tr.1"' in bounded
    assert 'file="' not in bounded


def test_read_tool_message_inlines_bounded_output(tmp_path):
    path = tmp_path / "large.txt"
    path.write_text("first\n" + "\n".join(f"middle-{index}" for index in range(20000)) + "\nlast\n", encoding="utf-8")
    s = session(tmp_path)
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)
    call_obj = call("Read", [{"path": "large.txt", "ranges": [[1, 0]]}])
    output = ReadTool(s, call_obj.args).call()
    key = s.store_tool_result("Read", call_obj.args, output)

    message = runner.tool_message(call_obj, key, output)

    # A whole-file read prints no range: 1:0 is the "to the end of the file" sentinel, and
    # echoing it raw would show the model an empty-looking range it never wrote.
    assert message.startswith("tool tr.1 Read large.txt\noutput:\n")
    assert "<Read" in message
    assert "<bounded_output" in message
    assert 'recall="tr.1"' in message
    assert "-> FILE STATE" not in message


def test_tool_error_records_keep_recent_failures(tmp_path):
    s = session(tmp_path)
    for index in range(7):
        s.record_tool_error(f"tr.{index}", "Bash", [str(index)], f"error {index}")

    assert [record.key for record in s.tool_errors] == ["tr.2", "tr.3", "tr.4", "tr.5", "tr.6"]


def test_working_context_does_not_repeat_durable_tool_errors(tmp_path):
    s = session(tmp_path)
    for index in range(6):
        s.record_tool_error(f"tr.{index}", "Bash", [f"cmd {index}"], f"error {index}")

    context = "\n".join(str(message.get("content") or "") for message in ContextManager(s).model_messages("sys"))

    assert "Recent tool errors:" not in context
    assert "error 5" not in context


def test_provider_context_limit_overrides_the_runtime_default(tmp_path):
    # Provider entries are effectively per-model, so a 1M-window model should not have to share one
    # global number with a 128K one. Unset (0) inherits, and the budget is resolved per call so
    # switching the active entry moves it.
    from minacode.config import Config, request_budget_for

    config = Config.from_dict(
        {
            "provider": {
                "active": "big",
                "big": {"model": "wide", "max_context_tokens": 1_048_576},
                "small": {"model": "narrow", "max_context_tokens": 131_072},
                "plain": {"model": "default"},
            },
            "runtime": {"max_context_tokens": 262_144},
        }
    )
    runtime_default = 262_144
    assert config.providers["big"].context_token_limit(runtime_default) == 1_048_576
    assert config.providers["small"].context_token_limit(runtime_default) == 131_072
    assert config.providers["plain"].context_token_limit(runtime_default) == runtime_default  # 0 inherits

    s = session(tmp_path)
    s.config = config
    s.settings.max_context_tokens = runtime_default
    context = ContextManager(s)
    wide = context.request_token_budget()

    s.config.active_provider = "small"  # what /provider does
    narrow = context.request_token_budget()

    assert wide > narrow, "the budget must follow the active provider entry"
    assert wide == request_budget_for(1_048_576, config.providers["big"].output_token_budget())
    assert narrow == request_budget_for(131_072, config.providers["small"].output_token_budget())


def test_the_context_budget_has_exactly_one_definition():
    """Every consumer must derive the budget from Session.request_token_budget().

    This is the regression guard, not a style rule. The budget is read by compaction, the usage
    recorder, and two renderers; each one that computes it itself is a place the provider override
    can silently stop applying, which is exactly how `runtime.max_context_tokens` used to leak back
    in. `request_budget_for` is pure and takes a plain int, so a new call site that passes
    `settings.max_context_tokens` type-checks, passes its own tests, and quietly ignores the entry's
    limit. Keep the call sites at one; route new readers through the session.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent / "minacode"
    callers = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*.py")
        if "request_budget_for(" in path.read_text(encoding="utf-8") and path.name != "config.py"  # config.py defines it
    )
    assert callers == ["session/__init__.py"], (
        f"the context budget is computed in {callers}; call Session.request_token_budget() instead so the "
        "per-provider max_context_tokens override cannot be bypassed"
    )


def test_provider_context_limit_reaches_every_budget_reader(tmp_path):
    # End to end, through the real Agent: a provider-level limit must drive the compaction budget and
    # the budget recorded for the status row, with no path still reading the runtime default.
    from minacode.config import Config, request_budget_for

    s = session(tmp_path)
    s.config = Config.from_dict({"provider": {"active": "wide", "wide": {"model": "m", "max_context_tokens": 1_048_576}}})
    s.settings.max_context_tokens = 262_144  # the runtime default, deliberately much smaller
    expected = request_budget_for(1_048_576, s.config.provider.output_token_budget())
    runtime_only = request_budget_for(262_144, s.config.provider.output_token_budget())
    assert expected != runtime_only

    agent = Agent(s, output_fn=lambda text: None)
    assert s.request_token_budget() == expected
    assert agent.context.request_token_budget() == expected

    # The recorded budget is what /status and the status bar divide by, so it has to agree.
    type(agent.model)._record_usage(agent.model, SimpleNamespace(prompt_tokens=1_000, completion_tokens=1))
    assert s.usage.last_prompt_budget == expected

    # And the fill really is measured against the wide window, not the runtime default.
    assert s.usage.last_prompt_tokens * 100 // s.usage.last_prompt_budget == 1_000 * 100 // expected


def test_provider_context_limit_shares_one_denominator_with_usage(tmp_path):
    # The compaction trigger and the status-bar fill must be measured against the same number, or
    # the bar reads 60% while the context manager is already compacting.
    from types import SimpleNamespace

    from minacode.config import Config, request_budget_for
    from minacode.model import ModelClient

    s = session(tmp_path)
    s.config = Config.from_dict({"provider": {"active": "big", "big": {"model": "wide", "max_context_tokens": 1_048_576}}})
    s.settings.max_context_tokens = 262_144
    ModelClient._record_usage(SimpleNamespace(session=s), SimpleNamespace(prompt_tokens=10, completion_tokens=1))

    assert s.usage.last_prompt_budget == ContextManager(s).request_token_budget()
    assert s.usage.last_prompt_budget > request_budget_for(262_144, 16_384), "the runtime default was used, not the entry's"


def test_compaction_uses_configured_context_budget(tmp_path):
    s = session(tmp_path)
    s.settings.max_context_tokens = 1
    s.messages = [
        {"role": "user", "content": "old user"},
        {"role": "assistant", "content": "old answer"},
        *({"role": "assistant", "content": f"recent {index}"} for index in range(8)),
        {"role": "user", "content": "latest"},
        {"role": "tool", "content": "tool kept"},
    ]
    context = ContextManager(s)
    compaction_phases = []
    context.on_compaction = lambda active, _error: compaction_phases.append(active)

    class FakeModel:
        def __init__(self):
            self.input = None

        def compact(self, text, *_args, **_kwargs):
            self.input = text
            return {"summary": "compact summary", "plan": ["next"], "known": ["fact"]}

    model = FakeModel()
    context.prepare_messages(model, "system", [{"role": "user", "content": "request"}])
    assert compaction_phases == [True, False]
    assert model.input is not None
    assert "Older Messages:" in model.input
    assert "old answer" in model.input
    assert "Recent Messages (rewrite briefly inside summary):" in model.input
    assert "recent 7" in model.input
    assert "latest" not in model.input
    assert "request" not in model.input
    assert s.state.summary == "compact summary"
    assert [vars(item) for item in s.state.plan] == [{"status": "todo", "text": "next"}]
    assert s.state.known == ["fact"]
    assert [message["role"] for message in s.messages] == ["user", "user", "tool"]
    assert s.messages[0]["content"].startswith(COMPACTION_SUMMARY_TITLE)
    assert "compact summary" in s.messages[0]["content"]
    assert s.messages[1]["content"] == "latest"
    assert s.messages[2]["content"] == "tool kept"
    assert all("recent 7" not in str(message.get("content") or "") for message in s.messages)


def test_default_budget_leaves_more_input_room_than_the_previous_240k_ceiling(tmp_path):
    """The output reserve trades against the input budget one for one, so the two defaults are one
    decision: doubling the output cap only pays off because the ceiling rose further."""
    s = session(tmp_path)
    context = ContextManager(s)

    assert s.settings.max_context_tokens == 256 * 1024
    assert s.config.provider.output_token_budget() == DEFAULT_OUTPUT_RESERVE_TOKENS
    assert context.request_token_budget() > 240 * 1024 - 8_192 - MIN_CONTEXT_SAFETY_TOKENS


def test_compaction_budget_reserves_output_and_safety(tmp_path):
    s = session(tmp_path)
    s.settings.max_context_tokens = 100_000
    context = ContextManager(s)

    assert context.request_token_budget() == 100_000 - DEFAULT_OUTPUT_RESERVE_TOKENS - MIN_CONTEXT_SAFETY_TOKENS

    s.config.provider.max_tokens = 10_000
    assert context.request_token_budget() == 100_000 - 10_000 - MIN_CONTEXT_SAFETY_TOKENS


def test_tool_schemas_can_trigger_compaction_before_context_ceiling(tmp_path):
    s = session(tmp_path)
    s.settings.max_context_tokens = 30_000
    s.messages = [
        {"role": "user", "content": "old request"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "latest request"},
    ]
    context = ContextManager(s)
    turn = [{"role": "user", "content": "continue"}]
    tools = [{"type": "function", "function": {"name": "Large", "description": "x" * 80_000, "parameters": {}}}]
    messages = context.model_messages("system", turn)
    assert context.request_tokens(messages) < context.request_token_budget()
    assert context.request_token_budget() <= context.request_tokens(messages, tools) < s.settings.max_context_tokens

    class FakeModel:
        def __init__(self):
            self.called = False

        def compact(self, text, *_args, **_kwargs):
            self.called = True
            return {"summary": "summary"}

    model = FakeModel()
    context.prepare_messages(model, "system", turn, tools)

    assert model.called is True


def test_compaction_parts_keep_latest_user_turn_after_prior_summary(tmp_path):
    s = session(tmp_path)
    summary = COMPACTION_SUMMARY_TITLE + "\nold summary"
    s.messages = [
        {"role": "user", "content": summary},
        {"role": "assistant", "content": "before"},
        {"role": "user", "content": "old request"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "latest request"},
        {"role": "user", "content": summary},
        {"role": "assistant", "content": "working"},
        {"role": "tool", "content": "tool tr.1"},
    ]

    compacted, keep = ContextManager(s).compaction_parts()

    assert [message["content"] for message in compacted] == ["before", "old request", "old answer"]
    assert [message["content"] for message in keep] == ["latest request", "working", "tool tr.1"]


def test_compaction_parts_compact_all_without_plain_user_message(tmp_path):
    s = session(tmp_path)
    s.messages = [
        {"role": "user", "content": COMPACTION_SUMMARY_TITLE + "\nold summary"},
        {"role": "assistant", "content": "answer"},
        {"role": "tool", "content": "tool tr.1"},
    ]

    compacted, keep = ContextManager(s).compaction_parts()

    assert compacted == s.messages[1:]
    assert keep == []


def test_compaction_selection_keeps_assistant_text_that_quotes_summary_marker(tmp_path):
    s = session(tmp_path)
    quoted = COMPACTION_SUMMARY_TITLE + "\nquoted by assistant"
    s.messages = [
        {"role": "user", "content": COMPACTION_SUMMARY_TITLE + "\nold summary"},
        {"role": "assistant", "content": quoted},
    ]

    compacted, keep = ContextManager(s).compaction_parts()

    assert compacted == [{"role": "assistant", "content": quoted}]
    assert keep == []


def test_prepare_messages_does_not_recompact_a_summary_by_itself(tmp_path):
    s = session(tmp_path)
    s.settings.max_context_tokens = 1
    summary = COMPACTION_SUMMARY_TITLE + "\nold summary"
    s.state.summary = "old summary"
    s.messages = [{"role": "user", "content": summary}]

    class FakeModel:
        def compact(self, text, *_args, **_kwargs):
            raise AssertionError(f"synthetic summary was compacted again: {text}")

    ContextManager(s).prepare_messages(FakeModel(), "system")

    assert s.messages == [{"role": "user", "content": summary}]
    assert s.state.compaction_count == 0
    assert s.history == []


def test_turn_compaction_does_not_recompact_a_prior_summary(tmp_path):
    context = ContextManager(session(tmp_path))
    summary = COMPACTION_SUMMARY_TITLE + "\nold summary"
    messages = [
        {"role": "user", "content": "current request"},
        {"role": "user", "content": summary},
        *({"role": "assistant", "content": f"step {index}"} for index in range(10)),
    ]

    compacted, keep = context.turn_compaction_parts(messages)

    assert [message["content"] for message in compacted] == ["step 0", "step 1"]
    assert keep[0]["content"] == "current request"
    assert all(message.get("content") != summary for message in [*compacted, *keep])


def test_turn_compaction_evicts_the_prefix_before_a_late_followup(tmp_path):
    context = ContextManager(session(tmp_path))
    messages = [
        {"role": "user", "content": "original request"},
        *({"role": "assistant", "content": f"old step {index}"} for index in range(20)),
        {"role": "user", "content": "late follow-up"},
        *({"role": "assistant", "content": f"new step {index}"} for index in range(10)),
    ]

    compacted, keep = context.turn_compaction_parts(messages)

    assert compacted[0]["content"] == "original request"
    assert "old step 19" in [message["content"] for message in compacted]
    assert keep[0]["content"] == "late follow-up"
    assert [message["content"] for message in keep[1:]] == [f"new step {index}" for index in range(2, 10)]


def test_prepare_request_persists_current_turn_compaction_without_pending_input(tmp_path):
    s = session(tmp_path)
    s.settings.max_context_tokens = 1
    agent = Agent(s, output_fn=lambda _text: None)
    turn = [
        {"role": "user", "content": "original request"},
        *({"role": "assistant", "content": f"old step {index}"} for index in range(20)),
        {"role": "user", "content": "continue"},
        *({"role": "assistant", "content": f"new step {index}"} for index in range(10)),
    ]
    agent.model.compact = lambda _text, *_args, **_kwargs: {"summary": "compact summary"}

    agent.prepare_request(turn)

    assert len(turn) == 10
    assert turn[0]["content"] == "continue"
    assert turn[1]["content"].startswith(COMPACTION_SUMMARY_TITLE)
    assert "original request" in s.history[-1].text


def test_accepted_followup_commits_staged_current_turn_compaction(tmp_path):
    s = session(tmp_path)
    agent = Agent(s, output_fn=lambda _text: None)
    turn = [
        {"role": "user", "content": "original request"},
        *({"role": "assistant", "content": f"old step {index}"} for index in range(20)),
    ]
    transcript = [agent.transcript_message(turn[0])]
    s.enqueue_user_input("late follow-up")
    agent.model.compact = lambda _text, *_args, **_kwargs: {"summary": "compact summary"}
    agent.context.request_token_budget = lambda: 10
    agent.context.request_tokens = lambda messages, tools=None: 100 if any("old step" in str(message.get("content") or "") for message in messages) else 1

    request = agent.prepare_request(turn)

    assert any("old step" in str(message.get("content") or "") for message in turn)
    assert not any("old step" in str(message.get("content") or "") for message in request.turn_messages)
    agent.accept_pending_inputs(turn, transcript, request.pending, request.turn_messages)
    assert turn == request.turn_messages
    assert transcript[-1]["content"].endswith("late follow-up")
    assert s.state.compaction_count == 1

    agent.prepare_request(turn)

    assert s.state.compaction_count == 1


def test_interrupted_current_turn_compaction_falls_back_before_cancelling(tmp_path):
    s = session(tmp_path)
    s.settings.max_context_tokens = 1
    context = ContextManager(s)
    turn = [{"role": "user", "content": "request"}, *({"role": "assistant", "content": f"step {index}"} for index in range(20))]
    phases = []
    context.on_compaction = lambda active, error: phases.append((active, error))

    class InterruptedModel:
        def compact(self, _text, *_args, **_kwargs):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        context.prepare_messages(InterruptedModel(), "system", turn)

    assert len(turn) == 10
    assert turn[0]["content"] == "request"
    assert turn[1]["content"].startswith(COMPACTION_SUMMARY_TITLE)
    assert CURRENT_TURN_CONTEXT_TRIMMED in turn[1]["content"]
    assert phases == [(True, ""), (False, "cancelled by user")]


def test_compaction_parts_bounds_the_work_after_the_last_request(tmp_path):
    """One request can drive dozens of tool calls. /compact must summarize that tail too, or a
    long turn leaves the context as large as it started."""
    s = session(tmp_path)
    s.messages = [{"role": "user", "content": "older"}, {"role": "assistant", "content": "older answer"}]
    s.messages.append({"role": "user", "content": "do the big thing"})
    for i in range(30):
        s.messages.append(
            {"role": "assistant", "content": f"step {i}", "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": "Read", "arguments": "{}"}}]}
        )
        s.messages.append({"role": "tool", "content": f"tool tr.{i}"})

    compacted, keep = ContextManager(s).compaction_parts()

    # The request that started the work is kept, plus a bounded window of what followed.
    assert keep[0] == {"role": "user", "content": "do the big thing"}
    assert len(keep) <= ContextManager.COMPACT_RECENT_MESSAGES + 1
    assert len(compacted) == len(s.messages) - len(keep)
    # A kept tool result never loses the call it answers.
    if keep[1].get("role") == "tool":
        raise AssertionError("kept tail starts with an orphaned tool result")


def test_compaction_parts_for_uses_last_fixed_window(tmp_path):
    messages = [{"role": "assistant", "content": f"m{index}"} for index in range(10)]

    older, recent = ContextManager(session(tmp_path)).compaction_parts_for(messages)

    assert [message["content"] for message in older] == ["m0", "m1"]
    assert [message["content"] for message in recent] == [f"m{index}" for index in range(2, 10)]


def test_prepare_messages_skips_compaction_when_context_under_budget(tmp_path):
    s = session(tmp_path)
    s.settings.max_context_tokens = 999_999
    s.messages = [{"role": "user", "content": "old"}, {"role": "assistant", "content": "answer"}]
    context = ContextManager(s)
    compaction_phases = []
    context.on_compaction = lambda active, _error: compaction_phases.append(active)

    class ExplodingModel:
        def compact(self, text, *_args, **_kwargs):
            raise AssertionError(text)

    context.prepare_messages(ExplodingModel(), "system", [{"role": "user", "content": "request"}])

    assert compaction_phases == []
    assert s.messages == [{"role": "user", "content": "old"}, {"role": "assistant", "content": "answer"}]


def _huge_history(tmp_path, steps, *, budget_tokens=200_000, chars=160_000):
    """A session that has already been compacted once, still over budget, with `steps` large
    assistant messages after the latest user message."""
    s = session(tmp_path)
    s.settings.max_context_tokens = budget_tokens
    s.state.summary = "old summary"
    s.messages = [
        {"role": "user", "content": COMPACTION_SUMMARY_TITLE + "\nold summary"},
        {"role": "user", "content": "keep going"},
        *({"role": "assistant", "content": f"step {index} " + "y" * chars} for index in range(steps)),
    ]
    return s, ContextManager(s)


class _CountingModel:
    def __init__(self):
        self.calls = 0

    def compact(self, _text, *_args, **_kwargs):
        self.calls += 1
        return {"summary": "new summary"}


def test_automatic_compaction_runs_once_until_new_messages_arrive(tmp_path):
    # The loop guard. Compaction shrinks history but cannot always get under budget, so "still over
    # budget" must not by itself justify compacting again: re-deciding on the same messages is the
    # runaway compaction this has regressed into before. One automatic pass per scope, until the
    # message list actually changes.
    s, context = _huge_history(tmp_path, steps=30)
    model = _CountingModel()

    for _ in range(5):
        context.prepare_messages(model, "system")

    assert model.calls == 1
    assert s.state.compaction_count == 1

    # A new message is new information, so one more pass is allowed -- and again only one.
    s.messages.append({"role": "assistant", "content": "another step " + "y" * 160_000})
    for _ in range(5):
        context.prepare_messages(model, "system")

    assert model.calls == 2


def test_a_short_tail_of_large_messages_is_still_compactable(tmp_path):
    # The recent window is a message count, not a size. Once a session has been compacted, the only
    # compactable head is what sits before the latest user message -- and that is just the previous
    # summary, which is filtered out. So a handful of large messages after that user message left an
    # empty head, and every following request went out over budget without compacting anything.
    for steps in (2, 6, 8):
        _s, context = _huge_history(tmp_path, steps=steps)
        budget = context.request_token_budget()
        before = context.request_tokens(context.model_messages("system"), None)
        model = _CountingModel()

        context.prepare_messages(model, "system")

        after = context.request_tokens(context.model_messages("system"), None)
        if before < budget:
            assert model.calls == 0, f"{steps} steps fit; nothing should have been compacted"
        else:
            assert model.calls == 1, f"{steps} large messages after the user message were left uncompacted"
            assert after < budget, f"{steps} steps: still over budget after compacting"


def test_over_budget_with_nothing_compactable_is_reported_once(tmp_path):
    # The irreducible case: the latest user message and one enormous tool result. The cut may not
    # land between a tool result and the call that produced it, so there is nothing to compact at
    # any window. Say so once rather than silently sending a request the provider will reject.
    s = session(tmp_path)
    s.settings.max_context_tokens = 200_000
    s.messages = [
        {"role": "user", "content": "read the file"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "t1", "function": {"name": "Read", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "x" * 1_000_000},
    ]
    context = ContextManager(s)
    reports = []
    context.on_compaction = lambda active, error: reports.append((active, error))
    model = _CountingModel()

    for _ in range(4):
        context.prepare_messages(model, "system")

    assert model.calls == 0  # nothing could be compacted, so nothing was sent to the model
    assert len(reports) == 1, "the dead end must be reported, and only once"
    assert reports[0][0] is False
    assert "nothing is left to compact" in reports[0][1]

    # A new message buys one more attempt, and reports again because it is still a dead end.
    s.messages.append({"role": "assistant", "content": "still working"})
    context.prepare_messages(model, "system")
    assert len(reports) == 2
    assert model.calls == 0

    # A new *user* message is different: everything before it becomes compactable again, so the
    # dead end ends and a real compaction runs.
    s.messages.append({"role": "user", "content": "carry on"})
    context.prepare_messages(model, "system")
    assert model.calls == 1
    assert reports[-2:] == [(True, ""), (False, "")]


def test_automatic_turn_compaction_runs_once_until_the_turn_grows(tmp_path):
    # Same guard for the current-turn pass, and it must not carry across turns: a fresh turn is a
    # different (shorter) list, and blocking it because the previous turn was longer would leave the
    # new one uncompactable.
    _s, context = _huge_history(tmp_path, steps=2)
    model = _CountingModel()
    turn = [{"role": "user", "content": "request"}, *({"role": "assistant", "content": "t " + "y" * 160_000} for _ in range(30))]

    for _ in range(5):
        context.prepare_messages(model, "system", turn)
    first = model.calls
    assert first >= 1

    for _ in range(5):
        context.prepare_messages(model, "system", turn)
    assert model.calls == first  # nothing changed, nothing recompacted

    next_turn = [{"role": "user", "content": "next"}, *({"role": "assistant", "content": "n " + "y" * 160_000} for _ in range(30))]
    context.prepare_messages(model, "system", next_turn)
    assert model.calls > first  # a new turn is not blocked by the previous turn's mark


def test_prepare_messages_builds_under_budget_context_once(tmp_path, monkeypatch):
    context = ContextManager(session(tmp_path))
    calls = 0
    original = context.model_messages

    def model_messages(base_system, turn_messages=None):
        nonlocal calls
        calls += 1
        return original(base_system, turn_messages)

    monkeypatch.setattr(context, "model_messages", model_messages)
    context.prepare_messages(object(), "system", [{"role": "user", "content": "request"}])

    assert calls == 1


def test_compaction_keeps_assistant_with_tool_results(tmp_path):
    context = ContextManager(session(tmp_path))
    messages = [
        *({"role": "user", "content": f"old {index}"} for index in range(3)),
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "tc.1", "type": "function", "function": {"name": "Read", "arguments": "{}"}},
                {"id": "tc.2", "type": "function", "function": {"name": "Read", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "tc.1", "content": "one"},
        {"role": "tool", "tool_call_id": "tc.2", "content": "two"},
        *({"role": "user", "content": f"recent {index}"} for index in range(6)),
    ]

    compacted, keep = context.compaction_parts_for(messages)

    assert compacted == messages[:3]
    assert keep == messages[3:]
    assert keep[0]["role"] == "assistant"
    assert [message["role"] for message in keep[1:3]] == ["tool", "tool"]


def test_compaction_keeps_tool_records_referenced_from_summary(tmp_path):
    s = session(tmp_path)
    context = ContextManager(s)
    kept = s.store_tool_result("Bash", ["kept"], "kept output")
    dropped = s.store_tool_result("Bash", ["dropped"], "dropped output")

    context.apply_compaction({"summary": f"Continue from {kept}."}, [])

    assert kept in s.tool_results
    assert dropped not in s.tool_results
    assert [record.key for record in s.tool_records] == [kept]


def test_compaction_prunes_unreferenced_tool_records(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("one\n", encoding="utf-8")
    s = session(tmp_path)
    context = ContextManager(s)
    old_key = s.store_tool_result("Bash", ["old"], "old output")
    read_key = s.store_tool_result("Read", [{"path": "a.txt", "ranges": [[0, 0]]}], ReadTool(s, [{"path": "a.txt", "ranges": [[0, 0]]}]).call())
    current_key = s.store_tool_result("Bash", ["current"], "current output")

    context.apply_compaction({"summary": "summary"}, [{"role": "tool", "content": f"tool {current_key} Bash current"}])

    assert old_key not in s.tool_results
    assert read_key not in s.tool_results
    assert {record.key for record in s.tool_records} == {current_key}
    assert set(s.tool_results) == {current_key}


def test_compaction_keeps_current_turn_tool_records(tmp_path):
    s = session(tmp_path)
    s.settings.max_context_tokens = 1
    s.messages = [{"role": "user", "content": "old"}, {"role": "assistant", "content": "old answer"}, {"role": "user", "content": "latest"}]
    old_key = s.store_tool_result("Bash", ["old"], "old output")
    current_key = s.store_tool_result("Bash", ["current"], "current output")

    class FakeModel:
        def compact(self, text, *_args, **_kwargs):
            return {"summary": "summary"}

    ContextManager(s).prepare_messages(FakeModel(), "system", [{"role": "tool", "content": f"tool {current_key} Bash current"}])

    assert old_key not in s.tool_results
    assert current_key in s.tool_results
    assert [record.key for record in s.tool_records] == [current_key]


def test_compaction_drops_unreferenced_read_edit_records(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")
    s = session(tmp_path)
    context = ContextManager(s)
    read_key = s.store_tool_result("Read", [{"path": "a.txt", "ranges": [[0, 0]]}], ReadTool(s, [{"path": "a.txt", "ranges": [[0, 0]]}]).call())
    edit_key = s.store_tool_result(
        "Edit",
        ["a.txt"],
        EditTool(s, ["a.txt", [{"op": "delete", "start": "1:" + ReadTool.line_hash("b\n"), "end": "1:" + ReadTool.line_hash("b\n")}]]).call(),
    )

    context.apply_compaction({"summary": "summary"}, [])

    assert read_key not in s.tool_results
    assert edit_key not in s.tool_results
    assert s.tool_records == []


def test_compaction_captures_a_history_segment(tmp_path):
    s = session(tmp_path)
    context = ContextManager(s)
    compacted = [
        {"role": "user", "content": "find the parser bug"},
        {"role": "assistant", "content": "looking into it"},
    ]

    context.apply_compaction({"summary": "summary"}, [], compacted=compacted)

    assert len(s.history) == 1
    segment = s.history[0]
    assert segment.key == "seg.1"
    assert segment.title == "find the parser bug"
    assert "find the parser bug" in segment.text
    assert "looking into it" in segment.text


def test_segment_takes_the_name_the_compactor_gave_it(tmp_path):
    # The compaction reply already describes the span, and the model is the only party that read
    # all of it. The deterministic name is the first user message of the window, which says little
    # once a span starts mid-work — here, "ok".
    s = session(tmp_path)
    context = ContextManager(s)
    compacted = [
        {"role": "user", "content": "ok"},
        {"role": "assistant", "content": "Extracted tokenize() and updated the imports."},
    ]

    context.apply_compaction({"title": "Tokenizer extraction", "summary": "s"}, [], compacted=compacted)

    assert s.history[0].title == "Tokenizer extraction"


def test_segment_title_is_flattened_and_bounded(tmp_path):
    # Free-form model text landing in a listing, a checkpoint line, and a viewer column.
    s = session(tmp_path)
    context = ContextManager(s)

    context.apply_compaction(
        {"title": '  "Parser refactor\n  and its tests"  ', "summary": "s"},
        [],
        compacted=[{"role": "user", "content": "x"}],
    )

    assert s.history[0].title == "Parser refactor and its tests"


def test_segment_title_falls_back_when_the_compactor_names_nothing(tmp_path):
    s = session(tmp_path)
    context = ContextManager(s)
    compacted = [{"role": "user", "content": "find the parser bug"}]

    for data in ({"summary": "s"}, {"title": "", "summary": "s"}, {"title": ["nope"], "summary": "s"}):
        s.history.clear()
        context.apply_compaction(data, [], compacted=compacted)
        assert s.history[0].title == "find the parser bug", data


def test_deterministic_trim_still_names_its_segment(tmp_path):
    # No model reply at all: the summarizer failed and the span was trimmed deterministically.
    s = session(tmp_path)
    context = ContextManager(s)

    context.apply_compaction(None, [], compacted=[{"role": "user", "content": "find the parser bug"}], fallback_note="trimmed")

    assert s.history[0].title == "find the parser bug"


def test_large_history_segment_has_no_self_referential_recall_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(context_module, "MAX_TOOL_OUTPUT_TOKENS", 10)
    s = session(tmp_path)
    context = ContextManager(s)

    context.apply_compaction({"summary": "summary"}, [], compacted=[{"role": "user", "content": "x" * 1000}])

    assert "<bounded_output" in s.history[0].text
    assert 'recall="seg.1"' not in s.history[0].text


def test_compaction_history_keys_increment(tmp_path):
    s = session(tmp_path)
    context = ContextManager(s)

    context.apply_compaction({"summary": "s"}, [], compacted=[{"role": "user", "content": "first task"}])
    context.apply_compaction({"summary": "s"}, [], compacted=[{"role": "user", "content": "second task"}])

    assert [segment.key for segment in s.history] == ["seg.1", "seg.2"]


def test_compaction_without_compacted_messages_captures_nothing(tmp_path):
    s = session(tmp_path)
    context = ContextManager(s)

    context.apply_compaction({"summary": "summary"}, [])

    assert s.history == []


def test_prepare_messages_captures_history_and_turn_segments_in_one_pass(tmp_path):
    """An over-budget request can cross both compaction stages in one prepare: the history before the
    latest request becomes seg.1, then the oversized current turn itself becomes seg.2."""
    s = session(tmp_path)
    s.settings.max_context_tokens = 1
    s.messages = [
        {"role": "user", "content": "old request"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "latest request"},
    ]
    context = ContextManager(s)
    turn = [{"role": "user", "content": "current request"}, *({"role": "assistant", "content": f"step {index}"} for index in range(20))]

    class FakeModel:
        def __init__(self):
            self.calls = 0

        def compact(self, text, *_args, **_kwargs):
            self.calls += 1
            return {"summary": f"summary {self.calls}"}

    model = FakeModel()
    context.prepare_messages(model, "system", turn)

    assert model.calls == 2
    assert [segment.key for segment in s.history] == ["seg.1", "seg.2"]
    assert "old request" in s.history[0].text
    assert "step 0" in s.history[1].text
    assert "step 11" in s.history[1].text
    # The turn keeps its request and recent window; the compacted prefix is replaced by the summary.
    assert turn[0]["content"] == "current request"
    assert turn[1]["content"].startswith(COMPACTION_SUMMARY_TITLE)


def test_history_title_skips_summary_blocks(tmp_path):
    context = ContextManager(session(tmp_path))
    messages = [
        {"role": "user", "content": COMPACTION_SUMMARY_TITLE + "\nold summary"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "the real request"},
    ]

    assert context.history_title(messages) == "the real request"


def test_history_index_and_memory_are_not_injected_into_each_request(tmp_path):
    s = session(tmp_path)
    s.skills = SkillLibrary({})  # this test isolates history projection from optional context
    s.messages.extend([{"role": "user", "content": "old request"}, {"role": "assistant", "content": "old answer"}])
    s.history.append(HistorySegment(key="seg.1", title="find the bug", text="..."))
    context = ContextManager(s)

    messages = context.model_messages("system", [{"role": "user", "content": "current request"}])
    contents = [str(message.get("content") or "") for message in messages]
    assert contents[2:] == ["old request", "old answer", "current request"]
    assert not any(content.startswith("--- History index ---") for content in contents)
    assert not any(content.startswith("--- Memory ---") for content in contents)


def test_compaction_fallback_trims_when_model_compact_fails(tmp_path):
    s = session(tmp_path)
    s.settings.max_context_tokens = 1
    s.state.summary = "existing"
    s.messages = [{"role": "user", "content": str(index)} for index in range(10)]
    context = ContextManager(s)
    compaction_phases = []
    context.on_compaction = lambda active, _error: compaction_phases.append(active)

    class FailingModel:
        def compact(self, text, *_args, **_kwargs):
            raise ModelError("failed")

    context.prepare_messages(FailingModel(), "system", [{"role": "user", "content": "request"}])

    assert compaction_phases == [True, False]
    assert s.state.summary != "existing"
    assert len(s.messages) == 2
    assert s.messages[0]["content"].startswith(COMPACTION_SUMMARY_TITLE)
    assert "deterministically trimmed" in s.messages[0]["content"]
    assert s.messages[1]["content"] == "9"
    # Even though summarization failed, the evicted conversation is still captured as a recallable
    # segment: the fallback summary is only a trim note, so this is the only way to recover it.
    assert [segment.key for segment in s.history] == ["seg.1"]
    assert "user:\n0" in s.history[0].text
    assert "user:\n8" in s.history[0].text


def test_manual_compact_inserts_summary_before_latest_user(tmp_path):
    s = session(tmp_path)
    s.messages = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "latest"},
        {"role": "tool", "content": "tool kept"},
    ]
    s.state.context_percent = 80
    loop = CommandLoop(Agent(s, output_fn=lambda text: None), output_fn=lambda text: None)
    transitions = []
    loop.tui = SimpleNamespace(set_running=transitions.append, set_dispatching=lambda: transitions.append("dispatch"))

    class FakeModel:
        last_compaction_model = ""

        def compact(self, text, *_args, **_kwargs):
            assert transitions == ["compacting context"]
            return {"summary": "summary", "plan": ["next"], "known": ["fact"]}

    loop.agent.model = FakeModel()
    result = compact(loop, "")

    assert [message["role"] for message in s.messages] == ["user", "user", "tool"]
    assert s.messages[0]["content"].startswith(COMPACTION_SUMMARY_TITLE)
    assert s.messages[1]["content"] == "latest"
    assert s.messages[2]["content"] == "tool kept"
    assert s.state.summary == "summary"
    assert transitions == ["compacting context", "dispatch"]
    assert "messages 4 -> 3" in result
    assert "prior summary inserted" in result


def test_agent_state_format_is_available_for_explicit_checkpoints(tmp_path):
    s = session(tmp_path)
    s.state.goal = "test goal"
    s.state.check = "all good"
    s.record_tool_error("tr.1", "Bash", ["bad"], "failed")

    ctx = s.state.format()

    assert "Goal:" in ctx
    assert "test goal" in ctx
    assert "Check:" in ctx
    assert "all good" in ctx
    assert "Recent tool errors:" not in ctx


def test_estimated_text_tokens_stays_on_characters_for_output_trimming(tmp_path):
    """estimated_text_tokens drives tool-output trimming (head/tail excerpts and the omitted marker),
    so it stays chars/4: UTF-8 bytes there would shrink the head/tail slice for CJK, overlapping the
    head and tail or inflating the bounded output marker. The request-level estimate is what counts
    bytes (test_cjk_payload_compacts_where_character_estimate_would_not)."""
    context = ContextManager(session(tmp_path))
    assert context.estimated_text_tokens("hello world") == (len("hello world") + 3) // 4
    # CJK stays at chars/4 too: 4 chars -> 1 estimated token, not 3.
    assert context.estimated_text_tokens("你好世界") == 1


def test_cjk_payload_compacts_where_character_estimate_would_not(tmp_path):
    """A CJK-heavy session that the chars/4 estimate kept under budget now compacts: the bytes/4
    estimate clears the same budget, closing the gap between the status-bar fill and the trigger."""
    import json

    s = session(tmp_path)
    s.settings.max_context_tokens = 23_000  # budget 2520: chars/4 estimate ~2017, bytes/4 5406
    s.messages = [
        {"role": "user", "content": "你好" * 300},
        {"role": "assistant", "content": "收到" * 300},
        *({"role": "assistant", "content": "继续" * 100} for _ in range(8)),
        {"role": "user", "content": "中文" * 2000},
    ]
    context = ContextManager(s)
    compaction_phases = []
    context.on_compaction = lambda active, _error: compaction_phases.append(active)

    class FakeModel:
        def compact(self, text, *_args, **_kwargs):
            return {"summary": "compact summary", "plan": ["next"], "known": ["fact"]}

    turn = [{"role": "user", "content": "请用中文回复"}]
    messages = context.model_messages("system", turn)
    raw = context.request_tokens(messages)
    budget = context.request_token_budget()
    # The chars/4 figure sits under the budget; the UTF-8 bytes/4 estimate clears it.
    assert len(json.dumps(messages, ensure_ascii=False)) // 4 < budget < raw

    context.prepare_messages(FakeModel(), "system", turn)
    assert compaction_phases == [True, False]
    assert s.state.compaction_count == 1


def test_overdue_usage_triggers_compaction_even_when_estimate_fits(tmp_path):
    """The last completed request filled >=99% of its budget, so the next one compacts even though the
    bytes/4 estimate still fits: a last line of defense when the estimate is off. Below 99% the
    estimate alone decides, so a small follow-up after an 80% request is not compacted."""
    s = session(tmp_path)
    s.settings.max_context_tokens = 21_000  # budget 520; the ASCII payload estimates ~326
    s.messages = [
        {"role": "user", "content": "old user"},
        {"role": "assistant", "content": "old answer"},
        *({"role": "assistant", "content": f"recent {index}"} for index in range(8)),
        {"role": "user", "content": "latest"},
    ]
    context = ContextManager(s)
    compaction_phases = []
    context.on_compaction = lambda active, _error: compaction_phases.append(active)

    class FakeModel:
        def compact(self, text, *_args, **_kwargs):
            return {"summary": "compact summary", "plan": ["next"], "known": ["fact"]}

    turn = [{"role": "user", "content": "request"}]
    assert context.request_tokens(context.model_messages("system", turn)) < context.request_token_budget()

    # 98%: estimate fits and nothing compacts.
    s.usage.last_prompt_budget = 520
    s.usage.last_prompt_tokens = 510
    context.prepare_messages(FakeModel(), "system", turn)
    assert compaction_phases == []
    assert s.state.compaction_count == 0

    # 100%: the overdue flag forces compaction despite the fitting estimate.
    s.usage.last_prompt_tokens = 520
    context.prepare_messages(FakeModel(), "system", turn)
    assert compaction_phases == [True, False]
    assert s.state.compaction_count == 1
    # Compaction cleared the last-* signals, so the next request is not double-compacted by the
    # guard (the compaction request's own usage was just wiped instead of being mistaken for an
    # ordinary 100%-full context).
    assert s.usage.last_prompt_tokens == 0
    assert s.usage.last_prompt_budget == 0
    context.prepare_messages(FakeModel(), "system", turn)
    assert compaction_phases == [True, False]
    assert s.state.compaction_count == 1

    # A fresh session with no recorded usage never trips the flag.
    s2 = session(tmp_path)
    context2 = ContextManager(s2)
    assert context2._overdue_by_usage() is False


def test_apply_compaction_clears_last_usage_but_keeps_cumulative(tmp_path):
    """Compaction rewrites history, so the recorded last-* usage no longer describes the next
    request. Clearing them (not the cumulative totals) makes the overdue guard and the status bar
    fall back to the local estimate until the next ordinary request reports real usage."""
    s = session(tmp_path)
    s.usage.last_prompt_tokens = 1234
    s.usage.last_prompt_budget = 1200
    s.usage.last_cached_prompt_tokens = 300
    s.usage.last_cache_write_prompt_tokens = 50
    s.usage.prompt_tokens = 9999
    s.usage.calls = 7
    s.messages = [
        {"role": "user", "content": "old user"},
        {"role": "assistant", "content": "old answer"},
        *({"role": "assistant", "content": f"recent {index}"} for index in range(8)),
        {"role": "user", "content": "latest"},
    ]
    context = ContextManager(s)
    compacted, keep = context.compaction_parts()
    context.apply_compaction({"summary": "compact summary", "plan": ["next"], "known": ["fact"]}, keep, compacted=compacted)

    assert s.usage.last_prompt_tokens == 0
    assert s.usage.last_prompt_budget == 0
    assert s.usage.last_cached_prompt_tokens == 0
    assert s.usage.last_cache_write_prompt_tokens == 0
    assert s.usage.prompt_tokens == 9999
    assert s.usage.calls == 7
    assert context._overdue_by_usage() is False


def test_compaction_override_does_not_change_request_budget(tmp_path):
    """The [compaction] entry never feeds the context budget: even pointing at a small-window entry,
    requests are still prepared against the active provider's window."""
    from minacode.config import Config, request_budget_for

    s = session(tmp_path)
    s.config = Config.from_dict(
        {
            "compaction": {"provider": "small"},
            "provider": {
                "active": "wide",
                "wide": {"model": "wide", "max_context_tokens": 1_048_576},
                "small": {"model": "small", "max_context_tokens": 16_384},
            },
        }
    )
    context = ContextManager(s)
    assert context.request_token_budget() == request_budget_for(1_048_576, DEFAULT_OUTPUT_RESERVE_TOKENS)
    assert context.request_token_budget() > request_budget_for(16_384, DEFAULT_OUTPUT_RESERVE_TOKENS)


# The request that opened a turn survives compaction because latest_user_index protects the last
# plain user message. Every user message the runtime generates on its own -- a mention expansion, a
# protocol correction -- therefore has to be marked as a session event, or it takes that protection
# for itself and the request it was expanding gets summarized away mid-turn. The worker is where
# this bites hardest: that message is the entire order (docs/worker.md), the worker cannot see the
# parent's history, and nothing re-sends it.
RUNTIME_GENERATED_EVENTS = ("mcp_mentions", "skill_mentions", "file_mentions", "tool_call_correction")


@pytest.mark.parametrize("event", RUNTIME_GENERATED_EVENTS)
def test_turn_compaction_keeps_the_request_a_runtime_message_follows(tmp_path, event):
    s = session(tmp_path)
    s.settings.max_context_tokens = 1
    context = ContextManager(s)
    turn = [
        {"role": "user", "content": "the whole order"},
        {"role": "user", "content": "runtime expansion", SESSION_EVENT_KEY: event},
        *({"role": "assistant", "content": f"step {index}"} for index in range(20)),
    ]

    class FakeModel:
        def compact(self, text, *_args, **_kwargs):
            return {"summary": "summary"}

    messages = context.prepare_messages(FakeModel(), "system", turn)

    assert turn[0]["content"] == "the whole order"
    assert any(message.get("content") == "the whole order" for message in messages)
    # Kept verbatim instead of summarized: the segment holds the steps, never the order itself.
    assert "the whole order" not in s.history[-1].text
    assert "step 0" in s.history[-1].text


def test_history_compaction_keeps_the_request_a_runtime_message_follows(tmp_path):
    s = session(tmp_path)
    s.messages = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "the whole order"},
        {"role": "user", "content": "runtime expansion", SESSION_EVENT_KEY: "skill_mentions"},
        {"role": "assistant", "content": "working"},
    ]

    compacted, keep = ContextManager(s).compaction_parts()

    assert [message["content"] for message in compacted] == ["old", "old answer"]
    assert [message["content"] for message in keep] == ["the whole order", "runtime expansion", "working"]


def test_turn_compaction_leaves_the_request_inside_the_cached_prefix(tmp_path):
    """Compaction is a cache break, and where it breaks is what it costs. Keeping the request in
    place puts the break behind it rather than on it, so the whole stable head -- system,
    environment, the request itself -- is still reused by the request that follows a compaction.
    The summary lands immediately after the request, so the expansion is what moves."""
    s = session(tmp_path)
    s.settings.max_context_tokens = 1
    context = ContextManager(s)
    turn = [
        {"role": "user", "content": "the whole order"},
        {"role": "user", "content": "runtime expansion", SESSION_EVENT_KEY: "skill_mentions"},
        *({"role": "assistant", "content": f"step {index}"} for index in range(20)),
    ]

    class FakeModel:
        def compact(self, text, *_args, **_kwargs):
            return {"summary": "summary"}

    client = ModelClient(s)
    before = client.chat_messages(context.model_messages("system", turn))
    after = client.chat_messages(context.prepare_messages(FakeModel(), "system", turn))
    shared = 0
    for old, new in zip(before, after):
        if old != new:
            break
        shared += 1

    # The break falls after the request: everything through it, the request included, is reused.
    assert before[shared - 1].get("content") == "the whole order"
    assert after[shared].get("content", "").startswith(COMPACTION_SUMMARY_TITLE)


def test_engine_marks_its_own_user_messages_as_session_events(tmp_path):
    """The engine half of the same rule: what run() appends around the request is not a request."""
    folder = os.path.join(tmp_path, ".minacode", "skills", "triage")
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "SKILL.md"), "w", encoding="utf-8") as handle:
        handle.write("---\nname: triage\ndescription: triage a bug\n---\nReproduce first.\n")
    with open(os.path.join(tmp_path, "issue.py"), "w", encoding="utf-8") as handle:
        handle.write("raise RuntimeError\n")
    s = session(tmp_path)  # discovers the skill written above
    s.mcp = SimpleNamespace(resolve_mentions=lambda text: "--- MCP MENTIONS ---", render_tools_index=lambda: "", tools={}, resources={})
    agent = Agent(s, output_fn=lambda text: None)

    class FakeModel:
        def request(self, messages, tools=None):
            return {"role": "assistant", "content": "done"}, [], "done"

    agent.model = FakeModel()
    assert agent.run("please $triage @file:issue.py") == "done"

    assert [message.get(SESSION_EVENT_KEY) for message in s.messages] == [None, "mcp_mentions", "skill_mentions", "file_mentions", None]
    assert ContextManager(s).latest_user_index(s.messages) == 0


def test_repeated_compaction_keeps_one_request_and_one_checkpoint(tmp_path):
    """Surviving one pass is not the same as surviving four. Each pass re-reads what the previous
    one left, so a request kept by accident (a second copy, a stale checkpoint it hides behind)
    would drift round by round: the invariant is one verbatim request, one checkpoint, and a
    message stream where every tool result still answers a call that is still there."""
    s = session(tmp_path)
    s.settings.max_context_tokens = 12000
    context = ContextManager(s)

    def steps(tag, count=8):
        messages = []
        for index in range(count):
            key = f"{tag}{index}"
            call_message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": key, "type": "function", "function": {"name": "Read", "arguments": "{}"}}],
            }
            messages.extend((call_message, {"role": "tool", "tool_call_id": key, "content": f"tr.{key} " + "filler " * 900}))
        return messages

    class FakeModel:
        def __init__(self):
            self.calls = 0

        def compact(self, text, *_args, **_kwargs):
            self.calls += 1
            return {"summary": f"summary {self.calls}"}

    model = FakeModel()
    turn = [
        {"role": "user", "content": "the whole order"},
        {"role": "user", "content": "--- SKILL MENTIONS ---\nbody", SESSION_EVENT_KEY: "skill_mentions"},
        {"role": "user", "content": "[Runtime protocol correction] ...", SESSION_EVENT_KEY: "tool_call_correction"},
        *steps("a"),
    ]
    for round_index in range(4):
        messages = context.prepare_messages(model, "system", turn)

        assert [message.get("content") for message in messages].count("the whole order") == 1
        assert turn[0]["content"] == "the whole order"
        assert sum(1 for message in turn if str(message.get("content") or "").startswith(COMPACTION_SUMMARY_TITLE)) == 1
        called = [raw["id"] for message in messages for raw in message.get("tool_calls") or []]
        answered = [message.get("tool_call_id") for message in messages if message.get("role") == "tool"]
        assert sorted(called) == sorted(answered), f"round {round_index} split a tool call from its result"
        turn.extend(steps(f"b{round_index}", 6))

    assert model.calls == 4  # one pass per round, never a compaction loop
    assert [segment.key for segment in s.history] == ["seg.1", "seg.2", "seg.3", "seg.4"]


def test_history_segments_keep_only_the_newest_window(tmp_path):
    """Every compaction stores a span and nothing used to drop one, so a long session carried each
    one it ever evicted. Keys keep counting past the bound: reusing a number the model has already
    seen would answer a stale recall with a different span instead of saying it is gone."""
    s = session(tmp_path)
    context = ContextManager(s)
    limit = ContextManager.MAX_HISTORY_SEGMENTS

    for index in range(limit + 5):
        context.store_history_segment([{"role": "user", "content": f"span {index}"}], scope="history", trigger="auto", fallback=False)

    assert len(s.history) == limit
    assert [segment.key for segment in s.history] == [f"seg.{number}" for number in range(6, limit + 6)]
    # The sixth span is the oldest one still retained; the five before it are gone with their text.
    assert [segment.text for segment in s.history] == [f"user:\nspan {index}" for index in range(5, limit + 5)]


def test_pruned_history_survives_a_snapshot_round_trip(tmp_path):
    """The snapshot writes segments as an append-only delta while the list only grows; pruning
    shortens it, and the digest guard has to notice and rewrite the whole list instead."""
    s = session(tmp_path)
    context = ContextManager(s)
    for index in range(ContextManager.MAX_HISTORY_SEGMENTS + 3):
        context.store_history_segment([{"role": "user", "content": f"span {index}"}], scope="history", trigger="auto", fallback=False)
        s.save_snapshot()

    restored = Session.load_snapshot(s.uid, config=s.config)

    assert [segment.key for segment in restored.history] == [segment.key for segment in s.history]
    assert [segment.text for segment in restored.history] == [segment.text for segment in s.history]


def test_compaction_reuses_the_agent_prefix_and_keeps_real_messages(tmp_path):
    """The summary request is the agent's own request with an instruction appended, so the provider
    cache already covers it -- and the compactor sees tool calls, which flattening drops."""
    live = session(tmp_path)
    live.messages = [
        {"role": "user", "content": "check the call sites"},
        {"role": "assistant", "content": "Looking.", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "Bash", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "tool tr.1 Bash rg -n _run_workflow"},
    ]
    context = ContextManager(live)
    compacted = list(live.messages)

    built = context.compaction_request(compacted)
    assert built is not None
    messages, tools = built

    header = context.model_header(live.system_prompt)
    assert messages[: len(header)] == Text.value(header)  # a real prefix of what the turn already paid for
    assert tools  # carried so the cached prefix matches; tool_choice is what stops their use
    # The tool call survives as structure, where messages_text() would have dropped it.
    assert any(message.get("tool_calls") for message in messages)
    assert "Compact the minacode working context." in messages[-1]["content"]
    assert "END OF CONVERSATION TO COMPACT" in messages[-1]["content"]


def test_compaction_falls_back_to_the_flat_payload_on_a_separate_provider(tmp_path):
    """A [compaction] entry elsewhere is a different cache namespace, so rebuilding the prefix for
    it would pay the whole history at full rate to save nothing."""
    live = session(tmp_path)
    live.messages = [{"role": "user", "content": "hello"}]
    live.config.compaction_provider = "cheap"
    assert ContextManager(live).compaction_request(list(live.messages)) is None


def test_compaction_leaves_tool_choice_exactly_as_an_ordinary_request_sets_it(tmp_path):
    """Forcing tool_choice would look safer and cost the prize: it invalidates the messages cache,
    which is the whole conversation this request exists to reuse. Every wire is treated alike."""
    live = session(tmp_path)
    live.messages = [{"role": "user", "content": "hello"}]
    for api in ("chat", "responses", "anthropic"):
        live.config.provider.api = api
        built = ContextManager(live).compaction_request(list(live.messages))
        assert built is not None, api  # no wire is excluded any more


def test_state_apply_takes_a_bare_string_where_a_list_was_asked_for(tmp_path):
    """Ignoring the wrong type is worse than accepting it: the previous compaction's value survives
    as though this one had confirmed it, and is fed back as current on the next pass."""
    live = session(tmp_path)
    live.state.apply({"known": ["stale fact"], "plan": [{"status": "done", "text": "old step"}]})

    live.state.apply({"known": "the API is rate limited", "plan": "finish Part B"})
    assert live.state.known == ["the API is rate limited"]
    assert [item.text for item in live.state.plan] == ["finish Part B"]

    # An empty string clears rather than silently keeping the old value.
    live.state.apply({"known": "   "})
    assert live.state.known == []
    # A type that is neither is still refused, as before.
    live.state.apply({"known": 17})
    assert live.state.known == []
