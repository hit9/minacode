"""Slash command implementations as free functions taking the CommandLoop.

Each handler has the signature `def name(loop, args) -> str | None` and is referenced directly
by the registry in minacode/cli/__init__.py. A `None` result means the handler rendered its own
UI (e.g. /diff's viewer). The multi-stage /worker flow lives in the WorkerFlow class below.
"""

from __future__ import annotations

import json
import os
import shlex
import sys
import time
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING

from minacode.base import (
    HTTP_USER_AGENT,
    SELECTION_BACK,
    SESSION_EVENT_KEY,
    ConfigError,
    ModelUsage,
    Text,
)
from minacode.cli.modals import choice_application, diff_viewer, mcp_manager, select_choice
from minacode.config import (
    IMAGE_INPUT_CHOICES,
    PROVIDER_API_CHOICES,
    REASONING_CHOICES,
    Config,
    ProviderConfig,
    RuntimeSettings,
)
from minacode.prompts import PREVIOUS_CONTEXT_TRIMMED
from minacode.provider_compat import builtin_tools_issue
from minacode.render import markdown_table
from minacode.session import SessionEntry, SessionSnapshotStore
from minacode.tools import CodeIndex
from minacode.tools.delegate import DelegateTool, refresh_worker_entry
from minacode.update import UpdateChecker

if TYPE_CHECKING:
    from minacode.cli import CommandLoop


# fmt: off


SetHandler = tuple[str, str, Callable[[str], int | float | None] | None]

# fmt: off
SET_HANDLERS: dict[str, SetHandler] = {
    "provider.temperature": ("provider", "temperature", lambda v: None if v == "off" else float(v)),
    "provider.max_tokens": ("provider", "max_tokens", lambda v: max(0, int(v))),
    "provider.timeout": ("provider", "timeout", lambda v: max(1, int(v))),
    "provider.response_timeout": ("provider", "response_timeout", lambda v: max(0, int(v))),
    "provider.stream": ("provider", "stream", lambda v: v == "on"),
    "provider.image_input": ("provider", "image_input", None),
    "runtime.max_agent_steps": ("settings", "max_steps", lambda v: max(1, int(v))),
    "runtime.max_context_tokens": ("settings", "max_context_tokens", lambda v: max(1, int(v))),
    "runtime.max_parallel_tools": ("settings", "max_parallel_tools", lambda v: max(1, int(v))),
    "runtime.shell_timeout": ("settings", "shell_timeout", lambda v: max(1, int(v))),
    "runtime.bash_wait_timeout": ("settings", "bash_wait_timeout", lambda v: max(0, int(v))),
    "runtime.worker": ("settings", "worker", lambda v: v == "on"),
}
SET_KEYS = tuple(SET_HANDLERS)
# Keys whose values are a closed set: rejected by /set when unknown, and offered whole as completions.
SET_CHOICES: dict[str, tuple[str, ...]] = {
    "provider.stream": ("on", "off"),
    "provider.image_input": IMAGE_INPUT_CHOICES,
    "runtime.worker": ("on", "off"),
}
SET_VALUES: dict[str, tuple[str, ...]] = {
    "provider.temperature": ("off",),
    **SET_CHOICES,
}
# fmt: on

WORKER_SUBCOMMANDS = ("status", "reset", "on", "off", "provider", "model", "reason", "api")


def _status_progress_bar(value: int, total: int, width: int = 14) -> str:
    ratio = min(1.0, max(0.0, value / total)) if total else 0.0
    eighths = int(ratio * width * 8 + 0.5)
    full, partial = divmod(eighths, 8)
    partials = "▏▎▍▌▋▊▉"
    return "[" + "█" * full + (partials[partial - 1] if partial else "") + "░" * (width - full - bool(partial)) + "]"


def _status_token_count(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(value)


def _status_render_table(rows: list[tuple[str, str]]) -> str:
    return "\n".join(
        [
            "| field | value |",
            "| --- | --- |",
            *(f"| {name} | {Text.clean(str(value)).replace(chr(10), ' ').replace('|', chr(92) + '|')} |" for name, value in rows),
        ]
    )


def _status_model_line(config: Config) -> str:
    active = config.provider
    return f"`{config.active_provider}/{active.model or '(empty)'}`; `{active.resolve().api}`; `{active.reasoning}`"


def _status_context_line(tokens: int, budget: int, percent: int) -> str:
    return f"`{_status_progress_bar(tokens, budget)}` `~{_status_token_count(tokens)} / {_status_token_count(budget)}` (`{percent}%`)"


def _status_cache_line(counts: ModelUsage) -> str:
    # The read ratios carry the useful signal; the raw token pairs made this the one row that
    # wrapped on a normal terminal. Writes stay, but only when there were any.
    def part(label: str, cached: int, prompt: int, written: int) -> str:
        write = f" (w {_status_token_count(written)})" if written else ""
        return f"{label} `{cached * 100 / prompt:.1f}%`{write}"

    return " ".join(
        [
            f"`{_status_progress_bar(counts.last_cached_prompt_tokens, counts.last_prompt_tokens)}`",
            part("last", counts.last_cached_prompt_tokens, counts.last_prompt_tokens, counts.last_cache_write_prompt_tokens) + ";"
            if counts.last_prompt_tokens
            else "last `n/a`;",
            part("session", counts.cached_prompt_tokens, counts.prompt_tokens, counts.cache_write_prompt_tokens),
        ]
    )


def resend_command(loop: CommandLoop, _args: str) -> str | None:
    """Resend the in-flight model request. Available only in the running queue-input region:
    typed while a turn works, it re-requests the current model call (same path as on_retry)."""
    if loop.tui is None or loop.tui.input_mode != "running":
        return "/resend re-requests the current model request — type it while a turn is working."
    if loop.session.state.current_model_call_started_at <= 0 or loop.session.state.model_retry_until > 0:
        return "Nothing to resend right now; /resend works while the model is generating."
    loop.tui.on_retry()
    return None


def mcp_command(loop: CommandLoop, args: str) -> str | None:
    mcp = loop.session.mcp
    if mcp is None:
        return "MCP not configured"

    parts = args.split()
    if not parts:
        if loop.tui is not None and loop.tui.input_mode != "running":
            return mcp_manager(loop)
        return mcp.render_server_status()

    sub = parts[0]
    rest = parts[1:]
    command = MCP_COMMANDS.get(sub)
    if command is None:
        return f"Unknown /mcp subcommand: {sub}. {MCP_HELP}"
    min_args, max_args, usage = command
    if not min_args <= len(rest) <= max_args:
        return usage

    if sub == "connect":
        return mcp.connect_servers(rest, interactive=loop.interactive_input, notify=loop.emit)
    if sub == "disconnect":
        return mcp.disconnect_server(rest[0])
    if sub == "tools":
        return mcp.render_tool_listing(rest[0] if rest else None)
    raise AssertionError("unreachable MCP subcommand")


def select_reasoning(loop: CommandLoop) -> str | object | None:
    current = loop.session.config.provider.reasoning
    labels = {"off": "off - disable reasoning"}
    labels[current] = labels.get(current, current) + " (current)"
    return select_choice(loop, "Reasoning effort", REASONING_CHOICES, labels=labels, current=current)


def select_api(loop: CommandLoop, model: str) -> str | object | None:
    # An endpoint that lists several model families rarely serves them all over one protocol, and
    # a /models listing does not say which. Confirm the wire alongside the model that needs it.
    provider = loop.session.config.provider
    current = provider.api
    inferred = replace(provider, api="auto", model=model).resolve().api
    labels = {"auto": f"auto - infer from the endpoint URL and model ({inferred})"}
    labels[current] = labels.get(current, current) + " (current)"
    return select_choice(loop, "Request API", PROVIDER_API_CHOICES, labels=labels, current=current)


def help(loop: CommandLoop, args: str) -> str:
    text = loop.HELP.rstrip()
    if loop.ui.color:
        return text
    text = text.replace("`", "")
    text = loop.HELP_HEADING_RE.sub(r"\1:", text)
    return loop.HELP_ENTRY_RE.sub(r"  \1  ", text)


def status(loop: CommandLoop, args: str) -> str:
    usage = loop.session.usage
    context_tokens = loop.agent.context.update_current_tokens(loop.agent.session.system_prompt)
    context_budget = loop.agent.context.request_token_budget()
    if usage.last_prompt_tokens and usage.last_prompt_budget:
        # Display the provider-reported tokens and the budget of the last request; the estimate
        # (state.context_percent) stays the fallback before any request exists.
        context_tokens = usage.last_prompt_tokens
        context_budget = usage.last_prompt_budget
        context_percent = min(100, context_tokens * 100 // context_budget)
    else:
        context_percent = loop.session.state.context_percent
    index = CodeIndex(loop.session)
    index_status, index_message = index.status(check=False)
    index.update_pending_async()
    if loop.session.state.code_index_refreshing:
        index_status, index_message = loop.session.state.code_index_notice or "syncing", ""
    elif loop.session.state.code_index_error:
        index_status, index_message = "error", loop.session.state.code_index_error
    if index_status in {"missing", "unavailable", "error"} and "run /index" not in index_message:
        index_message = (index_message + "; " if index_message else "") + "run /index"
    elif index_status == "stale" and "run /index" not in index_message:
        index_message = (index_message + "; " if index_message else "") + "run /index or wait for auto update"
    connected_mcp = sum(loop.session.mcp.connected(config.name) for config in loop.session.mcp.parse_configs()) if loop.session.mcp else 0
    activity: list[tuple[str, int | str]] = [
        ("history", len(loop.session.messages)),
        ("turn", loop.session.state.turn_messages),
        ("tools", len(loop.session.tool_results)),
        ("mcp", connected_mcp),
        ("skills", len(loop.session.skills.skills) if loop.session.skills else 0),
        ("known", len(loop.session.state.known)),
        ("compactions", loop.session.state.compaction_count),
    ]
    running_jobs = len(loop.session.running_jobs())
    if loop.session.jobs:
        activity.append(("jobs", f"{running_jobs}/{len(loop.session.jobs)}"))

    # One flat table: the session's own facts, then the parent's, then the worker's under
    # `worker*` labels. /status is an explicit query, so the worker rows appear whenever a
    # worker session exists, in flight or not.
    rows = [
        ("workspace", "`" + loop.session.cwd + "`"),
        ("session", "`" + loop.session.uid + "`"),
    ]
    if loop.session.state.goal:
        rows.append(("goal", loop.session.state.goal))
    runtime = [
        f"yolo {'on' if loop.session.settings.yolo else 'off'}",
        f"steps {loop.session.settings.max_steps}",
        CodeIndex.status_line(index_status, index_message),
    ]
    update = UpdateChecker(loop.session).status_line().removeprefix("update: ")
    if update not in {"current", "unknown"}:
        runtime.append("update " + update)
    rows.append(("runtime", "; ".join(f"`{value}`" for value in runtime)))

    rows.append(("model", _status_model_line(loop.session.config)))
    rows.append(("context", _status_context_line(context_tokens, context_budget, context_percent)))
    rows.append(("cache", _status_cache_line(usage) if usage.prompt_tokens else "(no requests yet)"))
    visible_activity = [(name, value) for name, value in activity if value]
    if visible_activity:
        rows.append(("activity", "; ".join(f"{name} `{value}`" for name, value in visible_activity)))
    if usage.calls:
        rows.append(("usage", f"calls `{usage.calls}`; total `{_status_token_count(usage.total_tokens)}`"))

    worker = loop.session.worker
    if worker is None:
        configured = loop.session.config.worker_provider
        rows.append(("worker", "`off` — `[worker] provider` " + (f"= `{configured}`" if configured else "unset")))
        return _status_render_table(rows)
    worker_usage = worker.usage
    state = f"`{'delegating' if worker._active_turn_messages else 'idle'}`, rounds `{worker.state.round_count}`"
    rows.append(("worker", _status_model_line(worker.config)))
    if worker_usage.last_prompt_tokens and worker_usage.last_prompt_budget:
        percent = min(100, worker_usage.last_prompt_tokens * 100 // worker_usage.last_prompt_budget)
        context = _status_context_line(worker_usage.last_prompt_tokens, worker_usage.last_prompt_budget, percent)
    else:
        context = "(no requests yet)"
    rows.append(("worker ctx", f"{context}; {state}"))
    if worker_usage.prompt_tokens:
        rows.append(("worker cache", _status_cache_line(worker_usage)))
    return _status_render_table(rows)


def skills_command(loop: CommandLoop, args: str) -> str:
    library = loop.session.skills
    skills = library.all() if library else []
    if not skills:
        return "No skills installed. Add `<name>/SKILL.md` under `.minacode/skills/` (project) or `~/.minacode/skills/` (user)."
    table = markdown_table(
        ["skill", "source", "description"],
        [(f"`{skill.name}`", skill.source, skill.description or "(no description)") for skill in skills],
    )
    return "\n".join([f"### Skills · {len(skills)}", "", "Load with `Skill(name)` or reference inline with `$name`.", "", table])


def ps_command(loop: CommandLoop, args: str) -> str:
    if args.strip():
        return "Usage: /ps"
    running = loop.session.running_jobs()
    if not running:
        total = len(loop.session.jobs)
        return f"No active jobs ({total} total)."
    rows = [(job.id, job.status, f"{job.elapsed():.1f}s", job.command[:80]) for job in running]
    table = markdown_table(["id", "status", "elapsed", "command"], rows)
    return f"### Active jobs · {len(running)}\n\n{table}"


def diff_command(loop: CommandLoop, args: str) -> str | None:
    if args.strip():
        return "Usage: /diff"
    if loop.interactive_input and loop.ui.color and (loop.tui is None or loop.tui.alternate_screen_available()):
        diff_viewer(loop)
        return None
    latest = loop.agent.session.latest_round_diff_sections()
    session = loop.agent.session.session_diff_sections()
    groups: list[tuple[str, list[tuple[str, str, str]]]] = []
    if latest is not None and latest[1]:
        round, sections = latest
        groups.append((f"Latest · Round {round}", sections))
    if session:
        groups.append(("Session", session))
    if not groups:
        return "No changes"
    lines: list[str] = []
    for title, sections in groups:
        lines.append("### " + title)
        for _status, path, diff in sections:
            lines.append(f"#### {path}")
            bounded, truncated = loop.bounded_diff(diff)
            lines.append(f"```diff\n{bounded}\n```")
            if truncated:
                lines.append("\n*Diff truncated. Full edit output is stored in the session.*")
    return "\n".join(lines)


def config(loop: CommandLoop, args: str) -> str:
    provider = loop.session.config.provider
    resolved = provider.resolve()
    configured_builtin_tools = ", ".join(str(entry.get("type") or "?") for entry in provider.builtin_tools) or "(off)"
    builtin_issue = builtin_tools_issue(resolved, provider.builtin_tools)
    if not provider.builtin_tools:
        resolved_builtin_tools = "(off)"
    elif builtin_issue is None:
        resolved_builtin_tools = "active: " + configured_builtin_tools
    elif builtin_issue.reason == "wire":
        resolved_builtin_tools = f"inactive on {resolved.api}: {configured_builtin_tools}"
    else:
        resolved_builtin_tools = "invalid: " + ", ".join(builtin_issue.configured)
    return "\n".join(
        [
            f"provider.active: {loop.session.config.active_provider}",
            f"provider.available: {', '.join(sorted(loop.session.config.providers))}",
            f"provider.url: {provider.url or '(empty)'}",
            f"provider.key: {'(set)' if provider.key else '(empty)'}",
            f"provider.model: {provider.model or '(empty)'}",
            f"provider.api: {provider.api}",
            f"provider.stream: {'on' if provider.stream else 'off'}",
            f"provider.image_input: {provider.image_input}",
            f"provider.resolved_api: {resolved.api}",
            f"provider.prompt_cache_key: {provider.prompt_cache_key}",
            f"provider.available_models: {', '.join(provider.available_models) or '(empty)'}",
            f"provider.reasoning: {provider.reasoning}",
            f"provider.resolved_reasoning_effort: {resolved.reasoning_effort or '(off)'}",
            f"provider.resolved_chat_reasoning: {resolved.chat_reasoning}",
            f"provider.chat_reasoning: {provider.chat_reasoning}",
            f"provider.temperature: {provider.temperature if provider.temperature is not None else '(off)'}",
            f"provider.max_tokens: {provider.max_tokens or '(server default)'}",
            f"provider.strict_tools: {provider.strict_tools} (active {resolved.strict_tools_active})",
            f"provider.extra_body: {json.dumps(provider.extra_body, ensure_ascii=False, sort_keys=True) if provider.extra_body else '(off)'}",
            f"provider.builtin_tools: {configured_builtin_tools}",
            f"provider.resolved_builtin_tools: {resolved_builtin_tools}",
            f"provider.timeout: {provider.timeout}",
            f"provider.response_timeout: {provider.response_timeout or '(off)'}",
            f"paths.data_dir: {loop.session.data_path()}",
            f"runtime.shell_timeout: {loop.session.settings.shell_timeout}",
            f"runtime.max_agent_steps: {loop.session.settings.max_steps}",
            f"runtime.max_context_tokens: {loop.session.settings.max_context_tokens}",
            f"runtime.max_parallel_tools: {loop.session.settings.max_parallel_tools}",
            f"runtime.session_retention_days: {loop.session.settings.session_retention_days}",
            f"runtime.yolo: {'on' if loop.session.settings.yolo else 'off'}",
            f"runtime.worker: {'on' if loop.session.settings.worker else 'off'}",
            f"runtime.language: {loop.session.settings.language}",
            f"worker.provider: {loop.session.config.worker_provider or '(off)'}",
            f"worker.model: {loop.session.config.worker_model or '(inherit)'}",
            f"worker.reasoning: {loop.session.config.worker_reasoning or '(inherit)'}",
            f"worker.api: {loop.session.config.worker_api or '(inherit)'}",
        ]
    )


def sessions_command(loop: CommandLoop, args: str) -> str | None:
    """Browse saved sessions and re-enter one. `/sessions all` widens past this project."""
    argument = args.strip().lower()
    if argument not in {"", "all"}:
        return "Usage: /sessions [all]"
    entries = SessionSnapshotStore.list_sessions(loop.session.config.data_dir, loop.session.cwd, all_projects=argument == "all")
    if not entries:
        return "No saved sessions yet."
    labels = {entry.uid: session_label(loop, entry, all_projects=argument == "all") for entry in entries}
    if loop.tui is None or not loop.interactive_input:
        return "\n".join(f"{entry.uid}  {labels[entry.uid]}" for entry in entries)
    title = "Sessions" + (" · all projects" if argument == "all" else "")
    # The preview renders on every frame, so it reads the list already in hand, never the store.
    by_uid = {entry.uid: entry for entry in entries}
    chosen = choice_application(
        loop, title, tuple(entry.uid for entry in entries), labels, loop.session.uid, set(), preview_fn=lambda uid: session_preview(loop, by_uid.get(uid))
    )
    if not isinstance(chosen, str) or chosen == loop.session.uid:
        return None
    loop.resume_request = chosen
    loop.save_and_emit_resume()
    return None


def session_label(loop: CommandLoop, entry: SessionEntry, *, all_projects: bool = False) -> str:
    rounds = f"{entry.rounds} round" + ("s" if entry.rounds > 1 else "") if entry.rounds else "no turns"
    parts = [Text.age(time.time() - entry.updated_at), rounds]
    if all_projects and entry.cwd:
        parts.append(os.path.basename(entry.cwd.rstrip(os.sep)) or entry.cwd)
    if entry.uid == loop.session.uid:
        parts.append("current")
    return f"{entry.label()}  ·  " + " · ".join(parts)


def session_preview(loop: CommandLoop, entry: SessionEntry | None) -> str:
    if entry is None:
        return ""
    return "\n".join([f"uid   {entry.uid}", f"start {entry.opening or '(no message)'}", f"where {entry.cwd or '(unknown)'}"])


def name_command(loop: CommandLoop, args: str) -> str:
    """Show or set the session's name, the label a later `--resume` can be given instead of a uid."""
    text = args.strip()
    if not text:
        current = loop.session.name
        source = {"user": "set by you", "goal": "from the current goal", "input": "from the opening message"}
        described = source.get(loop.session.state.name_source, "")
        return f"Session name: {current} ({described})" if current and described else f"Session name: {current or '(unnamed)'}"
    name = loop.session.rename(text)
    loop.session.save_snapshot()
    return f"Session named: {name}\nResume with: minacode --resume {shlex.quote(name)}"


def language_command(loop: CommandLoop, args: str) -> str:
    """Force this session's reply language, or show the current one. Session-scoped like other
    runtime switches: it never touches the config file, and workers inherit it from the parent."""
    if not args:
        current = loop.session.settings.language
        if current == "auto":
            return "Reply language: auto (follows your messages)"
        return f"Reply language: {current}"
    try:
        language = RuntimeSettings.clean_language(args)
    except ConfigError as error:
        return str(error)
    loop.session.settings.language = language
    if language == "auto":
        return "Reply language reset to auto"
    return f"Reply language set: {language}"


def compact(loop: CommandLoop, args: str) -> str:
    if args.strip():
        return "Usage: /compact"
    before = len(loop.session.messages)
    compacted, keep = loop.agent.context.compaction_parts()
    if not compacted:
        return "No prior conversation to compact"
    fallback = False
    fallback_error = ""
    loop.status_bar.begin()
    loop.compaction_active = True
    if loop.tui is not None:
        loop.tui.set_running("compacting context")
    else:
        loop.status_bar.start(reset=False)
    try:
        data = loop.agent.model.compact(loop.agent.context.compaction_input(compacted))
    except KeyboardInterrupt:
        return "Cancelled"
    except Exception as error:  # noqa: BLE001 - manual compaction uses the same deterministic fallback as automatic compaction.
        loop.agent.context.apply_compaction(None, keep, fallback_note=PREVIOUS_CONTEXT_TRIMMED, compacted=compacted)
        fallback = True
        fallback_error = Text.clip_width(" ".join(str(error).split()) or type(error).__name__, 220)
        data = None
    finally:
        loop.compaction_active = False
        if loop.tui is not None:
            loop.tui.set_dispatching()
        else:
            loop.status_bar.stop()
    if data is not None:
        loop.agent.context.apply_compaction(data, keep, compacted=compacted)
    loop.agent.context.update_current_tokens(loop.agent.session.system_prompt)
    # Compaction rewrites the history in place. Persist it now: leaving the session without
    # running another turn would otherwise resume from the log's pre-compaction state.
    loop.session.save_snapshot()
    fallback_note = f" (fallback: {fallback_error})" if fallback else ""
    return (
        f"Compacted context: messages {before} -> {len(loop.session.messages)}, "
        f"prior summary inserted, ctx {loop.session.state.context_percent}%{fallback_note}"
    )


def index(loop: CommandLoop, args: str) -> str:
    value = args.strip()
    if value not in {"", "force"}:
        return "Usage: /index [force]"
    try:
        loop.status_bar.start()
        return CodeIndex(loop.session).sync(force=value == "force")
    finally:
        loop.status_bar.stop()


def provider(loop: CommandLoop, args: str) -> str:
    parts = args.split()
    if len(parts) > 1:
        return "Usage: /provider [NAME]"
    if parts:
        return set_provider(loop, parts[0])
    choices = tuple(sorted(loop.session.config.providers))
    summary = "provider: " + loop.session.config.active_provider + "\nproviders: " + ", ".join(choices)
    current = loop.session.config.active_provider
    choice = select_choice(loop, "Provider", choices, labels={current: current + " (current)"}, current=current)
    if not isinstance(choice, str):
        return "No change" if choice is SELECTION_BACK else summary
    provider_result = set_provider(loop, choice)
    model_result = model(loop, "")
    return provider_result + ("\n" + model_result if model_result else "")


def set_provider(loop: CommandLoop, name: str) -> str:
    if name not in loop.session.config.providers:
        return "Unknown provider: " + name
    loop.session.config.active_provider = name
    return "Set provider = " + name


def model(loop: CommandLoop, args: str) -> str:
    parts = args.split()
    if len(parts) > 1:
        return "Usage: /model [MODEL]"
    if parts:
        result = set_model(loop, parts[0])
        return "No change" if result is SELECTION_BACK else str(result)
    provider = loop.session.config.provider
    configured = tuple(dict.fromkeys(provider.available_models))
    tui = loop.tui
    show_loading = tui is not None and bool(provider.url and provider.key)
    if show_loading and tui is not None:
        tui.set_dispatching("Loading models...")
    try:
        remote = tuple(model for model in remote_models(loop, provider) if model not in configured)
    finally:
        if show_loading and tui is not None:
            tui.set_dispatching()
    choices: list[str] = []
    if configured:
        choices.extend((MODEL_CONFIGURED_LABEL, *configured))
    if remote:
        choices.extend((MODEL_DISCOVERED_LABEL, *remote))
    choice_values = tuple(choices)
    if not choice_values:
        return "Current provider.model is " + (loop.session.config.provider.model or "(empty)")
    while True:
        current = loop.session.config.provider.model
        labels = {label: label for label in MODEL_LABELS if label in choice_values}
        labels.update({current: current + " (current)"} if current in choice_values else {})
        choice = select_choice(loop, "Model", choice_values, labels=labels, current=current, disabled=MODEL_LABELS)
        if choice is SELECTION_BACK:
            return "No change"
        if not isinstance(choice, str):
            return "Current provider.model is " + (loop.session.config.provider.model or "(empty)")
        if choice in MODEL_LABELS:
            continue
        result = set_model(loop, choice, back_to_model=True)
        if result is SELECTION_BACK:
            continue
        return str(result)


def remote_models(loop: CommandLoop, provider: ProviderConfig) -> tuple[str, ...]:
    if not provider.url or not provider.key:
        return ()
    try:
        # lazy import: /model discovery is the only OpenAI use here, so the SDK stays off the startup path
        from openai import OpenAI

        page = OpenAI(
            api_key=provider.key,
            base_url=provider.resolve().base_url,
            timeout=min(provider.timeout, 10),
            max_retries=0,
            default_headers={"User-Agent": HTTP_USER_AGENT},
        ).models.list()
    except Exception:  # noqa: BLE001 - remote model discovery is optional and provider SDKs expose varied failures.
        return ()
    names = []
    for item in getattr(page, "data", page) or []:
        name = item.get("id") if isinstance(item, dict) else getattr(item, "id", None)
        if isinstance(name, str) and name:
            names.append(name)
    return tuple(sorted(dict.fromkeys(names)))


def set_model(loop: CommandLoop, model: str, *, back_to_model: bool = False) -> str | object:
    while True:
        api = select_api(loop, model)
        if api is SELECTION_BACK:
            return SELECTION_BACK if back_to_model else "No change"
        reasoning = select_reasoning(loop)
        if reasoning is not SELECTION_BACK:
            break
    provider = loop.session.config.provider
    provider.model = model
    lines = ["Set provider.model = " + model]
    if isinstance(api, str):
        lines.append(set_api(loop, api))
    if isinstance(reasoning, str):
        provider.reasoning = reasoning
        lines.append("Set provider.reasoning = " + reasoning)
    return "\n".join(lines)


def reason(loop: CommandLoop, args: str) -> str:
    value = args.strip()
    if value:
        if value not in REASONING_CHOICES:
            return "Usage: /reason " + "|".join(REASONING_CHOICES)
        loop.session.config.provider.reasoning = value
        return "Set provider.reasoning = " + value
    choice = select_reasoning(loop)
    if isinstance(choice, str):
        loop.session.config.provider.reasoning = choice
        return "Set provider.reasoning = " + choice
    return "No change"


def api(loop: CommandLoop, args: str) -> str:
    value = args.strip()
    provider = loop.session.config.provider
    if value:
        if value not in PROVIDER_API_CHOICES:
            return "Usage: /api " + "|".join(PROVIDER_API_CHOICES)
        return set_api(loop, value)
    choice = select_api(loop, provider.model)
    return set_api(loop, choice) if isinstance(choice, str) else "No change"


def set_api(loop: CommandLoop, value: str) -> str:
    provider = loop.session.config.provider
    provider.api = value
    # "auto" is the usual choice, so name the wire it resolved to rather than echoing the setting back.
    resolved = provider.resolve()
    result = f"Set provider.api = {value} (wire: {resolved.api})"
    issue = builtin_tools_issue(resolved, provider.builtin_tools)
    if issue is not None:
        if issue.reason == "wire":
            result += f"; builtin_tools inactive on {resolved.api}"
        else:
            result += "; unsupported builtin_tools: " + ", ".join(issue.configured)
    return result


def yolo(loop: CommandLoop, args: str) -> str:
    loop.session.settings.yolo = not loop.session.settings.yolo
    return "yolo: " + ("on" if loop.session.settings.yolo else "off")


def hints(loop: CommandLoop, args: str) -> str:
    loop.session.settings.quick_hints = not loop.session.settings.quick_hints
    return "quick hints: " + ("on" if loop.session.settings.quick_hints else "off")


def strict(loop: CommandLoop, args: str) -> str:
    if args:
        return "Usage: /strict"
    provider = loop.session.config.provider
    provider.strict_tools = not provider.strict_tools
    state = "on" if provider.strict_tools else "off"
    if provider.strict_tools:
        resolved = provider.resolve()
        if not resolved.strict_tools_active:
            return f"strict_tools: {state} (inactive: {resolved.host or 'this provider'} does not support strict tool calling)"
    return f"strict_tools: {state}"


def set_value(loop: CommandLoop, args: str) -> str:
    key, _, value = args.partition(" ")
    if not key or not value:
        return "Usage: /set KEY VALUE"
    handler = SET_HANDLERS.get(key)
    if handler is None:
        return "Unknown config key: " + key
    target_name, attr, coerce = handler
    choices = SET_CHOICES.get(key)
    if choices is not None and value not in choices:
        return "Invalid value for " + key
    obj = loop.session.config.provider if target_name == "provider" else loop.session.settings
    try:
        if coerce is not None:
            value = coerce(value)
        setattr(obj, attr, value)
    except (ConfigError, ValueError):
        return "Invalid value for " + key
    return "Set " + key


# The single source of command metadata: dispatch, the completer's name tuple, and the
# queue-safe allowlist all derive from this registry (see CommandLoop.COMMANDS, COMMAND_LOOKUP,
# and QUEUE_SAFE_COMMANDS below).
# fmt: off


class WorkerFlow:
    """The multi-stage /worker configuration flow (provider -> model -> reason -> api).

    All stages go through the shared choice selector; backing out of any stage keeps the stages
    already set and reports what landed. Externally only `worker_command` is exposed.
    """

    def __init__(self, loop: CommandLoop) -> None:
        self.loop = loop


    def worker_command(self, args: str) -> str:
        parts = args.split()
        subcommand = parts[0].lower() if parts else ""
        rest = parts[1:]
        if subcommand == "reset" and not rest:
            result = DelegateTool(self.loop.session, [{"action": "reset"}]).call()
            if 'action="reset"' not in result:
                return result
            # The parent model does not know the user reset the worker; without this event the next
            # delegation would write "continue where you left off" against a clean context. Tail
            # append, ages with compaction, render-hidden, never filtered from the model history.
            self.loop.session.messages.append(
                {
                    "role": "user",
                    "content": "[Worker context was reset by the user. The next delegation starts from scratch.]",
                    SESSION_EVENT_KEY: "worker_reset",
                }
            )
            self.loop.session.save_snapshot()
            if 'alive="false"' in result:
                return "[worker] reset · no worker session to reset."
            return "[worker] reset · worker context cleared; file changes and merged diffs kept. The next delegation starts from scratch."
        if subcommand == "on" and not rest:
            self.loop.session.settings.worker = True
            return "worker: on (the tool block changes, so the prompt-cache scope is recompiled once)"
        if subcommand == "off" and not rest:
            self.loop.session.settings.worker = False
            return "worker: off (the worker's context stays on disk; /worker on resumes it)"
        if subcommand in {"", "status"} and not rest:
            return self._worker_status()
        if subcommand == "provider":
            if len(rest) > 1:
                return "Usage: /worker provider [NAME]"
            if not rest:
                return self._worker_provider_picker()
            return self._worker_set_provider(rest[0])
        if subcommand == "model":
            if len(rest) > 1:
                return "Usage: /worker model [MODEL]"
            if not rest:
                return self._worker_model_picker()
            return self._worker_set_model(rest[0])
        if subcommand == "reason":
            if len(rest) > 1:
                return "Usage: /worker reason [EFFORT]"
            if not rest:
                return self._worker_reason_picker()
            return self._worker_set_reasoning(rest[0])
        if subcommand == "api":
            if len(rest) > 1:
                return "Usage: /worker api [API]"
            if not rest:
                return self._worker_api_picker()
            return self._worker_set_api(rest[0])
        return "Usage: /worker [" + "|".join(WORKER_SUBCOMMANDS) + "]"


    def _worker_status(self) -> str:
        """Readable /worker status for the human; the model-facing envelope stays in DelegateTool."""
        worker = self.loop.session.worker
        if worker is None:
            return "worker: no active session\nworker provider: " + (self.loop.session.config.worker_provider or "(off)")
        usage = worker.usage
        percent = min(100, usage.last_prompt_tokens * 100 // usage.last_prompt_budget) if usage.last_prompt_budget else worker.state.context_percent
        provider = worker.config.provider
        state = "delegating" if worker._active_turn_messages else "idle"
        return "\n".join(
            [
                f"worker: {worker.config.active_provider}/{provider.model or '(no model)'}",
                "worker reasoning: " + provider.reasoning,
                "worker state: " + state,
                "worker rounds: " + str(worker.state.round_count),
                "worker context: " + str(percent) + "%",
            ]
        )


    def _worker_provider_picker(self) -> str:
        summary = "worker provider: " + (self.loop.session.config.worker_provider or "(off)") + "\nproviders: " + ", ".join(sorted(self.loop.session.config.providers))
        choices = tuple(sorted(self.loop.session.config.providers))
        if "off" not in choices:
            choices = (*choices, "off")
        current = self.loop.session.config.worker_provider
        choice = select_choice(self.loop, "Worker provider", choices, labels={current: current + " (current)"} if current else {}, current=current)
        if not isinstance(choice, str):
            return "No change" if choice is SELECTION_BACK else summary
        provider_result = self._worker_set_provider(choice)
        if self.loop.session.config.worker_provider != choice:
            # Picking "off" cleared the entry (or the set failed): there is no newly selected
            # provider entry to pick a model for, so the cascade stops after the provider set.
            return provider_result
        # One setup flow, like /provider: worker provider -> worker model -> worker reasoning.
        # Backing out of any stage keeps the stages already set and reports what landed.
        lines = [provider_result]
        set_ok, model_result = self._worker_model_stage()
        if not set_ok:
            lines.append("worker model: unchanged")
            return "\n".join(lines)
        lines.append(model_result)
        set_ok, reason_result = self._worker_reason_stage()
        if not set_ok:
            lines.append("worker reasoning: unchanged")
            return "\n".join(lines)
        lines.append(reason_result)
        return "\n".join(lines)


    def _worker_set_provider(self, name: str) -> str:
        if name == "off" and "off" not in self.loop.session.config.providers:
            # "off" names the clearing action unless a provider entry is literally named "off"
            # (existence in config.providers wins). The Delegate gate is frozen per session, so
            # this only clears the next spawn's provider; the live worker keeps running on its
            # current provider and the tool block never flips mid-session.
            self.loop.session.config.worker_provider = ""
            return "worker provider: off"
        if name not in self.loop.session.config.providers:
            return "Unknown provider: " + name
        self.loop.session.config.worker_provider = name
        refresh_worker_entry(self.loop.session.config, self.loop.session.worker, name)
        result = "Set worker provider = " + name
        if not self.loop.session.worker_tool_enabled:
            # Delegation was off at session start: the frozen gate keeps the tool block off no
            # matter what the live config says, so the change only counts after a restart.
            result += " (delegation is off this session; takes effect after a restart)"
        return result


    def _worker_simple_field(
        self,
        *,
        title: str,
        label: str,
        choices: tuple[str, ...],
        current: str,
        labels: dict[str, str],
        apply: Callable[[str], str],
    ) -> tuple[bool, str]:
        """Pick one value through the shared selector and apply it; returns (set, message) so both
        the standalone /worker pickers and the /worker provider cascade can tell a set from an
        abort. Shared by worker model/reasoning/api: same shape, no cascade, and each `apply`
        writes the config and refreshes a live worker itself."""
        choice = select_choice(self.loop, title, choices, labels=labels, current=current)
        if not isinstance(choice, str):
            return False, ("No change" if choice is SELECTION_BACK else (f"{label}: " + (current or "(inherit)")))
        return True, apply(choice)


    def _worker_model_picker(self) -> str:
        """Standalone /worker model picker: one selection, no cascade."""
        return self._worker_model_stage()[1]


    def _worker_model_stage(self) -> tuple[bool, str]:
        """Pick a worker model override; returns (set, message). Shared by /worker model and the
        /worker provider cascade so the cascade can tell a set from an abort."""
        entry = self.loop.session.config.providers[self.loop.session.config.worker_provider or self.loop.session.config.active_provider]
        configured = tuple(dict.fromkeys(entry.available_models))
        remote = tuple(model for model in remote_models(self.loop, entry) if model not in configured)
        override = self.loop.session.config.worker_model
        choices = [*configured, *remote]
        if override and override not in choices:
            choices.append(override)
        choices.append("default")
        choice_values = tuple(dict.fromkeys(choices))
        labels = {override: override + " (current)"} if override in choice_values else {}
        labels["default"] = "default - inherit the provider entry's model"
        return self._worker_simple_field(
            title="Worker model", label="worker model", choices=choice_values, current=override, labels=labels, apply=self._worker_set_model
        )


    def _worker_set_model(self, value: str) -> str:
        if value != "default":
            self.loop.session.config.worker_model = value
        else:
            self.loop.session.config.worker_model = ""
        refresh_worker_entry(self.loop.session.config, self.loop.session.worker)
        if value == "default":
            return "worker model: (inherit)"
        return "Set worker.model = " + value


    def _worker_reason_picker(self) -> str:
        """Standalone /worker reason picker: one selection, no cascade."""
        return self._worker_reason_stage()[1]


    def _worker_reason_stage(self) -> tuple[bool, str]:
        """Pick a worker reasoning effort; returns (set, message). Shared by /worker reason and
        the /worker provider cascade."""
        current = self.loop.session.config.worker_reasoning
        choices = (*REASONING_CHOICES, "default")
        labels = {"default": "default - inherit the provider entry's reasoning"}
        if current:
            labels[current] = current + " (current)"
        return self._worker_simple_field(
            title="Worker reasoning", label="worker reasoning", choices=choices, current=current, labels=labels, apply=self._worker_set_reasoning
        )


    def _worker_set_reasoning(self, value: str) -> str:
        if value != "default":
            # "off" is a valid effort, never the clearing word; only "default" clears.
            if value not in REASONING_CHOICES:
                return "Usage: /worker reason " + "|".join(REASONING_CHOICES)
            self.loop.session.config.worker_reasoning = value
        else:
            self.loop.session.config.worker_reasoning = ""
        refresh_worker_entry(self.loop.session.config, self.loop.session.worker)
        if value == "default":
            return "worker reasoning: (inherit)"
        return "Set worker.reasoning = " + value


    def _worker_api_picker(self) -> str:
        """Standalone /worker api picker: one selection, no cascade."""
        current = self.loop.session.config.worker_api
        choices = (*PROVIDER_API_CHOICES, "default")
        labels = {"default": "default - inherit the provider entry's api"}
        if current:
            labels[current] = current + " (current)"
        return self._worker_simple_field(title="Worker api", label="worker api", choices=choices, current=current, labels=labels, apply=self._worker_set_api)[1]


    def _worker_set_api(self, value: str) -> str:
        if value != "default":
            if value not in PROVIDER_API_CHOICES:
                return "Usage: /worker api " + "|".join(PROVIDER_API_CHOICES)
            self.loop.session.config.worker_api = value
        else:
            self.loop.session.config.worker_api = ""
        refresh_worker_entry(self.loop.session.config, self.loop.session.worker)
        if value == "default":
            return "worker api: (inherit)"
        return "Set worker.api = " + value


    def run_worker_config(self) -> None:
        """The Delegate confirm-time `c` loop: pick which worker knob to adjust with the shared
        choice selector (each field labeled with its current value, done preselected), then drive
        the corresponding /worker picker -- which writes the config and refreshes a live worker
        itself. done or Esc returns to the confirmation prompt; non-interactive runs return at
        once (select_choice yields nothing)."""
        while True:
            config = self.loop.session.config
            provider_name = config.worker_provider or config.active_provider
            entry = config.providers[provider_name]
            provider_value = config.worker_provider or f"(inherit) {provider_name}"
            model_value = config.worker_model or f"(inherit) {entry.model or '(no model)'}"
            effort_value = config.worker_reasoning or f"(inherit) {entry.reasoning}"
            api_value = config.worker_api or f"(inherit) {entry.api}"
            labels = {
                "provider": f"provider: {provider_value}",
                "model": f"model: {model_value}",
                "effort": f"effort: {effort_value}",
                "api": f"api: {api_value}",
                "done": "done - return to the confirmation prompt",
            }
            choice = select_choice(self.loop, "Worker config", ("provider", "model", "effort", "api", "done"), labels=labels, current="done")
            if choice == "provider":
                self._worker_provider_picker()
            elif choice == "model":
                self._worker_model_picker()
            elif choice == "effort":
                self._worker_reason_picker()
            elif choice == "api":
                self._worker_api_picker()
            else:
                return


def worker_command(loop: CommandLoop, args: str) -> str:
    """Dispatch a /worker subcommand through a fresh WorkerFlow."""
    return WorkerFlow(loop).worker_command(args)


MODEL_CONFIGURED_LABEL = "---- Configured models ----"
MODEL_DISCOVERED_LABEL = "---- Discovered models ----"
MODEL_LABELS = frozenset((MODEL_CONFIGURED_LABEL, MODEL_DISCOVERED_LABEL))
MCP_COMMANDS: dict[str, tuple[int, int, str]] = {
    "connect": (1, sys.maxsize, "Usage: /mcp connect <server> [server ...]"),
    "disconnect": (1, 1, "Usage: /mcp disconnect <server>"),
    "tools": (0, 1, "Usage: /mcp tools [server]"),
}
MCP_HELP = "Try /mcp, /mcp connect <server> [server ...], /mcp disconnect <server>, or /mcp tools [server]"
