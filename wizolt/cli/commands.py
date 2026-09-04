"""Slash command implementations as free functions taking the CommandLoop.

Each handler takes `(loop, args)` and is referenced directly by the registry in
`wizolt/cli/__init__.py`. Handlers that await a modal or the network are coroutines; the dispatcher
accepts either shape. A `None` result means the handler rendered its own UI (e.g. /diff's viewer).
The multi-stage /worker flow lives in the WorkerFlow class below.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import shutil
import sys
import time
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING

from prompt_toolkit.utils import get_cwidth

from wizolt import compaction
from wizolt.base import (
    SELECTION_BACK,
    ConfigError,
    LogBlock,
    LogEdge,
    LogLine,
    LogRole,
    ModelUsage,
    Text,
)
from wizolt.cli.modals import (
    choice_application,
    compaction_log_viewer,
    diff_viewer,
    mcp_manager,
    missing_summary_note,
    segment_columns,
    select_choice,
)
from wizolt.cli.update import UpdateChecker
from wizolt.config import (
    PROVIDER_API_CHOICES,
    Config,
    ProviderConfig,
    RuntimeSettings,
    compaction_provider_config,
)
from wizolt.prompts import PREVIOUS_CONTEXT_TRIMMED
from wizolt.providers.compat import builtin_tools_issue
from wizolt.providers.schema import CatalogSyncError
from wizolt.providers.sync import CATALOG_URL
from wizolt.render import markdown_table, progress_bar
from wizolt.session import Session, SessionEntry, SessionSnapshotStore
from wizolt.tools import CodeIndex

if TYPE_CHECKING:
    from prompt_toolkit.formatted_text import StyleAndTextTuples

    from wizolt.cli import CommandLoop
    from wizolt.session import Session

# fmt: off


SetHandler = tuple[str, str, Callable[[str], int | float | None] | None]

# fmt: off
SET_HANDLERS: dict[str, SetHandler] = {
    "provider.temperature": ("provider", "temperature", lambda v: None if v == "off" else float(v)),
    "provider.max_tokens": ("provider", "max_tokens", lambda v: max(0, int(v))),
    "provider.max_context_tokens": ("provider", "max_context_tokens", lambda v: max(0, int(v))),
    "provider.timeout": ("provider", "timeout", lambda v: max(1, int(v))),
    "provider.response_timeout": ("provider", "response_timeout", lambda v: max(0, int(v))),
    "provider.stream": ("provider", "stream", lambda v: v == "on"),
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
    "runtime.worker": ("on", "off"),
}
SET_VALUES: dict[str, tuple[str, ...]] = {
    "provider.temperature": ("off",),
    **SET_CHOICES,
}
# fmt: on


def _status_model_line(session: Session, config: Config) -> str:
    active = config.provider
    return f"`{config.active_provider}/{active.model or '(empty)'}`; `{session.policy.resolve(active).api}`; `{active.reasoning}`"


def _status_context_line(tokens: int, budget: int, percent: int) -> str:
    return f"`{progress_bar(tokens, budget)}` `~{Text.abbreviate_count(tokens)} / {Text.abbreviate_count(budget)}` (`{percent}%`)"


def _status_cache_line(counts: ModelUsage) -> str:
    # The read ratios carry the useful signal; the raw token pairs made this the one row that
    # wrapped on a normal terminal. Writes stay, but only when there were any.
    def part(label: str, cached: int, prompt: int, written: int) -> str:
        write = f" (w {Text.abbreviate_count(written)})" if written else ""
        return f"{label} `{cached * 100 / prompt:.1f}%`{write}"

    return " ".join(
        [
            f"`{progress_bar(counts.last_cached_prompt_tokens, counts.last_prompt_tokens)}`",
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


async def mcp_command(loop: CommandLoop, args: str) -> str | None:
    mcp = loop.session.mcp
    if mcp is None:
        return "MCP not configured"

    parts = args.split()
    if not parts:
        if loop.tui is not None and loop.tui.input_mode != "running":
            return await mcp_manager(loop)
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
        return await mcp.connect_servers(rest, interactive=loop.interactive_input, notify=loop.emit)
    if sub == "disconnect":
        return mcp.disconnect_server(rest[0])
    if sub == "tools":
        return mcp.render_tool_listing(rest[0] if rest else None)
    raise AssertionError("unreachable MCP subcommand")


async def select_reasoning(loop: CommandLoop, model: str = "") -> str | object | None:
    """Offer the efforts `model` accepts — the entry's own model when none is named.

    The list is the model's scale, not wizolt's: a level this model has no spelling for is not
    a choice, so nothing has to be rewritten between picking it and sending it. `model` is passed
    when the effort is being chosen for a model the entry has not switched to yet."""
    provider = loop.session.config.provider
    current = provider.reasoning
    policy = loop.session.policy
    choices = policy.reasoning_choices(provider, model)
    labels = {"off": "off - disable reasoning"}
    labels[current] = labels.get(current, current) + " (current)"
    # A shortened list raises the question the same screen should answer, so the reason it is
    # short is shown under it with the page it came from. The same footer under every row: it is
    # about the list, not about whichever level the cursor happens to be on. It opens by naming
    # itself, since text appearing under a list of choices otherwise reads as being about the
    # choice rather than about the list.
    why, evidence = loop.session.policy.effort_source(provider, model)
    # The explanation has its own quiet informational tone: visible enough to read, while the
    # selected row remains the strongest element in the modal.
    footer: StyleAndTextTuples = [("class:choice.explanation", "  │ " + line + "\n") for line in ("Why these levels", why, evidence) if line]
    return await select_choice(loop, "Reasoning effort", choices, labels=labels, current=current, preview_fn=(lambda _choice: footer) if why else None)


async def select_api(loop: CommandLoop, model: str) -> str | object | None:
    # An endpoint that lists several model families rarely serves them all over one protocol, and
    # a /models listing does not say which. Confirm the wire alongside the model that needs it.
    provider = loop.session.config.provider
    current = provider.api
    inferred = loop.session.policy.resolve(replace(provider, api="auto", model=model)).api
    labels = {"auto": f"auto - infer from the endpoint URL and model ({inferred})"}
    labels[current] = labels.get(current, current) + " (current)"
    return await select_choice(loop, "Request API", PROVIDER_API_CHOICES, labels=labels, current=current)


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
    context_percent = usage.context_percent(loop.session.state.context_percent)
    if usage.last_prompt_tokens and usage.last_prompt_budget:
        # Display the provider-reported tokens and the budget of the last request alongside the
        # percentage; `/config` reports the configured max_context_tokens immediately, while these
        # describe one real request and only catch up after the next one.
        context_tokens = usage.last_prompt_tokens
        context_budget = usage.last_prompt_budget
    index = CodeIndex(loop.session)
    index_status, index_message = index.status(check=False)
    index.schedule_pending_update()
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
    info = loop.session.system_info
    if loop.session.settings.agents_md:
        source = info.agents_md_source if info is not None else ""
        runtime.append(f"agents_md on ({source})" if source else "agents_md on (none)")
    else:
        runtime.append("agents_md off")
    update = UpdateChecker(loop.session).status_line().removeprefix("update: ")
    if update not in {"current", "unknown"}:
        runtime.append("update " + update)
    rows.append(("runtime", "; ".join(f"`{value}`" for value in runtime)))

    rows.append(("model", _status_model_line(loop.session, loop.session.config)))
    rows.append(("context", _status_context_line(context_tokens, context_budget, context_percent)))
    rows.append(("cache", _status_cache_line(usage) if usage.prompt_tokens else "(no requests yet)"))
    visible_activity = [(name, value) for name, value in activity if value]
    if visible_activity:
        rows.append(("activity", "; ".join(f"{name} `{value}`" for name, value in visible_activity)))
    if usage.calls:
        rows.append(("usage", f"calls `{usage.calls}`; total `{Text.abbreviate_count(usage.total_tokens)}`"))
    # Summaries are counted apart from the conversation, so each row can be multiplied by one
    # price: the entry they run on may be another account entirely. The row names the model for
    # the same reason, and stays hidden until a summary has actually run.
    compaction_usage = loop.session.compaction_usage
    if compaction_usage.calls:
        compaction_model = compaction_provider_config(loop.session.config).model or "(no model)"
        rows.append(
            (
                "compaction usage",
                f"calls `{compaction_usage.calls}`; total `{Text.abbreviate_count(compaction_usage.total_tokens)}`; `{compaction_model}`",
            )
        )
        # The summary request is built to ride the conversation's own cached prefix, and this is
        # the only place that says whether it did. Without it the reuse is unfalsifiable.
        rows.append(("compaction cache", _status_cache_line(compaction_usage)))

    worker = loop.session.worker
    if worker is None:
        configured = loop.session.config.worker_provider
        rows.append(("worker", "`off` — `[worker] provider` " + (f"= `{configured}`" if configured else "unset")))
        return markdown_table(["field", "value"], rows)
    worker_usage = worker.usage
    state = f"`{'delegating' if worker._active_turn_messages else 'idle'}`, rounds `{worker.state.round_count}`"
    rows.append(("worker", _status_model_line(worker, worker.config)))
    if worker_usage.last_prompt_tokens and worker_usage.last_prompt_budget:
        percent = worker_usage.context_percent()
        context = _status_context_line(worker_usage.last_prompt_tokens, worker_usage.last_prompt_budget, percent)
    else:
        context = "(no requests yet)"
    rows.append(("worker ctx", f"{context}; {state}"))
    if worker_usage.prompt_tokens:
        rows.append(("worker cache", _status_cache_line(worker_usage)))
    return markdown_table(["field", "value"], rows)


async def catalog_command(loop: CommandLoop, args: str) -> str:
    """Show the active provider catalog, or force a remote refresh with ``sync``.

    Async because ``sync`` reaches the network behind a cross-process lock: awaited here, the
    prompt and the status line stay live for the seconds a slow remote can take."""

    parts = args.split()
    if parts not in ([], ["status"], ["sync"]):
        return "Usage: /catalog [status|sync]"
    catalog = loop.session.catalog
    if catalog is None:
        return "catalog: bundled (no session catalog yet)"
    if parts == ["sync"]:
        previous = catalog.snapshot.version
        try:
            snapshot = await catalog.sync()
        except CatalogSyncError as error:
            return f"catalog sync failed: {error}"
        if snapshot.version > previous:
            return f"catalog synced: activated version `{snapshot.version}` from `{catalog.source}` (previous `{previous}`)"
        return f"catalog sync: current (no newer catalog; active is `{snapshot.version}`)"
    state = catalog.sync_state
    rows = [
        ("version", str(catalog.snapshot.version)),
        ("source", "`" + catalog.source + "`"),
        ("updated", catalog.snapshot.updated_at.isoformat()),
        ("schema", str(catalog.snapshot.schema_version)),
        ("scope", catalog.snapshot.maintenance_scope),
    ]
    bundled_version, cached_version = catalog.source_versions()
    rows.append(("bundled", str(bundled_version)))
    rows.append(("cached", str(cached_version) if cached_version is not None else "none"))
    if catalog.note:
        rows.append(("note", catalog.note))
    if state.error:
        rows.append(("sync", "error: " + state.error))
    elif state.checking:
        rows.append(("sync", "checking..."))
    elif state.last_synced_at:
        rows.append(("sync", "last " + time.strftime("%Y-%m-%d %H:%M", time.localtime(state.last_synced_at))))
    rows.append(("remote", CATALOG_URL))
    rows.append(("hint", "run `/catalog sync` to force a remote check"))
    return markdown_table(["field", "value"], rows)


def skills_command(loop: CommandLoop, args: str) -> str:
    library = loop.session.skills
    skills = library.all() if library else []
    if not skills:
        return "No skills installed. Add `<name>/SKILL.md` under `.wizolt/skills/` (project) or `~/.wizolt/skills/` (user)."
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


async def diff_command(loop: CommandLoop, args: str) -> str | None:
    if args.strip():
        return "Usage: /diff"
    if loop.interactive_input and loop.ui.color and (loop.tui is None or await loop.tui.alternate_screen_available()):
        await diff_viewer(loop)
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
        for _, path, diff in sections:
            lines.append(f"#### {path}")
            bounded, truncated = loop.bounded_diff(diff)
            lines.append(f"```diff\n{bounded}\n```")
            if truncated:
                lines.append("\n*Diff truncated. Full edit output is stored in the session.*")
    return "\n".join(lines)


def config(loop: CommandLoop, args: str) -> str:
    provider = loop.session.config.provider
    resolved = loop.session.policy.resolve(provider)
    compaction_effective = compaction_provider_config(loop.session.config)
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
            f"provider.resolved_api: {resolved.api}",
            f"provider.prompt_cache_key: {provider.prompt_cache_key}",
            f"provider.available_models: {', '.join(provider.available_models) or '(empty)'}",
            f"provider.reasoning: {provider.reasoning}",
            f"provider.resolved_reasoning_effort: {resolved.reasoning_effort or '(off)'}",
            f"provider.supported_reasoning: {', '.join(loop.session.policy.reasoning_choices(provider))}",
            f"provider.resolved_chat_reasoning: {resolved.chat_reasoning}",
            f"provider.chat_reasoning: {provider.chat_reasoning}",
            f"provider.reasoning_history: {provider.reasoning_history}",
            f"provider.resolved_reasoning_history: {resolved.reasoning_history}",
            f"provider.temperature: {provider.temperature if provider.temperature is not None else '(off)'}",
            f"provider.max_tokens: {provider.max_tokens or '(server default)'}",
            # Show the effective limit either way: the whole point of the key is which number the
            # compaction budget is measured against, and "(inherit)" alone does not answer that.
            f"provider.max_context_tokens: {provider.max_context_tokens or f'(inherit) {loop.session.settings.max_context_tokens}'}",
            f"provider.strict_tools: {provider.strict_tools} (active {resolved.strict_tools_active})",
            f"provider.extra_body: {json.dumps(provider.extra_body, ensure_ascii=False, sort_keys=True) if provider.extra_body else '(off)'}",
            f"provider.headers: {', '.join(f'{name}: {value}' for name, value in sorted(provider.headers.items())) or '(none)'}",
            f"provider.omit_body: {', '.join(provider.omit_body) or '(none)'}",
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
            f"runtime.agents_md: {'on' if loop.session.settings.agents_md else 'off'}",
            f"worker.provider: {loop.session.config.worker_provider or '(off)'}",
            f"worker.model: {loop.session.config.worker_model or '(inherit)'}",
            f"worker.reasoning: {loop.session.config.worker_reasoning or '(inherit)'}",
            f"worker.api: {loop.session.config.worker_api or '(inherit)'}",
            f"compaction.provider: {loop.session.config.compaction_provider or loop.session.config.active_provider}",
            f"compaction.model: {compaction_effective.model}",
            f"compaction.reasoning: {compaction_effective.reasoning}",
            f"compaction.api: {compaction_effective.api}",
        ]
    )


async def sessions_command(loop: CommandLoop, args: str) -> str | None:
    """Browse saved sessions and re-enter one. `/sessions all` widens past this project."""
    argument = args.strip().lower()
    if argument not in {"", "all"}:
        return "Usage: /sessions [all]"
    entries = SessionSnapshotStore.list_sessions(loop.session.config.data_dir, loop.session.cwd, all_projects=argument == "all")
    if not entries:
        return "No saved sessions yet."
    table, widths = session_table(loop, entries, all_projects=argument == "all")
    rows = session_rows(table, widths)
    if loop.tui is None or not loop.interactive_input:
        uid_width = max(get_cwidth(entry.uid) for entry in entries)
        return "\n".join(f"{entry.uid}{' ' * (uid_width - get_cwidth(entry.uid))}  {label}" for entry, label in zip(entries, rows))
    labels = {entry.uid: label for entry, label in zip(entries, rows)}
    title = "Sessions" + (" · all projects" if argument == "all" else "")
    # The preview renders on every frame, so it reads the list already in hand, never the store.
    # Each session's summary is read lazily the first time the cursor lands on it and cached, so
    # opening the picker costs nothing and a huge log is only read for sessions you actually look
    # at.
    by_uid = {entry.uid: entry for entry in entries}
    fields_by_uid = {entry.uid: row for entry, row in zip(entries, table)}
    height = shutil.get_terminal_size().lines
    summaries: dict[str, list[tuple[str, str]]] = {}

    def preview_fn(uid: str) -> StyleAndTextTuples:
        entry = by_uid.get(uid)
        if entry is None:
            return []
        if uid not in summaries:
            summaries[uid] = SessionSnapshotStore.tail_summary(entry.path)
        return session_preview(entry, summary=summaries[uid])

    chosen = await choice_application(
        loop,
        title,
        tuple(entry.uid for entry in entries),
        labels,
        loop.session.uid,
        set(),
        preview_fn=preview_fn,
        label_fn=session_label_fn(fields_by_uid, widths),
        exclusive=True,
        # A viewport over the list, like the Ctrl-O browser's: the picker fills the terminal, so
        # beyond this many rows the list scrolls instead of pushing the preview off the screen.
        max_rows=max(5, min(20, height - 12)),
    )
    if not isinstance(chosen, str) or chosen == loop.session.uid:
        return None
    loop.resume_request = chosen
    loop.save_and_emit_resume()
    return None


def _session_fields(loop: CommandLoop, entry: SessionEntry, *, all_projects: bool) -> list[str]:
    """One session's fields in display order: name, age, round count, then the project when
    browsing all projects and a `current` marker for the live session."""
    rounds = f"{entry.rounds} round{'s' if entry.rounds > 1 else ''}" if entry.rounds else "no turns"
    fields = [entry.label(), Text.age(time.time() - entry.updated_at), rounds]
    if all_projects and entry.cwd:
        fields.append(os.path.basename(entry.cwd.rstrip(os.sep)) or entry.cwd)
    if entry.uid == loop.session.uid:
        fields.append("current")
    return fields


def session_table(loop: CommandLoop, entries: list[SessionEntry], *, all_projects: bool = False) -> tuple[list[list[str]], list[int]]:
    """Each session's fields plus every column's display width, so the same table can be laid out
    as plain text or as styled fragments without recomputing the padding."""
    rows = [_session_fields(loop, entry, all_projects=all_projects) for entry in entries]
    widths = [0] * max((len(row) for row in rows), default=0)
    for row in rows:
        for index, field in enumerate(row):
            widths[index] = max(widths[index], get_cwidth(field))
    return rows, widths


def session_rows(rows: list[list[str]], widths: list[int]) -> list[str]:
    """The session list as table rows: every column padded to the widest value in it, so names,
    ages, and round counts line up in the picker instead of drifting with the label lengths.
    Padding is measured in display cells, so CJK names align too."""
    lines = []
    for row in rows:
        cells = [field if index == len(row) - 1 else field + " " * max(0, widths[index] - get_cwidth(field)) for index, field in enumerate(row)]
        lines.append("  ".join(cells))
    return lines


def session_label_fn(fields_by_uid: dict[str, list[str]], widths: list[int]) -> Callable[[str], StyleAndTextTuples]:
    """Style one session row for the picker: the name plain, the age and round count dim, the
    current session's marker in the live colour. Columns keep the same padding as the text layout."""

    def label_fn(uid: str) -> StyleAndTextTuples:
        fields = fields_by_uid[uid]
        parts: StyleAndTextTuples = []
        for index, field in enumerate(fields):
            text = field if index == len(fields) - 1 else field + " " * max(0, widths[index] - get_cwidth(field))
            if index == 0:
                style = ""
            elif field == "current":
                style = "class:choice.live"
            else:
                style = "class:choice.meta"
            parts.append((style, text + ("  " if index < len(fields) - 1 else "")))
        return parts

    return label_fn


def session_preview(entry: SessionEntry, *, summary: list[tuple[str, str]] | None = None) -> StyleAndTextTuples:
    """The picker's preview as fragments, laid out like the transcript itself: a user message is
    its bullet plus the same warm tone the transcript uses, an assistant reply sits indented in
    the default colour, and a collapsed tool line is dimmed. Newest exchange at the bottom, the
    way a conversation reads."""
    if not summary:
        return []
    width = max(40, shutil.get_terminal_size((120, 24)).columns - 4)
    messages = list(summary)
    # A tool-heavy session's newest turns are almost all assistant text; without a user message
    # the preview gives no hint of what the conversation was about. Anchor it with the opening
    # question when the recent window is all replies.
    if not any(role == "user" for role, _ in messages) and entry.opening:
        messages.append(("user", entry.opening))
    parts: StyleAndTextTuples = []
    for role, text in reversed(messages):
        line = Text.clip_width(text.replace("\n", " "), width)
        if role == "user":
            parts.append(("", "\n"))
            parts.append(("class:prompt", "• "))
            parts.append(("class:choice.user", line))
            parts.append(("", "\n"))
        elif role == "tool":
            parts.append(("class:choice.meta", "  " + line + "\n"))
        else:
            parts.append(("", "  " + line + "\n"))
    return parts


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
    return f"Session named: {name}\nResume with: wizolt --resume {shlex.quote(name)}"


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


async def compaction_log(loop: CommandLoop, args: str) -> str | LogBlock | None:
    """`/compact log [seg.N]`: review what compaction kept. The viewer is the interactive form; a
    headless run (piped input, no color, no alternate screen) gets the same segments as log lines,
    and naming one segment prints its whole summary, which the list can only show a title of.

    Neither form prints the stored excerpt. It is the raw conversation the summary already stands
    for, kept for the model to reach through RecallContext; paging it past a reader hides the one
    line they came for."""
    key = args.strip()
    segments = loop.session.history
    if key:
        segment = next((item for item in segments if item.key == key), None)
        if segment is None:
            return f"No stored segment {key}" if key.startswith("seg.") else "Usage: /compact log [seg.N]"
        when, kind, messages = segment_columns(segment)
        model = f" · model {segment.model}" if segment.model else ""
        return LogBlock(
            [
                LogLine(segment.key, segment.title, LogRole.FIELD, LogEdge.BRANCH),
                LogLine("compaction", f"{when} · {kind} · {messages}{model}", LogRole.FIELD, LogEdge.CONTINUE),
                *(
                    LogLine("summary" if index == 0 else "", line, LogRole.FIELD, LogEdge.CONTINUE)
                    for index, line in enumerate((segment.summary or missing_summary_note(segment)).splitlines())
                ),
            ]
        )
    if not segments:
        return "No compaction has stored a segment yet"
    # A TUI is required, not just assumed from interactive input: without one the viewer would
    # render nothing at all, and the log lines below say the same thing without a screen.
    if loop.interactive_input and loop.ui.color and loop.tui is not None and await loop.tui.alternate_screen_available():
        await compaction_log_viewer(loop)
        return None
    count = loop.session.state.compaction_count
    return LogBlock(
        [
            LogLine("compaction log", f"{count} compactions · {len(segments)} stored segments", LogRole.FIELD, LogEdge.BRANCH),
            *(
                LogLine(
                    segment.key,
                    " · ".join((*segment_columns(segment), *((f"model {segment.model}",) if segment.model else ()), segment.title)),
                    LogRole.FIELD,
                    LogEdge.CONTINUE,
                )
                for segment in reversed(segments)
            ),
        ]
    )


async def compact(loop: CommandLoop, args: str) -> str | LogBlock | None:
    """`/compact`: the one command that reaches the provider, so the only awaited one."""
    sub, _, rest = args.strip().partition(" ")
    if sub == "log":
        return await compaction_log(loop, rest)
    if args.strip():
        return "Usage: /compact [log [seg.N]]"
    before = len(loop.session.messages)
    compactor = compaction.Compactor(loop.agent.context, loop.agent.model)
    compacted, keep = compactor.parts()
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
        request = compactor.request(compacted)
        # Same pairing as the automatic path: the echo guard checks what the model is handed, and
        # the inline slice carries one message more than `compacted` does.
        sent = request[0][:-1] if request else compacted
        data = await compactor.compact(compactor.input(compacted), *(request or ()), echo_source=compactor.echo_source(sent))
    except (asyncio.CancelledError, KeyboardInterrupt):
        return "Cancelled"
    except Exception as error:  # noqa: BLE001 - manual compaction uses the same deterministic fallback as automatic compaction.
        loop.agent.context.apply_compaction(None, keep, fallback_note=PREVIOUS_CONTEXT_TRIMMED, compacted=compacted, trigger="manual")
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
        loop.agent.context.apply_compaction(
            data,
            keep,
            compacted=compacted,
            trigger="manual",
            model=loop.agent.model.last_compaction_model,
            title=compactor.title(data),
        )
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


async def provider(loop: CommandLoop, args: str) -> str:
    parts = args.split()
    if len(parts) > 1:
        return "Usage: /provider [NAME]"
    if parts:
        return set_provider(loop, parts[0])
    choices = tuple(sorted(loop.session.config.providers))
    summary = "provider: " + loop.session.config.active_provider + "\nproviders: " + ", ".join(choices)
    current = loop.session.config.active_provider
    choice = await select_choice(loop, "Provider", choices, labels={current: current + " (current)"}, current=current)
    if not isinstance(choice, str):
        return "No change" if choice is SELECTION_BACK else summary
    provider_result = set_provider(loop, choice)
    model_result = await model(loop, "")
    return provider_result + ("\n" + model_result if model_result else "")


def record_provider_override(session: Session, field: str, value: str) -> None:
    """Remember a runtime /provider /model /reason /api switch so a later --resume can restore it.

    model/reasoning/api are keyed by the provider entry they applied to (the active one at the time
    of the change); active_provider is global. url and key are never recorded, so the config file
    stays the only home of credentials."""
    if field == "active_provider":
        session.provider_overrides["active_provider"] = value
        return
    session.provider_overrides.setdefault("providers", {}).setdefault(session.config.active_provider, {})[field] = value


def realign_reasoning(loop: CommandLoop, model: str = "") -> str:
    """Move the stored effort onto `model`'s scale, and say so when it moves.

    The alternative to saying it is a request that silently sends something other than the effort
    on screen, which is what this replaced. It happens where the scale changes underneath a stored
    choice — switching entry or model — never per request."""
    provider = loop.session.config.provider
    aligned = loop.session.policy.normalized_reasoning(provider, model)
    if aligned == provider.reasoning:
        return ""
    previous, provider.reasoning = provider.reasoning, aligned
    record_provider_override(loop.session, "reasoning", aligned)
    return f"Reasoning {previous} is not offered by {model or provider.model}, using {aligned}"


def set_provider(loop: CommandLoop, name: str) -> str:
    if name not in loop.session.config.providers:
        return "Unknown provider: " + name
    loop.session.config.active_provider = name
    record_provider_override(loop.session, "active_provider", name)
    return "\n".join(line for line in ("Set provider = " + name, realign_reasoning(loop)) if line)


async def model(loop: CommandLoop, args: str) -> str:
    parts = args.split()
    if len(parts) > 1:
        return "Usage: /model [MODEL]"
    if parts:
        result = await set_model(loop, parts[0])
        return "No change" if result is SELECTION_BACK else str(result)
    provider = loop.session.config.provider
    configured = tuple(dict.fromkeys(provider.available_models))
    tui = loop.tui
    show_loading = tui is not None and bool(provider.url and provider.key)
    if show_loading and tui is not None:
        tui.set_dispatching("Loading models...")
    try:
        remote = tuple(model for model in await remote_models(loop, provider) if model not in configured)
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
        choice = await select_choice(loop, "Model", choice_values, labels=labels, current=current, disabled=MODEL_LABELS)
        if choice is SELECTION_BACK:
            return "No change"
        if not isinstance(choice, str):
            return "Current provider.model is " + (loop.session.config.provider.model or "(empty)")
        if choice in MODEL_LABELS:
            continue
        result = await set_model(loop, choice, back_to_model=True)
        if result is SELECTION_BACK:
            continue
        return str(result)


async def remote_models(loop: CommandLoop, provider: ProviderConfig) -> tuple[str, ...]:
    if not provider.url or not provider.key:
        return ()
    try:
        # lazy import: /model discovery is the only OpenAI use here, so the SDK stays off the startup path
        from openai import AsyncOpenAI

        from wizolt.model import ModelClient

        client = AsyncOpenAI(
            api_key=provider.key,
            base_url=loop.session.policy.resolve(provider).base_url,
            timeout=min(provider.timeout, 10),
            max_retries=0,
            default_headers=ModelClient.request_headers(provider),
        )
        try:
            page = await client.models.list()
        finally:
            await client.close()
    except Exception:  # noqa: BLE001 - remote model discovery is optional and provider SDKs expose varied failures.
        return ()
    names = []
    for item in getattr(page, "data", page) or []:
        name = item.get("id") if isinstance(item, dict) else getattr(item, "id", None)
        if isinstance(name, str) and name:
            names.append(name)
    return tuple(sorted(dict.fromkeys(names)))


async def set_model(loop: CommandLoop, model: str, *, back_to_model: bool = False) -> str | object:
    while True:
        api = await select_api(loop, model)
        if api is SELECTION_BACK:
            return SELECTION_BACK if back_to_model else "No change"
        # The model being switched to, not the one still configured: its scale is what the new
        # effort has to come from.
        reasoning = await select_reasoning(loop, model)
        if reasoning is not SELECTION_BACK:
            break
    provider = loop.session.config.provider
    provider.model = model
    record_provider_override(loop.session, "model", model)
    lines = ["Set provider.model = " + model]
    if isinstance(api, str):
        lines.append(set_api(loop, api))
    if isinstance(reasoning, str):
        provider.reasoning = reasoning
        record_provider_override(loop.session, "reasoning", reasoning)
        lines.append("Set provider.reasoning = " + reasoning)
    elif realigned := realign_reasoning(loop):
        lines.append(realigned)
    return "\n".join(lines)


def set_reasoning(loop: CommandLoop, value: str) -> str:
    provider = loop.session.config.provider
    provider.reasoning = value
    record_provider_override(loop.session, "reasoning", value)
    return "Set provider.reasoning = " + value


async def reason(loop: CommandLoop, args: str) -> str:
    value = args.strip()
    if value:
        # Typed efforts are held to the same list the picker offers, so `/reason` and the picker
        # cannot disagree about what this model takes.
        choices = loop.session.policy.reasoning_choices(loop.session.config.provider)
        if value not in choices:
            return "Usage: /reason " + "|".join(choices)
        return set_reasoning(loop, value)
    choice = await select_reasoning(loop)
    return set_reasoning(loop, choice) if isinstance(choice, str) else "No change"


async def api(loop: CommandLoop, args: str) -> str:
    value = args.strip()
    provider = loop.session.config.provider
    if value:
        if value not in PROVIDER_API_CHOICES:
            return "Usage: /api " + "|".join(PROVIDER_API_CHOICES)
        return set_api(loop, value)
    choice = await select_api(loop, provider.model)
    return set_api(loop, choice) if isinstance(choice, str) else "No change"


def set_api(loop: CommandLoop, value: str) -> str:
    provider = loop.session.config.provider
    provider.api = value
    record_provider_override(loop.session, "api", value)
    # "auto" is the usual choice, so name the wire it resolved to rather than echoing the setting back.
    resolved = loop.session.policy.resolve(provider)
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


def strict(loop: CommandLoop, args: str) -> str:
    if args:
        return "Usage: /strict"
    provider = loop.session.config.provider
    provider.strict_tools = not provider.strict_tools
    state = "on" if provider.strict_tools else "off"
    if provider.strict_tools:
        resolved = loop.session.policy.resolve(provider)
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


MODEL_CONFIGURED_LABEL = "---- Configured models ----"
MODEL_DISCOVERED_LABEL = "---- Discovered models ----"
MODEL_LABELS = frozenset((MODEL_CONFIGURED_LABEL, MODEL_DISCOVERED_LABEL))
MCP_COMMANDS: dict[str, tuple[int, int, str]] = {
    "connect": (1, sys.maxsize, "Usage: /mcp connect <server> [server ...]"),
    "disconnect": (1, 1, "Usage: /mcp disconnect <server>"),
    "tools": (0, 1, "Usage: /mcp tools [server]"),
}
MCP_HELP = "Try /mcp, /mcp connect <server> [server ...], /mcp disconnect <server>, or /mcp tools [server]"
