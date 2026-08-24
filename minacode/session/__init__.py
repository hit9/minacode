"""minacode session: live agent state, records, and the Session object."""

from __future__ import annotations

import contextlib
import difflib
import os
import re
import signal
import subprocess
import threading
import time
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, ClassVar, cast

from minacode.base import (
    IMAGE_ROUTE_TEXT_ONLY_LEARNED,
    IMAGE_ROUTE_TEXT_ONLY_STATIC,
    IMAGE_ROUTE_UNKNOWN,
    SESSION_EVENT_KEY,
    Json,
    ModelUsage,
    Text,
    ToolArgs,
    UpdateStatus,
    split_lines,
)
from minacode.config import PROVIDER_API_CHOICES, REASONING_CHOICES, Config, ConfigFile, RuntimeSettings, SystemInfo, request_budget_for
from minacode.image import IMAGE_REFS_KEY, ImageInputs, ImageRef, UserInput
from minacode.prompts import COMPACTION_SUMMARY_TITLE, LIVE_FOLLOWUP_PREFIX, SYSTEM_PROMPT, WORKING_STATE_CHECKPOINT_TITLE
from minacode.session.store import (
    CONTEXT_LAYOUT_VERSION,
    TRANSCRIPT_SYNC_VERSION,
    SessionEntry,
    SessionSnapshotCodec,
    SessionSnapshotStore,
    local_timestamp,
)

__all__ = [
    "TRANSCRIPT_SYNC_VERSION",
    "SessionEntry",
    "SessionSnapshotCodec",
    "SessionSnapshotStore",
    "local_timestamp",
]

if TYPE_CHECKING:
    from minacode.engine import Agent
    from minacode.mcp import MCPManager
    from minacode.mentions import FileMentions
    from minacode.skill import SkillLibrary


@dataclass
class PlanItem:
    _PLAN_LINE_RE: ClassVar[re.Pattern] = re.compile(r"\[( |x|X|~|-)\]\s+(.+)")
    STATUSES: ClassVar[tuple[str, ...]] = ("todo", "doing", "done", "blocked")
    SYMBOLS: ClassVar[dict[str, str]] = {"todo": " ", "doing": "~", "done": "x", "blocked": "-"}
    LEGACY_MARKERS: ClassVar[dict[str, str]] = {" ": "todo", "~": "doing", "x": "done", "X": "done", "-": "blocked"}

    status: str
    text: str

    @classmethod
    def parse(cls, value: object) -> PlanItem | None:
        if isinstance(value, cls):
            status, text = value.status, value.text
        elif isinstance(value, dict):
            status = str(value.get("status") or "todo").strip().lower()
            text = str(value.get("text") or "").strip()
        else:
            raw = str(value).strip()
            match = PlanItem._PLAN_LINE_RE.fullmatch(raw)
            status = cls.LEGACY_MARKERS[match.group(1)] if match else "todo"
            text = match.group(2).strip() if match else raw
        if not text:
            return None
        return cls(status if status in cls.STATUSES else "todo", text)

    def row(self, *, status: bool = False, style: str = "text") -> str:
        prefix = f"[{self.SYMBOLS[self.status]}] " if status and style == "symbol" else f"{self.status}: " if status else ""
        return "- " + prefix + self.text


@dataclass
class AgentState:
    goal: str = ""
    plan: list[PlanItem | Json | str] = field(default_factory=list)
    known: list[str] = field(default_factory=list)
    check: str = ""
    summary: str = ""
    # How this session is labelled when listed, and where that label came from. `apply` never sets
    # either: the name follows the user and the goal, not whatever a tool call happens to write.
    name: str = ""
    name_source: str = ""  # "" | user | goal | input
    code_index_status: str = ""
    code_index_error: str = ""
    code_index_notice: str = ""
    code_index_refreshing: bool = False
    code_index_checking: bool = False
    context_percent: int = 0
    turn_step: int = 0
    turn_messages: int = 0
    round_count: int = 0
    current_model_call_started_at: float = 0.0
    manual_model_retry_requested: bool = False
    model_retry_count: int = 0
    current_model_attempt: int = 0
    model_retry_reason: str = ""
    model_retry_until: float = 0.0  # monotonic deadline of the current retry wait; 0 when idle
    compaction_count: int = 0
    # `entry/model` of the provider entry a summary request is running on right now, "" when none
    # is. Live display state, like the retry and index fields above: set around the request in
    # ModelClient.compact and never persisted.
    compaction_entry: str = ""
    # True while an explicit ViewImage vision request is in flight. Its usage joins the session
    # totals but it is not a main-model request, so `_record_usage` must not let it overwrite the
    # last-request ctx/cache snapshot the status bar reads. Live request state, like
    # compaction_entry: never persisted.
    vision_observe_active: bool = False
    # The last delegation that failed on this worker, for `Delegate status` to tell the parent
    # (which cannot see the worker) why it stopped, instead of the parent having to remember.
    # Live display state, like compaction_entry: never persisted.
    last_error: str = ""
    last_error_round: int = 0
    # The current request's output stream, for the throughput the running divider shows. Characters
    # rather than tokens because token deltas are not on the wire: providers report usage once, when
    # the request is over. Reset at the start of every attempt and cleared when it ends, so the rate
    # belongs to the response being watched and never survives it. Live display state, never persisted.
    stream_started_at: float = 0.0
    stream_chars: int = 0

    def __post_init__(self) -> None:
        self.plan = cast(list[PlanItem | Json | str], self.plan_items(self.plan))

    @classmethod
    def plan_items(cls, items: Iterable[object]) -> list[PlanItem]:
        return [item for raw in items if (item := PlanItem.parse(raw))]

    @classmethod
    def plan_rows_for(cls, items: Iterable[object], *, status: bool = False, style: str = "text") -> list[str]:
        rows = [item.row(status=status, style=style) for item in cls.plan_items(items)]
        return rows or ["- (empty)"]

    def apply(self, data: Json) -> None:
        for attr in ("goal", "summary", "check"):
            if isinstance(data.get(attr), str):
                setattr(self, attr, str(data[attr]).strip())
        for attr in ("plan", "known"):
            value = data.get(attr)
            # One string where the schema asks for a list used to fall through untouched, which is
            # worse than being wrong: the previous compaction's value survives as though the model
            # had confirmed it, and gets fed back as current on the next pass. One string is one
            # item. Anything else that is not a list is still refused, as before.
            if isinstance(value, str):
                value = [value] if value.strip() else []
            if isinstance(value, list):
                items = list(filter(None, (str(item).strip() for item in value))) if attr == "known" else self.plan_items(value)
                setattr(self, attr, items)

    def format(self, *, include_summary: bool = False) -> str:
        known = ["- " + item for item in self.known] or ["- (empty)"]
        rows = [
            "Goal: " + (self.goal or "(empty)"),
            "Plan:",
            *self.plan_rows_for(self.plan, status=True),
            "Known:",
            *known,
            "Check: " + (self.check or "(empty)"),
        ]
        if include_summary:
            rows.extend(("Summary:", self.summary or "(empty)"))
        return "\n".join(rows)


@dataclass
class ToolResultRecord:
    key: str
    name: str
    args: ToolArgs
    output: str
    note: str = ""


@dataclass
class ToolErrorRecord:
    key: str
    name: str
    args: ToolArgs
    error: str


@dataclass
class TurnDiff:
    SNAPSHOT_CHAR_LIMIT: ClassVar[int] = 1_000_000
    TRANSCRIPT_CHAR_LIMIT: ClassVar[int] = 64 * 1024

    key: str
    turn: int
    path: str
    diff: str
    before: str = ""
    after: str = ""
    round: int = 0

    @classmethod
    def bounded_snapshots(cls, before: str, after: str) -> tuple[str, str]:
        """Cap each snapshot on its own. Snapshots are stored once per unique content, so a pair
        usually costs one new version rather than two, and summing the two would hold the ceiling at
        half the file size it can actually afford. Both are dropped together when either is too
        large: one alone would read as the file being created or deleted wholesale."""
        return ("", "") if max(len(before), len(after)) > cls.SNAPSHOT_CHAR_LIMIT else (before, after)

    @classmethod
    def bounded_transcript(cls, diff: str) -> str:
        if len(diff) <= cls.TRANSCRIPT_CHAR_LIMIT:
            return diff
        clipped = diff[: cls.TRANSCRIPT_CHAR_LIMIT].rsplit("\n", 1)[0]
        return clipped + "\n… diff preview truncated; see /diff for the retained session diff"


@dataclass
class HistorySegment:
    """One compacted span of conversation, retained for later recall. The evicted messages are
    captured once at compaction time (never re-summarized), so repeated compaction cannot compound
    loss; a bounded verbatim excerpt is stored as a content-addressed blob, and `RecallContext`
    lists, searches, or retrieves it on demand.

    The fields after `text` describe the compaction that produced the segment, for `/compact log`:
    the model never sees them (RecallContext returns key/title/text), and they are what makes an
    eviction reviewable afterwards. `summary` is the checkpoint summary as it stood at this
    compaction — the live checkpoint carries only the newest one, so without this copy every
    earlier summary would be unreachable once the next compaction replaced it."""

    key: str
    title: str
    text: str = ""
    created_at: str = ""
    scope: str = ""  # "history" (prior conversation) | "turn" (the running turn)
    trigger: str = ""  # "auto" (over budget) | "manual" (/compact)
    fallback: bool = False  # the summarizer failed and the span was trimmed deterministically
    messages: int = 0  # evicted message count
    summary: str = ""
    model: str = ""  # effective model the summary ran on; empty = fell back to trimming


@dataclass
class BackgroundJob:
    """A non-blocking shell process tracked by the session. Output is either redirected to a log
    file on disk (jobs started via `Job(start)`) or accumulated in an in-memory tail buffer by a
    drainer thread (jobs promoted from a running BashTool call after bash_wait_timeout). Both
    variants expose the same tail/status/wait/kill surface."""

    id: str
    command: str
    process: subprocess.Popen[bytes]
    log_path: str
    started_at: float
    status: str = "running"
    exit_code: int | None = None
    # Memory-backed tail, populated by BashTool.promote_to_job's drainer thread. When set, tail()
    # reads from here instead of log_path. Bounded at BUFFER_LIMIT chars by the drainer.
    stream_buffer: list[str] | None = None
    stream_lock: threading.Lock | None = None

    BUFFER_LIMIT: ClassVar[int] = 32 * 1024  # per-stream tail cap in chars

    def update_status(self) -> None:
        if self.status != "running":
            return
        code = self.process.poll()
        if code is not None:
            self.status = "done"
            self.exit_code = code

    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    def kill(self, grace: float = 3.0) -> None:
        """SIGTERM, wait grace seconds, then SIGKILL if still running. Removes the log file."""
        if self.status == "running":
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except OSError:
                self.process.terminate()
            try:
                self.process.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except OSError:
                    self.process.kill()
                self.process.wait()
            self.update_status()
            if self.status == "running":
                self.status = "killed"
                self.exit_code = -1
        if self.log_path:
            with contextlib.suppress(OSError):
                os.unlink(self.log_path)

    def tail(self, limit: int) -> str:
        """Return the last `limit` chars from the merged stdout+stderr log."""
        limit = max(0, limit)
        if self.stream_buffer is not None:
            with self.stream_lock or contextlib.nullcontext():
                text = "".join(self.stream_buffer)
        else:
            try:
                with open(self.log_path, "rb") as file:
                    file.seek(0, 2)
                    size = file.tell()
                    # UTF-8 is up to 4 bytes/char; read a little extra so decoding produces at least `limit` chars.
                    file.seek(max(0, size - limit * 4), 0)
                    text = file.read().decode("utf-8", errors="replace")
            except OSError:
                return ""
        if len(text) <= limit:
            return text
        if limit <= 3:
            return "." * limit
        return "..." + text[-(limit - 3) :]


class ImageRoute:
    """Unified image-delivery decision for the active main route; session-local.

    Static text-only evidence is folded by `ProviderConfig.resolve()` from the provider
    compatibility catalog. Learned evidence is created only when an eligible main request
    returns HTTP 400 for a request carrying a current-turn raw image, and is keyed by the full
    route identity (provider entry, resolved API, resolved base URL, model). It lives in memory
    for the live session only: snapshots never carry it and a resumed session starts unknown
    unless the catalog supplies static evidence.

    Attachments and ViewImage must ask this one decision which delivery is required instead of
    duplicating model matching or 400 learning; presentation only observes routing events.
    """

    def __init__(self, session: Session) -> None:
        self.session = session

    def identity(self) -> tuple[str, str, str, str]:
        """The full identity of the effective active provider entry, used as the learned-evidence key."""

        provider = self.session.config.provider
        resolved = provider.resolve()
        return (self.session.config.active_provider, resolved.api, resolved.base_url, provider.model.lower())

    def static_text_only(self) -> bool:
        return self.session.config.provider.resolve().text_only

    def learned_text_only(self) -> bool:
        return self.identity() in self.session.learned_text_only_routes

    def is_text_only(self) -> bool:
        return self.static_text_only() or self.learned_text_only()

    def state(self) -> str:
        if self.static_text_only():
            return IMAGE_ROUTE_TEXT_ONLY_STATIC
        if self.learned_text_only():
            return IMAGE_ROUTE_TEXT_ONLY_LEARNED
        return IMAGE_ROUTE_UNKNOWN

    def learn_text_only(self) -> None:
        """Record session-local evidence for the exact current main route."""

        self.session.learned_text_only_routes.add(self.identity())

    def delivery(self) -> str:
        """How a current image occurrence is delivered: `vision` when the route is text-only
        (static or learned) and a vision entry exists, else a raw attempt on the main model.

        A text-only route without `[vision]` deliberately keeps the raw attempt so the
        provider's real failure stays visible; no local image-disable error is invented.
        """

        if self.is_text_only() and self.session.config.vision_provider:
            return "vision"
        return "raw"


@dataclass(eq=False)
class QueuedInput:
    text: str
    images: tuple[ImageRef, ...] = ()
    draft: str = ""
    inflight: bool = False

    def to_json(self) -> str | Json:
        if not self.images:
            return self.text
        return {
            "text": self.text,
            "draft": self.draft,
            IMAGE_REFS_KEY: [image.to_json() for image in self.images],
        }

    @classmethod
    def from_json(cls, value: object) -> QueuedInput | None:
        if isinstance(value, str):
            return cls(value) if value.strip() else None
        if not isinstance(value, dict):
            return None
        text = str(value.get("text") or "")
        raw_images = value.get(IMAGE_REFS_KEY)
        images = tuple(image for raw in raw_images if (image := ImageRef.from_json(raw)) is not None) if isinstance(raw_images, list) else ()
        draft = str(value.get("draft") or text)
        if not text.strip():
            return None
        if draft.count("\ufffc") != len(images):
            return cls(text)
        return cls(text, images, draft)

    def user_input(self) -> UserInput:
        return UserInput(self.draft or self.text, self.images)

    def message(self, prefix: str = "") -> Json:
        message: Json = {"role": "user", "content": prefix + self.text}
        if self.images:
            message[IMAGE_REFS_KEY] = [image.to_json() for image in self.images]
        return message


@dataclass
class Session:
    """Protocol-neutral semantic state plus resources scoped to one running session.

    The durable source of truth includes messages, retained tool output, diffs, and usage. The same
    aggregate owns transient session resources such as jobs, provider/update state, capability
    managers, and caches, but `SessionSnapshotCodec` explicitly selects the subset sufficient to
    resume. Provider clients, stream fragments, and terminal layout are absent by design and are
    reconstructed.

    A turn in progress is staged apart from committed history, so an interrupted or crashed turn can
    be settled or dropped without leaving half a turn in the record.

    Queued input and snapshot writes are lock-guarded: input arrives on the UI thread while the agent
    runs on another.
    """

    cwd: str = field(default_factory=os.getcwd)
    system_info: SystemInfo | None = None
    config: Config = field(default_factory=Config)
    settings: RuntimeSettings = field(default_factory=RuntimeSettings)
    # Runtime /provider /model /reason /api switches, keyed for restore: {"active_provider": name,
    # "providers": {entry: {"model"/"reasoning"/"api": value}}}. Only fields the slash commands
    # changed are recorded, and never url/key; a resume applies them best-effort over the config file.
    provider_overrides: dict[str, Any] = field(default_factory=dict)
    messages: list[Json] = field(default_factory=list)
    state: AgentState = field(default_factory=AgentState)
    tool_results: dict[str, str] = field(default_factory=dict)
    tool_records: list[ToolResultRecord] = field(default_factory=list)
    tool_errors: list[ToolErrorRecord] = field(default_factory=list)
    pending_user_inputs: list[QueuedInput] = field(default_factory=list)
    quick_hints: tuple[str, ...] = field(default_factory=tuple)  # transient offered next-step inputs; never serialized, cleared each turn
    next_hints_available: bool = True  # transient frontend capability; false for the simple REPL, which has no chip UI
    # Worker handoff (see DESIGN.md): the second session this one delegates to, and its per-session
    # projection knobs. None of these are persisted — SessionSnapshotCodec.snapshot is an explicit
    # whitelist, so they return to their defaults on load and must be re-set by the delegate caller.
    system_prompt: str = SYSTEM_PROMPT  # role definition; the parent's default is unchanged
    tool_names: tuple[str, ...] = ()  # empty tuple = no filtering (parent behavior)
    listed: bool = True  # False -> no latest pointer, hidden from /sessions
    worker: Session | None = None  # runtime handle of the delegated session
    worker_tool_enabled: bool = False  # Delegate registration gate, frozen at construction from bool(config.worker_provider)
    _agent: Agent | None = None  # runtime handle of the worker's Agent; same lifetime as the worker Session
    tool_counter: int = 0
    turn_diffs: list[TurnDiff] = field(default_factory=list)
    history: list[HistorySegment] = field(default_factory=list)
    jobs: dict[str, BackgroundJob] = field(default_factory=dict)
    job_counter: int = 0
    usage: ModelUsage = field(default_factory=ModelUsage)
    # Summary requests are counted apart from the conversation: they can run on another provider
    # entry entirely, and one blended total cannot be multiplied by any single price. A worker's
    # spending is already separate by virtue of its own Session; this gives compaction the same.
    compaction_usage: ModelUsage = field(default_factory=ModelUsage)
    update: UpdateStatus = field(default_factory=UpdateStatus)
    mcp: MCPManager | None = None
    skills: SkillLibrary | None = None
    mentions: FileMentions | None = None  # runtime handle; holds the cached @file: path list
    images: ImageInputs = field(init=False, repr=False)
    # Session-local learned text-only route evidence, keyed by ImageRoute.identity(). Runtime
    # only: SessionSnapshotCodec is an explicit whitelist, so this never reaches a snapshot and
    # a resumed session starts unknown unless the catalog supplies static evidence.
    learned_text_only_routes: set[tuple[str, str, str, str]] = field(default_factory=set, repr=False)
    image_route: ImageRoute = field(init=False, repr=False)
    _gitignore_cache: dict[str, tuple[int, list[str]]] = field(default_factory=dict)
    uid: str = ""
    resumed: bool = False
    created_at: str = field(default_factory=local_timestamp)
    context_layout_version: int = CONTEXT_LAYOUT_VERSION
    transcript_messages: list[Json] = field(default_factory=list)
    transcript_tool_records: list[ToolResultRecord] = field(default_factory=list)  # legacy read-only replay bridge
    transcript_turn_diffs: list[TurnDiff] = field(default_factory=list)
    transcript_incomplete: bool = False
    _snapshot_saved: dict = field(default_factory=dict)
    _blobs_written: set[str] = field(default_factory=set)
    _meta_written: dict = field(default_factory=dict)
    _active_turn_messages: list[Json] = field(default_factory=list)
    _active_transcript_messages: list[Json] = field(default_factory=list)
    _queue_lock: threading.RLock = field(default_factory=threading.RLock)
    _snapshot_lock: threading.RLock = field(default_factory=threading.RLock)

    def __post_init__(self) -> None:
        self.images = ImageInputs(self)
        self.image_route = ImageRoute(self)
        if not self.uid:
            self.uid = datetime.now().strftime("%Y%m%d%H%M%S") + "-" + str(uuid.uuid4())[:12]  # noqa: DTZ005 - IDs intentionally use local wall time.
        if self.system_info is None:
            self.system_info = SystemInfo.detect(self.cwd)
        # The Delegate registration gate is frozen per session: computed once from the config this
        # session was constructed with, so a runtime /worker provider switch tunes an already-
        # enabled delegation and prepares the next session without flipping the tool block (and
        # thus the prompt-cache scope) mid-session. Recomputes on every load because the config
        # passed to SessionSnapshotStore.load is the caller's freshly built one.
        self.worker_tool_enabled = bool(self.config.worker_provider)
        self.apply_provider_overrides()

    def apply_provider_overrides(self) -> None:
        """Best-effort restore of the runtime /provider /model /reason /api switches saved with this
        session. Stale values are skipped, never fatal: a provider entry may have been removed or a
        choice renamed since the snapshot was written. model is a free string and applied as-is, so
        a model that no longer exists surfaces on the first request exactly as it would have live."""
        overrides = self.provider_overrides
        providers = self.config.providers
        for name, fields in (overrides.get("providers") or {}).items():
            entry = providers.get(name)
            if entry is None or not isinstance(fields, dict):
                continue
            reasoning = fields.get("reasoning")
            if reasoning and reasoning not in REASONING_CHOICES:
                reasoning = None
            api = fields.get("api")
            if api and api not in PROVIDER_API_CHOICES:
                api = None
            for attr, value in (("model", fields.get("model")), ("reasoning", reasoning), ("api", api)):
                if value is not None:
                    setattr(entry, attr, value)
        active = overrides.get("active_provider")
        if active and active in providers:
            self.config.active_provider = active

    def store_turn_diff(
        self,
        key: str,
        turn: int,
        path: str,
        diff: str,
        *,
        before: str = "",
        after: str = "",
        round: int = 0,
    ) -> None:
        before, after = TurnDiff.bounded_snapshots(before, after)
        record = TurnDiff(key, turn, path, diff, before, after, round)
        self.turn_diffs.append(record)
        self.transcript_turn_diffs.append(TurnDiff(key, turn, path, TurnDiff.bounded_transcript(diff), round=round))
        if len(self.turn_diffs) > 100:
            self.turn_diffs.pop(0)

    @classmethod
    def from_config_file(cls, *, path: str | None = None, yolo: bool = False, theme: str = "") -> Session:
        data = ConfigFile.load(path)
        session = cls(config=Config.from_dict(data), settings=RuntimeSettings.from_dict(data, yolo=yolo, theme=theme))
        bootstrap_features(session)
        return session

    def resolve_path(self, path: str) -> str:
        path = os.path.expanduser(path)
        return os.path.abspath(path if os.path.isabs(path) else os.path.join(self.cwd, path))

    def relpath(self, path: str) -> str:
        try:
            return os.path.relpath(path, self.cwd)
        except ValueError:
            return path

    def in_cwd(self, path: str) -> bool:
        try:
            return os.path.commonpath([os.path.realpath(self.cwd), os.path.realpath(path)]) == os.path.realpath(self.cwd)
        except ValueError:
            return False

    def owns_asset(self, path: str) -> bool:
        """True for a file in this session's own assets directory -- a materialized tool output or a
        stored image. Reading one back is minacode following a path it just handed the model, not the
        model reaching outside the workspace, so it is not what an out-of-workspace prompt is for."""
        try:
            directory = os.path.realpath(self.images.assets_dir())
            return os.path.commonpath([directory, os.path.realpath(path)]) == directory
        except (ValueError, OSError):
            return False

    def request_token_budget(self) -> int:
        """The input budget one request is measured against, under this session's *current* config.

        The single definition of the denominator. Cheap (no message projection), so renderers can
        call it per frame instead of reusing `usage.last_prompt_budget` -- that one is the budget a
        past request was prepared against, which is the right question for the overdue-by-usage
        guard and the wrong one for "how full am I now": it goes stale the moment the limit changes
        and is restored verbatim from a snapshot on resume.
        """
        provider = self.config.provider
        return request_budget_for(provider.context_token_limit(self.settings.max_context_tokens), provider.output_token_budget())

    def data_path(self, *parts: str) -> str:
        root = os.path.expanduser(self.config.data_dir)
        return os.path.abspath(os.path.join(root if os.path.isabs(root) else os.path.join(self.cwd, root), *parts))

    def running_jobs(self) -> list[BackgroundJob]:
        for job in self.jobs.values():
            job.update_status()
        return [job for job in self.jobs.values() if job.status == "running"]

    def missing_config(self) -> list[str]:
        return ["provider." + name for name in self.config.provider.missing_fields()]

    def store_tool_result(self, name: str, args: ToolArgs, output: str, note: str = "") -> str:
        self.tool_counter += 1
        key = f"tr.{self.tool_counter}"
        args, output = Text.value(list(args)), Text.clean(output)
        self.tool_results[key] = output
        record = ToolResultRecord(key, name, args, output, note)
        self.tool_records.append(record)
        if len(self.tool_results) > 400:
            old = self.tool_records.pop(0)
            self.tool_results.pop(old.key, None)
        return key

    def enqueue_user_input(self, value: str | UserInput) -> None:
        if isinstance(value, UserInput) and value.images:
            message = self.images.message(value)
            text = str(message.get("content") or "").strip()
            images = self.images.refs(message)
            draft = str(value)
        else:
            text = Text.clean(str(value).strip())
            images = ()
            draft = text
        if not text:
            return
        with self._queue_lock:
            self.pending_user_inputs.append(QueuedInput(text, images, draft))

    def claim_user_inputs(self) -> list[QueuedInput]:
        # claim/ack/release is a transaction across model retries; keep this boundary even though each step is small.
        with self._queue_lock:
            for item in self.pending_user_inputs:
                item.inflight = True
            return list(self.pending_user_inputs)

    def acknowledge_user_inputs(self, inputs: list[QueuedInput]) -> None:
        with self._queue_lock:
            self.pending_user_inputs = [item for item in self.pending_user_inputs if item not in inputs]

    def has_inflight_user_inputs(self) -> bool:
        with self._queue_lock:
            return any(item.inflight for item in self.pending_user_inputs)

    def release_user_inputs(self) -> None:
        with self._queue_lock:
            for item in self.pending_user_inputs:
                item.inflight = False

    def add_quick_hints(self, hints: list[str], *, limit: int = 4) -> None:
        """Merge more offered inputs into the current set: appended in call order, deduplicated,
        and capped at `limit`. Several `NextHints` calls in one batch must not overwrite each
        other, so the batch's suggestions accumulate instead of the last call winning."""
        with self._queue_lock:
            merged = [*self.quick_hints, *(hint for hint in hints if hint not in self.quick_hints)]
            self.quick_hints = tuple(merged[:limit])

    def clear_quick_hints(self) -> None:
        with self._queue_lock:
            self.quick_hints = ()

    @staticmethod
    def net_diff_for_path(status: str, path: str, before: str, after: str) -> tuple[str, str, str] | None:
        if before == after:
            return None
        text = "".join(difflib.unified_diff(split_lines(before), split_lines(after), fromfile="/dev/null" if not before else path, tofile=path))
        return (status, path, text) if text else None

    @classmethod
    def net_diff_sections(cls, diffs: list[TurnDiff], status: str, *, cwd: str = "") -> list[tuple[str, str, str]]:
        states: dict[str, tuple[str, str]] = {}
        legacy: dict[str, list[str]] = {}
        # Whether the most recent edit to each path carried snapshots. A path can hold both kinds
        # when a file grows past the snapshot size limit partway through a session, and the two
        # descriptions overlap — emitting both would repeat the file's changes.
        snapshot_tail: dict[str, bool] = {}
        paths: list[str] = []
        for diff in diffs:
            if diff.path not in paths:
                paths.append(diff.path)
            snapshot_tail[diff.path] = bool(diff.before or diff.after)
            if not diff.before and not diff.after:
                legacy.setdefault(diff.path, []).append(diff.diff)
                continue
            before, _ = states.get(diff.path, (diff.before, diff.after))
            states[diff.path] = (before, diff.after)

        # Bash can move a file between Edit calls. When one path's `.after` matches another path's
        # `.before` uniquely on both sides, that's the boundary of a move: merge into the target so
        # the logical history follows the file to its final path.
        while (move := cls._find_unambiguous_move(states, legacy)) is not None:
            source, target = move
            states[target] = (states[source][0], states[target][1])
            del states[source]

        sections = []
        for path in paths:
            chunk = cls.net_diff_chunk(path, status, states, legacy, snapshot_tail, cwd)
            if chunk:
                sections.append((status, path, chunk.rstrip("\n") + "\n"))
        return sections

    @classmethod
    def net_diff_chunk(
        cls,
        path: str,
        status: str,
        states: dict[str, tuple[str, str]],
        legacy: dict[str, list[str]],
        snapshot_tail: dict[str, bool],
        cwd: str,
    ) -> str:
        """One diff per path, from exactly one description of its history."""
        if path in states and snapshot_tail.get(path):
            # The last edit carried snapshots, so the recorded `after` is the file's final content.
            before, after = states[path]
            if legacy_chunks := legacy.get(path, []):
                # Snapshots cover only a suffix: snapshot-less edits ran before the first snapshot
                # (the file shrank past the limit mid-session), and their starting content isn't in
                # `states`. Walk their hunks back from the first snapshot's `before` to recover it so
                # the net diff spans the whole path. If they don't apply cleanly — they were
                # interleaved between snapshots, so the snapshot span already reflects them, or the
                # file was mutated outside Edit — the snapshot span stands as-is.
                original = cls._reverse_apply(before, legacy_chunks)
                if original is not None:
                    before = original
            section = cls.net_diff_for_path(status, path, before, after)
            return section[2] if section else ""
        if path in states and not snapshot_tail.get(path):
            # Snapshots stop partway through the path's history (the file grew past the limit); the
            # starting content is still known exactly. The end state is the file's current on-disk
            # content; if the file is gone, forward-apply the trailing snapshot-less hunks onto the
            # last snapshot's `after` to recover it, so the exactly-known snapshot history isn't
            # discarded. If neither is available, fall through to the raw-hunks fallback below.
            final = cls._current_content(cwd, path)
            if final is None:
                final = cls._forward_apply(states[path][1], legacy.get(path, []))
            if final is not None:
                section = cls.net_diff_for_path(status, path, states[path][0], final)
                return section[2] if section else ""
        legacy_chunks = legacy.get(path, [])
        if not legacy_chunks:
            return ""
        # No usable snapshots for this file. Best effort: reconstruct the pre-edit content by
        # reverse-applying the recorded per-Edit hunks to the file's current on-disk state, then emit
        # one clean synthesized diff. Falls back to the raw per-Edit hunks concatenated when
        # reconstruction can't uniquely locate a hunk (e.g. the file was mutated outside Edit).
        reconstructed = cls._reconstruct_legacy_diff(cwd, path, legacy_chunks, status) if cwd else None
        if reconstructed is not None:
            return reconstructed
        return "\n".join(chunk.rstrip("\n") for chunk in legacy_chunks)

    @staticmethod
    def _current_content(cwd: str, path: str) -> str | None:
        if not cwd:
            return None
        abspath = path if os.path.isabs(path) else os.path.join(cwd, path)
        try:
            with open(abspath, encoding="utf-8") as file:
                return file.read()
        except (OSError, UnicodeDecodeError):
            return None

    _HUNK_RE: ClassVar[re.Pattern[str]] = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@")

    @classmethod
    def _reverse_apply(cls, current: str, chunks: list[str]) -> str | None:
        """Walk `current` back to the state before the given per-Edit hunks by reverse-applying them
        in reverse chronological order. Each hunk's after-text must occur uniquely in the buffer; if
        not (external mutation, ambiguous context, or hunks that don't belong to this buffer's
        history), return None so the caller can fall back."""
        hunk_pairs: list[tuple[str, str]] = []
        for chunk in chunks:
            pairs = cls._split_hunks(chunk)
            if pairs is None:
                return None
            hunk_pairs.extend(pairs)
        for after_text, before_text in reversed(hunk_pairs):
            if not after_text or not before_text:
                return None
            if current.count(after_text) != 1:
                return None
            current = current.replace(after_text, before_text, 1)
        return current

    @classmethod
    def _forward_apply(cls, current: str, chunks: list[str]) -> str | None:
        """Apply the given per-Edit hunks forward to `current` in chronological order, deriving the
        content they produce. Each hunk's before-text must occur uniquely in the buffer; if not
        (external mutation or ambiguous context), return None so the caller can fall back. The mirror
        of `_reverse_apply`: used to recover a file's final content from its last snapshot when the
        file is no longer on disk."""
        hunk_pairs: list[tuple[str, str]] = []
        for chunk in chunks:
            pairs = cls._split_hunks(chunk)
            if pairs is None:
                return None
            hunk_pairs.extend(pairs)
        for after_text, before_text in hunk_pairs:
            if not after_text or not before_text:
                return None
            if current.count(before_text) != 1:
                return None
            current = current.replace(before_text, after_text, 1)
        return current

    @classmethod
    def _reconstruct_legacy_diff(cls, cwd: str, path: str, chunks: list[str], status: str) -> str | None:
        final = cls._current_content(cwd, path)
        if final is None:
            return None
        original = cls._reverse_apply(final, chunks)
        if original is None:
            return None
        section = cls.net_diff_for_path(status, path, original, final)
        return section[2] if section else ""

    @classmethod
    def _split_hunks(cls, chunk: str) -> list[tuple[str, str]] | None:
        pairs: list[tuple[str, str]] = []
        before_lines: list[str] | None = None
        after_lines: list[str] | None = None
        for line in chunk.splitlines():
            if line.startswith(("--- ", "+++ ")):
                continue
            if cls._HUNK_RE.match(line):
                if before_lines is not None and after_lines is not None:
                    pairs.append(("\n".join(after_lines), "\n".join(before_lines)))
                before_lines, after_lines = [], []
                continue
            if before_lines is None or after_lines is None:
                return None
            if line.startswith("+"):
                after_lines.append(line[1:])
            elif line.startswith("-"):
                before_lines.append(line[1:])
            elif line.startswith(" "):
                before_lines.append(line[1:])
                after_lines.append(line[1:])
            elif line == "\\ No newline at end of file":
                continue
            else:
                return None
        if before_lines is not None and after_lines is not None:
            pairs.append(("\n".join(after_lines), "\n".join(before_lines)))
        return pairs

    @staticmethod
    def _find_unambiguous_move(states: dict[str, tuple[str, str]], legacy: dict[str, list[str]]) -> tuple[str, str] | None:
        sources_by_after: dict[str, list[str]] = {}
        targets_by_before: dict[str, list[str]] = {}
        for path, (before, after) in states.items():
            if path in legacy:
                continue
            if after:
                sources_by_after.setdefault(after, []).append(path)
            if before:
                targets_by_before.setdefault(before, []).append(path)
        for content, sources in sources_by_after.items():
            targets = targets_by_before.get(content, [])
            if len(sources) == 1 and len(targets) == 1 and sources[0] != targets[0]:
                return sources[0], targets[0]
        return None

    def latest_round_diff_sections(self) -> tuple[int, list[tuple[str, str, str]]] | None:
        if not self.turn_diffs:
            return None
        round = max(diff.round or diff.turn for diff in self.turn_diffs)
        diffs = [diff for diff in self.turn_diffs if (diff.round or diff.turn) == round]
        return round, self.net_diff_sections(diffs, "edit", cwd=self.cwd)

    def session_diff_sections(self) -> list[tuple[str, str, str]]:
        return self.net_diff_sections(self.turn_diffs, "overall", cwd=self.cwd)

    def record_tool_error(self, key: str, name: str, args: ToolArgs, error: str) -> None:
        self.tool_errors.append(ToolErrorRecord(key, name, Text.value(list(args)), " ".join(Text.clean(error).split())))
        self.tool_errors = self.tool_errors[-5:]

    NAME_WIDTH: ClassVar[int] = 72

    @property
    def name(self) -> str:
        """What this session is called when it is listed. Empty only before the first message."""
        return self.state.name

    def rename(self, text: str) -> str:
        """Name the session explicitly. A user's name is never replaced by a derived one."""
        self.state.name, self.state.name_source = self.clip_name(text), "user"
        return self.state.name

    def refresh_name(self) -> str:
        """Latch a name, then let it follow the goal until the user sets one of their own.

        Deriving on every read would be simpler but wrong: compaction eventually drops the opening
        message, and a session listed under one name yesterday must not appear under another today
        just because its history was trimmed. A name is therefore decided once and only revised for
        a better source, never for a later one.
        """
        if self.state.name_source == "user":
            return self.state.name
        if self.state.name_source != "goal" and (goal := self.clip_name(self.state.goal)):
            self.state.name, self.state.name_source = goal, "goal"
        elif not self.state.name and (opening := self.opening_text()):
            self.state.name, self.state.name_source = self.clip_name(opening), "input"
        return self.state.name

    def opening_text(self) -> str:
        """The first thing the user asked for, as one line. Compaction summaries are not it."""
        for message in self.messages:
            content = message.get("content")
            if message.get("role") != "user" or not isinstance(content, str) or message.get(SESSION_EVENT_KEY):
                continue
            text = ImageInputs.label_text(message).strip()
            if text and not text.startswith(COMPACTION_SUMMARY_TITLE) and not text.startswith(LIVE_FOLLOWUP_PREFIX):
                return text.splitlines()[0]
        return ""

    def state_checkpoint_event(self) -> Json:
        return {
            "role": "user",
            "content": WORKING_STATE_CHECKPOINT_TITLE + "\n" + self.state.format(include_summary=True),
            SESSION_EVENT_KEY: "state_checkpoint",
        }

    @classmethod
    def clip_name(cls, text: str) -> str:
        return Text.clip_width(" ".join(str(text).split()), cls.NAME_WIDTH)

    def save_snapshot(self) -> str:
        # Session owns the persistence boundary; callers should not depend on the snapshot store.
        with self._snapshot_lock, self._queue_lock:
            self.refresh_name()
            return SessionSnapshotStore(self).save()

    @classmethod
    def load_snapshot(cls, uid: str, config: Config | None = None, settings: RuntimeSettings | None = None, cwd: str = "") -> Session:
        session = SessionSnapshotStore.load(uid, config=config, settings=settings, cwd=cwd)
        bootstrap_features(session)
        return session


def bootstrap_features(session: Session) -> None:
    """Attach the session's feature objects (MCP, skills, file mentions) when not already injected.

    Session itself stays feature-free: the dataclass constructor never reaches upward. Callers that
    need the features -- the runtime entry points and the worker handoff -- opt in explicitly after
    construction, so the feature packages sit above session/ without a module-scope cycle.
    """
    if session.mcp is None:
        from minacode.mcp import MCPManager  # local import: mcp is built on top of session

        session.mcp = MCPManager(session)
    if session.skills is None:
        from minacode.skill import SkillLibrary  # local import: skill is built on top of session

        session.skills = SkillLibrary.load(session)
    if session.mentions is None:
        from minacode.mentions import FileMentions  # local import: mentions is built on top of session

        session.mentions = FileMentions(session)
