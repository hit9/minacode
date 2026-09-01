"""wizolt session: live agent state, records, and the Session object."""

from __future__ import annotations

import os
import re
import threading
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, ClassVar, cast

from wizolt.base import (
    SESSION_EVENT_KEY,
    Json,
    ModelUsage,
    Text,
    ToolArgs,
    UpdateStatus,
)
from wizolt.config import PROVIDER_API_CHOICES, Config, ConfigFile, RuntimeSettings, SystemInfo, request_budget_for
from wizolt.image import ImageInputs, UserInput
from wizolt.prompts import COMPACTION_SUMMARY_TITLE, LIVE_FOLLOWUP_PREFIX, SYSTEM_PROMPT, WORKING_STATE_CHECKPOINT_TITLE
from wizolt.providers.compat import ProviderPolicy, bundled_policy
from wizolt.providers.sync import CatalogRuntime
from wizolt.session.codec import TRANSCRIPT_SYNC_VERSION, SessionSnapshotCodec
from wizolt.session.diffs import net_diff_sections
from wizolt.session.images import ImageRoute
from wizolt.session.jobs import BackgroundJob
from wizolt.session.queue import QueuedInput
from wizolt.session.store import (
    CONTEXT_LAYOUT_VERSION,
    SessionEntry,
    SessionSnapshotStore,
    local_timestamp,
)
from wizolt.source import SourceView, SourceViewDraft

__all__ = [
    "TRANSCRIPT_SYNC_VERSION",
    "SessionEntry",
    "SessionSnapshotCodec",
    "SessionSnapshotStore",
    "local_timestamp",
]

if TYPE_CHECKING:
    from wizolt.engine import Agent
    from wizolt.mcp import MCPManager
    from wizolt.mentions import FileMentions
    from wizolt.skill import SkillLibrary


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
    # is. Display state only: billing now rides api_request's billing=Billing.COMPACTION parameter.
    # Set around the request in Compactor.run and never persisted.
    compaction_entry: str = ""
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

    def apply_summary(self, data: Json) -> None:
        """Take the one field a compaction reply owns.

        `goal`, `plan`, `known` and `check` are `Note`'s, and a compaction used to overwrite all
        four from the same JSON. They did not need rescuing: they live here and survive eviction
        untouched, so handing them to a summarizer only let a prose re-derivation replace a
        validated structure -- and compound, since the next pass re-derived from that. `summary`
        is different: the compactor is the only party that read the evicted span, so it is the one
        thing it must produce. Anything it learned that belongs in `known` reaches the model
        through the summary, which can then call `Note` like any other writer.
        """
        if isinstance(data.get("summary"), str):
            self.summary = str(data["summary"]).strip()

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
    # Source views are owned by the Session: only it allocates keys and mutates the mapping, so
    # read-only tools can create immutable drafts on worker threads without a lock.
    source_view_counter: int = 0
    source_views: dict[str, SourceView] = field(default_factory=dict)
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
    # The provider/model catalog this session resolves against: snapshot + compiled policy + sync
    # state. Attached by bootstrap_features (which also owns the feature packages), so the session
    # dataclass itself stays feature-free; absent until a session is bootstrapped.
    catalog: CatalogRuntime | None = None
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
    _gitignore_lock: threading.RLock = field(default_factory=threading.RLock)

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

    @property
    def policy(self) -> ProviderPolicy:
        """The selected catalog policy, with a lazy bundled fallback for bare test/library sessions."""

        return self.catalog.policy if self.catalog is not None else bundled_policy()

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
            restored_model = fields.get("model")
            model = restored_model if isinstance(restored_model, str) else entry.model
            reasoning_choices = self.policy.reasoning_values(entry, model)
            reasoning = fields.get("reasoning")
            if not isinstance(reasoning, str) or reasoning not in reasoning_choices:
                reasoning = None
            api = fields.get("api")
            if api and api not in PROVIDER_API_CHOICES:
                api = None
            for attr, value in (("model", model if isinstance(restored_model, str) else None), ("reasoning", reasoning), ("api", api)):
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
        catalog = CatalogRuntime(Config.data_dir_from(data))
        session = cls(
            config=Config.from_dict(data, policy=catalog.policy),
            settings=RuntimeSettings.from_dict(data, yolo=yolo, theme=theme),
            catalog=catalog,
        )
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
        stored image. Reading one back is wizolt following a path it just handed the model, not the
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

    def register_source_drafts(self, drafts: list[SourceViewDraft]) -> list[str]:
        """Allocate `view.N` keys for `drafts` in call order, register them, return the keys.

        Only the runner's main thread calls this, in model tool-call order, so keys are
        deterministic regardless of parallel completion order. A key is never reused: the counter
        is monotonic and re-registering an existing key is refused. Identical drafts within one
        call share one key, so one Search call with several queries pointing at the same file
        resolves to a single view id. Drafts from earlier calls are never reused: they were
        captured from a different read, and their `total_lines` no longer describes what this call
        showed the model.
        """
        keys: list[str] = []
        existing: dict[tuple[object, ...], str] = {}
        for draft in drafts:
            identity = self._draft_identity(draft)
            if identity in existing:
                keys.append(existing[identity])
                continue
            self.source_view_counter += 1
            key = SourceView.make_key(self.source_view_counter)
            if key in self.source_views:
                raise RuntimeError(f"source view key already registered: {key}")
            self.source_views[key] = SourceView(
                key,
                draft.path,
                draft.display_path,
                draft.total_lines,
                draft.spans,
                draft.producer,
                self.state.round_count,
                self.state.turn_step,
            )
            existing[identity] = key
            keys.append(key)
        return keys

    @staticmethod
    def _draft_identity(draft: SourceViewDraft) -> tuple[object, ...]:
        return (draft.path, draft.total_lines, tuple((span.start, span.lines) for span in draft.spans))

    def get_source_view(self, key: str) -> SourceView | None:
        """The live view named by `key`, or None when it is unknown or expired."""
        return self.source_views.get(key)

    def prune_source_views(self, referenced: set[str]) -> int:
        """Drop views whose public ids are not in `referenced`; return the number removed.

        Referenced ids come from model-visible current state: committed messages, the active turn,
        and AgentState text. Transcript-only history is not a retention root, so a view expires
        once it leaves active model context.
        """
        before = len(self.source_views)
        self.source_views = {key: view for key, view in self.source_views.items() if key in referenced}
        return before - len(self.source_views)

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

    # The session owns the edit records; `diffs` owns what they mean. Both entry points pass the
    # records and the working directory and read nothing else off the session, which is what let
    # the reconstruction move out whole.
    def latest_round_diff_sections(self) -> tuple[int, list[tuple[str, str, str]]] | None:
        if not self.turn_diffs:
            return None
        round = max(diff.round or diff.turn for diff in self.turn_diffs)
        diffs = [diff for diff in self.turn_diffs if (diff.round or diff.turn) == round]
        return round, net_diff_sections(diffs, "edit", cwd=self.cwd)

    def session_diff_sections(self) -> list[tuple[str, str, str]]:
        return net_diff_sections(self.turn_diffs, "overall", cwd=self.cwd)

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
    def load_snapshot(
        cls,
        uid: str,
        config: Config | None = None,
        settings: RuntimeSettings | None = None,
        cwd: str = "",
        catalog: CatalogRuntime | None = None,
    ) -> Session:
        if config is None:
            data = ConfigFile.load()
            catalog = catalog or CatalogRuntime(Config.data_dir_from(data))
            config = Config.from_dict(data, policy=catalog.policy)
            if settings is None:
                settings = RuntimeSettings.from_dict(data)
        else:
            catalog = catalog or CatalogRuntime(config.data_dir)
        if settings is None:
            settings = RuntimeSettings()
        session = SessionSnapshotStore.load(uid, config=config, settings=settings, cwd=cwd)
        session.catalog = catalog
        bootstrap_features(session)
        return session


def bootstrap_features(session: Session) -> None:
    """Attach the session's feature objects (MCP, skills, file mentions) when not already injected.

    Session itself stays feature-free: the dataclass constructor never reaches upward. Callers that
    need the features -- the runtime entry points and the worker handoff -- opt in explicitly after
    construction, so the feature packages sit above session/ without a module-scope cycle.
    """
    if session.mcp is None:
        from wizolt.mcp import MCPManager  # local import: mcp is built on top of session

        session.mcp = MCPManager(session)
    if session.skills is None:
        from wizolt.skill import SkillLibrary  # local import: skill is built on top of session

        session.skills = SkillLibrary.load(session)
    if session.mentions is None:
        from wizolt.mentions import FileMentions  # local import: mentions is built on top of session

        session.mentions = FileMentions(session)
    if session.catalog is None:
        from wizolt.providers.sync import CatalogRuntime  # local import: keeps providers above session

        session.catalog = CatalogRuntime(session.config.data_dir)
