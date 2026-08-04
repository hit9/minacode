"""minacode session: agent state, records, and session persistence."""

from __future__ import annotations

import contextlib
import difflib
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
import uuid
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, ClassVar, cast

from minacode.base import (
    SESSION_EVENT_KEY,
    Config,
    ConfigFile,
    Json,
    MinacodeError,
    ModelUsage,
    RuntimeSettings,
    SystemInfo,
    Text,
    ToolArgs,
    UpdateStatus,
)
from minacode.image import IMAGE_REFS_KEY, ImageInputs, ImageRef, UserInput
from minacode.prompts import COMPACTION_SUMMARY_TITLE, LIVE_FOLLOWUP_PREFIX, SYSTEM_PROMPT, WORKING_STATE_CHECKPOINT_TITLE

if TYPE_CHECKING:
    from minacode.engine import Agent
    from minacode.mcp import MCPManager
    from minacode.skill import SkillLibrary


CONTEXT_LAYOUT_VERSION = 2


def local_timestamp(value: float | None = None) -> str:
    """A user-readable local wall-clock timestamp with its numeric UTC offset."""
    current = datetime.now().astimezone() if value is None else datetime.fromtimestamp(value).astimezone()
    return current.isoformat(timespec="seconds")


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
    compaction_count: int = 0

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


@dataclass
class HistorySegment:
    """One compacted span of conversation, retained for later recall. The evicted messages are
    captured once at compaction time (never re-summarized), so repeated compaction cannot compound
    loss; a bounded verbatim excerpt is stored as a content-addressed blob, and `RecallContext`
    lists, searches, or retrieves it on demand."""

    key: str
    title: str
    text: str = ""


class SessionSnapshotCodec:
    """Decide what is durable, and encode it so saving stays cheap as the session grows.

    A session is snapshotted after every response and tool batch, so rewriting all of it each time
    would make saving cost more the longer the session runs. Each save records lengths and digests of
    the append-only sequences, and the next save emits only what was appended; the loader replays
    those deltas onto the last full snapshot. A sequence that changed in any way other than growing is
    rewritten whole, so a stale prefix can never be persisted silently.

    Large repeated text — file snapshots behind diffs, message text evicted by compaction — is stored
    once per unique content and referenced by hash, because the same content routinely appears as one
    edit's `before` and the previous edit's `after`.

    Legacy system-role resume markers are filtered during migration. New lifecycle events are
    append-only user messages: durable model context with protocol-neutral metadata hidden from UI.
    """

    @staticmethod
    def digest(value: object) -> str:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @classmethod
    def marker(cls, session: Session) -> Json:
        messages = cls.snapshot_messages(session)
        records = [cls.tool_record(record) for record in session.tool_records]
        errors = [cls.tool_error(error) for error in session.tool_errors]
        turn_diff_keys = [diff.key for diff in session.turn_diffs]
        # fmt: off
        return {
            "messages_len": len(messages), "messages_digest": cls.digest(messages), "tool_counter": session.tool_counter,
            "pending_user_inputs_digest": cls.digest([item.to_json() for item in session.pending_user_inputs]),
            "tool_records_len": len(records), "tool_records_digest": cls.digest(records),
            "tool_errors_len": len(errors), "tool_errors_digest": cls.digest(errors),
            "turn_diffs_len": len(turn_diff_keys), "turn_diffs_keys_digest": cls.digest(turn_diff_keys),
            "history_len": len(session.history), "history_keys_digest": cls.digest([seg.key for seg in session.history]),
        }
        # fmt: on

    @classmethod
    def turn_diff(cls, diff: TurnDiff, blobs: dict[str, str]) -> Json:
        """File snapshots are stored by content hash, not inline. Editing one file repeatedly makes
        each version appear twice — as one edit's `after` and the next edit's `before` — and a
        rewrite of the retained window would otherwise re-serialize every snapshot again."""
        before, after = TurnDiff.bounded_snapshots(diff.before, diff.after)
        return {
            "key": diff.key,
            "turn": diff.turn,
            "path": diff.path,
            "diff": diff.diff,
            "before_blob": cls.blob_ref(before, blobs),
            "after_blob": cls.blob_ref(after, blobs),
            "round": diff.round,
        }

    @staticmethod
    def blob_ref(text: str, blobs: dict[str, str]) -> str:
        if not text:
            return ""
        ref = hashlib.sha256(text.encode("utf-8")).hexdigest()
        blobs[ref] = text
        return ref

    @staticmethod
    def tool_record(record: ToolResultRecord) -> Json:
        return asdict(record)

    @staticmethod
    def tool_error(error: ToolErrorRecord) -> Json:
        return asdict(error)

    @staticmethod
    def turn_diffs(data: list[Json], blobs: dict[str, str]) -> list[TurnDiff]:
        diffs: list[TurnDiff] = []
        for d in data:
            # A blob missing from the log leaves the snapshot empty, which `net_diff_sections`
            # already handles by reconstructing that path's diff from its recorded hunks.
            before = blobs.get(d.get("before_blob", ""), "")
            after = blobs.get(d.get("after_blob", ""), "")
            before, after = TurnDiff.bounded_snapshots(before, after)
            diffs.append(TurnDiff(key=d["key"], turn=d["turn"], path=d["path"], diff=d["diff"], before=before, after=after, round=d.get("round", 0)))
        return diffs

    @classmethod
    def history_segment(cls, segment: HistorySegment, blobs: dict[str, str]) -> Json:
        """The evicted-message text is a content-addressed blob, written once per unique content,
        so appending a segment never re-serializes prior ones."""
        return {"key": segment.key, "title": segment.title, "blob": cls.blob_ref(segment.text, blobs)}

    @staticmethod
    def history(data: list[Json], blobs: dict[str, str]) -> list[HistorySegment]:
        return [HistorySegment(key=d["key"], title=d.get("title", ""), text=blobs.get(d.get("blob", ""), "")) for d in data]

    @classmethod
    def has_content(cls, session: Session) -> bool:
        state = session.state
        return any(
            (
                bool(cls.snapshot_messages(session)),
                bool(session.pending_user_inputs),
                bool(session.tool_records),
                bool(session.tool_errors),
                bool(session.turn_diffs),
                bool(session.history),
                bool(state.goal or state.plan or state.known or state.check or state.summary),
            )
        )

    @staticmethod
    def is_internal_message(message: Json) -> bool:
        return SessionSnapshotCodec.is_legacy_internal_message(message) or bool(message.get(SESSION_EVENT_KEY))

    @staticmethod
    def is_legacy_internal_message(message: Json) -> bool:
        return message.get("role") == "system" and str(message.get("content") or "").startswith("[Session resumed:")

    @classmethod
    def persistable_messages(cls, messages: list[Json]) -> list[Json]:
        return [message for message in messages if not cls.is_legacy_internal_message(message)]

    @classmethod
    def snapshot_messages(cls, session: Session) -> list[Json]:
        return cls.persistable_messages([*session.messages, *session._active_turn_messages])

    @staticmethod
    def state(state: AgentState) -> Json:
        data = asdict(state)
        return {
            key: data[key]
            for key in (
                "goal",
                "plan",
                "known",
                "check",
                "summary",
                "name",
                "name_source",
                "compaction_count",
                "round_count",
            )
        }

    @staticmethod
    def usage(usage: ModelUsage) -> Json:
        return asdict(usage)

    @classmethod
    def snapshot(cls, session: Session, blobs: dict[str, str]) -> Json:
        # fmt: off
        return {
            "uid": session.uid, "cwd": session.cwd, "created_at": session.created_at,
            "context_layout_version": session.context_layout_version, "messages": cls.snapshot_messages(session),
            "pending_user_inputs": [item.to_json() for item in session.pending_user_inputs],
            "state": cls.state(session.state), "usage": cls.usage(session.usage), "tool_counter": session.tool_counter,
            "tool_records": [cls.tool_record(record) for record in session.tool_records], "tool_errors": [cls.tool_error(error) for error in session.tool_errors],
            "turn_diffs": [cls.turn_diff(diff, blobs) for diff in session.turn_diffs],
            "history": [cls.history_segment(segment, blobs) for segment in session.history],
        }
        # fmt: on

    @classmethod
    def delta(cls, session: Session, saved: Json, blobs: dict[str, str]) -> Json:
        delta: Json = {
            "tool_counter": session.tool_counter,
            "usage": cls.usage(session.usage),
            "state": cls.state(session.state),
            "created_at": session.created_at,
            "context_layout_version": session.context_layout_version,
        }
        cls.add_sequence_delta(delta, "messages", cls.snapshot_messages(session), saved, "messages_len", "messages_digest")
        pending_user_inputs = [item.to_json() for item in session.pending_user_inputs]
        if cls.digest(pending_user_inputs) != saved.get("pending_user_inputs_digest", cls.digest([])):
            delta["pending_user_inputs"] = pending_user_inputs
        cls.add_sequence_delta(
            delta,
            "tool_records",
            [cls.tool_record(record) for record in session.tool_records],
            saved,
            "tool_records_len",
            "tool_records_digest",
        )
        cls.add_sequence_delta(
            delta,
            "tool_errors",
            [cls.tool_error(error) for error in session.tool_errors],
            saved,
            "tool_errors_len",
            "tool_errors_digest",
        )
        cls.add_turn_diffs_delta(delta, session.turn_diffs, saved, blobs)
        cls.add_history_delta(delta, session.history, saved, blobs)
        return delta

    @classmethod
    def add_sequence_delta(cls, delta: Json, key: str, current: list[Json], saved: Json, len_key: str, digest_key: str) -> None:
        last_len = saved.get(len_key, 0)
        if cls.digest(current[:last_len]) == saved.get(digest_key):
            if len(current) > last_len:
                delta[key] = current[last_len:]
        elif cls.digest(current) != saved.get(digest_key):
            delta[key + "_replace"] = current

    @classmethod
    def add_turn_diffs_delta(cls, delta: Json, current: list[TurnDiff], saved: Json, blobs: dict[str, str]) -> None:
        keys = [diff.key for diff in current]
        last_len = int(saved.get("turn_diffs_len", 0) or 0)
        saved_digest = saved.get("turn_diffs_keys_digest")
        if cls.digest(keys[:last_len]) == saved_digest:
            if len(current) > last_len:
                delta["turn_diffs"] = [cls.turn_diff(diff, blobs) for diff in current[last_len:]]
        elif cls.digest(keys) != saved_digest:
            # Only the references are rewritten here; the snapshots they point at are already
            # in the log, so a window rewrite stays small however large the files were.
            delta["turn_diffs_replace"] = [cls.turn_diff(diff, blobs) for diff in current]

    @classmethod
    def add_history_delta(cls, delta: Json, current: list[HistorySegment], saved: Json, blobs: dict[str, str]) -> None:
        keys = [segment.key for segment in current]
        last_len = int(saved.get("history_len", 0) or 0)
        saved_digest = saved.get("history_keys_digest")
        if cls.digest(keys[:last_len]) == saved_digest:
            if len(current) > last_len:
                delta["history"] = [cls.history_segment(segment, blobs) for segment in current[last_len:]]
        elif cls.digest(keys) != saved_digest:
            delta["history_replace"] = [cls.history_segment(segment, blobs) for segment in current]

    @classmethod
    def merge(cls, data: Json, delta: Json) -> None:
        cls.merge_sequence(data, delta, "messages")
        cls.merge_sequence(data, delta, "tool_records")
        cls.merge_sequence(data, delta, "tool_errors")
        cls.merge_sequence(data, delta, "turn_diffs")
        cls.merge_sequence(data, delta, "history")
        if "tool_counter" in delta:
            data["tool_counter"] = delta["tool_counter"]
        if "usage" in delta:
            data["usage"] = delta["usage"]
        if "state" in delta:
            data["state"] = delta["state"]
        if "pending_user_inputs" in delta:
            data["pending_user_inputs"] = delta["pending_user_inputs"]
        for key in ("created_at", "context_layout_version"):
            if key in delta:
                data[key] = delta[key]

    @staticmethod
    def merge_sequence(data: Json, delta: Json, key: str) -> None:
        replace_key = key + "_replace"
        if replace_key in delta:
            data[key] = delta[replace_key]
        if key in delta:
            data.setdefault(key, []).extend(delta[key])

    @staticmethod
    def model_usage(data: Json) -> ModelUsage:
        usage = ModelUsage()
        usage.calls = data.get("calls", 0)
        usage.prompt_tokens = data.get("prompt_tokens", 0)
        usage.completion_tokens = data.get("completion_tokens", 0)
        usage.total_tokens = data.get("total_tokens", 0)
        usage.cached_prompt_tokens = data.get("cached_prompt_tokens", 0)
        usage.cache_write_prompt_tokens = data.get("cache_write_prompt_tokens", 0)
        usage.last_cached_prompt_tokens = data.get("last_cached_prompt_tokens", 0)
        usage.last_cache_write_prompt_tokens = data.get("last_cache_write_prompt_tokens", 0)
        usage.last_prompt_tokens = data.get("last_prompt_tokens", 0)
        usage.last_prompt_budget = data.get("last_prompt_budget", 0)
        return usage

    @staticmethod
    def tool_records(data: list[Json]) -> list[ToolResultRecord]:
        # fmt: off
        return [ToolResultRecord(key=rec["key"], name=rec["name"], args=rec.get("args", []), output=rec.get("output", ""), note=rec.get("note", "")) for rec in data]
        # fmt: on

    @staticmethod
    def tool_errors(data: list[Json]) -> list[ToolErrorRecord]:
        return [ToolErrorRecord(key=err["key"], name=err["name"], args=err.get("args", []), error=err.get("error", "")) for err in data]


@dataclass(frozen=True)
class SessionEntry:
    """One stored session as a listing sees it: labels and facts, no conversation."""

    uid: str
    name: str
    opening: str
    rounds: int
    cwd: str
    updated_at: float
    path: str

    def matches(self, query: str) -> bool:
        needle = query.strip().lower()
        return bool(needle) and (self.uid.lower().startswith(needle) or needle in (self.name + " " + self.opening).lower())

    def label(self) -> str:
        return self.name or self.opening or self.uid


class SessionSnapshotStore:
    """Session logs live at `<data_dir>/projects/<project>/<uid>.jsonl`, one directory per working
    directory, each holding its own `latest` pointer. Sharding keeps a resume scoped to the project
    it belongs to and makes per-project listing and deletion a directory operation.

    Each log starts with a header line (`{"v": 2, "uid", "cwd", "created_at"}`) that gates the
    format version and makes a log self-describing when read by hand. The full snapshot is line 2;
    `blob` lines and deltas append from line 3."""

    FORMAT_VERSION: ClassVar[int] = 2
    PROJECTS_DIR: ClassVar[str] = "projects"
    META_SUFFIX: ClassVar[str] = ".meta.json"
    _SLUG_RE: ClassVar[re.Pattern] = re.compile(r"[^A-Za-z0-9._-]+")

    def __init__(self, session: Session):
        self.session = session

    def save(self) -> str:
        if not self.session._snapshot_saved and not SessionSnapshotCodec.has_content(self.session):
            return ""
        path = self.session_path(self.session.config.data_dir, self.session.cwd, self.session.uid)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        blobs: dict[str, str] = {}
        if not self.session._snapshot_saved:
            self.write_jsonl(path, self.header(self.session), mode="w")
            record = SessionSnapshotCodec.snapshot(self.session, blobs)
        else:
            record = SessionSnapshotCodec.delta(self.session, self.session._snapshot_saved, blobs)
        self.write_blobs(path, blobs)
        self.write_jsonl(path, record, mode="a")
        self.session._snapshot_saved = SessionSnapshotCodec.marker(self.session)
        if self.session.listed:
            # Workers never claim the latest pointer: `-c` must keep landing on the parent session.
            self.write_latest(self.session.config.data_dir, self.session.cwd, self.session.uid)
        self.write_meta()
        self.garbage_collect_assets()
        return self.session.uid

    def write_meta(self) -> None:
        """Keep what a listing shows beside the log, so browsing sessions never parses one.

        The log stays the source of truth; this is a cache of values derived from it, rewritten only
        when one of them changes. A missing or unreadable file costs a listing its labels for that
        session and nothing else, which is why it is never read back into a resumed session.
        """
        meta: Json = {
            "name": self.session.name,
            "opening": self.session.clip_name(self.session.opening_text()),
            "rounds": self.session.state.round_count,
            "cwd": self.session.cwd,
        }
        if meta == self.session._meta_written:
            return
        path = self.meta_path(self.session.config.data_dir, self.session.cwd, self.session.uid)
        with contextlib.suppress(OSError):
            self.write_jsonl(path, meta, mode="w")
            self.session._meta_written = meta

    def garbage_collect_assets(self) -> None:
        directory = self.session.images.assets_dir()
        if not os.path.isdir(directory):
            return
        refs: set[str] = set()
        for message in SessionSnapshotCodec.snapshot_messages(self.session):
            raw_images = message.get(IMAGE_REFS_KEY)
            if not isinstance(raw_images, list):
                continue
            refs.update(image.ref for raw in raw_images if (image := ImageRef.from_json(raw)) is not None)
        refs.update(image.ref for item in self.session.pending_user_inputs for image in item.images)
        refs.update(self.session.images.retained_refs)
        with contextlib.suppress(OSError):
            for entry in os.scandir(directory):
                if entry.is_file() and entry.name not in refs:
                    os.unlink(entry.path)
            if not any(os.scandir(directory)):
                os.rmdir(directory)

    def write_blobs(self, path: str, blobs: dict[str, str]) -> None:
        """Blob lines precede the record that references them, and each content hash is written to
        the log once. Content the session has already stored costs nothing to reference again."""
        for ref, text in blobs.items():
            if ref in self.session._blobs_written:
                continue
            self.write_jsonl(path, {"blob": ref, "text": text}, mode="a")
            self.session._blobs_written.add(ref)

    @classmethod
    def header(cls, session: Session) -> Json:
        return {"v": cls.FORMAT_VERSION, "uid": session.uid, "cwd": session.cwd, "created_at": session.created_at}

    @staticmethod
    def write_jsonl(path: str, data: Json, *, mode: str) -> None:
        with open(path, mode, encoding="utf-8") as file:
            file.write(json.dumps(data, ensure_ascii=False) + "\n")

    @classmethod
    def project_slug(cls, cwd: str) -> str:
        """Readable basename plus a hash of the real path: browsable, and still unique across
        same-named directories."""
        real = os.path.realpath(cwd)
        name = SessionSnapshotStore._SLUG_RE.sub("-", os.path.basename(real)).strip("-") or "root"
        return name + "-" + hashlib.sha256(real.encode("utf-8")).hexdigest()[:10]

    @classmethod
    def project_dir(cls, data_dir: str, cwd: str) -> str:
        return cls.path_for(data_dir, cls.PROJECTS_DIR, cls.project_slug(cwd))

    @classmethod
    def session_path(cls, data_dir: str, cwd: str, uid: str) -> str:
        return os.path.join(cls.project_dir(data_dir, cwd), uid + ".jsonl")

    @classmethod
    def meta_path(cls, data_dir: str, cwd: str, uid: str) -> str:
        return os.path.join(cls.project_dir(data_dir, cwd), uid + cls.META_SUFFIX)

    @classmethod
    def read_meta(cls, directory: str, uid: str) -> Json:
        try:
            with open(os.path.join(directory, uid + cls.META_SUFFIX), encoding="utf-8") as file:
                data = json.loads(file.read())
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    @classmethod
    def list_sessions(cls, data_dir: str, cwd: str = "", *, all_projects: bool = False) -> list[SessionEntry]:
        """Every stored session, newest first, without opening a single log.

        One directory scan plus one small sidecar read per session. A session whose sidecar is
        missing still lists — under its uid — because the log on disk is what makes it real.
        """
        directories = cls.project_dirs(data_dir) if all_projects else [cls.project_dir(data_dir, cwd)]
        entries: list[SessionEntry] = []
        for directory in directories:
            try:
                found = list(os.scandir(directory))
            except OSError:
                continue
            for entry in found:
                if not entry.name.endswith(".jsonl") or not entry.is_file():
                    continue
                uid = entry.name[:-6]
                if uid.endswith(".w"):
                    # Worker sessions are subordinates, not resumable sessions: hidden from listings.
                    continue
                meta = cls.read_meta(directory, uid)
                try:
                    rounds = int(meta.get("rounds") or 0)
                except (TypeError, ValueError):
                    # A sidecar is a cache, never the record; a malformed one loses its turn count,
                    # not the whole listing (str() already shields the text fields above).
                    rounds = 0
                with contextlib.suppress(OSError):
                    entries.append(
                        SessionEntry(
                            uid=uid,
                            name=str(meta.get("name") or ""),
                            opening=str(meta.get("opening") or ""),
                            rounds=rounds,
                            cwd=str(meta.get("cwd") or ""),
                            updated_at=entry.stat().st_mtime,
                            path=entry.path,
                        )
                    )
        return sorted(entries, key=lambda item: item.updated_at, reverse=True)

    @classmethod
    def search_sessions(cls, query: str, data_dir: str, cwd: str = "") -> list[SessionEntry]:
        """Sessions matching a uid prefix or a word in the name, this project before the rest.

        Searching only the current project would hide the session the user means whenever they have
        moved directories, so a miss here widens rather than fails.
        """
        matches = [entry for entry in cls.list_sessions(data_dir, cwd) if entry.matches(query)]
        if matches:
            return matches
        # Widen only on a miss: the tuple form scanned every project even when this one matched.
        return [entry for entry in cls.list_sessions(data_dir, all_projects=True) if entry.matches(query)]

    @classmethod
    def project_dirs(cls, data_dir: str) -> list[str]:
        try:
            return [entry.path for entry in os.scandir(cls.path_for(data_dir, cls.PROJECTS_DIR)) if entry.is_dir()]
        except OSError:
            return []

    @classmethod
    def find_session_path(cls, data_dir: str, uid: str) -> str:
        """Locate a session by UID alone. Projects are few, so a scan beats an index file that can
        drift out of sync with the directories it describes."""
        for directory in cls.project_dirs(data_dir):
            path = os.path.join(directory, uid + ".jsonl")
            if os.path.isfile(path):
                return path
        return ""

    @classmethod
    def clean_expired(cls, session: Session) -> int:
        days = session.settings.session_retention_days
        if days <= 0:
            return 0
        cutoff = time.time() - days * 86400
        removed = 0
        for directory in cls.project_dirs(session.config.data_dir):
            try:
                entries = list(os.scandir(directory))
            except OSError:
                continue
            stale_latest = False
            for entry in entries:
                if not entry.name.endswith(".jsonl") or not entry.is_file():
                    continue
                uid = entry.name[:-6]
                if uid == session.uid:
                    continue
                try:
                    # A worker outlives its parent only by accident: once the parent log is gone the
                    # worker is an orphan and expires even if its own mtime is fresh.
                    orphan_worker = uid.endswith(".w") and not os.path.isfile(os.path.join(directory, uid[:-2] + ".jsonl"))
                    if entry.stat().st_mtime >= cutoff and not orphan_worker:
                        continue
                    os.unlink(entry.path)
                    shutil.rmtree(os.path.join(directory, uid + ".assets"), ignore_errors=True)
                    # The sidecar describes a log that no longer exists; it expires with it.
                    with contextlib.suppress(OSError):
                        os.unlink(os.path.join(directory, uid + cls.META_SUFFIX))
                    removed += 1
                    stale_latest = stale_latest or cls.read_latest(directory) == uid
                except OSError:
                    continue
            if stale_latest:
                cls.clear_latest_dir(directory)
            cls.prune_empty(directory)
        return removed

    @classmethod
    def prune_empty(cls, directory: str) -> None:
        """Drop a project directory once its last session expires, so the store does not accumulate
        an entry for every directory minacode was ever started in."""
        with contextlib.suppress(OSError):
            if not any(entry.name.endswith(".jsonl") for entry in os.scandir(directory)):
                cls.clear_latest_dir(directory)
                os.rmdir(directory)

    @classmethod
    def write_latest(cls, data_dir: str, cwd: str, uid: str) -> None:
        with open(os.path.join(cls.project_dir(data_dir, cwd), "latest"), "w", encoding="utf-8") as file:
            file.write(uid)

    @classmethod
    def read_latest(cls, directory: str) -> str:
        try:
            with open(os.path.join(directory, "latest"), encoding="utf-8") as file:
                return file.read().strip()
        except OSError:
            return ""

    @classmethod
    def latest_uid(cls, data_dir: str, cwd: str) -> str:
        """The most recent session for `cwd`. A single pointer read: no directory scan, and a
        resume can never cross into another project."""
        directory = cls.project_dir(data_dir, cwd)
        uid = cls.read_latest(directory)
        if uid and os.path.isfile(os.path.join(directory, uid + ".jsonl")):
            return uid
        return cls.newest_uid(directory)

    @classmethod
    def newest_uid(cls, directory: str) -> str:
        """Fallback for a missing or stale pointer: newest log in the project by mtime."""
        try:
            entries = [entry for entry in os.scandir(directory) if entry.name.endswith(".jsonl") and entry.is_file() and not entry.name.endswith(".w.jsonl")]
        except OSError:
            return ""
        newest = max(entries, key=lambda entry: entry.stat().st_mtime, default=None)
        return newest.name[:-6] if newest else ""

    @classmethod
    def clear_latest_dir(cls, directory: str) -> None:
        with contextlib.suppress(OSError):
            os.unlink(os.path.join(directory, "latest"))

    @classmethod
    def load(cls, uid: str, config: Config | None = None, settings: RuntimeSettings | None = None, cwd: str = "") -> Session:
        if config is None:
            config = Config.from_dict(ConfigFile.load())
        if settings is None:
            settings = RuntimeSettings()
        cwd = cwd or os.getcwd()
        uid = cls.resolve_uid(uid, config.data_dir, cwd)
        path = cls.find_session_path(config.data_dir, uid)
        if not path:
            raise MinacodeError(f"Session snapshot not found: {uid} under {cls.path_for(config.data_dir, cls.PROJECTS_DIR)}")
        data, blobs, header = cls.read_merged(path)
        tool_records = SessionSnapshotCodec.tool_records(data.get("tool_records", []))
        raw_created_at = data.get("created_at", header.get("created_at"))
        if isinstance(raw_created_at, (int, float)):
            created_at = local_timestamp(float(raw_created_at))
        elif isinstance(raw_created_at, str) and raw_created_at.strip():
            created_at = raw_created_at.strip()
        else:
            created_at = local_timestamp()
        session = Session(
            cwd=data.get("cwd", cwd),
            config=config,
            settings=settings,
            messages=SessionSnapshotCodec.persistable_messages(data.get("messages", [])),
            state=AgentState(**data.get("state", {})),
            usage=SessionSnapshotCodec.model_usage(data.get("usage", {})),
            tool_counter=data.get("tool_counter", 0),
            tool_results={record.key: record.output for record in tool_records},
            tool_records=tool_records,
            tool_errors=SessionSnapshotCodec.tool_errors(data.get("tool_errors", [])),
            turn_diffs=SessionSnapshotCodec.turn_diffs(data.get("turn_diffs", []), blobs),
            history=SessionSnapshotCodec.history(data.get("history", []), blobs),
            pending_user_inputs=[item for value in data.get("pending_user_inputs", []) if (item := QueuedInput.from_json(value)) is not None],
            uid=data.get("uid", uid),
            resumed=True,
            created_at=created_at,
            context_layout_version=int(data.get("context_layout_version", 1) or 1),
        )
        # Mark the loaded prefix before appending durable lifecycle/checkpoint events, so the next
        # snapshot writes them as an append-only delta.
        session._snapshot_saved = SessionSnapshotCodec.marker(session)
        if session.context_layout_version < CONTEXT_LAYOUT_VERSION:
            if session.state.goal or session.state.plan or session.state.known or session.state.check or session.state.summary:
                session.messages.append(session.state_checkpoint_event())
            session.context_layout_version = CONTEXT_LAYOUT_VERSION
        resumed_at = local_timestamp()
        session.messages.append(
            {
                "role": "user",
                "content": f'<session_event type="resumed" at="{resumed_at}" />',
                SESSION_EVENT_KEY: "resumed",
            }
        )
        session._blobs_written = set(blobs)
        return session

    @classmethod
    def resolve_uid(cls, uid: str, data_dir: str, cwd: str) -> str:
        """`latest`/`last` mean the latest session *in this project*, never one from elsewhere.

        Anything else is a uid, or failing that a search: nobody retypes a uid they can describe.
        An ambiguous search names its candidates rather than picking one of them.
        """
        if uid in {"latest", "last"}:
            resolved = cls.latest_uid(data_dir, cwd)
            if not resolved:
                raise MinacodeError(f"No previous session for this project: {cwd}")
            return resolved
        if cls.find_session_path(data_dir, uid):
            return uid
        matches = cls.search_sessions(uid, data_dir, cwd)
        if len(matches) == 1:
            return matches[0].uid
        if matches:
            listed = "\n".join(f"  {entry.uid}  {entry.label()}" for entry in matches[:5])
            more = f"\n  ... and {len(matches) - 5} more" if len(matches) > 5 else ""
            raise MinacodeError(f"{len(matches)} sessions match {uid!r}:\n{listed}{more}")
        return uid

    @classmethod
    def read_merged(cls, path: str) -> tuple[Json, dict[str, str], Json]:
        merged: Json | None = None
        blobs: dict[str, str] = {}
        header: Json = {}
        with open(path, encoding="utf-8") as file:
            for index, line in enumerate(file):
                line = line.strip()
                if not line:
                    continue
                parsed = json.loads(line)
                if index == 0:
                    cls.check_header(parsed, path)
                    header = parsed
                elif "blob" in parsed:
                    blobs[parsed["blob"]] = parsed.get("text", "")
                elif merged is None:
                    merged = parsed
                else:
                    SessionSnapshotCodec.merge(merged, parsed)
        if merged is None:
            raise MinacodeError(f"Empty session file: {path}")
        return merged, blobs, header

    @classmethod
    def check_header(cls, header: Json, path: str) -> None:
        version = header.get("v")
        if version != cls.FORMAT_VERSION:
            raise MinacodeError(f"Unsupported session format v{version} (expected v{cls.FORMAT_VERSION}): {path}")

    @staticmethod
    def path_for(data_dir: str, *parts: str) -> str:
        return os.path.abspath(os.path.join(os.path.expanduser(data_dir), *parts))


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
    messages: list[Json] = field(default_factory=list)
    state: AgentState = field(default_factory=AgentState)
    tool_results: dict[str, str] = field(default_factory=dict)
    tool_records: list[ToolResultRecord] = field(default_factory=list)
    tool_errors: list[ToolErrorRecord] = field(default_factory=list)
    pending_user_inputs: list[QueuedInput] = field(default_factory=list)
    quick_hints: tuple[str, ...] = field(default_factory=tuple)  # transient offered next-step inputs; never serialized, cleared each turn
    # Worker handoff (see DESIGN.md): the second session this one delegates to, and its per-session
    # projection knobs. None of these are persisted — SessionSnapshotCodec.snapshot is an explicit
    # whitelist, so they return to their defaults on load and must be re-set by the delegate caller.
    system_prompt: str = SYSTEM_PROMPT  # role definition; the parent's default is unchanged
    tool_names: tuple[str, ...] = ()  # empty tuple = no filtering (parent behavior)
    listed: bool = True  # False -> no latest pointer, hidden from /sessions
    worker: Session | None = None  # runtime handle of the delegated session
    _agent: Agent | None = None  # runtime handle of the worker's Agent; same lifetime as the worker Session
    tool_counter: int = 0
    turn_diffs: list[TurnDiff] = field(default_factory=list)
    history: list[HistorySegment] = field(default_factory=list)
    jobs: dict[str, BackgroundJob] = field(default_factory=dict)
    job_counter: int = 0
    usage: ModelUsage = field(default_factory=ModelUsage)
    update: UpdateStatus = field(default_factory=UpdateStatus)
    mcp: MCPManager | None = None
    skills: SkillLibrary | None = None
    images: ImageInputs = field(init=False, repr=False)
    _gitignore_cache: dict[str, tuple[int, list[str]]] = field(default_factory=dict)
    uid: str = ""
    resumed: bool = False
    created_at: str = field(default_factory=local_timestamp)
    context_layout_version: int = CONTEXT_LAYOUT_VERSION
    _snapshot_saved: dict = field(default_factory=dict)
    _blobs_written: set[str] = field(default_factory=set)
    _meta_written: dict = field(default_factory=dict)
    _active_turn_messages: list[Json] = field(default_factory=list)
    _queue_lock: threading.RLock = field(default_factory=threading.RLock)
    _snapshot_lock: threading.RLock = field(default_factory=threading.RLock)

    def __post_init__(self) -> None:
        self.images = ImageInputs(self)
        if not self.uid:
            self.uid = datetime.now().strftime("%Y%m%d%H%M%S") + "-" + str(uuid.uuid4())[:12]  # noqa: DTZ005 - IDs intentionally use local wall time.
        if self.system_info is None:
            self.system_info = SystemInfo.detect(self.cwd)
        if self.mcp is None:
            from minacode.mcp import MCPManager  # local import: mcp is built on top of session

            self.mcp = MCPManager(self)
        if self.skills is None:
            from minacode.skill import SkillLibrary  # local import: skill is built on top of session

            self.skills = SkillLibrary.load(self)

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
        self.turn_diffs.append(TurnDiff(key, turn, path, diff, before, after, round))
        if len(self.turn_diffs) > 100:
            self.turn_diffs.pop(0)

    @classmethod
    def from_config_file(cls, *, path: str | None = None, yolo: bool = False, theme: str = "") -> Session:
        data = ConfigFile.load(path)
        return cls(config=Config.from_dict(data), settings=RuntimeSettings.from_dict(data, yolo=yolo, theme=theme))

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

    def data_path(self, *parts: str) -> str:
        root = os.path.expanduser(self.config.data_dir)
        return os.path.abspath(os.path.join(root if os.path.isabs(root) else os.path.join(self.cwd, root), *parts))

    def running_jobs(self) -> list[BackgroundJob]:
        for job in self.jobs.values():
            job.update_status()
        return [job for job in self.jobs.values() if job.status == "running"]

    def missing_config(self) -> list[str]:
        provider = self.config.provider
        return [key for key, value in (("provider.url", provider.url), ("provider.key", provider.key), ("provider.model", provider.model)) if not value]

    def store_tool_result(self, name: str, args: ToolArgs, output: str, note: str = "") -> str:
        self.tool_counter += 1
        key = f"tr.{self.tool_counter}"
        args, output = Text.value(list(args)), Text.clean(output)
        self.tool_results[key] = output
        self.tool_records.append(ToolResultRecord(key, name, args, output, note))
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

    def set_quick_hints(self, hints: list[str]) -> None:
        """Transient next-step inputs offered at the idle prompt; replaced wholesale, never snapshotted."""
        with self._queue_lock:
            self.quick_hints = tuple(hints)

    def clear_quick_hints(self) -> None:
        with self._queue_lock:
            self.quick_hints = ()

    @staticmethod
    def net_diff_for_path(status: str, path: str, before: str, after: str) -> tuple[str, str, str] | None:
        from minacode.tools import ReadTool  # local import: tools is built on top of session

        if before == after:
            return None
        text = "".join(
            difflib.unified_diff(ReadTool.split_lines(before), ReadTool.split_lines(after), fromfile="/dev/null" if not before else path, tofile=path)
        )
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
        return SessionSnapshotStore.load(uid, config=config, settings=settings, cwd=cwd)
