"""minacode session store: durable snapshot encoding and persistence."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import TYPE_CHECKING, ClassVar

from minacode.base import SESSION_EVENT_KEY, Json, MinacodeError, ModelUsage, Text
from minacode.config import Config, ConfigFile, RuntimeSettings
from minacode.image import IMAGE_REFS_KEY, ImageInputs, ImageRef

if TYPE_CHECKING:
    from minacode.session import (
        AgentState,
        HistorySegment,
        Session,
        ToolErrorRecord,
        ToolResultRecord,
    )
    from minacode.session import (
        TurnDiff as TurnDiffT,
    )


CONTEXT_LAYOUT_VERSION = 2
TRANSCRIPT_SYNC_VERSION = 1


def local_timestamp(value: float | None = None) -> str:
    """A user-readable local wall-clock timestamp with its numeric UTC offset."""
    current = datetime.now().astimezone() if value is None else datetime.fromtimestamp(value).astimezone()
    return current.isoformat(timespec="seconds")


class SessionSnapshotCodec:
    """Decide what is durable, and encode it so saving stays cheap as the session grows.

    A session is snapshotted after every response and tool batch, so rewriting all of it each time
    would make saving cost more the longer the session runs. Bounded model state uses prefix digests;
    the unbounded visible transcript uses an append-only length and tail sentinel, while the active
    turn is replaced separately. The loader replays those deltas onto the last full snapshot.

    Large repeated text — file snapshots behind diffs, message text evicted by compaction — is stored
    once per unique content and referenced by hash, because the same content routinely appears as one
    edit's `before` and the previous edit's `after`.

    Legacy system-role resume markers are filtered during migration. New lifecycle events are
    append-only user messages: durable model context with protocol-neutral metadata hidden from UI.
    """

    TRANSCRIPT_TOOL_ARGUMENT_CHAR_LIMIT: ClassVar[int] = 64 * 1024

    @staticmethod
    def digest(value: object) -> str:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @classmethod
    def marker(cls, session: Session) -> Json:
        messages = cls.snapshot_messages(session)
        transcript_messages = cls.snapshot_transcript_messages(session)
        active_transcript_messages = cls.active_transcript_messages(session)
        records = [cls.tool_record(record) for record in session.tool_records]
        errors = [cls.tool_error(error) for error in session.tool_errors]
        turn_diff_keys = [diff.key for diff in session.turn_diffs]
        transcript_diff_len = len(session.transcript_turn_diffs)
        transcript_diff_tail = cls.transcript_turn_diff(session.transcript_turn_diffs[-1]) if session.transcript_turn_diffs else None
        # fmt: off
        return {
            "messages_len": len(messages), "messages_digest": cls.digest(messages), "tool_counter": session.tool_counter,
            "transcript_messages_len": len(transcript_messages), "transcript_messages_tail_digest": cls.tail_digest(transcript_messages),
            "active_transcript_messages_digest": cls.digest(active_transcript_messages),
            "pending_user_inputs_digest": cls.digest([item.to_json() for item in session.pending_user_inputs]),
            "tool_records_len": len(records), "tool_records_digest": cls.digest(records),
            "tool_errors_len": len(errors), "tool_errors_digest": cls.digest(errors),
            "turn_diffs_len": len(turn_diff_keys), "turn_diffs_keys_digest": cls.digest(turn_diff_keys),
            "transcript_turn_diffs_len": transcript_diff_len, "transcript_turn_diffs_tail_digest": cls.digest(transcript_diff_tail),
            "history_len": len(session.history), "history_keys_digest": cls.digest([seg.key for seg in session.history]),
        }
        # fmt: on

    @classmethod
    def tail_digest(cls, values: list[Json]) -> str:
        return cls.digest(values[-1] if values else None)

    @classmethod
    def turn_diff(cls, diff: TurnDiffT, blobs: dict[str, str]) -> Json:
        from minacode.session import TurnDiff

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
    def turn_diffs(data: list[Json], blobs: dict[str, str]) -> list[TurnDiffT]:
        from minacode.session import TurnDiff

        diffs = []
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
        from minacode.session import HistorySegment

        return [HistorySegment(key=d["key"], title=d.get("title", ""), text=blobs.get(d.get("blob", ""), "")) for d in data]

    @classmethod
    def has_content(cls, session: Session) -> bool:
        state = session.state
        return any(
            (
                bool(cls.snapshot_messages(session)),
                bool(cls.snapshot_transcript_messages(session) or cls.active_transcript_messages(session)),
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
    def transcript_message(cls, message: Json) -> Json | None:
        """Project a model message into visible, provider-neutral resume history."""
        if cls.is_internal_message(message) or ImageInputs.is_tool_observation(message):
            return None
        role = str(message.get("role") or "")
        if role == "user":
            return {"role": "user", "content": ImageInputs.label_text(message)}
        if role == "assistant":
            projected: Json = {"role": "assistant", "content": message.get("content")}
            calls: list[Json] = []
            for raw in message.get("tool_calls") or []:
                if not isinstance(raw, dict):
                    continue
                raw_function = raw.get("function")
                if not isinstance(raw_function, dict):
                    continue
                arguments = raw_function.get("arguments", "")
                serialized_arguments = arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
                arguments_truncated = len(serialized_arguments) > cls.TRANSCRIPT_TOOL_ARGUMENT_CHAR_LIMIT
                function = {
                    "name": str(raw_function.get("name") or ""),
                    "arguments": "{}" if arguments_truncated else arguments,
                }
                call = {"id": str(raw.get("id") or ""), "type": "function", "function": function}
                if arguments_truncated:
                    call["arguments_truncated"] = True
                calls.append(call)
            if calls:
                projected["tool_calls"] = calls
            return Text.value(projected)
        if role == "tool":
            projected = {"role": "tool", "tool_call_id": str(message.get("tool_call_id") or "")}
            if "status" in message:
                projected["result_key"] = str(message.get("result_key") or "")
                projected["status"] = str(message.get("status") or "unknown")
                return projected
            content = str(message.get("content") or "")
            if content.startswith("tool "):
                first_line = content.splitlines()[0]
                key_match = re.match(r"tool (tr\.\d+)\b", first_line)
                projected["result_key"] = key_match.group(1) if key_match else ""
                projected["status"] = "failed" if first_line.startswith("tool - ") else "ok"
            return projected
        return None

    @classmethod
    def transcript_messages(cls, messages: list[Json]) -> list[Json]:
        return [projected for message in messages if (projected := cls.transcript_message(message)) is not None]

    @classmethod
    def snapshot_messages(cls, session: Session) -> list[Json]:
        return cls.persistable_messages([*session.messages, *session._active_turn_messages])

    @classmethod
    def snapshot_transcript_messages(cls, session: Session) -> list[Json]:
        # Runtime transcript messages are already semantic projections and never internal events.
        # Returning the append-only list directly keeps checkpoint bookkeeping O(1).
        return session.transcript_messages

    @classmethod
    def active_transcript_messages(cls, session: Session) -> list[Json]:
        return cls.persistable_messages(session._active_transcript_messages)

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
            "transcript_messages": cls.snapshot_transcript_messages(session),
            "active_transcript_messages": cls.active_transcript_messages(session), "transcript_sync": TRANSCRIPT_SYNC_VERSION,
            "pending_user_inputs": [item.to_json() for item in session.pending_user_inputs],
            "state": cls.state(session.state), "usage": cls.usage(session.usage), "tool_counter": session.tool_counter,
            "tool_records": [cls.tool_record(record) for record in session.tool_records], "tool_errors": [cls.tool_error(error) for error in session.tool_errors],
            "turn_diffs": [cls.turn_diff(diff, blobs) for diff in session.turn_diffs],
            "transcript_turn_diffs": [cls.transcript_turn_diff(diff) for diff in session.transcript_turn_diffs],
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
            "transcript_sync": TRANSCRIPT_SYNC_VERSION,
        }
        cls.add_sequence_delta(delta, "messages", cls.snapshot_messages(session), saved, "messages_len", "messages_digest")
        cls.add_append_only_delta(delta, "transcript_messages", cls.snapshot_transcript_messages(session), saved)
        active_transcript_messages = cls.active_transcript_messages(session)
        if cls.digest(active_transcript_messages) != saved.get("active_transcript_messages_digest", cls.digest([])):
            delta["active_transcript_messages_replace"] = active_transcript_messages
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
        cls.add_transcript_turn_diffs_delta(delta, session.transcript_turn_diffs, saved)
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
    def add_append_only_delta(cls, delta: Json, key: str, current: list[Json], saved: Json) -> None:
        last_len = int(saved.get(key + "_len", 0) or 0)
        saved_tail = saved.get(key + "_tail_digest", cls.digest(None))
        prefix_is_saved = len(current) >= last_len and (last_len == 0 or cls.digest(current[last_len - 1]) == saved_tail)
        if prefix_is_saved:
            if len(current) > last_len:
                delta[key] = current[last_len:]
        else:
            delta[key + "_replace"] = current

    @classmethod
    def add_turn_diffs_delta(cls, delta: Json, current: list[TurnDiffT], saved: Json, blobs: dict[str, str]) -> None:
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

    @staticmethod
    def transcript_turn_diff(diff: TurnDiffT) -> Json:
        from minacode.session import TurnDiff

        return {"key": diff.key, "turn": diff.turn, "path": diff.path, "diff": TurnDiff.bounded_transcript(diff.diff), "round": diff.round}

    @classmethod
    def add_transcript_turn_diffs_delta(cls, delta: Json, current: list[TurnDiffT], saved: Json) -> None:
        key = "transcript_turn_diffs"
        last_len = int(saved.get(key + "_len", 0) or 0)
        saved_tail = saved.get(key + "_tail_digest", cls.digest(None))
        prefix_is_saved = len(current) >= last_len and (last_len == 0 or cls.digest(cls.transcript_turn_diff(current[last_len - 1])) == saved_tail)
        if prefix_is_saved:
            if len(current) > last_len:
                delta[key] = [cls.transcript_turn_diff(diff) for diff in current[last_len:]]
        else:
            delta[key + "_replace"] = [cls.transcript_turn_diff(diff) for diff in current]

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
        cls.merge_sequence(data, delta, "transcript_messages")
        cls.merge_sequence(data, delta, "active_transcript_messages")
        cls.merge_sequence(data, delta, "tool_records")
        cls.merge_sequence(data, delta, "tool_errors")
        cls.merge_sequence(data, delta, "turn_diffs")
        cls.merge_sequence(data, delta, "transcript_turn_diffs")
        cls.merge_sequence(data, delta, "history")
        if "tool_counter" in delta:
            data["tool_counter"] = delta["tool_counter"]
        if "usage" in delta:
            data["usage"] = delta["usage"]
        if "state" in delta:
            data["state"] = delta["state"]
        if "pending_user_inputs" in delta:
            data["pending_user_inputs"] = delta["pending_user_inputs"]
        for key in ("created_at", "context_layout_version", "transcript_sync"):
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
        from minacode.session import ToolResultRecord

        # fmt: off
        return [ToolResultRecord(key=rec["key"], name=rec["name"], args=rec.get("args", []), output=rec.get("output", ""), note=rec.get("note", "")) for rec in data]
        # fmt: on

    @staticmethod
    def tool_errors(data: list[Json]) -> list[ToolErrorRecord]:
        from minacode.session import ToolErrorRecord

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
            expiring_parents: set[str] = set()
            for entry in entries:
                if not entry.name.endswith(".jsonl") or not entry.is_file():
                    continue
                uid = entry.name[:-6]
                if uid == session.uid or uid.endswith(".w"):
                    continue
                with contextlib.suppress(OSError):
                    if entry.stat().st_mtime < cutoff:
                        expiring_parents.add(uid)
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
                    orphan_worker = uid.endswith(".w") and (uid[:-2] in expiring_parents or not os.path.isfile(os.path.join(directory, uid[:-2] + ".jsonl")))
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
        from minacode.session import AgentState, QueuedInput, Session, local_timestamp

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
        messages = SessionSnapshotCodec.persistable_messages(data.get("messages", []))
        tool_records = SessionSnapshotCodec.tool_records(data.get("tool_records", []))
        turn_diffs = SessionSnapshotCodec.turn_diffs(data.get("turn_diffs", []), blobs)
        raw_transcript_messages = data.get("transcript_messages", [])
        raw_active_transcript_messages = data.get("active_transcript_messages", [])
        has_transcript = any(key in data for key in ("transcript_messages", "active_transcript_messages", "transcript_turn_diffs", "transcript_sync"))
        if has_transcript:
            committed_transcript_messages = SessionSnapshotCodec.transcript_messages(raw_transcript_messages)
            active_transcript_messages = SessionSnapshotCodec.transcript_messages(raw_active_transcript_messages)
            transcript_messages = [*committed_transcript_messages, *active_transcript_messages]
            # Read-only bridge for the first transcript snapshot shape; new semantic tool events
            # carry their own call id/status/key and never write this duplicate metadata.
            transcript_tool_records = SessionSnapshotCodec.tool_records(data.get("transcript_tool_records", []))
            transcript_turn_diffs = SessionSnapshotCodec.turn_diffs(data.get("transcript_turn_diffs", []), {})
        else:
            # Older snapshots used model context as their only transcript. Preserve what still
            # exists there; conversation already removed by an old compaction cannot be recovered.
            committed_transcript_messages = []
            active_transcript_messages = []
            transcript_messages = SessionSnapshotCodec.transcript_messages(messages)
            transcript_tool_records = list(tool_records)
            transcript_turn_diffs = list(turn_diffs)
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
            messages=messages,
            transcript_messages=transcript_messages,
            state=AgentState(**data.get("state", {})),
            usage=SessionSnapshotCodec.model_usage(data.get("usage", {})),
            tool_counter=data.get("tool_counter", 0),
            tool_results={record.key: record.output for record in tool_records},
            tool_records=tool_records,
            transcript_tool_records=transcript_tool_records,
            tool_errors=SessionSnapshotCodec.tool_errors(data.get("tool_errors", [])),
            turn_diffs=turn_diffs,
            transcript_turn_diffs=transcript_turn_diffs,
            transcript_incomplete=bool(data.get("_transcript_incomplete")),
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
        # Active transcript data is flattened into committed memory on load. Keep the marker at
        # the on-disk boundary so the next save appends that partial turn once, then clears active.
        session._snapshot_saved.update(
            {
                "transcript_messages_len": len(committed_transcript_messages),
                "transcript_messages_tail_digest": SessionSnapshotCodec.tail_digest(committed_transcript_messages),
                "active_transcript_messages_digest": SessionSnapshotCodec.digest(active_transcript_messages),
            }
        )
        if not has_transcript:
            session._snapshot_saved.update(
                {
                    "transcript_messages_len": 0,
                    "transcript_messages_tail_digest": SessionSnapshotCodec.digest(None),
                    "transcript_turn_diffs_len": 0,
                    "transcript_turn_diffs_tail_digest": SessionSnapshotCodec.digest(None),
                }
            )
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
        transcript_sync_seen = False
        transcript_incomplete = False
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
                    transcript_sync_seen = "transcript_sync" in parsed
                else:
                    if transcript_sync_seen and "transcript_sync" not in parsed:
                        transcript_incomplete = True
                    SessionSnapshotCodec.merge(merged, parsed)
                    transcript_sync_seen = transcript_sync_seen or "transcript_sync" in parsed
        if merged is None:
            raise MinacodeError(f"Empty session file: {path}")
        if transcript_incomplete:
            merged["_transcript_incomplete"] = True
        return merged, blobs, header

    @classmethod
    def check_header(cls, header: Json, path: str) -> None:
        version = header.get("v")
        if version != cls.FORMAT_VERSION:
            raise MinacodeError(f"Unsupported session format v{version} (expected v{cls.FORMAT_VERSION}): {path}")

    @staticmethod
    def path_for(data_dir: str, *parts: str) -> str:
        return os.path.abspath(os.path.join(os.path.expanduser(data_dir), *parts))
