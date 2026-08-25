"""minacode session snapshot codec: what is durable, and how it is encoded.

Pure translation between a `Session` and the JSON a log line carries -- no filesystem, no
process state, every method a `@staticmethod` or `@classmethod`. `SessionSnapshotStore` owns
the writing; keeping the two apart means the encoding can be reasoned about and tested without
a directory, and the store can be read as file handling without the schema in the way.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, fields
from typing import TYPE_CHECKING, ClassVar

from minacode.base import SESSION_EVENT_KEY, Json, ModelUsage, Text
from minacode.image import ImageInputs

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


TRANSCRIPT_SYNC_VERSION = 1


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
            "provider_overrides_digest": cls.digest(session.provider_overrides),
        }
        # fmt: on

    @classmethod
    def tail_digest(cls, values: list[Json]) -> str:
        return cls.digest(values[-1] if values else None)

    @classmethod
    def turn_diff(cls, diff: TurnDiffT, blobs: dict[str, str]) -> Json:
        """File snapshots are stored by content hash, not inline. Editing one file repeatedly makes
        each version appear twice — as one edit's `after` and the next edit's `before` — and a
        rewrite of the retained window would otherwise re-serialize every snapshot again."""
        from minacode.session import TurnDiff

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
        so appending a segment never re-serializes prior ones. The compaction metadata is small and
        stays inline; a snapshot written before it existed decodes with the defaults, and every
        reader treats an empty field as "not recorded" rather than a value."""
        return {
            "key": segment.key,
            "title": segment.title,
            "blob": cls.blob_ref(segment.text, blobs),
            "created_at": segment.created_at,
            "scope": segment.scope,
            "trigger": segment.trigger,
            "fallback": segment.fallback,
            "messages": segment.messages,
            "summary": segment.summary,
            "model": segment.model,
        }

    @staticmethod
    def history(data: list[Json], blobs: dict[str, str]) -> list[HistorySegment]:
        from minacode.session import HistorySegment

        return [
            HistorySegment(
                key=d["key"],
                title=d.get("title", ""),
                text=blobs.get(d.get("blob", ""), ""),
                created_at=str(d.get("created_at", "") or ""),
                scope=str(d.get("scope", "") or ""),
                trigger=str(d.get("trigger", "") or ""),
                fallback=bool(d.get("fallback", False)),
                messages=int(d.get("messages", 0) or 0),
                summary=str(d.get("summary", "") or ""),
                model=str(d.get("model", "") or ""),
            )
            for d in data
        ]

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
    def agent_state(value: object) -> AgentState:
        """Decode known state fields and ignore fields retired by newer versions."""

        from minacode.session import AgentState

        data = value if isinstance(value, dict) else {}
        known = {item.name for item in fields(AgentState)}
        return AgentState(**{key: item for key, item in data.items() if key in known})

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
            "compaction_usage": cls.usage(session.compaction_usage),
            "tool_records": [cls.tool_record(record) for record in session.tool_records], "tool_errors": [cls.tool_error(error) for error in session.tool_errors],
            "turn_diffs": [cls.turn_diff(diff, blobs) for diff in session.turn_diffs],
            "transcript_turn_diffs": [cls.transcript_turn_diff(diff) for diff in session.transcript_turn_diffs],
            "history": [cls.history_segment(segment, blobs) for segment in session.history],
            "provider_overrides": dict(session.provider_overrides),
        }
        # fmt: on

    @classmethod
    def delta(cls, session: Session, saved: Json, blobs: dict[str, str]) -> Json:
        delta: Json = {
            "tool_counter": session.tool_counter,
            "usage": cls.usage(session.usage),
            "compaction_usage": cls.usage(session.compaction_usage),
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
        if cls.digest(session.provider_overrides) != saved.get("provider_overrides_digest", cls.digest({})):
            delta["provider_overrides"] = dict(session.provider_overrides)
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
        if "compaction_usage" in delta:
            data["compaction_usage"] = delta["compaction_usage"]
        if "state" in delta:
            data["state"] = delta["state"]
        if "pending_user_inputs" in delta:
            data["pending_user_inputs"] = delta["pending_user_inputs"]
        if "provider_overrides" in delta:
            data["provider_overrides"] = delta["provider_overrides"]
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
