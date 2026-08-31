"""wizolt session store: where snapshots live on disk and how they are written.

File handling only -- paths, the append-only log, blob dedup, windowing, listing, GC. The schema
those lines carry belongs to `SessionSnapshotCodec` in codec.py, which this calls and never
reimplements.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, ClassVar

from wizolt.base import SESSION_EVENT_KEY, TOOL_OUTPUT_ASSET_SUFFIX, Json, WizoltError
from wizolt.image import IMAGE_REFS_KEY, ImageRef
from wizolt.session.codec import SessionSnapshotCodec

if TYPE_CHECKING:
    from wizolt.config import Config, RuntimeSettings
    from wizolt.session import Session


CONTEXT_LAYOUT_VERSION = 2
# A ".image-*" staging file older than this is crash residue, not an in-flight write: the
# copy+replace window in ImageInputs._store is milliseconds, so GC may collect it (and clear
# the way for the assets directory's removal) without racing a real save.
IMAGE_STAGING_MAX_AGE = 60.0


def local_timestamp(value: float | None = None) -> str:
    """A user-readable local wall-clock timestamp with its numeric UTC offset."""
    current = datetime.now().astimezone() if value is None else datetime.fromtimestamp(value).astimezone()
    return current.isoformat(timespec="seconds")


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

    # v3: source views (view.N) replaced content-hashed line anchors; older sessions are refused
    # because their assistant messages and tool results teach the removed anchor schema.
    FORMAT_VERSION: ClassVar[int] = 3
    PROJECTS_DIR: ClassVar[str] = "projects"
    META_SUFFIX: ClassVar[str] = ".meta.json"
    _SLUG_RE: ClassVar[re.Pattern] = re.compile(r"[^A-Za-z0-9._-]+")
    # A session preview reads only the tail of the log, starting from a small window and widening
    # it geometrically until it holds enough text or hits the budget: most sessions are fully
    # covered by the first small read, and only a log whose newest turns are megabytes of tool
    # output costs the bigger reads.
    TAIL_START: ClassVar[int] = 64 * 1024
    TAIL_BUDGET: ClassVar[int] = 8 * 1024 * 1024

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
        # Images are not the only thing in here: ContextManager.materialize_output writes a
        # truncated tool result's full text as "<key>.txt", and the marker in the conversation
        # promises that path. Retain one for as long as its tool result is retained, so the two
        # expire together and the promise is never left pointing at a deleted file.
        refs.update(key + TOOL_OUTPUT_ASSET_SUFFIX for key in self.session.tool_results)
        with contextlib.suppress(OSError):
            for entry in os.scandir(directory):
                if entry.name.startswith("."):
                    # ImageInputs._store stages uploads as ".image-*" (mkstemp in this dir) before
                    # os.replace; deleting one mid-flight breaks the rename, so a recent staging
                    # file is spared. One older than IMAGE_STAGING_MAX_AGE is crash residue -- the
                    # copy+replace window is milliseconds -- and is collected so it cannot pile up
                    # or block the assets directory's removal.
                    if entry.name.startswith(".image-") and entry.is_file() and time.time() - entry.stat().st_mtime > IMAGE_STAGING_MAX_AGE:
                        os.unlink(entry.path)
                    continue
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
        an entry for every directory wizolt was ever started in."""
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
    def load(cls, uid: str, config: Config, settings: RuntimeSettings, cwd: str = "") -> Session:
        from wizolt.session import QueuedInput, Session, local_timestamp

        cwd = cwd or os.getcwd()
        uid = cls.resolve_uid(uid, config.data_dir, cwd)
        path = cls.find_session_path(config.data_dir, uid)
        if not path:
            raise WizoltError(f"Session snapshot not found: {uid} under {cls.path_for(config.data_dir, cls.PROJECTS_DIR)}")
        data, blobs, header = cls.read_merged(path)
        messages = SessionSnapshotCodec.persistable_messages(data.get("messages", []))
        tool_records = SessionSnapshotCodec.tool_records(data.get("tool_records", []))
        turn_diffs = SessionSnapshotCodec.turn_diffs(data.get("turn_diffs", []), blobs)
        source_views = SessionSnapshotCodec.source_views(data.get("source_views", []), blobs)
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
            provider_overrides=data.get("provider_overrides") or {},
            messages=messages,
            transcript_messages=transcript_messages,
            state=SessionSnapshotCodec.agent_state(data.get("state", {})),
            usage=SessionSnapshotCodec.model_usage(data.get("usage", {})),
            compaction_usage=SessionSnapshotCodec.model_usage(data.get("compaction_usage", {})),
            tool_counter=data.get("tool_counter", 0),
            tool_results={record.key: record.output for record in tool_records},
            tool_records=tool_records,
            transcript_tool_records=transcript_tool_records,
            tool_errors=SessionSnapshotCodec.tool_errors(data.get("tool_errors", [])),
            turn_diffs=turn_diffs,
            transcript_turn_diffs=transcript_turn_diffs,
            transcript_incomplete=bool(data.get("_transcript_incomplete")),
            history=SessionSnapshotCodec.history(data.get("history", []), blobs),
            source_views={view.key: view for view in source_views},
            source_view_counter=max((int(view.key.split(".", 1)[1]) for view in source_views), default=0),
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
                raise WizoltError(f"No previous session for this project: {cwd}")
            return resolved
        if cls.find_session_path(data_dir, uid):
            return uid
        matches = cls.search_sessions(uid, data_dir, cwd)
        if len(matches) == 1:
            return matches[0].uid
        if matches:
            listed = "\n".join(f"  {entry.uid}  {entry.label()}" for entry in matches[:5])
            more = f"\n  ... and {len(matches) - 5} more" if len(matches) > 5 else ""
            raise WizoltError(f"{len(matches)} sessions match {uid!r}:\n{listed}{more}")
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
            raise WizoltError(f"Empty session file: {path}")
        if transcript_incomplete:
            merged["_transcript_incomplete"] = True
        return merged, blobs, header

    @classmethod
    def check_header(cls, header: Json, path: str) -> None:
        version = header.get("v")
        if version != cls.FORMAT_VERSION:
            raise WizoltError(f"Unsupported session format v{version} (expected v{cls.FORMAT_VERSION}): {path}")

    @staticmethod
    def path_for(data_dir: str, *parts: str) -> str:
        return os.path.abspath(os.path.join(os.path.expanduser(data_dir), *parts))

    @staticmethod
    def _summary_from_tail(path: str, size: int, window: int, limit: int) -> tuple[list[tuple[str, str]], Counter[str]]:
        """Parse the log's last `window` bytes into text messages and tool counts. The window may start
        mid-record or inside a multi-byte character; the binary read and the dropped first slice make
        that harmless."""
        start = max(0, size - window)
        try:
            with open(path, "rb") as file:
                file.seek(start)
                chunk = file.read()
        except OSError:
            return [], Counter()
        lines = chunk.split(b"\n")
        if not lines:
            return [], Counter()
        # The first slice may start mid-record (the seek point, or a line cut by it); the rest are
        # whole JSON lines.
        lines = lines[1:]
        picked: list[tuple[str, str]] = []
        tool_counts: Counter[str] = Counter()
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                continue
            messages = parsed.get("messages")
            if not isinstance(messages, list):
                continue
            for message in reversed(messages):
                if message.get("role") not in {"user", "assistant"}:
                    continue
                if SessionSnapshotCodec.is_internal_message(message):
                    continue
                content = message.get("content")
                if isinstance(content, str) and content:
                    picked.append((str(message.get("role")), content))
                    if len(picked) >= limit:
                        break
                elif message.get("role") == "assistant":
                    # A turn that only ran tools has no text; count its tools and let one merged line
                    # summarise them, without crowding out the conversation.
                    calls = message.get("tool_calls")
                    if isinstance(calls, list):
                        for call in calls:
                            if isinstance(call, dict):
                                name = str(call.get("function", {}).get("name") or "")
                                if name:
                                    tool_counts[name] += 1
            if len(picked) >= limit:
                break
        return picked, tool_counts

    @classmethod
    def tail_summary(cls, path: str, limit: int = 5) -> list[tuple[str, str]]:
        """The most recent messages as `(role, text)` pairs, newest first, read from the tail of the
        log. A full decode of the session is never needed; the tail window starts small and widens
        until it holds enough text or hits `TAIL_BUDGET`, so a tool-heavy log whose newest turns run
        to megabytes still yields the conversation. A line that does not parse is skipped. Tool-only
        turns collapse into a single counted line (role `"tool"`), so a tool-heavy session stays
        identifiable instead of reading as a wall of names."""
        try:
            size = os.path.getsize(path)
        except OSError:
            return []
        if size <= 0:
            return []
        window = min(size, cls.TAIL_START)
        while True:
            picked, tool_counts = cls._summary_from_tail(path, size, window, limit)
            if len(picked) >= limit or window >= size or window >= cls.TAIL_BUDGET:
                break
            window = min(cls.TAIL_BUDGET, window * 4)
        if len(picked) < limit and tool_counts:
            tools = ", ".join(name if count == 1 else f"{name} ×{count}" for name, count in tool_counts.most_common())
            picked.append(("tool", "→ " + tools))
        return picked
