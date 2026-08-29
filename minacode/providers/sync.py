"""Provider catalog sync: source selection, the session runtime, and remote refresh.

``CatalogRepository`` validates each available source (the bundled ``catalog.json`` and the cached
copy) and picks the whole document with the highest ``version``; the same-version invariant is
enforced. ``CatalogRuntime`` is what a ``Session`` holds -- the selected snapshot, its compiled
``ProviderPolicy``, and sync state. Nothing here imports CLI or Session; the startup layer passes
``data_dir`` in.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from minacode.base import HTTP_USER_AGENT, Text
from minacode.providers.catalog import CatalogCodec, decode_bundled
from minacode.providers.compat import ProviderPolicy
from minacode.providers.schema import CatalogError, CatalogSnapshot, CatalogSourceError, CatalogSyncError, CatalogVersionConflict

# The published copy of the bundled catalog; the remote is the single sync target.
CATALOG_URL = "https://raw.githubusercontent.com/hit9/minacode/master/minacode/providers/catalog.json"
CATALOG_DIR = "catalog"
CATALOG_CACHE_FILE = "catalog.json"
CATALOG_ETAG_FILE = "catalog-etag.txt"
CATALOG_STATE_FILE = "sync.json"
CATALOG_LOCK_FILE = "catalog.lock"
SYNC_INTERVAL_SECONDS = 72 * 3600
REMOTE_TIMEOUT = 5
MAX_REMOTE_BYTES = 2 * 1024 * 1024


@dataclass
class CatalogSyncState:
    """Transient background-sync status, mirroring the update checker's shape."""

    checking: bool = False
    error: str = ""
    last_synced_at: float = 0.0
    last_source: str = ""
    last_version: int = 0
    # When the last automatic check happened (success or failure); gates the 72h interval across
    # processes, persisted in ``sync.json``.
    checked_at: float = 0.0


class CatalogRepository:
    """Read, validate, and choose among the bundled and cached catalog documents."""

    def __init__(self, data_dir: str):
        self.data_dir = os.path.expanduser(data_dir)
        self.catalog_dir = os.path.join(self.data_dir, CATALOG_DIR)
        self.cache_path = os.path.join(self.catalog_dir, CATALOG_CACHE_FILE)
        self.etag_path = os.path.join(self.catalog_dir, CATALOG_ETAG_FILE)
        self.state_path = os.path.join(self.catalog_dir, CATALOG_STATE_FILE)
        self._codec = CatalogCodec()

    def bundled(self) -> CatalogSnapshot:
        try:
            return decode_bundled()
        except (OSError, CatalogError) as error:
            raise CatalogSourceError(f"bundled provider catalog is corrupt: {error}") from error

    def cached(self) -> CatalogSnapshot | None:
        """The cached document, or ``None`` when absent or invalid (a corrupt cache is ignored)."""

        return self._cached()[0]

    def _cached(self) -> tuple[CatalogSnapshot | None, str]:
        """The cached snapshot and a diagnostic when a present copy is unreadable."""

        try:
            with open(self.cache_path, "rb") as file:
                return self._codec.decode(file.read(), "cached"), ""
        except FileNotFoundError:
            return None, ""
        except (OSError, CatalogError, ValueError) as error:
            return None, f"cached catalog ignored: invalid ({Text.clean(str(error))})"

    def select(self) -> tuple[CatalogSnapshot, str, str]:
        """(snapshot, source, note) for the highest-version whole document.

        ``source`` is ``"bundled"`` or ``"cached"``. A cached document never overrides a bundled
        one with an equal version unless the bytes are identical; a corrupt cache falls back to the
        bundled copy without failing startup.
        """

        bundled = self.bundled()
        cached, cached_note = self._cached()
        if cached is None:
            return bundled, "bundled", cached_note
        if cached.version > bundled.version:
            return cached, "cached", ""
        if cached.version == bundled.version:
            if cached.content_hash != bundled.content_hash:
                return bundled, "bundled", "cached catalog ignored: same version but different content"
            return bundled, "bundled", ""
        return bundled, "bundled", ""

    def fetch(self) -> CatalogSnapshot:
        """Download the remote catalog, validate it, and atomically cache it on success.

        Sends the cached ETag only when the current cache is still valid, so an unchanged remote
        answers 304 and the local copy stays. The fetch, re-read of the current cache, version
        comparison and write happen inside one cross-process lock, and the cache is never
        downgraded to an older version. The download is bounded and the write is atomic, so a
        failure never leaves a half file.
        """

        try:
            with self._locked():
                request = Request(CATALOG_URL, headers={"User-Agent": HTTP_USER_AGENT, "Accept": "application/json"})
                if self.cached() is not None:
                    with contextlib.suppress(OSError):
                        with open(self.etag_path, encoding="utf-8") as file:
                            etag = file.read().strip()
                        if etag:
                            request.add_header("If-None-Match", etag)
                try:
                    response_context = urlopen(request, timeout=REMOTE_TIMEOUT)
                except HTTPError as error:
                    if error.code == 304:
                        return self.select()[0]
                    raise
                with response_context as response:
                    if getattr(response, "status", 200) == 304:
                        return self.select()[0]
                    payload = response.read(MAX_REMOTE_BYTES + 1)
                    if len(payload) > MAX_REMOTE_BYTES:
                        raise CatalogSyncError(f"remote catalog exceeds {MAX_REMOTE_BYTES} bytes")
                    snapshot = self._codec.decode(payload, "cached")
                    current = self.select()[0]
                    if snapshot.version < current.version:
                        return current
                    if snapshot.version == current.version:
                        if snapshot.content_hash != current.content_hash:
                            raise CatalogVersionConflict(f"remote catalog has the same version {snapshot.version} but different content")
                        return current
                    self._write_cache(payload, response.headers.get("ETag", ""))
                    return snapshot
        except CatalogSyncError:
            raise
        except (OSError, URLError, HTTPError, ValueError, CatalogError) as error:
            raise CatalogSyncError(f"catalog sync failed: {error}") from error

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        """Best-effort cross-process lock around fetch/write; skipped where fcntl is unavailable."""

        try:
            import fcntl
        except ImportError:
            yield
            return
        os.makedirs(self.catalog_dir, exist_ok=True)
        lock_path = os.path.join(self.catalog_dir, CATALOG_LOCK_FILE)
        with open(lock_path, "a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _write_cache(self, payload: bytes, etag: str) -> None:
        os.makedirs(self.catalog_dir, exist_ok=True)
        self._atomic_write(self.cache_path, payload)
        if etag:
            self._atomic_write(self.etag_path, etag.encode())
        else:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(self.etag_path)

    def load_sync_state(self) -> dict[str, object]:
        """Return persisted runtime status; malformed state is equivalent to no state."""

        try:
            with open(self.state_path, encoding="utf-8") as file:
                value = json.load(file)
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError):
            return {}

    def save_sync_state(self, state: dict[str, object]) -> None:
        self._atomic_write(self.state_path, json.dumps(state, separators=(",", ":")).encode())

    @staticmethod
    def _atomic_write(path: str, payload: bytes) -> None:
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=os.path.basename(path) + ".", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(descriptor, "wb") as file:
                file.write(payload)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, path)
            with contextlib.suppress(OSError):
                directory_fd = os.open(directory, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary)


class CatalogRuntime:
    """The catalog a session lives against: snapshot + compiled policy + sync state.

    Not a global: each top-level ``Session`` owns one selected from its ``data_dir``. Delegate
    workers share that exact runtime, so one conversation cannot resolve against two catalogs.
    """

    def __init__(self, data_dir: str):
        self.repository = CatalogRepository(data_dir)
        self.snapshot, self.source, self.note = self.repository.select()
        self.policy = ProviderPolicy(self.snapshot)
        self.sync_state = CatalogSyncState()
        self._load_state()
        if not self.sync_state.last_source:
            self.sync_state.last_source = self.source
        if not self.sync_state.last_version:
            self.sync_state.last_version = self.snapshot.version

    def source_versions(self) -> tuple[int, int | None]:
        """The bundled and valid cached versions for status presentation."""

        bundled = self.repository.bundled().version
        cached = self.repository.cached()
        return bundled, cached.version if cached is not None else None

    def sync_now(self) -> CatalogSnapshot:
        """Fetch the remote catalog and, if newer, activate it at the command boundary.

        This is the manual-sync path (``/catalog sync``): the active snapshot and policy swap only
        here, never in the background refresh.
        """

        remote = self.repository.fetch()
        if remote.version > self.snapshot.version:
            self.snapshot = remote
            self.source = "cached"
            self.note = ""
            self.policy = ProviderPolicy(remote)
        self.sync_state.last_synced_at = time.time()
        self.sync_state.checked_at = time.time()
        self.sync_state.last_source = self.source
        self.sync_state.last_version = self.snapshot.version
        self.sync_state.error = ""
        self._save_state()
        return self.snapshot

    def start_background_sync(self) -> None:
        """Refresh from the remote on a daemon thread, at most once per 72h."""

        state = self.sync_state
        if state.checking or time.time() - state.checked_at < SYNC_INTERVAL_SECONDS:
            return
        state.checking = True

        def run() -> None:
            try:
                # The automatic refresh only updates the cache; the active policy is never
                # hot-swapped, so a long turn's requests keep one catalog version. A newer cache
                # is picked up on the next startup (see CatalogRepository.select).
                snapshot = self.repository.fetch()
                state.error = ""
                state.last_synced_at = time.time()
                state.last_source = "cached" if snapshot.version > self.repository.bundled().version else "bundled"
                state.last_version = snapshot.version
            except Exception as error:  # noqa: BLE001 - background sync failures must not escape the worker.
                state.error = Text.clean(str(error))
            finally:
                state.checking = False
                state.checked_at = time.time()
                self._save_state()

        threading.Thread(target=run, daemon=True).start()

    def _load_state(self) -> None:
        """Restore the persisted gate and last result; a corrupt file counts as never checked."""

        data = self.repository.load_sync_state()
        state = self.sync_state
        checked_at = data.get("checked_at", 0)
        last_synced_at = data.get("last_synced_at", 0)
        last_version = data.get("last_version", 0)
        last_source = data.get("last_source", "")
        error = data.get("error", "")
        state.checked_at = float(checked_at) if isinstance(checked_at, (int, float)) and not isinstance(checked_at, bool) else 0.0
        state.last_synced_at = float(last_synced_at) if isinstance(last_synced_at, (int, float)) and not isinstance(last_synced_at, bool) else 0.0
        state.last_source = last_source if isinstance(last_source, str) else ""
        state.last_version = last_version if isinstance(last_version, int) and not isinstance(last_version, bool) else 0
        state.error = error if isinstance(error, str) else ""

    def _save_state(self) -> None:
        """Persist the gate and last result; a write failure must not break a sync."""

        state = self.sync_state
        payload = {
            "checked_at": state.checked_at,
            "last_synced_at": state.last_synced_at,
            "last_source": state.last_source,
            "last_version": state.last_version,
            "error": state.error,
        }
        with contextlib.suppress(Exception):
            self.repository.save_sync_state(payload)
