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
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from minacode.base import HTTP_USER_AGENT, MinacodeError, Text
from minacode.providers.catalog import CatalogCodec, decode_bundled
from minacode.providers.compat import ProviderPolicy
from minacode.providers.schema import CatalogError, CatalogSnapshot, CatalogSyncError

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
        self._codec = CatalogCodec()

    def bundled(self) -> CatalogSnapshot:
        try:
            return decode_bundled()
        except CatalogError as error:
            raise MinacodeError(f"bundled provider catalog is corrupt: {error}") from error

    def cached(self) -> CatalogSnapshot | None:
        """The cached document, or ``None`` when absent or invalid (a corrupt cache is ignored)."""

        try:
            with open(self.cache_path, "rb") as file:
                return self._codec.decode(file.read(), "cached")
        except (OSError, CatalogError, ValueError):
            return None

    def select(self) -> tuple[CatalogSnapshot, str, str]:
        """(snapshot, source, note) for the highest-version whole document.

        ``source`` is ``"bundled"`` or ``"cached"``. A cached document never overrides a bundled
        one with an equal version unless the bytes are identical; a corrupt cache falls back to the
        bundled copy without failing startup.
        """

        bundled = self.bundled()
        cached = self.cached()
        if cached is None:
            return bundled, "bundled", ""
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

        request = Request(
            CATALOG_URL,
            headers={"User-Agent": HTTP_USER_AGENT, "Accept": "application/json"},
        )
        if self.cached() is not None:
            with contextlib.suppress(OSError):
                with open(self.etag_path, encoding="utf-8") as file:
                    etag = file.read().strip()
                if etag:
                    request.add_header("If-None-Match", etag)
        try:
            with self._locked(), urlopen(request, timeout=REMOTE_TIMEOUT) as response:
                if getattr(response, "status", 200) == 304:
                    return self.select()[0]
                payload = response.read(MAX_REMOTE_BYTES + 1)
                if len(payload) > MAX_REMOTE_BYTES:
                    raise CatalogSyncError(f"remote catalog exceeds {MAX_REMOTE_BYTES} bytes")
                snapshot = self._codec.decode(payload, "cached")
                self._write_cache_monotonic(payload, response.headers.get("ETag", ""), snapshot)
                return snapshot
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
        try:
            with open(lock_path, "w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lock, fcntl.LOCK_UN)
        finally:
            # The lock file is transient; remove it so a session data dir stays clean.
            with contextlib.suppress(OSError):
                os.unlink(lock_path)

    def _write_cache_monotonic(self, payload: bytes, etag: str, snapshot: CatalogSnapshot) -> None:
        """Write the fetched document only when it is strictly newer than the current cache."""

        current = self.cached()
        if current is not None and snapshot.version <= current.version:
            return
        self._write_cache(payload, etag)

    def _write_cache(self, payload: bytes, etag: str) -> None:
        os.makedirs(self.catalog_dir, exist_ok=True)
        tmp = self.cache_path + ".tmp"
        with open(tmp, "wb") as file:
            file.write(payload)
        os.replace(tmp, self.cache_path)
        if etag:
            with open(self.etag_path, "w", encoding="utf-8") as file:
                file.write(etag)


class CatalogRuntime:
    """The catalog a session lives against: snapshot + compiled policy + sync state.

    Not a global: each ``Session`` owns one, built from its own ``data_dir`` in
    ``bootstrap_features``, so a worker session can point at a different data directory.
    """

    def __init__(self, data_dir: str):
        self.repository = CatalogRepository(data_dir)
        try:
            self.snapshot, self.source, self.note = self.repository.select()
        except MinacodeError as error:
            # Startup must never die on a corrupt install; the bundled copy is the floor.
            self.snapshot = decode_bundled()
            self.source = "bundled"
            self.note = str(error)
        self.policy = ProviderPolicy(self.snapshot)
        self.sync_state = CatalogSyncState()
        self._load_state()
        if not self.sync_state.last_source:
            self.sync_state.last_source = self.source
        if not self.sync_state.last_version:
            self.sync_state.last_version = self.snapshot.version

    @property
    def version(self) -> int:
        return self.snapshot.version

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
                self.repository.fetch()
                state.error = ""
            except Exception as error:  # noqa: BLE001 - background sync failures must not escape the worker.
                state.error = Text.clean(str(error))
            finally:
                state.checking = False
                state.checked_at = time.time()
                self._save_state()

        threading.Thread(target=run, daemon=True).start()

    # -- persisted sync state (``<data_dir>/catalog/sync.json``) ------------------

    def _state_path(self) -> str:
        return os.path.join(self.repository.catalog_dir, CATALOG_STATE_FILE)

    def _load_state(self) -> None:
        """Restore the persisted gate and last result; a corrupt file counts as never checked."""

        with contextlib.suppress(Exception):
            with open(self._state_path(), encoding="utf-8") as file:
                data = json.load(file)
            if isinstance(data, dict):
                state = self.sync_state
                state.checked_at = float(data.get("checked_at") or 0)
                state.last_synced_at = float(data.get("last_synced_at") or 0)
                state.last_source = str(data.get("last_source") or "")
                state.last_version = int(data.get("last_version") or 0)
                state.error = str(data.get("error") or "")

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
            os.makedirs(self.repository.catalog_dir, exist_ok=True)
            tmp = self._state_path() + ".tmp"
            with open(tmp, "w", encoding="utf-8") as file:
                json.dump(payload, file)
            os.replace(tmp, self._state_path())
