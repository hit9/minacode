"""Filesystem-backed token storage for MCP OAuth credentials."""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from typing import ClassVar

from wizolt.base import Json, run_blocking


class MCPFileTokenStore:
    DEFAULT_COLLECTION = "default_collection"
    _locks: ClassVar[dict[str, threading.Lock]] = {}
    _locks_guard: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, path: str):
        self.path = os.path.abspath(os.path.expanduser(path))
        with self._locks_guard:
            self.lock = self._locks.setdefault(self.path, threading.Lock())

    def token_key(self, server_url: str, suffix: str) -> str:
        return server_url.rstrip("/") + suffix

    async def has_server_tokens(self, server_url: str) -> bool:
        return await run_blocking(lambda: self._has_server_tokens_sync(server_url))

    async def clear_server(self, server_url: str) -> None:
        await run_blocking(lambda: self._clear_server_sync(server_url))

    def _has_server_tokens_sync(self, server_url: str) -> bool:
        key = self.token_key(server_url, "/tokens")
        collection = "mcp-oauth-token"
        with self.lock:
            entry = self.load().get(collection, {}).get(key)
            return bool(entry and not self.expired(entry))

    def _clear_server_sync(self, server_url: str) -> None:
        with self.lock:
            data = self.load()
            for collection, key in (
                ("mcp-oauth-token", self.token_key(server_url, "/tokens")),
                ("mcp-oauth-client-info", self.token_key(server_url, "/client_info")),
                ("mcp-oauth-token-expiry", self.token_key(server_url, "/token_expiry")),
            ):
                data.get(collection, {}).pop(key, None)
            self.save(data)

    async def get(self, key: str, *, collection: str | None = None) -> Json | None:
        return await run_blocking(lambda: self._get_sync(key, collection=collection or self.DEFAULT_COLLECTION))

    def _get_sync(self, key: str, *, collection: str) -> Json | None:
        with self.lock:
            data = self.load()
            entry = data.get(collection, {}).get(key)
            if entry is None:
                return None
            if self.expired(entry):
                data.get(collection, {}).pop(key, None)
                self.save(data)
                return None
            value = entry.get("value")
            return dict(value) if isinstance(value, dict) else None

    # Called dynamically through the MCP OAuth token-storage protocol; static call graphs will not see it.
    async def put(self, key: str, value: Json, *, collection: str | None = None, ttl: float | None = None) -> None:
        await run_blocking(lambda: self._put_sync(key, value, collection=collection or self.DEFAULT_COLLECTION, ttl=ttl))

    def _put_sync(self, key: str, value: Json, *, collection: str, ttl: float | None) -> None:
        expires_at = time.time() + float(ttl) if ttl is not None else None
        with self.lock:
            data = self.load()
            data.setdefault(collection, {})[key] = {"value": dict(value), "expires_at": expires_at}
            self.save(data)

    async def delete(self, key: str, *, collection: str | None = None) -> bool:
        return await run_blocking(lambda: self._delete_sync(key, collection=collection or self.DEFAULT_COLLECTION))

    def _delete_sync(self, key: str, *, collection: str) -> bool:
        with self.lock:
            data = self.load()
            removed = data.get(collection, {}).pop(key, None) is not None
            if removed:
                self.save(data)
            return removed

    @staticmethod
    def expired(entry: Json) -> bool:
        expires_at = entry.get("expires_at")
        return isinstance(expires_at, int | float) and expires_at <= time.time()

    def load(self) -> dict[str, dict[str, Json]]:
        try:
            with open(self.path, encoding="utf-8") as file:
                data = json.load(file)
        except FileNotFoundError:
            return {}
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def save(self, data: dict[str, dict[str, Json]]) -> None:
        directory = os.path.dirname(self.path)
        os.makedirs(directory, mode=0o700, exist_ok=True)
        with contextlib.suppress(OSError):
            os.chmod(directory, 0o700)
        tmp = self.path + ".tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            try:
                file = os.fdopen(fd, "w", encoding="utf-8")
            except Exception:
                # os.fdopen doesn't close fd on failure; do it ourselves so the descriptor doesn't leak.
                os.close(fd)
                raise
            with file:
                json.dump(data, file, ensure_ascii=False, sort_keys=True)
        except Exception:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
        os.replace(tmp, self.path)
        with contextlib.suppress(OSError):
            os.chmod(self.path, 0o600)
