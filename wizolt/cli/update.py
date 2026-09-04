"""wizolt update check: the background PyPI version probe and its cached status."""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time
from typing import ClassVar

import httpx2

from wizolt.base import (
    HTTP_USER_AGENT,
    Text,
    UpdateStatus,
    WizoltError,
    __version__,
    run_blocking,
)
from wizolt.session import Session


class UpdateChecker:
    PYPI_URL = "https://pypi.org/pypi/wizolt/json"
    CACHE_FILE = "update.json"
    TIMEOUT = 5
    INTERVAL_SECONDS = 24 * 3600
    HEADERS: ClassVar[dict[str, str]] = {"Accept": "application/json", "User-Agent": HTTP_USER_AGENT}

    def __init__(self, session: Session):
        self.session = session
        self.cache_path = session.data_path(self.CACHE_FILE)

    def load_cached(self) -> bool:
        """Publish the cached version, and say whether a remote check is due.

        Split from `check` on purpose: this is small, local, bounded data the first status display
        needs, so startup reads it directly. Only the network half is scheduled, and only when the
        interval says it is due -- so a session opened twice in a minute makes no request at all."""

        cached_at, cached_latest = self._load()
        self.session.update.latest = cached_latest
        if self.session.update.checking or time.time() - cached_at < self.INTERVAL_SECONDS:
            return False
        self.session.update.checking = True
        return True

    async def check(self) -> None:
        """Ask PyPI what the latest version is, and record the answer or why there is none.

        Every ending is contained here: an unreachable index, a proxy returning HTML, a timeout --
        none of them are the session's problem, and all of them leave a concise status behind. The
        session fields are written on the loop this coroutine runs on, never from a worker."""

        try:
            self.session.update.latest = await self.fetch_latest()
            self.session.update.error = ""
        except Exception as error:  # noqa: BLE001 - an expected maintenance failure; the status is the report.
            self.session.update.error = Text.clean(str(error))
        finally:
            self.session.update.checking = False
            await run_blocking(self._save)

    def _load(self) -> tuple[float, str]:
        with contextlib.suppress(Exception):
            with open(self.cache_path, encoding="utf-8") as file:
                data = json.load(file)
            latest = str(data.get("latest") or "")
            if UpdateStatus.version_tuple(latest):
                return float(data.get("checked_at") or 0), latest
        return 0.0, ""

    def _save(self) -> None:
        with contextlib.suppress(Exception):
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as file:
                json.dump({"checked_at": time.time(), "latest": self.session.update.latest}, file)

    @staticmethod
    async def fetch_latest() -> str:
        """The PyPI probe, on the caller's loop. Cancelling it closes the client and the request."""
        # `async with`, which is how HTTPX documents the async client: the connection pool has to be
        # closed, and a cancellation here must not leave a socket to a finalizer. httpx2 rather
        # than httpx: it is the continuation of the same project, and it is already the client the
        # provider SDKs in this process speak, so the tree keeps one HTTP stack.
        async with httpx2.AsyncClient(timeout=UpdateChecker.TIMEOUT, headers=UpdateChecker.HEADERS) as client:
            response = await client.get(UpdateChecker.PYPI_URL)
            response.raise_for_status()
            return UpdateChecker.parse_latest(response.content)

    @staticmethod
    def fetch_latest_sync() -> str:
        """The same probe for `wizolt update`, which is a standalone synchronous command."""
        with httpx2.Client(timeout=UpdateChecker.TIMEOUT, headers=UpdateChecker.HEADERS) as client:
            response = client.get(UpdateChecker.PYPI_URL)
            response.raise_for_status()
            return UpdateChecker.parse_latest(response.content)

    @staticmethod
    def parse_latest(payload: bytes) -> str:
        """The version in a PyPI JSON body. Pure, so both transports agree on what is acceptable."""
        with contextlib.suppress(ValueError):
            data = json.loads(payload.decode("utf-8", "replace"))
            version = data.get("info", {}).get("version") if isinstance(data, dict) else ""
            if isinstance(version, str) and UpdateStatus.version_tuple(version):
                return version
        raise WizoltError("invalid PyPI version response")

    def status_line(self) -> str:
        update = self.session.update
        if update.checking:
            return "update: checking"
        if update.newer_than(__version__):
            return f"update: {__version__} -> {update.latest}"
        if update.error:
            return "update: error"
        return "update: current" if update.latest else "update: unknown"

    @staticmethod
    def upgrade_command() -> list[str]:
        """Best-effort package-manager command to upgrade wizolt, based on how it was installed."""
        # Match on sys.prefix, not realpath(sys.executable): in uv tool and pipx venvs,
        # bin/python is a symlink to the base interpreter, and realpath escapes the venv
        # that identifies the install source.
        prefix = sys.prefix.replace(os.sep, "/")
        if "/uv/tools/" in prefix:
            return ["uv", "tool", "upgrade", "wizolt"]
        if "/pipx/venvs/" in prefix:
            return ["pipx", "upgrade", "wizolt"]
        return [sys.executable, "-m", "pip", "install", "--upgrade", "wizolt"]
