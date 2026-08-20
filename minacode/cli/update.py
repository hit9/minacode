"""minacode update check: the background PyPI version probe and its cached status."""

from __future__ import annotations

import contextlib
import json
import os
import sys
import threading
import time
from urllib.request import Request, urlopen

from minacode.base import (
    HTTP_USER_AGENT,
    MinacodeError,
    Text,
    UpdateStatus,
    __version__,
)
from minacode.session import Session


class UpdateChecker:
    PYPI_URL = "https://pypi.org/pypi/minacode/json"
    CACHE_FILE = "update.json"
    TIMEOUT = 5
    INTERVAL_SECONDS = 24 * 3600

    def __init__(self, session: Session):
        self.session = session
        self.cache_path = session.data_path(self.CACHE_FILE)

    def start(self) -> None:
        cached_at, cached_latest = self._load()
        self.session.update.latest = cached_latest
        if self.session.update.checking or time.time() - cached_at < self.INTERVAL_SECONDS:
            return
        self.session.update.checking = True
        threading.Thread(target=self.check, daemon=True).start()

    def check(self) -> None:
        try:
            self.session.update.latest = self.fetch_latest()
            self.session.update.error = ""
        except Exception as error:  # noqa: BLE001 - background update failures must not escape the worker.
            self.session.update.error = Text.clean(str(error))
        finally:
            self.session.update.checking = False
            self._save()

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
    def fetch_latest() -> str:
        request = Request(UpdateChecker.PYPI_URL, headers={"Accept": "application/json", "User-Agent": HTTP_USER_AGENT})
        with urlopen(request, timeout=UpdateChecker.TIMEOUT) as response:
            data = json.loads(response.read().decode("utf-8", "replace"))
        version = data.get("info", {}).get("version") if isinstance(data, dict) else ""
        if not isinstance(version, str) or not UpdateStatus.version_tuple(version):
            raise MinacodeError("invalid PyPI version response")
        return version

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
        """Best-effort package-manager command to upgrade minacode, based on how it was installed."""
        executable = os.path.realpath(sys.executable).replace(os.sep, "/")
        if "/uv/tools/" in executable:
            return ["uv", "tool", "upgrade", "minacode"]
        if "/pipx/venvs/" in executable:
            return ["pipx", "upgrade", "minacode"]
        return [sys.executable, "-m", "pip", "install", "--upgrade", "minacode"]
