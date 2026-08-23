"""The /mcp command surface: subcommands, tab completion, and end-to-end user scenarios."""

import asyncio
import threading
import time
from types import SimpleNamespace
from typing import ClassVar

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
from mcp_harness import _fake_resource, mcp_cfg, mcp_tool_info

import minacode.cli.commands as commands_mod
from minacode.base import SELECTION_BACK
from minacode.cli import CommandCompleter, CommandLoop
from minacode.cli.commands import mcp_command
from minacode.cli.modals import mcp_manager
from minacode.cli.update import UpdateChecker
from minacode.config import (
    Config,
)
from minacode.engine import Agent
from minacode.mcp import MCPFileTokenStore, MCPManager
from minacode.render import StatusBar, UiPrinter
from minacode.session import Session, SessionSnapshotStore
from minacode.tools import CodeIndex
from minacode.tui import TUI_MODAL_PENDING, ChoiceViewState


def oauth_value(store: MCPFileTokenStore, url: str, collection: str, suffix: str) -> dict | None:
    entry = store.load().get(collection, {}).get(store.token_key(url, suffix))
    return entry.get("value") if entry else None


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


def oauth_store(tmp_path, states: dict[str, str]) -> MCPFileTokenStore:
    """Create a real token store containing one token/client pair per server URL."""
    store = MCPFileTokenStore(str(tmp_path / "mcp-oauth.json"))
    for url, label in states.items():
        put_oauth_state(store, url, label)
    return store


def put_oauth_state(store: MCPFileTokenStore, url: str, label: str) -> None:
    data = store.load()
    data.setdefault("mcp-oauth-token", {})[store.token_key(url, "/tokens")] = {"value": {"access_token": label + "-token", "token_type": "Bearer"}}
    data.setdefault("mcp-oauth-client-info", {})[store.token_key(url, "/client_info")] = {
        "value": {"client_id": label + "-client", "redirect_uris": ["http://localhost:12345/callback"]}
    }
    store.save(data)




# ---------------------------------------------------------------------------
# Tab completion
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# MCPManager — discover_server with nonexistent server
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# render_tools_index truncation
# ---------------------------------------------------------------------------


