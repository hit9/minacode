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
from minacode.config import (
    Config,
)
from minacode.engine import Agent
from minacode.mcp import MCPFileTokenStore, MCPManager
from minacode.render import StatusBar, UiPrinter
from minacode.session import Session, SessionSnapshotStore
from minacode.tools import CodeIndex
from minacode.tui import TUI_MODAL_PENDING, ChoiceViewState
from minacode.cli.update import UpdateChecker


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


class TestMCPCommands:
    def test_start_session_discovers_auto_servers(self, monkeypatch):
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg()))
        calls = []
        monkeypatch.setattr(s.mcp, "discover_auto", lambda: calls.append("auto"))
        monkeypatch.setattr(UpdateChecker, "start", lambda _checker: None)
        monkeypatch.setattr(SessionSnapshotStore, "clean_expired", lambda _session: 0)
        monkeypatch.setattr(CodeIndex, "refresh_existing_async", lambda _index: False)

        CommandLoop(Agent(s), input_fn=lambda _: "", output_fn=lambda _: None).start_session()

        assert calls == ["auto"]

    def test_mcp_command_no_args_shows_status(self, monkeypatch):
        """/mcp returns server status."""
        raw = mcp_cfg()
        s = Session(cwd="/tmp", config=Config.from_dict(raw))

        class FakeTool:
            name = "echo"
            description = "Echo"
            inputSchema: ClassVar[dict] = {"type": "object", "properties": {}, "required": []}
            annotations = None

        async def fake_list(url, headers):
            return [FakeTool()]

        monkeypatch.setattr(s.mcp, "_list_tools", fake_list)
        s.mcp.discover_auto()

        loop = CommandLoop(Agent(s), input_fn=lambda _: "", output_fn=lambda _: None)
        result = mcp_command(loop, "")
        assert "test" in result
        assert "| `test` | auto | ● connected | 1     |" in result

    def test_mcp_tools_shows_listing(self, monkeypatch):
        """/mcp tools returns tool listing."""
        raw = mcp_cfg()
        s = Session(cwd="/tmp", config=Config.from_dict(raw))

        class FakeTool:
            name = "echo"
            description = "Echo"
            inputSchema: ClassVar[dict] = {"type": "object", "properties": {}, "required": []}
            annotations = None

        async def fake_list(url, headers):
            return [FakeTool()]

        monkeypatch.setattr(s.mcp, "_list_tools", fake_list)
        s.mcp.discover_auto()

        loop = CommandLoop(Agent(s), input_fn=lambda _: "", output_fn=lambda _: None)
        result = mcp_command(loop, "tools")
        assert "### `test`" in result
        assert "echo" in result

    def test_mcp_tools_without_name_does_not_discover(self, monkeypatch):
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg()))
        calls = []
        monkeypatch.setattr(s.mcp, "discover_server", lambda name: calls.append(name))

        loop = CommandLoop(Agent(s), input_fn=lambda _: "", output_fn=lambda _: None)
        result = mcp_command(loop, "tools")

        assert calls == []
        assert result == "(no connected MCP servers)"

    def test_mcp_connect_oauth_failure_includes_mcp_url(self, monkeypatch):
        """Interactive connect shows a fallback URL when OAuth does not provide one."""
        raw = mcp_cfg(auth="oauth")
        s = Session(cwd="/tmp", config=Config.from_dict(raw))

        async def fake_login(config, headers, operation, *, long_timeout=False, interactive=False, notify=None):
            raise RuntimeError("Unexpected content type: text/html")

        monkeypatch.setattr(s.mcp, "_run_op", fake_login)
        result = s.mcp.connect_server("test", interactive=True)

        assert "Unexpected content type: text/html" in result
        assert "Open MCP URL: http://localhost:9999/mcp" in result

    def test_mcp_connect_authenticates_then_loads_capabilities(self, monkeypatch):
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg(auth="oauth")))
        authenticated = False
        calls = []
        cleared = []

        monkeypatch.setattr(s.mcp._oauth_token_store, "has_server_tokens", lambda _url: authenticated)
        monkeypatch.setattr(s.mcp._oauth_token_store, "clear_server", cleared.append)

        def fake_auth(config, notify=None):
            nonlocal authenticated
            authenticated = True
            calls.append((config.name, notify))

        def fake_discover(name):
            assert authenticated
            s.mcp.tools[name] = [mcp_tool_info(name, "echo")]
            s.mcp.resources[name] = []

        monkeypatch.setattr(s.mcp, "_authenticate_oauth", fake_auth)
        monkeypatch.setattr(s.mcp, "discover_server", fake_discover)

        result = s.mcp.connect_server("test", interactive=True)

        assert result == "MCP server connected: test; tools=1; resources=0"
        assert calls == [("test", None)]
        assert cleared == ["http://localhost:9999/mcp"]
        assert s.mcp.connected("test")
        assert "echo" in s.mcp.render_tools_index()

    def test_mcp_connect_reauthorizes_rejected_cached_oauth_session(self, monkeypatch):
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg(auth="oauth")))
        authorized = False
        attempts = []
        cleared = []
        monkeypatch.setattr(s.mcp._oauth_token_store, "has_server_tokens", lambda _url: True)
        monkeypatch.setattr(s.mcp._oauth_token_store, "clear_server", cleared.append)

        async def request(_config, _headers, _operation, *, long_timeout=False, interactive=False, notify=None):
            nonlocal authorized
            attempts.append(interactive)
            if interactive:
                authorized = True
                return []
            if not authorized:
                raise RuntimeError("authentication required; run /mcp connect test")
            return []

        monkeypatch.setattr(s.mcp, "_run_op", request)
        loop = CommandLoop(Agent(s), input_fn=lambda _: "", output_fn=lambda _text: None)
        loop.interactive_input = True

        result = mcp_command(loop, "connect test")

        assert result == "MCP server connected: test; tools=0; resources=0"
        assert attempts.count(True) == 1
        assert attempts.index(True) >= 1
        assert cleared == ["http://localhost:9999/mcp"]
        assert s.mcp.connected("test")

    def test_mcp_connect_keeps_valid_cached_oauth_session(self, monkeypatch):
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg(auth="oauth")))
        monkeypatch.setattr(s.mcp._oauth_token_store, "has_server_tokens", lambda _url: True)
        monkeypatch.setattr(s.mcp._oauth_token_store, "clear_server", lambda _url: pytest.fail("valid credentials were cleared"))
        monkeypatch.setattr(s.mcp, "_authenticate_oauth", lambda *_args, **_kwargs: pytest.fail("valid credentials triggered login"))

        async def tools(_config, _headers):
            return [SimpleNamespace(name="echo", description="Echo", inputSchema={}, annotations=None)]

        async def resources(_config, _headers):
            return []

        monkeypatch.setattr(s.mcp, "_list_tools", tools)
        monkeypatch.setattr(s.mcp, "_list_resources", resources)

        result = s.mcp.connect_server("test", interactive=True)

        assert result == "MCP server connected: test; tools=1; resources=0"

    def test_mcp_connect_does_not_reauthorize_on_non_auth_failure(self, monkeypatch):
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg(auth="oauth")))
        monkeypatch.setattr(s.mcp._oauth_token_store, "has_server_tokens", lambda _url: True)
        monkeypatch.setattr(s.mcp._oauth_token_store, "clear_server", lambda _url: pytest.fail("credentials were cleared"))
        monkeypatch.setattr(s.mcp, "_authenticate_oauth", lambda *_args, **_kwargs: pytest.fail("connection error triggered login"))

        async def offline(_config, _headers):
            raise ConnectionError("service unavailable")

        monkeypatch.setattr(s.mcp, "_list_tools", offline)
        monkeypatch.setattr(s.mcp, "_list_resources", offline)

        result = s.mcp.connect_server("test", interactive=True)

        assert result == "MCP server error: test: service unavailable"

    @pytest.mark.parametrize("rejection", ["invalid_request", "invalid client", "invalid_token", "HTTP 403 forbidden"])
    def test_mcp_connect_recognizes_cached_oauth_rejection_variants(self, rejection, monkeypatch):
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg(auth="oauth")))
        authorized = False
        logins = []
        monkeypatch.setattr(s.mcp._oauth_token_store, "has_server_tokens", lambda _url: True)
        monkeypatch.setattr(s.mcp._oauth_token_store, "clear_server", lambda _url: None)

        async def tools(_config, _headers):
            if not authorized:
                raise RuntimeError(rejection)
            return []

        async def resources(_config, _headers):
            return []

        def authorize(config, notify=None):
            nonlocal authorized
            logins.append(config.name)
            authorized = True

        monkeypatch.setattr(s.mcp, "_list_tools", tools)
        monkeypatch.setattr(s.mcp, "_list_resources", resources)
        monkeypatch.setattr(s.mcp, "_authenticate_oauth", authorize)

        result = s.mcp.connect_server("test", interactive=True)

        assert result == "MCP server connected: test; tools=0; resources=0"
        assert logins == ["test"]

    def test_mcp_connect_oauth_requires_interactive_session(self):
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg(auth="oauth")))

        result = s.mcp.connect_server("test")

        assert result == "MCP server authentication required: test; run /mcp connect test interactively"
        assert not s.mcp.connected("test")

    def test_mcp_connect_discovers_and_rediscovers_server(self, monkeypatch):
        """Repeated /mcp connect NAME calls reconnect that server."""
        calls = []
        raw = mcp_cfg()
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        monkeypatch.setattr(s.mcp, "discover_server", lambda name: calls.append(name))

        loop = CommandLoop(Agent(s), input_fn=lambda _: "", output_fn=lambda _: None)
        mcp_command(loop, "connect test")
        mcp_command(loop, "connect test")

        assert calls == ["test", "test"]

    def test_mcp_connects_multiple_servers_concurrently_in_argument_order(self, monkeypatch):
        raw = {
            "mcp": {
                "alpha": {"url": "https://alpha.example/mcp"},
                "beta": {"url": "https://beta.example/mcp"},
            }
        }
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        barrier = threading.Barrier(2)
        started = []

        def fake_discover(config):
            started.append(config.name)
            barrier.wait(timeout=1)
            s.mcp.tools[config.name] = []
            s.mcp.resources[config.name] = []

        monkeypatch.setattr(s.mcp, "_discover_one", fake_discover)

        result = s.mcp.connect_servers(["alpha", "beta", "alpha"])

        assert set(started) == {"alpha", "beta"}
        assert result == ("MCP connection results:\n\n- ● connected  `alpha` — 0 tools\n- ● connected  `beta` — 0 tools")

    def test_mcp_batch_serializes_oauth_only(self, monkeypatch):
        raw = {
            "mcp": {
                "alpha": {"url": "https://alpha.example/mcp", "auth": "oauth"},
                "beta": {"url": "https://beta.example/mcp", "auth": "oauth"},
            }
        }
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        authenticated: set[str] = set()
        active = 0
        maximum = 0
        state_lock = threading.Lock()
        monkeypatch.setattr(s.mcp._oauth_token_store, "has_server_tokens", lambda url: url in authenticated)

        def authenticate(config, notify=None):
            nonlocal active, maximum
            with state_lock:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.02)
            authenticated.add(config.url)
            with state_lock:
                active -= 1

        def discover(config):
            s.mcp.tools[config.name] = []
            s.mcp.resources[config.name] = []

        monkeypatch.setattr(s.mcp, "_authenticate_oauth", authenticate)
        monkeypatch.setattr(s.mcp, "_discover_one", discover)

        s.mcp.connect_servers(["alpha", "beta"], interactive=True)

        assert maximum == 1

    def test_mcp_batch_keeps_oauth_failure_compact_and_connects_other_servers(self, monkeypatch):
        raw = {
            "mcp": {
                "oauth": {"url": "https://oauth.example/mcp", "auth": "oauth"},
                "plain": {"url": "https://plain.example/mcp"},
            }
        }
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        monkeypatch.setattr(s.mcp._oauth_token_store, "clear_server", lambda _url: None)
        monkeypatch.setattr(
            s.mcp,
            "_authenticate_oauth",
            lambda config, notify=None: "\n".join(
                [
                    "MCP OAuth authentication failed for oauth: authorization denied",
                    "No authorization URL was provided by the server.",
                    "Open MCP URL: " + config.url,
                ]
            ),
        )

        async def tools(_config, _headers):
            return [SimpleNamespace(name="echo", description="Echo", inputSchema={}, annotations=None)]

        async def no_resources(_config, _headers):
            return []

        monkeypatch.setattr(s.mcp, "_list_tools", tools)
        monkeypatch.setattr(s.mcp, "_list_resources", no_resources)

        result = s.mcp.connect_servers(["oauth", "plain"], interactive=True)

        assert result == (
            "MCP connection results:\n\n"
            "- ● error  `oauth` — authorization denied\n"
            "    No authorization URL was provided by the server.\n"
            "    Open MCP URL: https://oauth.example/mcp\n"
            "- ● connected  `plain` — 1 tool"
        )
        assert not s.mcp.connected("oauth")
        assert s.mcp.connected("plain")

    def test_noninteractive_batch_never_starts_missing_oauth_login(self, monkeypatch):
        raw = {
            "mcp": {
                "oauth": {"url": "https://oauth.example/mcp", "auth": "oauth"},
                "plain": {"url": "https://plain.example/mcp"},
            }
        }
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        monkeypatch.setattr(s.mcp, "_authenticate_oauth", lambda *_args, **_kwargs: pytest.fail("batch opened OAuth"))

        async def tools(_config, _headers):
            return []

        async def no_resources(_config, _headers):
            return []

        monkeypatch.setattr(s.mcp, "_list_tools", tools)
        monkeypatch.setattr(s.mcp, "_list_resources", no_resources)

        result = s.mcp.connect_servers(["oauth", "plain"], interactive=False)

        assert "● error  `oauth` — authentication required" in result
        assert "● connected  `plain` — 0 tools" in result
        assert not s.mcp.connected("oauth")
        assert s.mcp.connected("plain")

    def test_mcp_batch_connect_formats_failures_as_separate_list_items(self, monkeypatch):
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg()))
        monkeypatch.setattr(s.mcp, "_discover_one", lambda config: s.mcp.set_server_error(config.name, "offline"))

        result = s.mcp.connect_servers(["test", "missing"])

        assert result == ("MCP connection results:\n\n- ● error  `test` — offline\n- ● error  `missing` — server not found")

    def test_mcp_connect_command_accepts_multiple_servers(self, monkeypatch):
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg()))
        calls = []
        monkeypatch.setattr(
            s.mcp,
            "connect_servers",
            lambda names, **kwargs: calls.append((names, kwargs)) or "connected batch",
        )
        loop = CommandLoop(Agent(s), input_fn=lambda _: "", output_fn=lambda _: None)

        result = mcp_command(loop, "connect alpha beta")

        assert result == "connected batch"
        assert calls == [(["alpha", "beta"], {"interactive": False, "notify": loop.emit})]

    def test_mcp_connect_rejects_missing_server(self):
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg()))
        loop = CommandLoop(Agent(s), input_fn=lambda _: "", output_fn=lambda _: None)

        assert mcp_command(loop, "connect missing") == "MCP server not found: missing"

    def test_mcp_disconnect_removes_connected_server(self):
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg()))
        s.mcp.tools["test"] = [mcp_tool_info("test", "echo")]
        s.mcp.resources["test"] = []
        loop = CommandLoop(Agent(s), input_fn=lambda _: "", output_fn=lambda _: None)

        assert mcp_command(loop, "disconnect test") == "MCP server disconnected: test"
        assert not s.mcp.connected("test")

    def test_mcp_disconnect_oauth_also_clears_authentication(self, monkeypatch):
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg(auth="oauth")))
        s.mcp.tools["test"] = [mcp_tool_info("test", "echo")]
        s.mcp.resources["test"] = []
        cleared = []
        monkeypatch.setattr(s.mcp._oauth_token_store, "clear_server", cleared.append)

        result = s.mcp.disconnect_server("test")

        assert result == "MCP server disconnected: test"
        assert cleared == ["http://localhost:9999/mcp"]
        assert not s.mcp.connected("test")

    def test_bare_mcp_opens_manager_in_tui(self, monkeypatch):
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg()))
        loop = CommandLoop(Agent(s), input_fn=lambda _: "", output_fn=lambda _: None)
        loop.tui = SimpleNamespace(input_mode="idle")
        calls = []
        monkeypatch.setattr(commands_mod, "mcp_manager", lambda _loop: calls.append("manager"))

        assert mcp_command(loop, "") is None
        assert calls == ["manager"]

    def test_mcp_manager_connects_selected_server(self, monkeypatch):
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg()))
        loop = CommandLoop(Agent(s), input_fn=lambda _: "", output_fn=lambda _: None)
        connected = threading.Event()
        release = threading.Event()
        repainted = threading.Event()

        def connect(name, **_kwargs):
            release.wait(1)
            s.mcp.tools[name] = []
            s.mcp.resources[name] = []
            return "connected " + name

        def show_modal(fragments, handle_key):
            assert handle_key("enter") is TUI_MODAL_PENDING
            assert "● connecting" in "".join(text for _style, text in fragments())
            release.set()
            assert repainted.wait(1)
            assert "● connected" in "".join(text for _style, text in fragments())
            connected.set()
            return SELECTION_BACK

        loop.tui = SimpleNamespace(show_modal=show_modal, invalidate=repainted.set)
        monkeypatch.setattr(s.mcp, "connect_server", connect)

        mcp_manager(loop, )

        assert connected.is_set()
        assert s.mcp.connected("test")

    def test_mcp_manager_disconnects_selected_server(self, monkeypatch):
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg()))
        s.mcp.tools["test"] = []
        s.mcp.resources["test"] = []
        loop = CommandLoop(Agent(s), input_fn=lambda _: "", output_fn=lambda _: None)
        release = threading.Event()
        repainted = threading.Event()

        def disconnect(name):
            release.wait(1)
            return MCPManager.disconnect_server(s.mcp, name)

        def show_modal(fragments, handle_key):
            assert handle_key("enter") is TUI_MODAL_PENDING
            assert "● disconnecting" in "".join(text for _style, text in fragments())
            release.set()
            assert repainted.wait(1)
            assert "● disconnected" in "".join(text for _style, text in fragments())
            return SELECTION_BACK

        loop.tui = SimpleNamespace(show_modal=show_modal, invalidate=repainted.set)
        monkeypatch.setattr(s.mcp, "disconnect_server", disconnect)

        mcp_manager(loop, )

        assert not s.mcp.connected("test")

    def test_mcp_manager_starts_multiple_connections_concurrently(self, monkeypatch):
        raw = {"mcp": {"a": {"url": "http://a/mcp"}, "b": {"url": "http://b/mcp"}}}
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        loop = CommandLoop(Agent(s), input_fn=lambda _: "", output_fn=lambda _: None)
        started = {name: threading.Event() for name in ("a", "b")}
        release = threading.Event()

        def connect(name, **_kwargs):
            started[name].set()
            release.wait(1)
            s.mcp.tools[name] = []
            s.mcp.resources[name] = []
            return "connected " + name

        def show_modal(_fragments, handle_key):
            assert handle_key("enter") is TUI_MODAL_PENDING
            assert handle_key("j") is TUI_MODAL_PENDING
            assert handle_key("enter") is TUI_MODAL_PENDING
            assert all(event.wait(1) for event in started.values())
            release.set()
            return SELECTION_BACK

        loop.tui = SimpleNamespace(show_modal=show_modal, invalidate=lambda: None)
        monkeypatch.setattr(s.mcp, "connect_server", connect)

        mcp_manager(loop, )

        assert all(event.is_set() for event in started.values())

    def test_mcp_manager_emits_late_result_without_repainting_closed_modal(self, monkeypatch):
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg(auto_connect=False)))
        release = threading.Event()
        emitted = threading.Event()
        outputs = []
        modal_closed = False

        def output(text):
            outputs.append(str(text))
            emitted.set()

        def connect(name, **_kwargs):
            release.wait(1)
            s.mcp.tools[name] = []
            s.mcp.resources[name] = []
            return "MCP server connected: " + name + "; tools=0; resources=0"

        def show_modal(_fragments, handle_key):
            assert handle_key("enter") is TUI_MODAL_PENDING
            return SELECTION_BACK

        def invalidate():
            assert not modal_closed, "completed worker repainted a closed modal"

        loop = CommandLoop(Agent(s), input_fn=lambda _: "", output_fn=output)
        loop.tui = SimpleNamespace(show_modal=show_modal, invalidate=invalidate)
        monkeypatch.setattr(s.mcp, "connect_server", connect)

        mcp_manager(loop, )
        modal_closed = True
        release.set()

        assert emitted.wait(1)
        assert any("MCP server connected: test" in text for text in outputs)

    def test_mcp_manager_aligns_server_labels(self, monkeypatch):
        raw = {
            "mcp": {
                "a": {"url": "https://a.example/mcp"},
                "much-longer": {"url": "https://long.example/mcp", "auto_connect": True},
            }
        }
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        s.mcp.tools["a"] = []
        s.mcp.resources["a"] = []
        loop = CommandLoop(Agent(s), input_fn=lambda _: "", output_fn=lambda _: None)
        captured = {}

        def show_modal(fragments, _handle_key):
            captured["text"] = "".join(text for _style, text in fragments())
            return SELECTION_BACK

        loop.tui = SimpleNamespace(show_modal=show_modal, invalidate=lambda: None)

        mcp_manager(loop, )

        lines = captured["text"].splitlines()
        connected = next(line for line in lines if " a " in line)
        disconnected = next(line for line in lines if "much-longer" in line)
        assert connected.index("●") == disconnected.index("●")
        assert connected.index("manual") == disconnected.index("auto")
        assert connected.rindex("tools") == disconnected.rindex("tools")
        assert lines[0] == "MCP servers · Enter toggles connection"

    def test_mcp_status_dots_use_semantic_terminal_colors(self):
        text = "● connected  ● connecting  ● disconnected  ● disconnecting  ● error  ● skipped"

        colored = UiPrinter.colorize_mcp_status(text)

        assert "\x1b[32m●\x1b[39m connected" in colored
        assert "\x1b[32m●\x1b[39m connecting" in colored
        assert "\x1b[33m●\x1b[39m disconnected" in colored
        assert "\x1b[33m●\x1b[39m disconnecting" in colored
        assert "\x1b[31m●\x1b[39m error" in colored
        assert "\x1b[90m●\x1b[39m skipped" in colored

    def test_mcp_manager_status_dots_receive_selector_styles(self):
        state = ChoiceViewState(
            choices=("up", "busy", "down"),
            labels={"up": "up    ● connected", "busy": "busy  ● connecting", "down": "down  ● disconnected"},
            disabled=set(),
        )

        fragments = state.fragments("MCP servers")

        assert ("class:choice.selected class:choice.status.connected", "●") in fragments
        assert ("class:choice.status.connecting", "●") in fragments
        assert ("class:choice.status.disconnected", "●") in fragments

    def test_unknown_mcp_subcommand(self):
        """Bad /mcp subcommand returns error."""
        s = Session(cwd="/tmp")
        loop = CommandLoop(Agent(s), input_fn=lambda _: "", output_fn=lambda _: None)
        result = mcp_command(loop, "bad_subcommand")
        assert "Unknown" in result

    def test_mcp_subcommands_reject_extra_args(self):
        """MCP subcommands do not silently ignore extra args."""
        s = Session(cwd="/tmp")
        loop = CommandLoop(Agent(s), input_fn=lambda _: "", output_fn=lambda _: None)

        assert mcp_command(loop, "tools a b") == "Usage: /mcp tools [server]"
        assert mcp_command(loop, "connect") == "Usage: /mcp connect <server> [server ...]"
        assert mcp_command(loop, "disconnect") == "Usage: /mcp disconnect <server>"
        assert mcp_command(loop, "disconnect a b") == "Usage: /mcp disconnect <server>"
        assert "Unknown" in mcp_command(loop, "login test")
        assert "Unknown" in mcp_command(loop, "logout test")

    def test_no_mcp_config(self):
        """No MCP config returns message."""
        s = Session(cwd="/tmp")
        s.mcp = None
        loop = CommandLoop(Agent(s), input_fn=lambda _: "", output_fn=lambda _: None)
        result = mcp_command(loop, "")
        assert "not configured" in result


# ---------------------------------------------------------------------------
# Tab completion
# ---------------------------------------------------------------------------


class TestMCPCommandsByName:
    def test_mcp_tools_specific_server(self, monkeypatch):
        """/mcp tools NAME points disconnected servers to connect."""
        raw = {"mcp": {"a": {"url": "http://a/mcp"}, "b": {"url": "http://b/mcp"}}}
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        discovered = []

        monkeypatch.setattr(s.mcp, "discover_server", lambda name: discovered.append(name))

        loop = CommandLoop(Agent(s), input_fn=lambda _: "", output_fn=lambda _: None)
        result = mcp_command(loop, "tools a")

        assert discovered == []
        assert result == "MCP server 'a' is not connected; run /mcp connect a"


# ---------------------------------------------------------------------------
# MCPManager — discover_server with nonexistent server
# ---------------------------------------------------------------------------


class TestMCPTabCompletion:
    def test_mcp_command_completion(self):
        """/mcp completes with connect and inspection commands."""
        completer = CommandCompleter()
        from prompt_toolkit.document import Document

        doc = Document("/mcp ")
        completions = list(completer.get_completions(doc, None))
        texts = [c.text for c in completions]
        assert "tools" in texts
        assert "connect" in texts
        assert "disconnect" in texts
        assert "refresh" not in texts
        assert "login" not in texts
        assert "logout" not in texts

    def test_mcp_completion_prefix_filtering(self):
        """Prefix filters subcommands."""
        completer = CommandCompleter()
        from prompt_toolkit.document import Document

        doc = Document("/mcp c")
        completions = list(completer.get_completions(doc, None))
        texts = [c.text for c in completions]
        assert "tools" not in texts
        assert texts == ["connect"]

    def test_mcp_tools_completion_uses_connected_servers(self):
        """/mcp tools completes only connected MCP server names."""
        completer = CommandCompleter(
            mcp_servers=lambda: ("plain", "oauthOne"),
            mcp_connected_servers=lambda: ("oauthOne",),
        )
        from prompt_toolkit.document import Document

        completions = list(completer.get_completions(Document("/mcp tools o"), None))
        texts = [c.text for c in completions]
        assert texts == ["oauthOne"]

    def test_mcp_connect_completion_uses_all_servers(self):
        completer = CommandCompleter(mcp_servers=lambda: ("plain", "oauthOne"))
        from prompt_toolkit.document import Document

        completions = list(completer.get_completions(Document("/mcp connect p"), None))
        assert [c.text for c in completions] == ["plain"]

    def test_mcp_connect_completion_advances_and_omits_selected_servers(self):
        completer = CommandCompleter(mcp_servers=lambda: ("plain", "oauthOne", "other"))
        from prompt_toolkit.document import Document

        completions = list(completer.get_completions(Document("/mcp connect plain o"), None))

        assert [c.text for c in completions] == ["oauthOne", "other"]
        assert all(c.start_position == -1 for c in completions)

        completions = list(completer.get_completions(Document("/mcp connect plain "), None))
        assert [c.text for c in completions] == ["oauthOne", "other"]

    def test_mcp_disconnect_completion_uses_all_servers(self):
        completer = CommandCompleter(mcp_servers=lambda: ("plain", "oauthOne"))
        from prompt_toolkit.document import Document

        completions = list(completer.get_completions(Document("/mcp disconnect o"), None))
        assert [c.text for c in completions] == ["oauthOne"]


# ---------------------------------------------------------------------------
# render_tools_index truncation
# ---------------------------------------------------------------------------


class TestMCPUserScenarios:
    @staticmethod
    def tool(name, description):
        return SimpleNamespace(
            name=name,
            description=description,
            inputSchema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            annotations=None,
        )

    @staticmethod
    def model():
        class RecordingModel:
            def __init__(self):
                self.requests = []
                self.tools = []

            def request(self, messages, tools=None):
                self.requests.append(messages)
                self.tools.append(tools or [])
                return {"role": "assistant", "content": "done"}, [], "done"

        return RecordingModel()

    @staticmethod
    def mcp_context(model):
        return next(
            (message["content"] for message in model.requests[-1] if str(message.get("content", "")).startswith("--- MCP TOOLS ---")),
            "",
        )

    @staticmethod
    def wait_until(predicate, timeout=1):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.001)
        return predicate()

    def test_startup_manual_connect_and_disconnect_update_next_model_request(self, monkeypatch):
        raw = {
            "mcp": {
                "search": {"url": "https://search.example/mcp", "auto_connect": True},
                "docs": {"url": "https://docs.example/mcp"},
            }
        }
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        tools = {
            "search": [self.tool("find", "Search source code")],
            "docs": [self.tool("lookup", "Look up documentation")],
        }

        async def list_tools(config, _headers):
            return tools[config.name]

        async def list_resources(config, _headers):
            return [_fake_resource(uri="docs://guide.md", description="Project guide")] if config.name == "docs" else []

        monkeypatch.setattr(s.mcp, "_list_tools", list_tools)
        monkeypatch.setattr(s.mcp, "_list_resources", list_resources)
        s.mcp.discover_auto()
        agent = Agent(s, output_fn=lambda _text: None)
        agent.model = self.model()
        loop = CommandLoop(agent, input_fn=lambda _: "", output_fn=lambda _text: None)

        assert agent.run("Search the project") == "done"
        assert "[search]" in self.mcp_context(agent.model)
        assert "[docs]" not in self.mcp_context(agent.model)

        assert mcp_command(loop, "connect docs") == "MCP server connected: docs; tools=1; resources=1"
        assert agent.run("Read the project guide") == "done"
        context = self.mcp_context(agent.model)
        assert "[search]" in context
        assert "[docs]" in context
        assert "docs://guide.md" in context

        assert mcp_command(loop, "disconnect search") == "MCP server disconnected: search"
        assert agent.run("Continue with the documentation") == "done"
        context = self.mcp_context(agent.model)
        assert "[search]" not in context
        assert "[docs]" in context

    def test_resource_only_mention_connects_server_and_reaches_model(self, monkeypatch):
        raw = {"mcp": {"handbook": {"url": "https://handbook.example/mcp"}}}
        s = Session(cwd="/tmp", config=Config.from_dict(raw))

        async def no_tools(_config, _headers):
            return []

        async def handbook(_config, _headers):
            return [_fake_resource(uri="handbook://operations.md", description="Operations handbook")]

        monkeypatch.setattr(s.mcp, "_list_tools", no_tools)
        monkeypatch.setattr(s.mcp, "_list_resources", handbook)
        agent = Agent(s, output_fn=lambda _text: None)
        agent.model = self.model()

        assert agent.run("Use @handbook to check the deployment process") == "done"

        request_text = "\n".join(str(message.get("content", "")) for message in agent.model.requests[-1])
        assert "--- MCP MENTIONS ---" in request_text
        assert "handbook://operations.md" in request_text
        assert "MCP" in {schema["function"]["name"] for schema in agent.model.tools[-1]}

    def test_reauthorization_replaces_cached_token_and_client_as_one_unit(self, tmp_path, monkeypatch):
        url = "https://metabase.example/mcp"
        raw = {"mcp": {"metabase": {"url": url, "auth": "oauth"}}}
        s = Session(cwd=str(tmp_path), config=Config.from_dict(raw))
        store = oauth_store(tmp_path, {url: "stale"})
        s.mcp._oauth_token_store = store
        authorized = False

        async def list_tools(_config, _headers):
            if not authorized:
                raise RuntimeError("authentication required; run /mcp connect metabase")
            return [self.tool("query", "Query analytics")]

        async def no_resources(_config, _headers):
            return []

        def authorize(_config, notify=None):
            nonlocal authorized
            assert oauth_value(store, url, "mcp-oauth-token", "/tokens") is None
            assert oauth_value(store, url, "mcp-oauth-client-info", "/client_info") is None
            put_oauth_state(store, url, "fresh")
            authorized = True

        monkeypatch.setattr(s.mcp, "_list_tools", list_tools)
        monkeypatch.setattr(s.mcp, "_list_resources", no_resources)
        monkeypatch.setattr(s.mcp, "_authenticate_oauth", authorize)
        loop = CommandLoop(Agent(s), input_fn=lambda _: "", output_fn=lambda _text: None)
        loop.interactive_input = True

        result = mcp_command(loop, "connect metabase")

        assert result == "MCP server connected: metabase; tools=1; resources=0"
        assert oauth_value(store, url, "mcp-oauth-token", "/tokens")["access_token"] == "fresh-token"
        assert oauth_value(store, url, "mcp-oauth-client-info", "/client_info")["client_id"] == "fresh-client"

    def test_noninteractive_connect_preserves_rejected_cached_oauth_state(self, tmp_path, monkeypatch):
        url = "https://metabase.example/mcp"
        raw = {"mcp": {"metabase": {"url": url, "auth": "oauth"}}}
        s = Session(cwd=str(tmp_path), config=Config.from_dict(raw))
        store = oauth_store(tmp_path, {url: "cached"})
        s.mcp._oauth_token_store = store

        async def rejected(_config, _headers):
            raise RuntimeError("authentication required; run /mcp connect metabase")

        monkeypatch.setattr(s.mcp, "_list_tools", rejected)
        monkeypatch.setattr(s.mcp, "_list_resources", rejected)
        monkeypatch.setattr(s.mcp, "_authenticate_oauth", lambda *_args, **_kwargs: pytest.fail("non-interactive connect opened OAuth"))
        loop = CommandLoop(Agent(s), input_fn=lambda _: "", output_fn=lambda _text: None)

        result = mcp_command(loop, "connect metabase")

        assert result == "MCP server error: metabase: authentication required; run /mcp connect metabase"
        assert oauth_value(store, url, "mcp-oauth-token", "/tokens")["access_token"] == "cached-token"
        assert oauth_value(store, url, "mcp-oauth-client-info", "/client_info")["client_id"] == "cached-client"

    def test_oauth_mention_reports_rejection_without_starting_login(self, tmp_path, monkeypatch):
        url = "https://metabase.example/mcp"
        raw = {"mcp": {"metabase": {"url": url, "auth": "oauth"}}}
        s = Session(cwd=str(tmp_path), config=Config.from_dict(raw))
        store = oauth_store(tmp_path, {url: "cached"})
        s.mcp._oauth_token_store = store

        async def rejected(_config, _headers):
            raise RuntimeError("authentication required; run /mcp connect metabase")

        monkeypatch.setattr(s.mcp, "_list_tools", rejected)
        monkeypatch.setattr(s.mcp, "_list_resources", rejected)
        monkeypatch.setattr(s.mcp, "_authenticate_oauth", lambda *_args, **_kwargs: pytest.fail("mention opened OAuth"))
        agent = Agent(s, output_fn=lambda _text: None)
        agent.model = self.model()

        assert agent.run("Use @metabase to inspect the dashboard") == "done"

        request_text = "\n".join(str(message.get("content", "")) for message in agent.model.requests[-1])
        assert "[metabase] unavailable: authentication required" in request_text
        assert oauth_value(store, url, "mcp-oauth-client-info", "/client_info")["client_id"] == "cached-client"

    def test_mixed_batch_reauthorizes_only_rejected_oauth_server(self, tmp_path, monkeypatch):
        valid_url = "https://valid.example/mcp"
        stale_url = "https://stale.example/mcp"
        raw = {
            "mcp": {
                "valid": {"url": valid_url, "auth": "oauth"},
                "stale": {"url": stale_url, "auth": "oauth"},
                "plain": {"url": "https://plain.example/mcp"},
            }
        }
        s = Session(cwd=str(tmp_path), config=Config.from_dict(raw))
        store = oauth_store(tmp_path, {valid_url: "valid", stale_url: "stale"})
        s.mcp._oauth_token_store = store
        refreshed: set[str] = set()
        authorized = []

        async def list_tools(config, _headers):
            if config.name == "stale" and config.name not in refreshed:
                raise RuntimeError("HTTP 401 unauthorized")
            return [self.tool(config.name + "_tool", "Tool for " + config.name)]

        async def no_resources(_config, _headers):
            return []

        def authorize(config, notify=None):
            authorized.append(config.name)
            assert config.name == "stale"
            assert oauth_value(store, stale_url, "mcp-oauth-client-info", "/client_info") is None
            assert oauth_value(store, valid_url, "mcp-oauth-client-info", "/client_info")["client_id"] == "valid-client"
            put_oauth_state(store, stale_url, "fresh")
            refreshed.add(config.name)

        monkeypatch.setattr(s.mcp, "_list_tools", list_tools)
        monkeypatch.setattr(s.mcp, "_list_resources", no_resources)
        monkeypatch.setattr(s.mcp, "_authenticate_oauth", authorize)
        loop = CommandLoop(Agent(s), input_fn=lambda _: "", output_fn=lambda _text: None)
        loop.interactive_input = True

        result = mcp_command(loop, "connect valid stale plain")

        assert authorized == ["stale"]
        assert all(s.mcp.connected(name) for name in ("valid", "stale", "plain"))
        assert result.index("`valid`") < result.index("`stale`") < result.index("`plain`")
        assert oauth_value(store, valid_url, "mcp-oauth-client-info", "/client_info")["client_id"] == "valid-client"
        assert oauth_value(store, stale_url, "mcp-oauth-client-info", "/client_info")["client_id"] == "fresh-client"

    def test_batch_connection_isolates_failed_server_from_model_context(self, monkeypatch):
        raw = {
            "mcp": {
                "catalog": {"url": "https://catalog.example/mcp"},
                "offline": {"url": "https://offline.example/mcp"},
            }
        }
        s = Session(cwd="/tmp", config=Config.from_dict(raw))

        async def list_tools(config, _headers):
            if config.name == "offline":
                raise ConnectionError("service unavailable")
            return [self.tool("search", "Search the catalog")]

        async def no_resources(_config, _headers):
            return []

        monkeypatch.setattr(s.mcp, "_list_tools", list_tools)
        monkeypatch.setattr(s.mcp, "_list_resources", no_resources)
        agent = Agent(s, output_fn=lambda _text: None)
        agent.model = self.model()
        loop = CommandLoop(agent, input_fn=lambda _: "", output_fn=lambda _text: None)

        result = mcp_command(loop, "connect catalog offline")
        assert "● connected  `catalog`" in result
        assert "● error  `offline` — service unavailable" in result

        assert agent.run("Search available products") == "done"
        context = self.mcp_context(agent.model)
        assert "[catalog]" in context
        assert "[offline]" not in context

    def test_batch_command_reports_live_progress_until_every_server_finishes(self, monkeypatch):
        raw = {
            "mcp": {
                "alpha": {"url": "https://alpha.example/mcp"},
                "beta": {"url": "https://beta.example/mcp"},
            }
        }
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        started = {name: threading.Event() for name in ("alpha", "beta")}
        release = {name: threading.Event() for name in ("alpha", "beta")}

        async def list_tools(config, _headers):
            started[config.name].set()
            while not release[config.name].is_set():
                await asyncio.sleep(0.001)
            return [self.tool("run", "Run workflow")]

        async def no_resources(_config, _headers):
            return []

        monkeypatch.setattr(s.mcp, "_list_tools", list_tools)
        monkeypatch.setattr(s.mcp, "_list_resources", no_resources)
        loop = CommandLoop(Agent(s), input_fn=lambda _: "", output_fn=lambda _text: None)
        result = []
        worker = threading.Thread(target=lambda: result.append(mcp_command(loop, "connect alpha beta")))
        worker.start()
        assert all(event.wait(1) for event in started.values())
        assert StatusBar(s).mcp_status().startswith("mcp 0/2")

        release["alpha"].set()
        assert self.wait_until(lambda: s.mcp.connected("alpha"))
        assert s.mcp.discovery_status == "discovering"
        assert StatusBar(s).mcp_status().startswith("mcp 1/2")

        release["beta"].set()
        worker.join(1)
        assert not worker.is_alive()
        assert s.mcp.discovery_status == "ready"
        assert StatusBar(s).mcp_status() == "mcp 2"
        assert result and "`alpha`" in result[0] and "`beta`" in result[0]
