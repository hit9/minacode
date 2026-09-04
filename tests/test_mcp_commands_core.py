"""mcp commands core (split from tests/test_mcp_commands.py)."""
import asyncio
from types import SimpleNamespace
from typing import ClassVar

import pytest
from mcp_harness import as_async, mcp_cfg, mcp_tool_info

import wizolt.cli.commands as commands_mod
from wizolt.base import SELECTION_BACK
from wizolt.cli import CommandCompleter, CommandLoop
from wizolt.cli.commands import mcp_command
from wizolt.cli.modals import mcp_manager
from wizolt.cli.update import UpdateChecker
from wizolt.config import (
    Config,
)
from wizolt.engine import Agent
from wizolt.render import UiPrinter
from wizolt.session import Session, SessionSnapshotStore, bootstrap_features
from wizolt.tools import CodeIndex
from wizolt.tui import TUI_MODAL_PENDING, ChoiceViewState


class TestMCPCommands:
    async def test_startup_discovers_auto_servers(self, monkeypatch):
        """The frontends start one discovery task; startup itself does not wait on any server."""
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg()))
        bootstrap_features(s)
        calls = []
        monkeypatch.setattr(s.mcp, "discover_auto", as_async(lambda: calls.append("auto")))
        monkeypatch.setattr(UpdateChecker, "load_cached", lambda _checker: False)
        monkeypatch.setattr(SessionSnapshotStore, "clean_expired", lambda _session: 0)
        monkeypatch.setattr(CodeIndex, "schedule_existing_refresh", lambda _index: False)
        command_loop = CommandLoop(Agent(s), input_fn=lambda _: "", output_fn=lambda _: None)

        command_loop.start_session()
        assert calls == []

        await command_loop.discover_mcp()
        assert calls == ["auto"]

    async def test_mcp_command_no_args_shows_status(self, monkeypatch):
        """/mcp returns server status."""
        raw = mcp_cfg()
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        bootstrap_features(s)

        class FakeTool:
            name = "echo"
            description = "Echo"
            inputSchema: ClassVar[dict] = {"type": "object", "properties": {}, "required": []}
            annotations = None

        async def fake_list(url, headers):
            return [FakeTool()]

        monkeypatch.setattr(s.mcp, "_list_tools", fake_list)
        await s.mcp.discover_auto()

        loop = CommandLoop(Agent(s), input_fn=lambda _: "", output_fn=lambda _: None)
        result = await mcp_command(loop, "")
        assert "test" in result
        assert "| `test` | auto | ● connected | 1     |" in result

    async def test_mcp_tools_shows_listing(self, monkeypatch):
        """/mcp tools returns tool listing."""
        raw = mcp_cfg()
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        bootstrap_features(s)

        class FakeTool:
            name = "echo"
            description = "Echo"
            inputSchema: ClassVar[dict] = {"type": "object", "properties": {}, "required": []}
            annotations = None

        async def fake_list(url, headers):
            return [FakeTool()]

        monkeypatch.setattr(s.mcp, "_list_tools", fake_list)
        await s.mcp.discover_auto()

        loop = CommandLoop(Agent(s), input_fn=lambda _: "", output_fn=lambda _: None)
        result = await mcp_command(loop, "tools")
        assert "### `test`" in result
        assert "echo" in result

    async def test_mcp_tools_without_name_does_not_discover(self, monkeypatch):
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg()))
        bootstrap_features(s)
        calls = []
        monkeypatch.setattr(s.mcp, "discover_server", as_async(lambda name: calls.append(name)))

        loop = CommandLoop(Agent(s), input_fn=lambda _: "", output_fn=lambda _: None)
        result = await mcp_command(loop, "tools")

        assert calls == []
        assert result == "(no connected MCP servers)"

    async def test_mcp_connect_oauth_failure_includes_mcp_url(self, monkeypatch):
        """Interactive connect shows a fallback URL when OAuth does not provide one."""
        raw = mcp_cfg(auth="oauth")
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        bootstrap_features(s)

        async def fake_login(config, headers, operation, *, long_timeout=False, interactive=False, notify=None):
            raise RuntimeError("Unexpected content type: text/html")

        monkeypatch.setattr(s.mcp, "_run_op", fake_login)
        result = await s.mcp.connect_server("test", interactive=True)

        assert "Unexpected content type: text/html" in result
        assert "Open MCP URL: http://localhost:9999/mcp" in result

    async def test_mcp_connect_authenticates_then_loads_capabilities(self, monkeypatch):
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg(auth="oauth")))
        bootstrap_features(s)
        authenticated = False
        calls = []
        cleared = []

        monkeypatch.setattr(s.mcp._oauth_token_store, "has_server_tokens", lambda _url: authenticated)
        monkeypatch.setattr(s.mcp._oauth_token_store, "clear_server", cleared.append)

        async def fake_auth(config, notify=None):
            nonlocal authenticated
            authenticated = True
            calls.append((config.name, notify))

        async def fake_discover(name):
            assert authenticated
            s.mcp.tools[name] = [mcp_tool_info(name, "echo")]
            s.mcp.resources[name] = []

        monkeypatch.setattr(s.mcp, "_authenticate_oauth", fake_auth)
        monkeypatch.setattr(s.mcp, "discover_server", fake_discover)

        result = await s.mcp.connect_server("test", interactive=True)

        assert result == "MCP server connected: test; tools=1; resources=0"
        assert calls == [("test", None)]
        assert cleared == ["http://localhost:9999/mcp"]
        assert s.mcp.connected("test")
        assert "echo" in s.mcp.render_tools_index()

    async def test_mcp_connect_reauthorizes_rejected_cached_oauth_session(self, monkeypatch):
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg(auth="oauth")))
        bootstrap_features(s)
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

        result = await mcp_command(loop, "connect test")

        assert result == "MCP server connected: test; tools=0; resources=0"
        assert attempts.count(True) == 1
        assert attempts.index(True) >= 1
        assert cleared == ["http://localhost:9999/mcp"]
        assert s.mcp.connected("test")

    async def test_mcp_connect_keeps_valid_cached_oauth_session(self, monkeypatch):
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg(auth="oauth")))
        bootstrap_features(s)
        monkeypatch.setattr(s.mcp._oauth_token_store, "has_server_tokens", lambda _url: True)
        monkeypatch.setattr(s.mcp._oauth_token_store, "clear_server", lambda _url: pytest.fail("valid credentials were cleared"))
        monkeypatch.setattr(s.mcp, "_authenticate_oauth", as_async(lambda *_args, **_kwargs: pytest.fail("valid credentials triggered login")))

        async def tools(_config, _headers):
            return [SimpleNamespace(name="echo", description="Echo", inputSchema={}, annotations=None)]

        async def resources(_config, _headers):
            return []

        monkeypatch.setattr(s.mcp, "_list_tools", tools)
        monkeypatch.setattr(s.mcp, "_list_resources", resources)

        result = await s.mcp.connect_server("test", interactive=True)

        assert result == "MCP server connected: test; tools=1; resources=0"

    async def test_mcp_connect_does_not_reauthorize_on_non_auth_failure(self, monkeypatch):
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg(auth="oauth")))
        bootstrap_features(s)
        monkeypatch.setattr(s.mcp._oauth_token_store, "has_server_tokens", lambda _url: True)
        monkeypatch.setattr(s.mcp._oauth_token_store, "clear_server", lambda _url: pytest.fail("credentials were cleared"))
        monkeypatch.setattr(s.mcp, "_authenticate_oauth", as_async(lambda *_args, **_kwargs: pytest.fail("connection error triggered login")))

        async def offline(_config, _headers):
            raise ConnectionError("service unavailable")

        monkeypatch.setattr(s.mcp, "_list_tools", offline)
        monkeypatch.setattr(s.mcp, "_list_resources", offline)

        result = await s.mcp.connect_server("test", interactive=True)

        assert result == "MCP server error: test: service unavailable"

    @pytest.mark.parametrize("rejection", ["invalid_request", "invalid client", "invalid_token", "HTTP 403 forbidden"])
    async def test_mcp_connect_recognizes_cached_oauth_rejection_variants(self, rejection, monkeypatch):
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg(auth="oauth")))
        bootstrap_features(s)
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

        async def authorize(config, notify=None):
            nonlocal authorized
            logins.append(config.name)
            authorized = True

        monkeypatch.setattr(s.mcp, "_list_tools", tools)
        monkeypatch.setattr(s.mcp, "_list_resources", resources)
        monkeypatch.setattr(s.mcp, "_authenticate_oauth", authorize)

        result = await s.mcp.connect_server("test", interactive=True)

        assert result == "MCP server connected: test; tools=0; resources=0"
        assert logins == ["test"]

    async def test_mcp_connect_oauth_requires_interactive_session(self):
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg(auth="oauth")))
        bootstrap_features(s)

        result = await s.mcp.connect_server("test")

        assert result == "MCP server authentication required: test; run /mcp connect test interactively"
        assert not s.mcp.connected("test")

    async def test_mcp_connect_discovers_and_rediscovers_server(self, monkeypatch):
        """Repeated /mcp connect NAME calls reconnect that server."""
        calls = []
        raw = mcp_cfg()
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        bootstrap_features(s)
        monkeypatch.setattr(s.mcp, "discover_server", as_async(lambda name: calls.append(name)))

        loop = CommandLoop(Agent(s), input_fn=lambda _: "", output_fn=lambda _: None)
        await mcp_command(loop, "connect test")
        await mcp_command(loop, "connect test")

        assert calls == ["test", "test"]

    async def test_mcp_connects_multiple_servers_concurrently_in_argument_order(self, monkeypatch):
        raw = {
            "mcp": {
                "alpha": {"url": "https://alpha.example/mcp"},
                "beta": {"url": "https://beta.example/mcp"},
            }
        }
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        bootstrap_features(s)
        both_started = asyncio.Event()
        started = []

        async def fake_discover(config):
            started.append(config.name)
            if len(started) == 2:
                both_started.set()
            # Neither server can finish until the other has started: a batch that connected them
            # one after another would never get past this.
            await asyncio.wait_for(both_started.wait(), 1)
            s.mcp.tools[config.name] = []
            s.mcp.resources[config.name] = []

        monkeypatch.setattr(s.mcp, "_discover_one", fake_discover)

        result = await s.mcp.connect_servers(["alpha", "beta", "alpha"])

        assert set(started) == {"alpha", "beta"}
        assert result == ("MCP connection results:\n\n- ● connected  `alpha` — 0 tools\n- ● connected  `beta` — 0 tools")

    async def test_mcp_batch_serializes_oauth_only(self, monkeypatch):
        raw = {
            "mcp": {
                "alpha": {"url": "https://alpha.example/mcp", "auth": "oauth"},
                "beta": {"url": "https://beta.example/mcp", "auth": "oauth"},
            }
        }
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        bootstrap_features(s)
        authenticated: set[str] = set()
        active = 0
        maximum = 0
        monkeypatch.setattr(s.mcp._oauth_token_store, "has_server_tokens", lambda url: url in authenticated)

        async def authenticate(config, notify=None):
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            await asyncio.sleep(0.02)  # a login yields, so a second one would overlap unless gated
            authenticated.add(config.url)
            active -= 1

        async def discover(config):
            s.mcp.tools[config.name] = []
            s.mcp.resources[config.name] = []

        monkeypatch.setattr(s.mcp, "_authenticate_oauth", authenticate)
        monkeypatch.setattr(s.mcp, "_discover_one", discover)

        await s.mcp.connect_servers(["alpha", "beta"], interactive=True)

        assert maximum == 1

    async def test_mcp_batch_keeps_oauth_failure_compact_and_connects_other_servers(self, monkeypatch):
        raw = {
            "mcp": {
                "oauth": {"url": "https://oauth.example/mcp", "auth": "oauth"},
                "plain": {"url": "https://plain.example/mcp"},
            }
        }
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        bootstrap_features(s)
        monkeypatch.setattr(s.mcp._oauth_token_store, "clear_server", lambda _url: None)
        monkeypatch.setattr(
            s.mcp,
            "_authenticate_oauth",
            as_async(lambda config, notify=None: "\n".join(
                [
                    "MCP OAuth authentication failed for oauth: authorization denied",
                    "No authorization URL was provided by the server.",
                    "Open MCP URL: " + config.url,
                ]
            )),
        )

        async def tools(_config, _headers):
            return [SimpleNamespace(name="echo", description="Echo", inputSchema={}, annotations=None)]

        async def no_resources(_config, _headers):
            return []

        monkeypatch.setattr(s.mcp, "_list_tools", tools)
        monkeypatch.setattr(s.mcp, "_list_resources", no_resources)

        result = await s.mcp.connect_servers(["oauth", "plain"], interactive=True)

        assert result == (
            "MCP connection results:\n\n"
            "- ● error  `oauth` — authorization denied\n"
            "    No authorization URL was provided by the server.\n"
            "    Open MCP URL: https://oauth.example/mcp\n"
            "- ● connected  `plain` — 1 tool"
        )
        assert not s.mcp.connected("oauth")
        assert s.mcp.connected("plain")

    async def test_noninteractive_batch_never_starts_missing_oauth_login(self, monkeypatch):
        raw = {
            "mcp": {
                "oauth": {"url": "https://oauth.example/mcp", "auth": "oauth"},
                "plain": {"url": "https://plain.example/mcp"},
            }
        }
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        bootstrap_features(s)
        monkeypatch.setattr(s.mcp, "_authenticate_oauth", as_async(lambda *_args, **_kwargs: pytest.fail("batch opened OAuth")))

        async def tools(_config, _headers):
            return []

        async def no_resources(_config, _headers):
            return []

        monkeypatch.setattr(s.mcp, "_list_tools", tools)
        monkeypatch.setattr(s.mcp, "_list_resources", no_resources)

        result = await s.mcp.connect_servers(["oauth", "plain"], interactive=False)

        assert "● error  `oauth` — authentication required" in result
        assert "● connected  `plain` — 0 tools" in result
        assert not s.mcp.connected("oauth")
        assert s.mcp.connected("plain")

    async def test_mcp_batch_connect_formats_failures_as_separate_list_items(self, monkeypatch):
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg()))
        bootstrap_features(s)
        monkeypatch.setattr(s.mcp, "_discover_one", as_async(lambda config: s.mcp.set_server_error(config.name, "offline")))

        result = await s.mcp.connect_servers(["test", "missing"])

        assert result == ("MCP connection results:\n\n- ● error  `test` — offline\n- ● error  `missing` — server not found")

    async def test_mcp_connect_command_accepts_multiple_servers(self, monkeypatch):
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg()))
        bootstrap_features(s)
        calls = []
        monkeypatch.setattr(
            s.mcp,
            "connect_servers",
            as_async(lambda names, **kwargs: calls.append((names, kwargs)) or "connected batch"),
        )
        loop = CommandLoop(Agent(s), input_fn=lambda _: "", output_fn=lambda _: None)

        result = await mcp_command(loop, "connect alpha beta")

        assert result == "connected batch"
        assert calls == [(["alpha", "beta"], {"interactive": False, "notify": loop.emit})]

    async def test_mcp_connect_rejects_missing_server(self):
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg()))
        bootstrap_features(s)
        loop = CommandLoop(Agent(s), input_fn=lambda _: "", output_fn=lambda _: None)

        assert await mcp_command(loop, "connect missing") == "MCP server not found: missing"

    async def test_mcp_disconnect_removes_connected_server(self):
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg()))
        bootstrap_features(s)
        s.mcp.tools["test"] = [mcp_tool_info("test", "echo")]
        s.mcp.resources["test"] = []
        loop = CommandLoop(Agent(s), input_fn=lambda _: "", output_fn=lambda _: None)

        assert await mcp_command(loop, "disconnect test") == "MCP server disconnected: test"
        assert not s.mcp.connected("test")

    def test_mcp_disconnect_oauth_also_clears_authentication(self, monkeypatch):
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg(auth="oauth")))
        bootstrap_features(s)
        s.mcp.tools["test"] = [mcp_tool_info("test", "echo")]
        s.mcp.resources["test"] = []
        cleared = []
        monkeypatch.setattr(s.mcp._oauth_token_store, "clear_server", cleared.append)

        result = s.mcp.disconnect_server("test")

        assert result == "MCP server disconnected: test"
        assert cleared == ["http://localhost:9999/mcp"]
        assert not s.mcp.connected("test")

    async def test_bare_mcp_opens_manager_in_tui(self, monkeypatch):
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg()))
        bootstrap_features(s)
        loop = CommandLoop(Agent(s), input_fn=lambda _: "", output_fn=lambda _: None)
        loop.tui = SimpleNamespace(input_mode="idle")
        calls = []
        monkeypatch.setattr(commands_mod, "mcp_manager", as_async(lambda _loop: calls.append("manager")))

        assert await mcp_command(loop, "") is None
        assert calls == ["manager"]

    async def test_mcp_manager_connects_selected_server(self, monkeypatch):
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg()))
        bootstrap_features(s)
        loop = CommandLoop(Agent(s), input_fn=lambda _: "", output_fn=lambda _: None)
        connected = asyncio.Event()
        release = asyncio.Event()
        repainted = asyncio.Event()

        async def connect(name, **_kwargs):
            await release.wait()
            s.mcp.tools[name] = []
            s.mcp.resources[name] = []
            return "connected " + name

        async def show_modal(fragments, handle_key):
            assert handle_key("enter") is TUI_MODAL_PENDING
            assert "● connecting" in "".join(text for _, text in fragments())
            release.set()
            await asyncio.wait_for(repainted.wait(), 1)
            assert "● connected" in "".join(text for _, text in fragments())
            connected.set()
            return SELECTION_BACK

        loop.tui = SimpleNamespace(show_modal=show_modal, invalidate=repainted.set)
        monkeypatch.setattr(s.mcp, "connect_server", connect)

        await mcp_manager(loop)

        assert connected.is_set()
        assert s.mcp.connected("test")

    async def test_mcp_manager_disconnects_selected_server(self, monkeypatch):
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg()))
        bootstrap_features(s)
        s.mcp.tools["test"] = []
        s.mcp.resources["test"] = []
        loop = CommandLoop(Agent(s), input_fn=lambda _: "", output_fn=lambda _: None)
        repainted = asyncio.Event()

        async def show_modal(fragments, handle_key):
            assert handle_key("enter") is TUI_MODAL_PENDING
            # The toggle is a task, so it has not run yet: the row shows the transition the key
            # press recorded, and only the repaint that follows shows the result.
            assert "● disconnecting" in "".join(text for _, text in fragments())
            await asyncio.wait_for(repainted.wait(), 1)
            assert "● disconnected" in "".join(text for _, text in fragments())
            return SELECTION_BACK

        loop.tui = SimpleNamespace(show_modal=show_modal, invalidate=repainted.set)

        await mcp_manager(loop)

        assert not s.mcp.connected("test")

    async def test_mcp_manager_starts_multiple_connections_concurrently(self, monkeypatch):
        raw = {"mcp": {"a": {"url": "http://a/mcp"}, "b": {"url": "http://b/mcp"}}}
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        bootstrap_features(s)
        loop = CommandLoop(Agent(s), input_fn=lambda _: "", output_fn=lambda _: None)
        started = {name: asyncio.Event() for name in ("a", "b")}
        release = asyncio.Event()

        async def connect(name, **_kwargs):
            started[name].set()
            await release.wait()
            s.mcp.tools[name] = []
            s.mcp.resources[name] = []
            return "connected " + name

        async def show_modal(_fragments, handle_key):
            assert handle_key("enter") is TUI_MODAL_PENDING
            assert handle_key("j") is TUI_MODAL_PENDING
            assert handle_key("enter") is TUI_MODAL_PENDING
            # Both toggles are in flight before either is allowed to finish.
            await asyncio.wait_for(asyncio.gather(*(event.wait() for event in started.values())), 1)
            release.set()
            return SELECTION_BACK

        loop.tui = SimpleNamespace(show_modal=show_modal, invalidate=lambda: None)
        monkeypatch.setattr(s.mcp, "connect_server", connect)

        await mcp_manager(loop)

        assert all(event.is_set() for event in started.values())

    async def test_mcp_manager_emits_late_result_without_repainting_closed_modal(self, monkeypatch):
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg(auto_connect=False)))
        bootstrap_features(s)
        outputs = []

        async def connect(name, **_kwargs):
            s.mcp.tools[name] = []
            s.mcp.resources[name] = []
            return "MCP server connected: " + name + "; tools=0; resources=0"

        async def show_modal(_fragments, handle_key):
            # Closes immediately: the toggle it started is a task that has not run yet, so it can
            # only finish during the drain, with the modal already gone.
            assert handle_key("enter") is TUI_MODAL_PENDING
            return SELECTION_BACK

        loop = CommandLoop(Agent(s), input_fn=lambda _: "", output_fn=outputs.append)
        loop.tui = SimpleNamespace(
            show_modal=show_modal,
            invalidate=lambda: pytest.fail("a completed toggle repainted a closed modal"),
        )
        monkeypatch.setattr(s.mcp, "connect_server", connect)

        await mcp_manager(loop)

        assert any("MCP server connected: test" in str(text) for text in outputs)

    async def test_mcp_manager_aligns_server_labels(self, monkeypatch):
        raw = {
            "mcp": {
                "a": {"url": "https://a.example/mcp"},
                "much-longer": {"url": "https://long.example/mcp", "auto_connect": True},
            }
        }
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        bootstrap_features(s)
        s.mcp.tools["a"] = []
        s.mcp.resources["a"] = []
        loop = CommandLoop(Agent(s), input_fn=lambda _: "", output_fn=lambda _: None)
        captured = {}

        async def show_modal(fragments, _handle_key):
            captured["text"] = "".join(text for _, text in fragments())
            return SELECTION_BACK

        loop.tui = SimpleNamespace(show_modal=show_modal, invalidate=lambda: None)

        await mcp_manager(loop)

        lines = captured["text"].splitlines()
        connected = next(line for line in lines if " a " in line)
        disconnected = next(line for line in lines if "much-longer" in line)
        assert connected.index("●") == disconnected.index("●")
        assert connected.index("manual") == disconnected.index("auto")
        assert connected.rindex("tools") == disconnected.rindex("tools")
        # The modal opens with a blank spacer line, and its title takes the same indent as its rows.
        assert lines[:2] == ["", "  MCP servers · Enter toggles connection"]

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

    async def test_unknown_mcp_subcommand(self):
        """Bad /mcp subcommand returns error."""
        s = Session(cwd="/tmp")
        bootstrap_features(s)
        loop = CommandLoop(Agent(s), input_fn=lambda _: "", output_fn=lambda _: None)
        result = await mcp_command(loop, "bad_subcommand")
        assert "Unknown" in result

    async def test_mcp_subcommands_reject_extra_args(self):
        """MCP subcommands do not silently ignore extra args."""
        s = Session(cwd="/tmp")
        bootstrap_features(s)
        loop = CommandLoop(Agent(s), input_fn=lambda _: "", output_fn=lambda _: None)

        assert await mcp_command(loop, "tools a b") == "Usage: /mcp tools [server]"
        assert await mcp_command(loop, "connect") == "Usage: /mcp connect <server> [server ...]"
        assert await mcp_command(loop, "disconnect") == "Usage: /mcp disconnect <server>"
        assert await mcp_command(loop, "disconnect a b") == "Usage: /mcp disconnect <server>"
        assert "Unknown" in await mcp_command(loop, "login test")
        assert "Unknown" in await mcp_command(loop, "logout test")

    async def test_no_mcp_config(self):
        """No MCP config returns message."""
        s = Session(cwd="/tmp")
        bootstrap_features(s)
        s.mcp = None
        loop = CommandLoop(Agent(s), input_fn=lambda _: "", output_fn=lambda _: None)
        result = await mcp_command(loop, "")
        assert "not configured" in result

class TestMCPCommandsByName:
    async def test_mcp_tools_specific_server(self, monkeypatch):
        """/mcp tools NAME points disconnected servers to connect."""
        raw = {"mcp": {"a": {"url": "http://a/mcp"}, "b": {"url": "http://b/mcp"}}}
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        bootstrap_features(s)
        discovered = []

        monkeypatch.setattr(s.mcp, "discover_server", as_async(lambda name: discovered.append(name)))

        loop = CommandLoop(Agent(s), input_fn=lambda _: "", output_fn=lambda _: None)
        result = await mcp_command(loop, "tools a")

        assert discovered == []
        assert result == "MCP server 'a' is not connected; run /mcp connect a"

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
