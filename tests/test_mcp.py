"""Tests for minacode MCP client integration."""

import asyncio
import os
import threading
import time
from types import SimpleNamespace

import pytest

from minacode.base import SELECTION_BACK, Config, ToolCall, ToolError
from minacode.engine import Agent, ContextManager, ToolRunner, UpdateChecker
from minacode.loop import CommandCompleter, CommandLoop
from minacode.mcp import MCPFileTokenStore, MCPManager, MCPResourceInfo, MCPServerConfig, MCPToolInfo
from minacode.render import StatusBar, UiPrinter
from minacode.session import Session, SessionSnapshotStore
from minacode.tools import CodeIndex, MCPTool, Tool
from minacode.tui import TUI_MODAL_PENDING, ChoiceViewState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def session(tmp_path):
    return Session(cwd=str(tmp_path))


def mcp_cfg(**overrides) -> dict:
    """Return a full [mcp.x] config dict for one server."""
    cfg = {
        "mcp": {
            "test": {
                "url": "http://localhost:9999/mcp",
                "auto_connect": True,
            }
        }
    }
    server = cfg["mcp"]["test"]
    server.update(overrides)
    return cfg


def parse_one(raw: dict) -> MCPServerConfig | None:
    """Parse a config dict and return the first server config."""
    config = Config.from_dict(raw)
    s = Session(cwd="/tmp", config=config)
    configs = s.mcp.parse_configs()
    return configs[0] if configs else None


def mcp_tool_info(server: str, name: str, **kw) -> MCPToolInfo:
    """Create an MCPToolInfo suitable for tests."""
    return MCPToolInfo(
        server=server,
        name=name,
        description=kw.pop("description", "A test tool."),
        input_schema=kw.pop(
            "input_schema",
            {
                "type": "object",
                "properties": {"text": {"type": "string", "description": "Input text."}},
                "required": ["text"],
            },
        ),
        annotations=kw.pop("annotations", {}),
        **kw,
    )


def put_oauth_state(store: MCPFileTokenStore, url: str, label: str) -> None:
    data = store.load()
    data.setdefault("mcp-oauth-token", {})[store.token_key(url, "/tokens")] = {"value": {"access_token": label + "-token", "token_type": "Bearer"}}
    data.setdefault("mcp-oauth-client-info", {})[store.token_key(url, "/client_info")] = {
        "value": {"client_id": label + "-client", "redirect_uris": ["http://localhost:12345/callback"]}
    }
    store.save(data)


def oauth_store(tmp_path, states: dict[str, str]) -> MCPFileTokenStore:
    """Create a real token store containing one token/client pair per server URL."""
    store = MCPFileTokenStore(str(tmp_path / "mcp-oauth.json"))
    for url, label in states.items():
        put_oauth_state(store, url, label)
    return store


def oauth_value(store: MCPFileTokenStore, url: str, collection: str, suffix: str) -> dict | None:
    entry = store.load().get(collection, {}).get(store.token_key(url, suffix))
    return entry.get("value") if entry else None


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


class TestConfigParsing:
    def test_parse_basic_server(self):
        """Parse [mcp.x] with explicit automatic connection."""
        cfg = parse_one(mcp_cfg())
        assert cfg is not None
        assert cfg.name == "test"
        assert cfg.url == "http://localhost:9999/mcp"
        assert cfg.auto_connect is True
        assert cfg.bearer_token_env_var == ""
        assert cfg.error == ""

    def test_parse_with_bearer_token_env_var(self):
        """Parse with bearer_token_env_var set."""
        cfg = parse_one(mcp_cfg(bearer_token_env_var="MY_TOKEN"))
        assert cfg is not None
        assert cfg.bearer_token_env_var == "MY_TOKEN"

    def test_auto_connect_defaults_false(self):
        cfg = parse_one({"mcp": {"test": {"url": "http://localhost:9999/mcp"}}})
        assert cfg is not None
        assert cfg.auto_connect is False

    @pytest.mark.parametrize("legacy", [True, False])
    def test_legacy_enabled_is_ignored(self, legacy):
        cfg = parse_one({"mcp": {"test": {"url": "http://localhost:9999/mcp", "enabled": legacy}}})
        assert cfg is not None
        assert cfg.auto_connect is False

    def test_missing_url(self):
        """Missing URL stores a server configuration error."""
        raw = mcp_cfg()
        raw["mcp"]["test"].pop("url")
        cfg = parse_one(raw)
        assert cfg is not None
        assert cfg.error is not None

    def test_empty_mcp_config(self):
        """No [mcp.*] sections returns empty list."""
        s = session("/tmp")
        assert s.mcp.parse_configs() == []

    def test_multiple_servers(self):
        """Multiple MCP servers are all parsed."""
        raw = {
            "mcp": {
                "a": {"url": "http://a/mcp"},
                "b": {"url": "http://b/mcp", "enabled": False},
            }
        }
        config = Config.from_dict(raw)
        s = Session(cwd="/tmp", config=config)
        configs = s.mcp.parse_configs()
        assert len(configs) == 2
        assert {c.name for c in configs} == {"a", "b"}

    def test_env_http_headers_valid(self):
        """Valid env_http_headers map is parsed."""
        cfg = parse_one(mcp_cfg(env_http_headers={"X-API-Key": "MCP_KEY"}))
        assert cfg is not None
        assert cfg.env_http_headers == {"X-API-Key": "MCP_KEY"}

    def test_env_http_headers_invalid_type(self):
        """Non-dict env_http_headers sets error."""
        cfg = parse_one(mcp_cfg(env_http_headers="bad"))
        assert cfg is not None
        assert cfg.error is not None

    def test_use_mcp_config_via_config(self):
        """MCPServerConfig stores bearer_token and env_http_headers correctly."""
        raw = mcp_cfg(bearer_token_env_var="TOKEN", env_http_headers={"X-Key": "KEY_VAR"})
        cfg = parse_one(raw)
        assert cfg.bearer_token_env_var == "TOKEN"
        assert cfg.env_http_headers == {"X-Key": "KEY_VAR"}


class TestStdioConfig:
    def test_parse_stdio_server(self):
        """command/args/env are parsed for a stdio server."""
        cfg = parse_one({"mcp": {"x": {"command": "npx", "args": ["-y", "srv"], "env": {"A": "b"}}}})
        assert cfg.command == "npx"
        assert cfg.args == ("-y", "srv")
        assert cfg.env == {"A": "b"}
        assert cfg.error == ""

    def test_url_and_command_mutually_exclusive(self):
        """Providing both url and command is an error; so is neither."""
        assert "exactly one" in parse_one({"mcp": {"x": {"url": "http://x/mcp", "command": "npx"}}}).error
        assert "exactly one" in parse_one({"mcp": {"x": {"auto_connect": True}}}).error

    def test_stdio_rejects_http_auth(self):
        """stdio servers cannot use HTTP auth/headers."""
        assert parse_one({"mcp": {"x": {"command": "npx", "auth": "oauth"}}}).error
        assert parse_one({"mcp": {"x": {"command": "npx", "bearer_token_env_var": "T"}}}).error

    def test_bad_args_type(self):
        """Non-string-list args sets an error."""
        assert "args must be a string list" in parse_one({"mcp": {"x": {"command": "npx", "args": "nope"}}}).error

    def test_transport_selection(self):
        """_transport builds a stdio transport for command servers, http otherwise."""
        from fastmcp.client.transports import StdioTransport, StreamableHttpTransport

        s = Session(cwd="/tmp", config=Config.from_dict({"mcp": {"x": {"command": "npx", "args": ["srv"]}}}))
        assert isinstance(s.mcp._transport(s.mcp.parse_configs()[0], {}), StdioTransport)
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg()))
        assert isinstance(s.mcp._transport(s.mcp.parse_configs()[0], {}), StreamableHttpTransport)


# ---------------------------------------------------------------------------
# MCPManager header/auth building
# ---------------------------------------------------------------------------


class TestMCPManagerHeaders:
    def test_bearer_token_success(self, monkeypatch):
        """bearer_token_env_var reads from environment."""
        monkeypatch.setenv("MY_TEST_TOKEN", "secret123")
        raw = mcp_cfg(bearer_token_env_var="MY_TEST_TOKEN")
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        config = s.mcp.parse_configs()[0]
        headers = s.mcp._build_mcp_headers(config)
        assert headers == {"Authorization": "Bearer secret123"}

    def test_bearer_token_missing_var(self, monkeypatch):
        """Missing bearer env var returns error string."""
        monkeypatch.delenv("MISSING_TOKEN", raising=False)
        raw = mcp_cfg(bearer_token_env_var="MISSING_TOKEN")
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        config = s.mcp.parse_configs()[0]
        result = s.mcp._build_mcp_headers(config)
        assert isinstance(result, str)
        assert "missing" in result.lower()

    def test_env_http_headers_success(self, monkeypatch):
        """env_http_headers reads header values from environment."""
        monkeypatch.setenv("MY_HEADER_VAL", "xyz")
        raw = mcp_cfg(env_http_headers={"X-Custom": "MY_HEADER_VAL"})
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        config = s.mcp.parse_configs()[0]
        headers = s.mcp._build_mcp_headers(config)
        assert headers == {"X-Custom": "xyz"}

    def test_authorization_env_http_header_success(self, monkeypatch):
        """Authorization is allowed via env_http_headers when it is the only auth source."""
        monkeypatch.setenv("AUTH_VAL", "Bearer custom")
        raw = mcp_cfg(env_http_headers={"Authorization": "AUTH_VAL"})
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        config = s.mcp.parse_configs()[0]
        headers = s.mcp._build_mcp_headers(config)
        assert headers == {"Authorization": "Bearer custom"}

    def test_env_http_headers_missing_var(self, monkeypatch):
        """Missing env_http_headers env var returns error string."""
        monkeypatch.delenv("MISSING_HEADER_VAL", raising=False)
        raw = mcp_cfg(env_http_headers={"X-Custom": "MISSING_HEADER_VAL"})
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        config = s.mcp.parse_configs()[0]
        result = s.mcp._build_mcp_headers(config)
        assert isinstance(result, str)

    def test_conflicting_bearer_and_authorization_header(self, monkeypatch):
        """Bearer + explicit Authorization header produces config error."""
        monkeypatch.setenv("TOKEN", "t")
        monkeypatch.setenv("AUTH_VAL", "Bearer x")
        raw = mcp_cfg(
            bearer_token_env_var="TOKEN",
            env_http_headers={"Authorization": "AUTH_VAL"},
        )
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        config = s.mcp.parse_configs()[0]
        result = s.mcp._build_mcp_headers(config)
        assert isinstance(result, str)
        assert "conflicting" in result.lower()

    def test_no_auth_config(self):
        """No auth config produces empty headers dict."""
        raw = mcp_cfg()
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        config = s.mcp.parse_configs()[0]
        headers = s.mcp._build_mcp_headers(config)
        assert headers == {}


# ---------------------------------------------------------------------------
# MCPManager discover & state
# ---------------------------------------------------------------------------


class TestMCPManagerDiscovery:
    def test_discovery_status_tracking(self, monkeypatch):
        """Fresh MCPManager starts stale, discovering, then ready/error."""
        s = session("/tmp")
        assert s.mcp.discovery_status == "stale"

    def test_no_mcp_config_no_tools(self):
        """No MCP servers configured means no tools."""
        s = session("/tmp")
        assert s.mcp.parse_configs() == []
        assert s.mcp.tools == {}

    def test_async_runner_reuses_one_loop(self):
        """MCP async work is scheduled onto one manager-owned event loop."""
        s = session("/tmp")

        async def loop_id():
            return id(asyncio.get_running_loop())

        assert s.mcp.run_async(loop_id()) == s.mcp.run_async(loop_id())

    def test_oauth_token_store_lock_is_shared_by_path(self, tmp_path):
        """Token storage keeps one store and one lock per token file path."""
        s = session(tmp_path)
        store = s.mcp._oauth_token_store
        same_path_store = MCPFileTokenStore(store.path)

        assert same_path_store.lock is store.lock

    def test_token_store_put_get_roundtrip(self, tmp_path):
        """put persists a value that get returns — put is the OAuth storage protocol's writer."""
        import asyncio

        store = MCPFileTokenStore(str(tmp_path / "tokens.json"))

        async def roundtrip():
            await store.put("k", {"v": 1}, collection="mcp-oauth-token")
            return await store.get("k", collection="mcp-oauth-token")

        assert asyncio.run(roundtrip()) == {"v": 1}

    def test_clear_server_matches_fastmcp_oauth_storage_contract(self, tmp_path):
        from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

        url = "https://mcp.example.com/mcp"
        s = Session(cwd=str(tmp_path), config=Config.from_dict({"mcp": {"test": {"url": url, "auth": "oauth"}}}))
        store = MCPFileTokenStore(str(tmp_path / "tokens.json"))
        s.mcp._oauth_token_store = store
        auth = s.mcp.oauth_client(s.mcp.find_config("test"))
        auth._bind(url)
        adapter = auth.token_storage_adapter

        async def roundtrip():
            await adapter.set_tokens(OAuthToken(access_token="stale", token_type="Bearer", expires_in=3600))
            await adapter.set_client_info(OAuthClientInformationFull(client_id="old-client", redirect_uris=["http://localhost:12345/callback"]))
            assert await adapter.get_tokens() is not None
            assert await adapter.get_client_info() is not None
            s.mcp._oauth_token_store.clear_server(url)
            return await adapter.get_tokens(), await adapter.get_client_info(), await adapter.get_token_expiry()

        assert asyncio.run(roundtrip()) == (None, None, None)

    def test_oauth_redirect_requires_explicit_interactive_connection(self):
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg(auth="oauth")))
        auth = s.mcp.oauth_client(s.mcp.find_config("test"), interactive=False)

        with pytest.raises(RuntimeError, match=r"authentication required; run /mcp connect test"):
            asyncio.run(auth.redirect_handler("https://login.example/authorize"))

    def test_interactive_oauth_redirect_notifies_before_delegating(self, monkeypatch):
        from fastmcp.client.auth import OAuth

        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg(auth="oauth")))
        notices = []
        delegated = []

        async def redirect(_auth, url):
            delegated.append(url)

        monkeypatch.setattr(OAuth, "redirect_handler", redirect)
        auth = s.mcp.oauth_client(s.mcp.find_config("test"), interactive=True, notify=notices.append)

        asyncio.run(auth.redirect_handler("https://login.example/authorize"))

        assert notices == ["Open this URL to authorize MCP server `test`:\nhttps://login.example/authorize"]
        assert delegated == ["https://login.example/authorize"]

    def test_interactive_run_op_builds_interactive_oauth_client(self, monkeypatch):
        import fastmcp.client

        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg(auth="oauth")))
        config = s.mcp.find_config("test")
        marker = object()
        auth_calls = []
        client_args = []

        def oauth_client(received, *, interactive=False, notify=None):
            auth_calls.append((received, interactive, notify))
            return marker

        class Client:
            def __init__(self, transport, *, auth, timeout, init_timeout):
                client_args.append((transport, auth, timeout, init_timeout))

            async def __aenter__(self):
                return SimpleNamespace(ping=lambda: asyncio.sleep(0, result="pong"))

            async def __aexit__(self, *_args):
                return None

        def notify(_text):
            return None

        monkeypatch.setattr(s.mcp, "oauth_client", oauth_client)
        monkeypatch.setattr(s.mcp, "_transport", lambda *_args: "transport")
        monkeypatch.setattr(fastmcp.client, "Client", Client)

        result = asyncio.run(s.mcp._run_op(config, {}, lambda client: client.ping(), interactive=True, notify=notify))

        assert result == "pong"
        assert auth_calls == [(config, True, notify)]
        assert client_args == [("transport", marker, s.mcp.call_timeout(), s.mcp.call_timeout())]

    def test_save_closes_fd_when_fdopen_fails(self, tmp_path, monkeypatch):
        """os.fdopen doesn't close its fd on failure — save() must close it or the descriptor leaks."""
        store = MCPFileTokenStore(str(tmp_path / "tokens.json"))
        closed: list[int] = []
        real_close = os.close

        def track_close(fd: int) -> None:
            closed.append(fd)
            real_close(fd)

        def fdopen_raises(*args, **kwargs):
            raise OSError("simulated fdopen failure")

        monkeypatch.setattr(os, "fdopen", fdopen_raises)
        monkeypatch.setattr(os, "close", track_close)

        try:
            store.save({"c": {"k": {"value": {"v": 1}}}})
            raised = False
        except OSError:
            raised = True

        assert raised, "save should propagate fdopen failure"
        assert closed, "fd must be closed when fdopen raises, otherwise it leaks"

    def test_discover_auto_stale_to_discovering(self, monkeypatch):
        """discover_auto sets status to discovering then ready."""
        raw = mcp_cfg()
        s = Session(cwd="/tmp", config=Config.from_dict(raw))

        # Inject a fake _list_tools to avoid real HTTP
        async def fake_list(url, headers):
            return []

        monkeypatch.setattr(s.mcp, "_list_tools", fake_list)

        assert s.mcp.discovery_status == "stale"
        s.mcp.discover_auto()
        assert s.mcp.discovery_status == "ready"

    def test_discover_auto_error_sets_status(self, monkeypatch):
        """Failed discovery sets error status."""
        raw = mcp_cfg()
        s = Session(cwd="/tmp", config=Config.from_dict(raw))

        async def fake_fail(url, headers):
            raise Exception("connection refused")

        monkeypatch.setattr(s.mcp, "_list_tools", fake_fail)

        s.mcp.discover_auto()
        assert s.mcp.discovery_status == "ready"
        assert s.mcp.server_errors.get("test") is not None

    def test_discover_auto_ignores_cancelled_server(self, monkeypatch):
        raw = mcp_cfg()
        s = Session(cwd="/tmp", config=Config.from_dict(raw))

        async def cancelled(_config, _headers):
            raise asyncio.CancelledError

        monkeypatch.setattr(s.mcp, "_gather_assets", cancelled)

        s.mcp.discover_auto()

        assert s.mcp.discovery_status == "ready"
        assert "test" not in s.mcp.server_errors

    def test_cancelled_error_detection_handles_groups_and_cyclic_causes(self):
        cancelled = BaseExceptionGroup("cancelled", [asyncio.CancelledError()])
        mixed = BaseExceptionGroup("mixed", [asyncio.CancelledError(), RuntimeError("failed")])
        first = RuntimeError("first")
        second = RuntimeError("second")
        first.__cause__ = second
        second.__cause__ = first

        assert MCPManager.is_cancelled_error(cancelled)
        assert not MCPManager.is_cancelled_error(mixed)
        assert not MCPManager.is_cancelled_error(first)

    def test_discover_auto_skips_missing_bearer_env(self, monkeypatch):
        """Missing bearer_token_env_var skips discovery without an error log."""
        monkeypatch.delenv("MISSING_TOKEN", raising=False)
        raw = mcp_cfg(bearer_token_env_var="MISSING_TOKEN")
        s = Session(cwd="/tmp", config=Config.from_dict(raw))

        s.mcp.discover_auto()

        assert "test" not in s.mcp.server_errors
        assert "test" in s.mcp.server_skips
        assert "missing environment variable MISSING_TOKEN" in s.mcp.render_server_status()

    def test_discover_auto_loads_servers_in_parallel(self, monkeypatch):
        """Multiple automatic servers are discovered in parallel."""
        raw = {
            "mcp": {
                "a": {"url": "http://a/mcp", "auto_connect": True},
                "b": {"url": "http://b/mcp", "auto_connect": True},
            }
        }
        s = Session(cwd="/tmp", config=Config.from_dict(raw))

        async def fake_list(url, headers):
            await asyncio.sleep(0.1)
            return []

        monkeypatch.setattr(s.mcp, "_list_tools", fake_list)
        monkeypatch.setattr(s.mcp, "_list_resources", fake_list)
        started = time.monotonic()
        s.mcp.discover_auto()
        elapsed = time.monotonic() - started

        assert elapsed < 0.18
        assert s.mcp.discovery_status == "ready"

    def test_discover_auto_skips_manual_servers(self, monkeypatch):
        raw = {
            "mcp": {
                "automatic": {"url": "http://a/mcp", "auto_connect": True},
                "manual": {"url": "http://b/mcp"},
            }
        }
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        discovered = []

        def fake_discover(config):
            discovered.append(config.name)
            s.mcp.tools[config.name] = []
            s.mcp.resources[config.name] = []

        monkeypatch.setattr(s.mcp, "_discover_one", fake_discover)

        s.mcp.discover_auto()

        assert discovered == ["automatic"]
        assert "manual" not in s.mcp.tools

    def test_tools_are_cached_after_discovery(self, monkeypatch):
        """Listed tools are cached after discovery."""
        raw = mcp_cfg()
        s = Session(cwd="/tmp", config=Config.from_dict(raw))

        class FakeTool:
            name = "echo"
            description = "Echo input"
            inputSchema = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}
            annotations = None

        async def fake_list(url, headers):
            return [FakeTool()]

        monkeypatch.setattr(s.mcp, "_list_tools", fake_list)
        s.mcp.discover_auto()

        assert "test" in s.mcp.tools
        assert len(s.mcp.tools["test"]) == 1
        assert s.mcp.tools["test"][0].name == "echo"

    def test_discover_auto_preserves_manual_and_removes_stale_servers(self, monkeypatch):
        raw = {
            "mcp": {
                "auto_server": {"url": "http://a/mcp", "auto_connect": True},
                "manual_server": {"url": "http://b/mcp", "enabled": False},
            }
        }
        s = Session(cwd="/tmp", config=Config.from_dict(raw))

        async def fake_list(url, headers):
            return []

        monkeypatch.setattr(s.mcp, "_list_tools", fake_list)
        monkeypatch.setattr(s.mcp, "_list_resources", fake_list)

        # Manually add pre-existing stale data
        s.mcp.tools["manual_server"] = [mcp_tool_info("manual_server", "kept")]
        s.mcp.tools["stale_server"] = [mcp_tool_info("stale_server", "old")]

        s.mcp.discover_auto()

        assert "stale_server" not in s.mcp.tools
        assert "manual_server" in s.mcp.tools
        assert "auto_server" in s.mcp.tools


# ---------------------------------------------------------------------------
# MCPManager render_tools_index & _format_tool_line
# ---------------------------------------------------------------------------


class TestToolIndexRendering:
    def test_render_tools_index_empty(self):
        """Empty tools returns empty string."""
        s = session("/tmp")
        assert s.mcp.render_tools_index() == ""

    def test_format_tool_line_with_type(self):
        """_format_tool_line shows name: type."""
        info = mcp_tool_info(
            "test",
            "echo",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        )
        s = session("/tmp")
        line = s.mcp._format_tool_line("test", info)
        assert "text: string" in line

    def test_format_tool_line_requires_args(self):
        """Required args appear before semicolon."""
        info = mcp_tool_info(
            "test",
            "echo",
            input_schema={
                "type": "object",
                "properties": {
                    "a": {"type": "string"},
                    "b": {"type": "integer"},
                },
                "required": ["a"],
            },
        )
        s = session("/tmp")
        line = s.mcp._format_tool_line("test", info)
        assert "a: string" in line
        assert "b: integer" in line
        # a is required, b is optional → semicolon
        assert "; " in line

    def test_format_tool_line_no_args(self):
        """Tools with no input_schema have empty parens."""
        info = mcp_tool_info("test", "ping", input_schema={})
        s = session("/tmp")
        line = s.mcp._format_tool_line("test", info)
        assert "ping()" in line

    def test_format_tool_line_description_truncation(self):
        """Long description is truncated."""
        long_desc = "x " * 50
        info = mcp_tool_info("test", "tool", description=long_desc)
        s = session("/tmp")
        line = s.mcp._format_tool_line("test", info)
        # Description lives on the first line; the schema is appended on a following line.
        summary = line.split("\n")[0]
        assert len(summary.split(" - ")[-1]) <= 83

    def test_index_contains_mcp_tools_header(self, monkeypatch):
        """render_tools_index includes the MCP TOOLS header."""
        raw = mcp_cfg()
        s = Session(cwd="/tmp", config=Config.from_dict(raw))

        class FakeTool:
            name = "echo"
            description = "Echo"
            inputSchema = {"type": "object", "properties": {"t": {"type": "string"}}, "required": ["t"]}
            annotations = None

        async def fake_list(url, headers):
            return [FakeTool()]

        monkeypatch.setattr(s.mcp, "_list_tools", fake_list)
        s.mcp.discover_auto()

        idx = s.mcp.render_tools_index()
        assert "--- MCP TOOLS ---" in idx
        assert "[test]" in idx

    def test_legacy_enabled_does_not_connect_server(self, monkeypatch):
        raw = {"mcp": {"test": {"url": "http://x/mcp", "enabled": False}}}
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        s.mcp.discover_auto()
        idx = s.mcp.render_tools_index()
        assert idx == ""


# ---------------------------------------------------------------------------
# MCPManager render_tools_index budget degradation (regression: a verbose
# server must never hide later servers from the model)
# ---------------------------------------------------------------------------


def _index_session(servers):
    """Build a session with the given {server: [(tool_name, n_schema_fields), ...]}."""
    s = Session(cwd="/tmp", config=Config.from_dict({"mcp": {name: {"url": f"https://{name}/mcp", "auto_connect": True} for name in servers}}))
    for name, tools in servers.items():
        s.mcp.tools[name] = [
            MCPToolInfo(
                server=name,
                name=tool_name,
                description="A tool.",
                input_schema={
                    "type": "object",
                    "properties": {f"p{i}": {"type": "string", "description": "d" * 40} for i in range(nfields)},
                    "required": [f"p{i}" for i in range(min(2, nfields))],
                },
                annotations={},
            )
            for tool_name, nfields in tools
        ]
    s.mcp.discovery_status = "ready"
    return s


class TestToolIndexBudget:
    def test_verbose_server_does_not_hide_later_servers(self):
        """Regression: a first server whose schemas exceed the whole budget must not
        truncate later servers out of the index entirely."""
        s = _index_session(
            {
                "alpha": [(f"q{i}", 30) for i in range(60)],  # huge: full schemas blow the cap
                "beta": [("beta_tool", 2)],
                "gamma": [("gamma_tool", 2)],
            }
        )
        idx = s.mcp.render_tools_index()
        assert len(idx) <= MCPManager.INDEX_TOTAL_LIMIT
        # Every server stays visible...
        for header in ("[alpha]", "[beta]", "[gamma]"):
            assert header in idx
        # ...and the small servers' tools are not lost behind the verbose one.
        assert "beta_tool" in idx
        assert "gamma_tool" in idx

    def test_tier1_inlines_schemas_when_small(self):
        """Small configs keep full per-tool schemas inline (no degradation note)."""
        s = _index_session({"alpha": [("one", 2)], "beta": [("two", 2)]})
        idx = s.mcp.render_tools_index()
        assert "\n  schema: {" in idx
        assert "Schemas omitted to fit" not in idx
        assert "Only tool names shown to fit" not in idx
        assert "one" in idx and "two" in idx

    def test_tier2_drops_schemas_but_keeps_all_tools(self):
        """When full schemas overflow, schemas are dropped but every server and tool name stay."""
        s = _index_session(
            {
                "alpha": [(f"q{i}", 25) for i in range(40)],
                "beta": [("beta_a", 3), ("beta_b", 3)],
                "slack": [("post", 3)],
            }
        )
        idx = s.mcp.render_tools_index()
        assert len(idx) <= MCPManager.INDEX_TOTAL_LIMIT
        assert "Schemas omitted to fit" in idx
        assert "\n  schema: {" not in idx  # no per-tool schema lines
        for header in ("[alpha]", "[beta]", "[slack]"):
            assert header in idx
        for tool in ("q0", "q39", "beta_a", "beta_b", "post"):
            assert tool in idx

    def test_tier3_names_only_lists_every_tool(self):
        """When even arg summaries overflow, fall back to name-only with all tools listed."""
        s = _index_session(
            {
                "alpha": [(f"q{i}", 30) for i in range(120)],
                "github": [(f"gh{i}", 30) for i in range(40)],
                "jira": [(f"j{i}", 30) for i in range(40)],
            }
        )
        idx = s.mcp.render_tools_index()
        assert len(idx) <= MCPManager.INDEX_TOTAL_LIMIT
        assert "Only tool names shown to fit" in idx
        for header in ("[alpha]", "[github]", "[jira]"):
            assert header in idx
        # Spot-check first/last tool of each server are all present.
        for tool in ("q0", "q119", "gh0", "gh39", "j0", "j39"):
            assert tool in idx

    def test_tier4_sets_truncated_flag(self):
        """Tier 4 (even name-only overflows) flags index_truncated so the CLI can warn;
        tiers 1-3 clear it."""
        big = _index_session({x: [(f"{x}_long_tool_name_{i}", 30) for i in range(800)] for x in ("a", "b", "c", "d")})
        big.mcp.render_tools_index()
        assert big.mcp.index_truncated is True

        small = _index_session({"a": [("t", 2)]})
        small.mcp.index_truncated = True  # stale value from a previous render
        small.mcp.render_tools_index()
        assert small.mcp.index_truncated is False

    def test_unconnected_server_stays_out_of_model_index(self):
        s = Session(
            cwd="/tmp",
            config=Config.from_dict(
                {
                    "mcp": {
                        "github": {"url": "https://g/mcp", "auto_connect": True},
                        "metabase": {"url": "https://m/api/mcp", "auth": "oauth", "auto_connect": True},
                    }
                }
            ),
        )
        s.mcp.tools["github"] = [
            MCPToolInfo(
                server="github",
                name="search",
                description="Search.",
                input_schema={"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
                annotations={},
            )
        ]
        s.mcp.server_errors["metabase"] = "authentication required; run /mcp connect metabase"
        s.mcp.discovery_status = "ready"
        idx = s.mcp.render_tools_index()
        assert "[github]" in idx
        assert "metabase" not in idx
        assert "authentication required" not in idx

    def test_mcp_context_and_tool_schema_require_activation(self):
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg()))

        assert s.mcp.render_tools_index() == ""
        assert "MCP" not in {schema["function"]["name"] for schema in Tool.resolved_schemas(s)}

        s.mcp.tools["test"] = [mcp_tool_info("test", "echo")]
        s.mcp.resources["test"] = []

        assert "[test]" in s.mcp.render_tools_index()
        assert "MCP" in {schema["function"]["name"] for schema in Tool.resolved_schemas(s)}

    def test_disconnect_removes_server_from_model_context(self):
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg()))
        s.mcp.tools["test"] = [mcp_tool_info("test", "echo")]
        s.mcp.resources["test"] = []

        result = s.mcp.disconnect_server("test")

        assert result == "MCP server disconnected: test"
        assert s.mcp.render_tools_index() == ""
        assert "MCP" not in {schema["function"]["name"] for schema in Tool.resolved_schemas(s)}


# ---------------------------------------------------------------------------
# MCPManager render_server_status & render_tool_listing
# ---------------------------------------------------------------------------


class TestServerStatusRendering:
    def test_render_server_status_no_servers(self):
        """No servers returns placeholder."""
        s = session("/tmp")
        status = s.mcp.render_server_status()
        assert status == "(no MCP servers configured)"

    def test_render_server_status_connected(self, monkeypatch):
        """Connected server shows mode and tool count."""
        raw = mcp_cfg()
        s = Session(cwd="/tmp", config=Config.from_dict(raw))

        class FakeTool:
            name = "echo"
            description = "Echo"
            inputSchema = {"type": "object", "properties": {}, "required": []}
            annotations = None

        async def fake_list(url, headers):
            return [FakeTool()]

        monkeypatch.setattr(s.mcp, "_list_tools", fake_list)
        s.mcp.discover_auto()

        status = s.mcp.render_server_status()
        assert "test" in status
        assert "● connected" in status
        assert "| `test` | auto | ● connected | 1     |" in status
        assert "`/mcp`" in status
        assert "`@NAME`" in status
        # No secrets leaked
        assert "localhost" not in status

    def test_render_server_status_manual(self):
        """Legacy enabled=false is ignored and the server remains manual."""
        raw = {"mcp": {"test": {"url": "http://x/mcp", "enabled": False}}}
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        status = s.mcp.render_server_status()
        assert "test" in status
        assert "manual" in status
        assert "● disconnected" in status

    def test_render_server_status_aligns_columns(self):
        raw = {
            "mcp": {
                "a": {"url": "https://a.example/mcp"},
                "much-longer": {"url": "https://long.example/mcp", "auto_connect": True},
            }
        }
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        s.mcp.tools["a"] = []
        s.mcp.resources["a"] = []

        lines = [line for line in s.mcp.render_server_status().splitlines() if line.startswith("|")]

        assert len({tuple(index for index, char in enumerate(line) if char == "|") for line in lines}) == 1
        assert "● connected" in lines[2]
        assert "● disconnected" in lines[3]

    def test_render_tool_listing_all(self, monkeypatch):
        """render_tool_listing shows connected servers."""
        raw = mcp_cfg()
        s = Session(cwd="/tmp", config=Config.from_dict(raw))

        class FakeTool:
            name = "echo"
            description = "Echo input back"
            inputSchema = {"type": "object", "properties": {"t": {"type": "string"}}, "required": ["t"]}
            annotations = None

        async def fake_list(url, headers):
            return [FakeTool()]

        monkeypatch.setattr(s.mcp, "_list_tools", fake_list)
        s.mcp.discover_auto()

        listing = s.mcp.render_tool_listing()
        assert "### `test`" in listing
        assert "| tool | args | description |" in listing
        assert "echo" in listing

    def test_render_tool_listing_specific_server(self, monkeypatch):
        """render_tool_listing('test') filters to one server."""
        raw = {"mcp": {"a": {"url": "http://a/mcp", "auto_connect": True}, "b": {"url": "http://b/mcp", "auto_connect": True}}}
        s = Session(cwd="/tmp", config=Config.from_dict(raw))

        async def fake_list(url, headers):
            return []

        monkeypatch.setattr(s.mcp, "_list_tools", fake_list)
        monkeypatch.setattr(s.mcp, "_list_resources", fake_list)
        s.mcp.discover_auto()

        listing = s.mcp.render_tool_listing("a")
        assert "### `a`" in listing
        assert "### `b`" not in listing

    def test_render_tool_listing_no_servers(self):
        """No servers returns placeholder."""
        s = session("/tmp")
        assert s.mcp.render_tool_listing() == "(no MCP servers configured)"

    def test_render_tool_listing_omits_disconnected_servers(self):
        raw = {"mcp": {"connected": {"url": "http://a/mcp"}, "offline": {"url": "http://b/mcp"}}}
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        s.mcp.tools["connected"] = []
        s.mcp.resources["connected"] = []

        listing = s.mcp.render_tool_listing()

        assert "### `connected`" in listing
        assert "offline" not in listing

    def test_render_tool_listing_disconnected_server_has_connect_hint(self):
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg()))

        assert s.mcp.render_tool_listing("test") == "MCP server 'test' is not connected; run /mcp connect test"

    def test_render_tool_listing_includes_resources(self):
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg()))
        s.mcp.tools["test"] = []
        s.mcp.resources["test"] = [MCPResourceInfo("test", "docs://guide", "guide", "Usage guide", "text/plain")]

        listing = s.mcp.render_tool_listing("test")

        assert "no tools discovered" in listing
        assert "docs://guide" in listing
        assert "Usage guide" in listing


# ---------------------------------------------------------------------------
# MCPTool — needs_confirmation
# ---------------------------------------------------------------------------


class TestMCPToolConfirmation:
    def test_describe_does_not_require_confirmation(self):
        """MCP(action='describe') → no confirmation."""
        payload = {"action": "describe", "server": "test", "tool": "echo"}
        tool = MCPTool(None, [payload])
        assert tool.needs_confirmation() is False

    def test_call_requires_confirmation(self, monkeypatch):
        """MCP(action='call') on an undiscovered tool → confirmation needed by default."""
        raw = mcp_cfg()
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        payload = {"action": "call", "server": "test", "tool": "echo", "arguments": {"text": "hi"}}
        tool = MCPTool(s, [payload])
        # No info yet (not discovered) → confirm by default
        assert tool.needs_confirmation() is True

    def test_call_without_annotations_requires_confirmation(self, monkeypatch):
        """A discovered tool with no annotations → confirmation needed by default."""
        raw = mcp_cfg()
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        s.mcp.tools["test"] = [mcp_tool_info("test", "echo", annotations={})]
        payload = {"action": "call", "server": "test", "tool": "echo", "arguments": {"text": "hi"}}
        tool = MCPTool(s, [payload])
        assert tool.needs_confirmation() is True

    def test_call_with_non_destructive_hint_no_confirmation(self, monkeypatch):
        """destructiveHint=false → no confirmation needed."""
        raw = mcp_cfg()
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        s.mcp.tools["test"] = [mcp_tool_info("test", "echo", annotations={"destructiveHint": False})]
        payload = {"action": "call", "server": "test", "tool": "echo", "arguments": {"text": "hi"}}
        tool = MCPTool(s, [payload])
        assert tool.needs_confirmation() is False

    def test_call_with_readonly_hint_no_confirmation(self, monkeypatch):
        """readOnlyHint=true → no confirmation needed."""
        raw = mcp_cfg()
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        # Pre-populate tools with readOnlyHint
        info = mcp_tool_info("test", "echo", annotations={"readOnlyHint": True})
        s.mcp.tools["test"] = [info]

        payload = {"action": "call", "server": "test", "tool": "echo", "arguments": {"text": "hi"}}
        tool = MCPTool(s, [payload])
        assert tool.needs_confirmation() is False

    def test_call_with_destructive_hint_requires_confirmation(self, monkeypatch):
        """destructiveHint=true → confirmation needed."""
        raw = mcp_cfg()
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        info = mcp_tool_info("test", "delete", annotations={"destructiveHint": True})
        s.mcp.tools["test"] = [info]

        payload = {"action": "call", "server": "test", "tool": "delete", "arguments": {"id": "1"}}
        tool = MCPTool(s, [payload])
        assert tool.needs_confirmation() is True

    def test_invalid_payload_raises_tool_error(self):
        """Non-dict payload raises ToolError."""
        tool = MCPTool(None, ["bad"])
        with pytest.raises(ToolError, match="named fields"):
            tool.payload()

    def test_payload_parsing(self):
        """payload() returns the raw dict."""
        payload = {"action": "call", "server": "x", "tool": "y"}
        tool = MCPTool(None, [payload])
        assert tool.payload() == payload


# ---------------------------------------------------------------------------
# MCPTool short_args
# ---------------------------------------------------------------------------


class TestMCPToolShortArgs:
    def test_short_args_call(self):
        """call action shows 'call server.tool'."""
        payload = {"action": "call", "server": "test", "tool": "echo"}
        tool = MCPTool(None, [payload])
        args = tool.short_args()
        assert any("call" in str(a) or "test.echo" in str(a) for a in args)

    def test_short_args_describe(self):
        """describe action shows 'describe server.tool'."""
        payload = {"action": "describe", "server": "test", "tool": "echo"}
        tool = MCPTool(None, [payload])
        args = tool.short_args()
        assert any("describe" in str(a) for a in args)


# ---------------------------------------------------------------------------
# StatusBar mcp_status
# ---------------------------------------------------------------------------


class TestStatusBarMCPStatus:
    def test_stale_shows_nothing(self):
        """Stale status → empty string."""
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg()))
        bar = StatusBar(s)
        assert bar.mcp_status() == ""

    def test_discovering_shows_spinner(self, monkeypatch):
        """Discovering status → loaded/total + spinner."""
        s = Session(cwd="/tmp", config=Config.from_dict({"mcp": {"test": {"url": "http://x/mcp"}}}))
        s.mcp.discovery_status = "discovering"
        bar = StatusBar(s)
        monkeypatch.setattr(time, "monotonic", lambda: 0.0)
        status = bar.mcp_status()
        # First spinner char
        assert status == "mcp 0/0" + bar.INDEX_SPINNER[0]

    def test_discovering_shows_loaded_and_total(self, monkeypatch):
        """Discovering status includes loaded and configured server counts."""
        raw = {
            "mcp": {
                "a": {"url": "http://a/mcp", "auto_connect": True},
                "b": {"url": "http://b/mcp", "auto_connect": True},
            }
        }
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        s.mcp.discovery_status = "discovering"
        s.mcp.tools["a"] = [mcp_tool_info("a", "echo")]
        bar = StatusBar(s)
        monkeypatch.setattr(time, "monotonic", lambda: 0.0)
        assert bar.mcp_status() == "mcp 1/2" + bar.INDEX_SPINNER[0]

    def test_ready_shows_server_count(self, monkeypatch):
        """Ready status → 'MCP N' where N is server count."""
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg()))
        s.mcp.discovery_status = "ready"
        s.mcp.tools["test"] = [mcp_tool_info("test", "echo")]
        bar = StatusBar(s)
        assert bar.mcp_status() == "mcp 1"

    def test_ready_zero_servers(self):
        """Ready with no servers → 'mcp 0'."""
        s = Session(cwd="/tmp", config=Config.from_dict({"mcp": {"test": {"url": "http://x/mcp"}}}))
        s.mcp.discovery_status = "ready"
        bar = StatusBar(s)
        assert bar.mcp_status() == "mcp 0"

    def test_error_shows_error(self):
        """Error status → 'mcp err'."""
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg()))
        s.mcp.discovery_status = "error"
        bar = StatusBar(s)
        assert bar.mcp_status() == "mcp err"

    def test_discovering_statusbar_spinner_animates(self, monkeypatch):
        """Discovering spinner changes over time."""
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg()))
        s.mcp.discovery_status = "discovering"
        bar = StatusBar(s)

        monkeypatch.setattr(time, "monotonic", lambda: 0.0)
        first = bar.mcp_status()

        monkeypatch.setattr(time, "monotonic", lambda: StatusBar.INTERVAL)
        second = bar.mcp_status()

        assert first != second


# ---------------------------------------------------------------------------
# ContextManager — MCP context blocks
# ---------------------------------------------------------------------------


class TestMCPContextBlocks:
    def test_mcp_tools_context_empty(self):
        """No MCP tools → empty string."""
        s = Session(cwd="/tmp")
        ctx = ContextManager(s)
        assert ctx.mcp_tools_context() == ""

    def test_mcp_tools_context_includes_tools(self, monkeypatch):
        """MCP tools present in index."""
        raw = mcp_cfg()
        s = Session(cwd="/tmp", config=Config.from_dict(raw))

        class FakeTool:
            name = "echo"
            description = "Echo"
            inputSchema = {"type": "object", "properties": {"t": {"type": "string"}}, "required": ["t"]}
            annotations = None

        async def fake_list(url, headers):
            return [FakeTool()]

        monkeypatch.setattr(s.mcp, "_list_tools", fake_list)
        s.mcp.discover_auto()

        ctx = ContextManager(s)
        result = ctx.mcp_tools_context()
        assert "--- MCP TOOLS ---" in result
        assert "[test]" in result

    def test_mcp_describe_result_inline_in_history(self):
        """A describe result renders inline like any tool output, not a tail pointer."""
        s = Session(cwd="/tmp")
        runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)
        call = ToolCall("c", "MCP", [{"action": "describe", "server": "test", "tool": "echo"}])
        desc = '<MCPDescribe server="test" tool="echo">\n<description>\nEcho input back.</description>\n</MCPDescribe>'

        msg = runner.tool_message(call, "tr.1", desc)
        assert "-> MCP TOOL DETAILS" not in msg
        assert "<MCPDescribe" in msg
        assert "tr.1" in msg

    def test_mcp_in_context_order(self):
        """MCP TOOLS appears after Environment and before Memory; no separate details block."""
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg()))
        s.mcp.tools["test"] = [mcp_tool_info("test", "echo")]

        ctx = ContextManager(s)
        msgs = ctx.model_messages("sys")
        texts = [m["content"] for m in msgs if m.get("role") == "user"]

        env_idx = next(i for i, t in enumerate(texts) if t.startswith("--- Environment ---"))
        mcp_tools_idx = next(i for i, t in enumerate(texts) if t.startswith("--- MCP TOOLS ---"))
        memory_idx = next(i for i, t in enumerate(texts) if t.startswith("--- Memory ---"))

        assert env_idx < mcp_tools_idx < memory_idx
        assert not any(t.startswith("--- MCP TOOL DETAILS ---") for t in texts)
        assert not any(t.startswith("--- FILE STATE ---") for t in texts)

    @staticmethod
    def _describe_msg(call_id: str, key: str, tool: str, body: str) -> dict:
        desc = f'<MCPDescribe server="test" tool="{tool}">\n{body}\n</MCPDescribe>'
        return {"role": "tool", "tool_call_id": call_id, "content": f"tool {key} MCP(describe, test, {tool})\noutput:\n{desc}"}

    def test_dedup_collapses_repeated_describe(self):
        """A second describe of the same tool collapses to a pointer at the first; the first stays full."""
        s = Session(cwd="/tmp")
        ctx = ContextManager(s)
        m1 = self._describe_msg("a", "tr.1", "echo", "schema")
        m2 = self._describe_msg("b", "tr.2", "echo", "schema")

        out = ctx.dedup_mcp_describes([m1, m2])

        assert "<MCPDescribe" in out[0]["content"]  # first kept full
        assert "<MCPDescribe" not in out[1]["content"]  # second collapsed
        assert "repeat describe of test.echo" in out[1]["content"]
        assert "tr.1" in out[1]["content"]  # points back to the first
        assert "tr.2" in out[1]["content"]  # head/recall key preserved
        assert m2["content"].count("<MCPDescribe") == 1  # input not mutated (pure transform)

    def test_dedup_keeps_distinct_tools(self):
        """Different tools each keep their full schema."""
        s = Session(cwd="/tmp")
        ctx = ContextManager(s)
        out = ctx.dedup_mcp_describes([self._describe_msg("a", "tr.1", "echo", "s1"), self._describe_msg("b", "tr.2", "ping", "s2")])

        assert all("<MCPDescribe" in m["content"] for m in out)

    def test_model_messages_dedups_describe(self):
        """model_messages applies the dedup to sent context without touching stored history."""
        s = Session(cwd="/tmp")
        s.messages = [self._describe_msg("a", "tr.1", "echo", "schema"), self._describe_msg("b", "tr.2", "echo", "schema")]
        ctx = ContextManager(s)

        tool_texts = [m["content"] for m in ctx.model_messages("sys") if m.get("role") == "tool"]

        assert sum("<MCPDescribe" in t for t in tool_texts) == 1
        assert sum("<MCPDescribe" in m["content"] for m in s.messages) == 2  # history untouched


# ---------------------------------------------------------------------------
# MCPManager — describe_tool
# ---------------------------------------------------------------------------


class TestDescribeTool:
    def test_describe_uses_cached_metadata(self, monkeypatch):
        """describe returns rendered metadata from cache."""
        raw = mcp_cfg()
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        info = mcp_tool_info("test", "echo")
        s.mcp.tools["test"] = [info]

        result = s.mcp.describe_tool("test", "echo")
        assert "<MCPDescribe server=" in result
        assert "echo" in result

    def test_describe_unknown_tool_raises_error(self):
        """Unknown tool raises ToolError."""
        s = Session(cwd="/tmp")
        s.mcp.tools["test"] = [mcp_tool_info("test", "echo")]
        with pytest.raises(ToolError, match="not found"):
            s.mcp.describe_tool("test", "missing_tool")

    def test_describe_unknown_server_raises_error(self):
        """Unknown server raises ToolError."""
        s = Session(cwd="/tmp")
        with pytest.raises(ToolError, match="not found"):
            s.mcp.describe_tool("unknown", "echo")

    def test_describe_requires_connected_server(self, monkeypatch):
        raw = mcp_cfg()
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        calls = []
        monkeypatch.setattr(s.mcp, "discover_server", lambda name: calls.append(name))

        with pytest.raises(ToolError, match="not connected"):
            s.mcp.describe_tool("test", "echo")
        assert calls == []


# ---------------------------------------------------------------------------
# MCPManager — call_tool
# ---------------------------------------------------------------------------


class TestCallTool:
    def test_call_unknown_server_raises_error(self):
        """Unknown server raises ToolError."""
        s = Session(cwd="/tmp")
        with pytest.raises(ToolError, match="not found"):
            s.mcp.call_tool("unknown", "echo", {})

    def test_call_disconnected_server_does_not_rediscover(self, monkeypatch):
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg()))
        calls = []
        monkeypatch.setattr(s.mcp, "discover_server", lambda name: calls.append(name))

        with pytest.raises(ToolError, match="not connected"):
            s.mcp.call_tool("test", "echo", {})

        assert calls == []

    def test_call_server_with_error_raises(self, monkeypatch):
        """Server with prior error raises ToolError."""
        raw = mcp_cfg()
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        s.mcp.server_errors["test"] = "connection failed"

        with pytest.raises(ToolError, match="error"):
            s.mcp.call_tool("test", "echo", {})

    def test_call_without_url(self):
        """Server without URL raises ToolError."""
        raw = {"mcp": {"test": {"url": "", "auto_connect": True}}}
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        with pytest.raises(ToolError, match="url"):
            s.mcp.call_tool("test", "echo", {})

    def test_call_and_resource_paths_share_oauth_gate(self):
        """call_tool and the resource path both reject an OAuth server with no stored authentication
        (both route through the shared _resolve_server)."""
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg(auth="oauth")))
        with pytest.raises(ToolError, match="requires authentication"):
            s.mcp.call_tool("test", "echo", {})
        with pytest.raises(ToolError, match="requires authentication"):
            s.mcp.list_resources("test")


# ---------------------------------------------------------------------------
# CommandLoop — /mcp commands
# ---------------------------------------------------------------------------


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
            inputSchema = {"type": "object", "properties": {}, "required": []}
            annotations = None

        async def fake_list(url, headers):
            return [FakeTool()]

        monkeypatch.setattr(s.mcp, "_list_tools", fake_list)
        s.mcp.discover_auto()

        loop = CommandLoop(Agent(s), input_fn=lambda _: "", output_fn=lambda _: None)
        result = loop.mcp_command("")
        assert "test" in result
        assert "| `test` | auto | ● connected | 1     |" in result

    def test_mcp_tools_shows_listing(self, monkeypatch):
        """/mcp tools returns tool listing."""
        raw = mcp_cfg()
        s = Session(cwd="/tmp", config=Config.from_dict(raw))

        class FakeTool:
            name = "echo"
            description = "Echo"
            inputSchema = {"type": "object", "properties": {}, "required": []}
            annotations = None

        async def fake_list(url, headers):
            return [FakeTool()]

        monkeypatch.setattr(s.mcp, "_list_tools", fake_list)
        s.mcp.discover_auto()

        loop = CommandLoop(Agent(s), input_fn=lambda _: "", output_fn=lambda _: None)
        result = loop.mcp_command("tools")
        assert "### `test`" in result
        assert "echo" in result

    def test_mcp_tools_without_name_does_not_discover(self, monkeypatch):
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg()))
        calls = []
        monkeypatch.setattr(s.mcp, "discover_server", lambda name: calls.append(name))

        loop = CommandLoop(Agent(s), input_fn=lambda _: "", output_fn=lambda _: None)
        result = loop.mcp_command("tools")

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

        result = loop.mcp_command("connect test")

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
        loop.mcp_command("connect test")
        loop.mcp_command("connect test")

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

        result = loop.mcp_command("connect alpha beta")

        assert result == "connected batch"
        assert calls == [(["alpha", "beta"], {"interactive": False, "notify": loop.emit})]

    def test_mcp_connect_rejects_missing_server(self):
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg()))
        loop = CommandLoop(Agent(s), input_fn=lambda _: "", output_fn=lambda _: None)

        assert loop.mcp_command("connect missing") == "MCP server not found: missing"

    def test_mcp_disconnect_removes_connected_server(self):
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg()))
        s.mcp.tools["test"] = [mcp_tool_info("test", "echo")]
        s.mcp.resources["test"] = []
        loop = CommandLoop(Agent(s), input_fn=lambda _: "", output_fn=lambda _: None)

        assert loop.mcp_command("disconnect test") == "MCP server disconnected: test"
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
        monkeypatch.setattr(loop, "mcp_manager", lambda: calls.append("manager"))

        assert loop.mcp_command("") is None
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

        loop.mcp_manager()

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

        loop.mcp_manager()

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

        loop.mcp_manager()

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

        loop.mcp_manager()
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

        loop.mcp_manager()

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
        result = loop.mcp_command("bad_subcommand")
        assert "Unknown" in result

    def test_mcp_subcommands_reject_extra_args(self):
        """MCP subcommands do not silently ignore extra args."""
        s = Session(cwd="/tmp")
        loop = CommandLoop(Agent(s), input_fn=lambda _: "", output_fn=lambda _: None)

        assert loop.mcp_command("tools a b") == "Usage: /mcp tools [server]"
        assert loop.mcp_command("connect") == "Usage: /mcp connect <server> [server ...]"
        assert loop.mcp_command("disconnect") == "Usage: /mcp disconnect <server>"
        assert loop.mcp_command("disconnect a b") == "Usage: /mcp disconnect <server>"
        assert "Unknown" in loop.mcp_command("login test")
        assert "Unknown" in loop.mcp_command("logout test")

    def test_no_mcp_config(self):
        """No MCP config returns message."""
        s = Session(cwd="/tmp")
        s.mcp = None
        loop = CommandLoop(Agent(s), input_fn=lambda _: "", output_fn=lambda _: None)
        result = loop.mcp_command("")
        assert "not configured" in result


# ---------------------------------------------------------------------------
# Tab completion
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


class TestToolIndexTruncation:
    def test_index_truncation_long_block(self, monkeypatch):
        """Long index block is bounded by INDEX_TOTAL_LIMIT."""
        raw = mcp_cfg()
        s = Session(cwd="/tmp", config=Config.from_dict(raw))

        # Create many tools to exceed budget
        class FakeTool:
            name = "tool"
            description = "Desc"
            inputSchema = {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}
            annotations = None

        many_tools = []
        for i in range(200):
            t = type(
                "FakeTool",
                (),
                {
                    "name": f"tool{i}",
                    "description": "x" * 80,
                    "inputSchema": {"type": "object", "properties": {"p": {"type": "string", "description": "x" * 100}}, "required": ["p"]},
                    "annotations": None,
                },
            )()
            many_tools.append(t)

        s.mcp.tools["test"] = []

        async def fake_list(url, headers):
            return many_tools

        monkeypatch.setattr(s.mcp, "_list_tools", fake_list)
        s.mcp.discover_auto()

        idx = s.mcp.render_tools_index()
        assert len(idx) <= MCPManager.INDEX_TOTAL_LIMIT + 100
        assert "truncated" in idx  # 200 tools with schemas exceed the budget

    def test_format_tool_line_long_args(self):
        """Long args list is truncated."""
        props = {f"p{i}": {"type": "string"} for i in range(20)}
        required = [f"p{i}" for i in range(20)]
        info = mcp_tool_info(
            "test",
            "big",
            input_schema={
                "type": "object",
                "properties": props,
                "required": required,
            },
        )
        s = session("/tmp")
        line = s.mcp._format_tool_line("test", info)
        assert "..." in line


# ---------------------------------------------------------------------------
# MCPManager — normalize_result
# ---------------------------------------------------------------------------


class TestNormalizeResult:
    def test_string_content(self):
        """String content is passed through."""
        s = session("/tmp")
        result = s.mcp.normalize_result("hello world")
        assert result == "hello world"

    def test_dict_text_type(self):
        """Dict with type='text' extracts text field."""
        s = session("/tmp")
        result = s.mcp.normalize_result({"type": "text", "text": "hello from mcp"})
        assert result == "hello from mcp"

    def test_dict_resource_type(self):
        """Dict with type='resource' dumps resource field."""
        s = session("/tmp")
        resource = {"uri": "file:///tmp/test.txt", "text": "contents"}
        result = s.mcp.normalize_result({"type": "resource", "resource": resource})
        assert "file:///tmp/test.txt" in result
        assert "contents" in result

    def test_dict_other_type(self):
        """Dict with unknown type is dumped as JSON."""
        s = session("/tmp")
        result = s.mcp.normalize_result({"type": "image", "data": "...", "mimeType": "image/png"})
        assert "image/png" in result

    def test_object_text_type(self):
        """Object with type='text' extracts text attribute."""
        s = session("/tmp")
        item = SimpleNamespace(type="text", text="object text")
        result = s.mcp.normalize_result(item)
        assert result == "object text"

    def test_object_resource_type(self):
        """Object with type='resource' converts resource to string."""
        s = session("/tmp")
        item = SimpleNamespace(type="resource", resource={"uri": "test://uri"})
        result = s.mcp.normalize_result(item)
        assert "test://uri" in result

    def test_list_of_items(self):
        """List of content items is joined."""
        s = session("/tmp")
        result = s.mcp.normalize_result(
            [
                {"type": "text", "text": "first"},
                {"type": "text", "text": "second"},
            ]
        )
        assert "first" in result
        assert "second" in result

    def test_object_model_dump(self):
        """Object with model_dump is serialized."""
        s = session("/tmp")
        obj = SimpleNamespace(model_dump=lambda mode="json": {"result": "ok", "value": 42})
        result = s.mcp.normalize_result(obj)
        assert "ok" in result
        assert "42" in result

    def test_long_output_truncation(self, monkeypatch):
        """Output exceeding RAW_OUTPUT_LIMIT is truncated."""
        s = session("/tmp")
        monkeypatch.setattr(s.mcp, "RAW_OUTPUT_LIMIT", 100)
        long_result = "x" * 200
        text = s.mcp.normalize_result(long_result)
        assert len(text) <= 150  # 100 + truncated marker
        assert "<MCPOutputTruncated" in text
        assert "200" in text


# ---------------------------------------------------------------------------
# MCPManager — call_tool success path
# ---------------------------------------------------------------------------


class TestCallToolSuccess:
    def test_call_success_mocked(self, monkeypatch):
        """call_tool returns wrapped output on success."""
        raw = mcp_cfg()
        s = Session(cwd="/tmp", config=Config.from_dict(raw))

        async def fake_call(url, headers, name, arguments):
            return {"type": "text", "text": f"called {name} with {arguments}"}

        monkeypatch.setattr(s.mcp, "_call_tool", fake_call)
        s.mcp.tools["test"] = [mcp_tool_info("test", "echo")]

        result = s.mcp.call_tool("test", "echo", {"text": "hi"})
        assert "<MCPCall server=" in result
        assert 'tool="echo"' in result
        assert "called echo" in result
        assert "</MCPCall>" in result

    def test_call_content_list(self, monkeypatch):
        """call_tool with multi-item content list."""
        raw = mcp_cfg()
        s = Session(cwd="/tmp", config=Config.from_dict(raw))

        async def fake_call(url, headers, name, arguments):
            return {
                "content": [
                    {"type": "text", "text": "part one"},
                    {"type": "text", "text": "part two"},
                ]
            }

        monkeypatch.setattr(s.mcp, "_call_tool", fake_call)
        s.mcp.tools["test"] = [mcp_tool_info("test", "multi")]

        result = s.mcp.call_tool("test", "multi", {})
        assert "part one" in result
        assert "part two" in result

    def test_call_from_running_event_loop(self, monkeypatch):
        """Synchronous call_tool still works when the caller already has an event loop."""
        raw = mcp_cfg()
        s = Session(cwd="/tmp", config=Config.from_dict(raw))

        async def fake_call(config, headers, name, arguments):
            return {"type": "text", "text": "ok"}

        async def run_call():
            return s.mcp.call_tool("test", "echo", {})

        monkeypatch.setattr(s.mcp, "_call_tool", fake_call)
        s.mcp.tools["test"] = [mcp_tool_info("test", "echo")]

        assert "ok" in asyncio.run(run_call())


# ---------------------------------------------------------------------------
# ContextManager — prune_tool_records preserves MCP describe records
# ---------------------------------------------------------------------------


class TestMCPPruning:
    def test_prune_keeps_describe_record_referenced_in_messages(self):
        """A describe record is retained when its tr.N is referenced in history (normal path)."""
        s = Session(cwd="/tmp")
        s.mcp.tools["test"] = [mcp_tool_info("test", "echo")]
        desc = '<MCPDescribe server="test" tool="echo">\n<description>\nEcho back.</description>\n</MCPDescribe>'
        s.store_tool_result("MCP", [{"action": "describe", "server": "test", "tool": "echo"}], desc)
        ctx = ContextManager(s)

        key = s.tool_records[0].key
        keep_messages = [{"role": "tool", "content": f"tool {key} MCP(describe)\noutput:\n{desc}"}]
        ctx.prune_tool_records(keep_messages)

        assert [r.key for r in s.tool_records] == [key]

    def test_prune_drops_unreferenced_describe_record(self):
        """With the tail digest gone, an unreferenced describe record prunes like any other."""
        s = Session(cwd="/tmp")
        s.mcp.tools["test"] = [mcp_tool_info("test", "echo")]
        desc = '<MCPDescribe server="test" tool="echo">\n<description>\nEcho.</description>\n</MCPDescribe>'
        s.store_tool_result("MCP", [{"action": "describe", "server": "test", "tool": "echo"}], desc)
        ctx = ContextManager(s)

        ctx.prune_tool_records([])

        assert s.tool_records == []

    def test_prune_drops_non_mcp_records(self):
        """prune_tool_records drops records not referenced in messages."""
        s = Session(cwd="/tmp")
        s.store_tool_result("Find", [], "results from find")
        s.store_tool_result("Read", [], "read output")
        ctx = ContextManager(s)

        ctx.prune_tool_records([])

        assert len(s.tool_records) == 0


# ---------------------------------------------------------------------------
# CommandLoop — /mcp tools NAME
# ---------------------------------------------------------------------------


class TestMCPCommandsByName:
    def test_mcp_tools_specific_server(self, monkeypatch):
        """/mcp tools NAME points disconnected servers to connect."""
        raw = {"mcp": {"a": {"url": "http://a/mcp"}, "b": {"url": "http://b/mcp"}}}
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        discovered = []

        monkeypatch.setattr(s.mcp, "discover_server", lambda name: discovered.append(name))

        loop = CommandLoop(Agent(s), input_fn=lambda _: "", output_fn=lambda _: None)
        result = loop.mcp_command("tools a")

        assert discovered == []
        assert result == "MCP server 'a' is not connected; run /mcp connect a"


# ---------------------------------------------------------------------------
# MCPManager — discover_server with nonexistent server
# ---------------------------------------------------------------------------


class TestMCPDiscoverServer:
    def test_discover_nonexistent_server_sets_error(self):
        """discover_server for a server not in config sets server_errors."""
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg()))
        s.mcp.discover_server("nonexistent")
        assert "nonexistent" in s.mcp.server_errors
        assert "not found" in s.mcp.server_errors["nonexistent"]

    def test_discover_nonexistent_server_removes_tools(self):
        """discover_server for a server not in config clears its stale tools."""
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg()))
        s.mcp.tools["gone"] = [mcp_tool_info("gone", "old_tool")]
        s.mcp.discover_server("gone")
        assert "gone" not in s.mcp.tools
        assert "gone" in s.mcp.server_errors


# ---------------------------------------------------------------------------
# MCP resources (list / read)
# ---------------------------------------------------------------------------


def _fake_resource(uri="docs://x.md", name="x", description="A doc", mime="text/markdown"):
    return SimpleNamespace(uri=uri, name=name, description=description, mimeType=mime)


class TestMCPResources:
    def _server_with_resources(self, monkeypatch, resources):
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg()))

        class FakeTool:
            name = "query"
            description = "Run a program."
            inputSchema = {"type": "object", "properties": {"operations": {"type": "array"}}, "required": ["operations"]}
            annotations = None

        async def fake_tools(url, headers):
            return [FakeTool()]

        async def fake_resources(url, headers):
            return resources

        monkeypatch.setattr(s.mcp, "_list_tools", fake_tools)
        monkeypatch.setattr(s.mcp, "_list_resources", fake_resources)
        s.mcp.discover_auto()
        return s

    def test_action_schema_includes_resource_actions(self):
        schema = MCPTool.params_schema()
        assert {"call", "describe", "list_resources", "read_resource"} <= set(schema["properties"]["action"]["enum"])
        assert "uri" in schema["properties"]
        assert schema["required"] == ["action", "server"]

    def test_discovery_populates_resources(self, monkeypatch):
        s = self._server_with_resources(monkeypatch, [_fake_resource(uri="metabase://docs/construct-query.md")])
        assert [r.uri for r in s.mcp.resources["test"]] == ["metabase://docs/construct-query.md"]

    def test_index_lists_resources(self, monkeypatch):
        s = self._server_with_resources(monkeypatch, [_fake_resource(uri="metabase://docs/construct-query.md")])
        idx = s.mcp.render_tools_index()
        assert "metabase://docs/construct-query.md" in idx
        assert "read_resource" in idx

    def test_resources_best_effort_on_failure(self, monkeypatch):
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg()))

        class FakeTool:
            name = "t"
            description = "d"
            inputSchema = {"type": "object", "properties": {}}
            annotations = None

        async def fake_tools(url, headers):
            return [FakeTool()]

        async def boom(url, headers):
            raise RuntimeError("resources not supported")

        monkeypatch.setattr(s.mcp, "_list_tools", fake_tools)
        monkeypatch.setattr(s.mcp, "_list_resources", boom)
        s.mcp.discover_auto()
        assert s.mcp.tools["test"]  # tool discovery still succeeded
        assert s.mcp.resources["test"] == []
        assert "test" not in s.mcp.server_errors

    def test_read_resource_dispatch(self, monkeypatch):
        s = self._server_with_resources(monkeypatch, [_fake_resource(uri="docs://a.md")])

        async def fake_read(config, headers, uri):
            return [SimpleNamespace(text="hello " + uri, blob=None)]

        monkeypatch.setattr(s.mcp, "_read_resource", fake_read)
        out = MCPTool(s, [{"action": "read_resource", "server": "test", "uri": "docs://a.md"}]).call()
        assert '<MCPResource server="test" uri="docs://a.md">' in out
        assert "hello docs://a.md" in out

    def test_read_resource_requires_uri(self, monkeypatch):
        s = self._server_with_resources(monkeypatch, [])
        with pytest.raises(ToolError, match="requires a uri"):
            MCPTool(s, [{"action": "read_resource", "server": "test"}]).call()

    def test_read_resource_is_read_only(self, monkeypatch):
        s = self._server_with_resources(monkeypatch, [])
        tool = MCPTool(s, [{"action": "read_resource", "server": "test", "uri": "docs://a.md"}])
        assert tool.needs_confirmation() is False

    def test_list_resources_dispatch(self, monkeypatch):
        s = self._server_with_resources(monkeypatch, [_fake_resource(uri="docs://a.md", description="Doc A")])
        out = MCPTool(s, [{"action": "list_resources", "server": "test"}]).call()
        assert "docs://a.md" in out and "Doc A" in out

    def test_normalize_resource_blob(self):
        mgr = MCPManager.__new__(MCPManager)
        out = mgr.normalize_resource([SimpleNamespace(text=None, blob=b"\x00\x01", mimeType="application/pdf")])
        assert "binary" in out and "application/pdf" in out

    def test_action_defaults_to_call_when_omitted(self):
        assert MCPTool.resolved_action({"tool": "x", "server": "s"}) == "call"
        assert MCPTool.resolved_action({"arguments": {}, "server": "s"}) == "call"
        assert MCPTool.resolved_action({"server": "s"}) == ""
        assert MCPTool.resolved_action({"action": "describe", "server": "s"}) == "describe"

    def test_omitted_action_invokes_tool(self, monkeypatch):
        s = self._server_with_resources(monkeypatch, [])

        async def fake_call(config, headers, name, arguments):
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="ok " + name)])

        monkeypatch.setattr(s.mcp, "_call_tool", fake_call)
        out = MCPTool(s, [{"server": "test", "tool": "query", "arguments": {"q": 1}}]).call()
        assert "ok query" in out

    def test_unknown_action_error_is_actionable(self, monkeypatch):
        s = self._server_with_resources(monkeypatch, [])
        with pytest.raises(ToolError, match=r"tool=.search"):
            MCPTool(s, [{"action": "search", "server": "test", "arguments": {}}]).call()

    def test_extract_uris_from_description(self):
        text = "See metabase://docs/cq.md for syntax. Also https://x.io/a, and (file://y.txt)."
        assert MCPManager._extract_uris(text) == ["metabase://docs/cq.md", "https://x.io/a", "file://y.txt"]

    def test_index_surfaces_description_uris(self, monkeypatch):
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg()))

        class FakeTool:
            name = "query"
            description = "Run a program. " + "x" * 200 + " See metabase://docs/construct-query.md for syntax."
            inputSchema = {"type": "object", "properties": {}}
            annotations = None

        async def fake_tools(url, headers):
            return [FakeTool()]

        async def empty(url, headers):
            return []

        monkeypatch.setattr(s.mcp, "_list_tools", fake_tools)
        monkeypatch.setattr(s.mcp, "_list_resources", empty)
        s.mcp.discover_auto()
        idx = s.mcp.render_tools_index()
        # URI survives even though the description is truncated to 80 chars on the main line.
        assert "metabase://docs/construct-query.md" in idx
        assert "refs" in idx

    def test_mention_block_lists_resources(self, monkeypatch):
        s = self._server_with_resources(monkeypatch, [_fake_resource(uri="docs://a.md", description="Doc A")])
        block = s.mcp._mention_block("test", "")
        assert "docs://a.md" in block and "read_resource" in block

    def test_mention_block_lists_resources_without_tools(self):
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg()))
        s.mcp.tools["test"] = []
        s.mcp.resources["test"] = [MCPResourceInfo("test", "docs://guide.md", "guide", "Usage guide", "text/markdown")]

        block = s.mcp._mention_block("test", "")

        assert "docs://guide.md" in block
        assert "no tools or resources" not in block

    def test_resource_only_server_renders_in_index(self):
        """A connected server with resources but zero tools is listed (not dumped into pending)."""
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg()))
        s.mcp.tools["test"] = []
        s.mcp.resources["test"] = [MCPResourceInfo("test", "docs://guide.md", "guide", "Usage guide", "text/markdown")]
        s.mcp.discovery_status = "ready"
        idx = s.mcp.render_tools_index()
        assert "[test]" in idx
        assert "docs://guide.md" in idx
        assert "not connected" not in idx

    def test_pending_status_connected_but_empty(self):
        """A ready server with neither tools nor resources is reported as connected, not 'not connected'."""
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg()))
        s.mcp.tools["test"] = []
        s.mcp.discovery_status = "ready"
        assert s.mcp._pending_status("test") == "connected; no tools or resources advertised"

    def _server_with_doc_tool(self, monkeypatch, description, read_calls):
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg()))

        class FakeTool:
            name = "query"
            inputSchema = {"type": "object", "properties": {}}
            annotations = None

        FakeTool.description = description

        async def fake_tools(url, headers):
            return [FakeTool()]

        async def fake_resources(url, headers):
            return [_fake_resource(uri="metabase://docs/cq.md")]

        async def fake_read(config, headers, uri):
            read_calls.append(uri)
            return [SimpleNamespace(text="GRAMMAR DOC", blob=None)]

        monkeypatch.setattr(s.mcp, "_list_tools", fake_tools)
        monkeypatch.setattr(s.mcp, "_list_resources", fake_resources)
        monkeypatch.setattr(s.mcp, "_read_resource", fake_read)
        s.mcp.discover_auto()
        return s

    def test_auto_read_injects_doc_on_first_call(self, monkeypatch):
        reads = []
        s = self._server_with_doc_tool(monkeypatch, "Run. See metabase://docs/cq.md for syntax.", reads)

        async def ok(config, headers, name, arguments):
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="ROWS")])

        monkeypatch.setattr(s.mcp, "_call_tool", ok)
        out1 = MCPTool(s, [{"action": "call", "server": "test", "tool": "query", "arguments": {}}]).call()
        assert "MCPAutoResources" in out1 and "GRAMMAR DOC" in out1 and "ROWS" in out1
        # injected once: a second call neither re-reads nor re-injects
        out2 = MCPTool(s, [{"action": "call", "server": "test", "tool": "query", "arguments": {}}]).call()
        assert "MCPAutoResources" not in out2
        assert reads == ["metabase://docs/cq.md"]

    def test_auto_read_attaches_doc_to_failed_call(self, monkeypatch):
        reads = []
        s = self._server_with_doc_tool(monkeypatch, "Run. See metabase://docs/cq.md for syntax.", reads)

        async def boom(config, headers, name, arguments):
            raise RuntimeError("Invalid body")

        monkeypatch.setattr(s.mcp, "_call_tool", boom)
        with pytest.raises(ToolError) as exc:
            MCPTool(s, [{"action": "call", "server": "test", "tool": "query", "arguments": {}}]).call()
        assert "Invalid body" in str(exc.value) and "GRAMMAR DOC" in str(exc.value)

    def test_auto_read_skips_web_links(self, monkeypatch):
        reads = []
        s = self._server_with_doc_tool(monkeypatch, "Run. Docs at https://web.example/guide.", reads)

        async def ok(config, headers, name, arguments):
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="ROWS")])

        monkeypatch.setattr(s.mcp, "_call_tool", ok)
        out = MCPTool(s, [{"action": "call", "server": "test", "tool": "query", "arguments": {}}]).call()
        assert "MCPAutoResources" not in out and reads == []


# ---------------------------------------------------------------------------
# User scenarios — public commands through model-visible context
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

        assert loop.mcp_command("connect docs") == "MCP server connected: docs; tools=1; resources=1"
        assert agent.run("Read the project guide") == "done"
        context = self.mcp_context(agent.model)
        assert "[search]" in context
        assert "[docs]" in context
        assert "docs://guide.md" in context

        assert loop.mcp_command("disconnect search") == "MCP server disconnected: search"
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

        result = loop.mcp_command("connect metabase")

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

        result = loop.mcp_command("connect metabase")

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

        result = loop.mcp_command("connect valid stale plain")

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

        result = loop.mcp_command("connect catalog offline")
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
        worker = threading.Thread(target=lambda: result.append(loop.mcp_command("connect alpha beta")))
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


# ---------------------------------------------------------------------------
# Smoke: py_compile
# ---------------------------------------------------------------------------


def test_py_compile():
    """Package compiles without errors."""
    import py_compile
    from pathlib import Path

    for source in sorted(Path("minacode").glob("*.py")):
        py_compile.compile(str(source), doraise=True)
