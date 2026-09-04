"""MCP server configuration, connection management, discovery, and status reporting."""

import asyncio
import os
import time
from types import SimpleNamespace
from typing import ClassVar

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
from mcp_harness import mcp_cfg, mcp_tool_info, session

from wizolt.config import (
    Config,
)
from wizolt.context import ContextManager
from wizolt.mcp import MCPFileTokenStore, MCPManager, MCPResourceInfo, MCPServerConfig
from wizolt.render import StatusBar
from wizolt.session import Session, bootstrap_features


def parse_one(raw: dict) -> MCPServerConfig | None:
    """Parse a config dict and return the first server config."""
    config = Config.from_dict(raw)
    s = Session(cwd="/tmp", config=config)
    bootstrap_features(s)
    configs = s.mcp.parse_configs()
    return configs[0] if configs else None


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
        bootstrap_features(s)
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
        bootstrap_features(s)
        assert isinstance(s.mcp._transport(s.mcp.parse_configs()[0], {}), StdioTransport)
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg()))
        bootstrap_features(s)
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
        bootstrap_features(s)
        config = s.mcp.parse_configs()[0]
        headers = s.mcp._build_mcp_headers(config)
        assert headers == {"Authorization": "Bearer secret123"}

    def test_bearer_token_missing_var(self, monkeypatch):
        """Missing bearer env var returns error string."""
        monkeypatch.delenv("MISSING_TOKEN", raising=False)
        raw = mcp_cfg(bearer_token_env_var="MISSING_TOKEN")
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        bootstrap_features(s)
        config = s.mcp.parse_configs()[0]
        result = s.mcp._build_mcp_headers(config)
        assert isinstance(result, str)
        assert "missing" in result.lower()

    def test_env_http_headers_success(self, monkeypatch):
        """env_http_headers reads header values from environment."""
        monkeypatch.setenv("MY_HEADER_VAL", "xyz")
        raw = mcp_cfg(env_http_headers={"X-Custom": "MY_HEADER_VAL"})
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        bootstrap_features(s)
        config = s.mcp.parse_configs()[0]
        headers = s.mcp._build_mcp_headers(config)
        assert headers == {"X-Custom": "xyz"}

    def test_authorization_env_http_header_success(self, monkeypatch):
        """Authorization is allowed via env_http_headers when it is the only auth source."""
        monkeypatch.setenv("AUTH_VAL", "Bearer custom")
        raw = mcp_cfg(env_http_headers={"Authorization": "AUTH_VAL"})
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        bootstrap_features(s)
        config = s.mcp.parse_configs()[0]
        headers = s.mcp._build_mcp_headers(config)
        assert headers == {"Authorization": "Bearer custom"}

    def test_env_http_headers_missing_var(self, monkeypatch):
        """Missing env_http_headers env var returns error string."""
        monkeypatch.delenv("MISSING_HEADER_VAL", raising=False)
        raw = mcp_cfg(env_http_headers={"X-Custom": "MISSING_HEADER_VAL"})
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        bootstrap_features(s)
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
        bootstrap_features(s)
        config = s.mcp.parse_configs()[0]
        result = s.mcp._build_mcp_headers(config)
        assert isinstance(result, str)
        assert "conflicting" in result.lower()

    def test_no_auth_config(self):
        """No auth config produces empty headers dict."""
        raw = mcp_cfg()
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        bootstrap_features(s)
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

    async def test_operations_run_on_the_callers_loop(self):
        """Every MCP operation is a coroutine on whichever loop awaited it.

        There is no manager-owned loop left to schedule onto: that is what let a client be entered
        on one loop and torn down on another, and what kept a cancelled turn from reaching it."""
        s = session("/tmp")

        async def loop_id():
            return id(asyncio.get_running_loop())

        assert await s.mcp._bounded(loop_id()) == id(asyncio.get_running_loop())

    async def test_timed_out_call_swallows_its_late_failure(self):
        """A call abandoned at the timeout stays quiet when its teardown then fails — the caller
        already has its timeout, and asyncio would otherwise print that error at collection."""
        import gc

        from wizolt.base import ToolError

        s = session("/tmp")
        unretrieved: list[dict] = []
        asyncio.get_running_loop().set_exception_handler(lambda _loop, context: unretrieved.append(context))

        async def fails_while_cancelled():
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                raise RuntimeError("client teardown timed out") from None

        with pytest.raises(ToolError, match="timed out after 1s"):
            await s.mcp._bounded(fails_while_cancelled(), timeout=1)

        gc.collect()
        assert unretrieved == []

    def test_oauth_token_store_lock_is_shared_by_path(self, tmp_path):
        """Token storage keeps one store and one lock per token file path."""
        s = session(tmp_path)
        store = s.mcp._oauth_token_store
        same_path_store = MCPFileTokenStore(store.path)

        assert same_path_store.lock is store.lock

    async def test_token_store_put_get_roundtrip(self, tmp_path):
        """put persists a value that get returns — put is the OAuth storage protocol's writer."""
        store = MCPFileTokenStore(str(tmp_path / "tokens.json"))

        async def roundtrip():
            await store.put("k", {"v": 1}, collection="mcp-oauth-token")
            return await store.get("k", collection="mcp-oauth-token")

        assert await roundtrip() == {"v": 1}

    async def test_clear_server_matches_fastmcp_oauth_storage_contract(self, tmp_path):
        from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

        url = "https://mcp.example.com/mcp"
        s = Session(cwd=str(tmp_path), config=Config.from_dict({"mcp": {"test": {"url": url, "auth": "oauth"}}}))
        bootstrap_features(s)
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

        assert await roundtrip() == (None, None, None)

    def test_oauth_redirect_requires_explicit_interactive_connection(self):
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg(auth="oauth")))
        bootstrap_features(s)
        auth = s.mcp.oauth_client(s.mcp.find_config("test"), interactive=False)

        with pytest.raises(RuntimeError, match=r"authentication required; run /mcp connect test"):
            asyncio.run(auth.redirect_handler("https://login.example/authorize"))

    def test_interactive_oauth_redirect_notifies_before_delegating(self, monkeypatch):
        from fastmcp.client.auth import OAuth

        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg(auth="oauth")))
        bootstrap_features(s)
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
        bootstrap_features(s)
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

    async def test_discover_auto_stale_to_discovering(self, monkeypatch):
        """discover_auto sets status to discovering then ready."""
        raw = mcp_cfg()
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        bootstrap_features(s)

        # Inject a fake _list_tools to avoid real HTTP
        async def fake_list(url, headers):
            return []

        monkeypatch.setattr(s.mcp, "_list_tools", fake_list)

        assert s.mcp.discovery_status == "stale"
        await s.mcp.discover_auto_async()
        assert s.mcp.discovery_status == "ready"

    async def test_discover_auto_error_sets_status(self, monkeypatch):
        """Failed discovery sets error status."""
        raw = mcp_cfg()
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        bootstrap_features(s)

        async def fake_fail(url, headers):
            raise RuntimeError("connection refused")

        monkeypatch.setattr(s.mcp, "_list_tools", fake_fail)

        await s.mcp.discover_auto_async()
        assert s.mcp.discovery_status == "ready"
        assert s.mcp.server_errors.get("test") is not None

    async def test_discover_auto_ignores_cancelled_server(self, monkeypatch):
        raw = mcp_cfg()
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        bootstrap_features(s)

        async def cancelled(_config, _headers):
            raise asyncio.CancelledError

        monkeypatch.setattr(s.mcp, "_gather_assets", cancelled)

        await s.mcp.discover_auto_async()

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

    async def test_discover_auto_skips_missing_bearer_env(self, monkeypatch):
        """Missing bearer_token_env_var skips discovery without an error log."""
        monkeypatch.delenv("MISSING_TOKEN", raising=False)
        raw = mcp_cfg(bearer_token_env_var="MISSING_TOKEN")
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        bootstrap_features(s)

        await s.mcp.discover_auto_async()

        assert "test" not in s.mcp.server_errors
        assert "test" in s.mcp.server_skips
        assert "missing environment variable MISSING_TOKEN" in s.mcp.render_server_status()

    async def test_discover_auto_loads_servers_in_parallel(self, monkeypatch):
        """Multiple automatic servers are discovered in parallel."""
        raw = {
            "mcp": {
                "a": {"url": "http://a/mcp", "auto_connect": True},
                "b": {"url": "http://b/mcp", "auto_connect": True},
            }
        }
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        bootstrap_features(s)

        async def fake_list(url, headers):
            await asyncio.sleep(0.1)
            return []

        monkeypatch.setattr(s.mcp, "_list_tools", fake_list)
        monkeypatch.setattr(s.mcp, "_list_resources", fake_list)
        started = time.monotonic()
        await s.mcp.discover_auto_async()
        elapsed = time.monotonic() - started

        assert elapsed < 0.18
        assert s.mcp.discovery_status == "ready"

    async def test_discover_auto_skips_manual_servers(self, monkeypatch):
        raw = {
            "mcp": {
                "automatic": {"url": "http://a/mcp", "auto_connect": True},
                "manual": {"url": "http://b/mcp"},
            }
        }
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        bootstrap_features(s)
        discovered = []

        async def fake_discover(config):
            discovered.append(config.name)
            s.mcp.tools[config.name] = []
            s.mcp.resources[config.name] = []

        monkeypatch.setattr(s.mcp, "_discover_one_async", fake_discover)

        await s.mcp.discover_auto_async()

        assert discovered == ["automatic"]
        assert "manual" not in s.mcp.tools

    async def test_tools_are_cached_after_discovery(self, monkeypatch):
        """Listed tools are cached after discovery."""
        raw = mcp_cfg()
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        bootstrap_features(s)

        class FakeTool:
            name = "echo"
            description = "Echo input"
            inputSchema: ClassVar[dict] = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}
            annotations = None

        async def fake_list(url, headers):
            return [FakeTool()]

        monkeypatch.setattr(s.mcp, "_list_tools", fake_list)
        await s.mcp.discover_auto_async()

        assert "test" in s.mcp.tools
        assert len(s.mcp.tools["test"]) == 1
        assert s.mcp.tools["test"][0].name == "echo"

    async def test_discover_auto_preserves_manual_and_removes_stale_servers(self, monkeypatch):
        raw = {
            "mcp": {
                "auto_server": {"url": "http://a/mcp", "auto_connect": True},
                "manual_server": {"url": "http://b/mcp", "enabled": False},
            }
        }
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        bootstrap_features(s)

        async def fake_list(url, headers):
            return []

        monkeypatch.setattr(s.mcp, "_list_tools", fake_list)
        monkeypatch.setattr(s.mcp, "_list_resources", fake_list)

        # Manually add pre-existing stale data
        s.mcp.tools["manual_server"] = [mcp_tool_info("manual_server", "kept")]
        s.mcp.tools["stale_server"] = [mcp_tool_info("stale_server", "old")]

        await s.mcp.discover_auto_async()

        assert "stale_server" not in s.mcp.tools
        assert "manual_server" in s.mcp.tools
        assert "auto_server" in s.mcp.tools


# ---------------------------------------------------------------------------
# MCPManager render_tools_index & _format_tool_line
# ---------------------------------------------------------------------------


class TestMCPDiscoverServer:
    async def test_discover_nonexistent_server_sets_error(self):
        """discover_server for a server not in config sets server_errors."""
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg()))
        bootstrap_features(s)
        await s.mcp.discover_server_async("nonexistent")
        assert "nonexistent" in s.mcp.server_errors
        assert "not found" in s.mcp.server_errors["nonexistent"]

    async def test_discover_nonexistent_server_removes_tools(self):
        """discover_server for a server not in config clears its stale tools."""
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg()))
        bootstrap_features(s)
        s.mcp.tools["gone"] = [mcp_tool_info("gone", "old_tool")]
        await s.mcp.discover_server_async("gone")
        assert "gone" not in s.mcp.tools
        assert "gone" in s.mcp.server_errors


# ---------------------------------------------------------------------------
# MCP resources (list / read)
# ---------------------------------------------------------------------------


class TestMCPPruning:
    def test_prune_keeps_describe_record_referenced_in_messages(self):
        """A describe record is retained when its tr.N is referenced in history (normal path)."""
        s = Session(cwd="/tmp")
        bootstrap_features(s)
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
        bootstrap_features(s)
        s.mcp.tools["test"] = [mcp_tool_info("test", "echo")]
        desc = '<MCPDescribe server="test" tool="echo">\n<description>\nEcho.</description>\n</MCPDescribe>'
        s.store_tool_result("MCP", [{"action": "describe", "server": "test", "tool": "echo"}], desc)
        ctx = ContextManager(s)

        ctx.prune_tool_records([])

        assert s.tool_records == []

    def test_prune_drops_non_mcp_records(self):
        """prune_tool_records drops records not referenced in messages."""
        s = Session(cwd="/tmp")
        bootstrap_features(s)
        s.store_tool_result("Find", [], "results from find")
        s.store_tool_result("Read", [], "read output")
        ctx = ContextManager(s)

        ctx.prune_tool_records([])

        assert len(s.tool_records) == 0


# ---------------------------------------------------------------------------
# CommandLoop — /mcp tools NAME
# ---------------------------------------------------------------------------


class TestServerStatusRendering:
    def test_render_server_status_no_servers(self):
        """No servers returns placeholder."""
        s = session("/tmp")
        status = s.mcp.render_server_status()
        assert status == "(no MCP servers configured)"

    async def test_render_server_status_connected(self, monkeypatch):
        """Connected server shows mode and tool count."""
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
        await s.mcp.discover_auto_async()

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
        bootstrap_features(s)
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
        bootstrap_features(s)
        s.mcp.tools["a"] = []
        s.mcp.resources["a"] = []

        lines = [line for line in s.mcp.render_server_status().splitlines() if line.startswith("|")]

        assert len({tuple(index for index, char in enumerate(line) if char == "|") for line in lines}) == 1
        assert "● connected" in lines[2]
        assert "● disconnected" in lines[3]

    async def test_render_tool_listing_all(self, monkeypatch):
        """render_tool_listing shows connected servers."""
        raw = mcp_cfg()
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        bootstrap_features(s)

        class FakeTool:
            name = "echo"
            description = "Echo input back"
            inputSchema: ClassVar[dict] = {"type": "object", "properties": {"t": {"type": "string"}}, "required": ["t"]}
            annotations = None

        async def fake_list(url, headers):
            return [FakeTool()]

        monkeypatch.setattr(s.mcp, "_list_tools", fake_list)
        await s.mcp.discover_auto_async()

        listing = s.mcp.render_tool_listing()
        assert "### `test`" in listing
        assert "| tool | args | description |" in listing
        assert "echo" in listing

    async def test_render_tool_listing_specific_server(self, monkeypatch):
        """render_tool_listing('test') filters to one server."""
        raw = {"mcp": {"a": {"url": "http://a/mcp", "auto_connect": True}, "b": {"url": "http://b/mcp", "auto_connect": True}}}
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        bootstrap_features(s)

        async def fake_list(url, headers):
            return []

        monkeypatch.setattr(s.mcp, "_list_tools", fake_list)
        monkeypatch.setattr(s.mcp, "_list_resources", fake_list)
        await s.mcp.discover_auto_async()

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
        bootstrap_features(s)
        s.mcp.tools["connected"] = []
        s.mcp.resources["connected"] = []

        listing = s.mcp.render_tool_listing()

        assert "### `connected`" in listing
        assert "offline" not in listing

    def test_render_tool_listing_disconnected_server_has_connect_hint(self):
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg()))
        bootstrap_features(s)

        assert s.mcp.render_tool_listing("test") == "MCP server 'test' is not connected; run /mcp connect test"

    def test_render_tool_listing_includes_resources(self):
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg()))
        bootstrap_features(s)
        s.mcp.tools["test"] = []
        s.mcp.resources["test"] = [MCPResourceInfo("test", "docs://guide", "guide", "Usage guide", "text/plain")]

        listing = s.mcp.render_tool_listing("test")

        assert "no tools discovered" in listing
        assert "docs://guide" in listing
        assert "Usage guide" in listing


# ---------------------------------------------------------------------------
# MCPTool — needs_confirmation
# ---------------------------------------------------------------------------


class TestStatusBarMCPStatus:
    def test_stale_shows_nothing(self):
        """Stale status → empty string."""
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg()))
        bootstrap_features(s)
        bar = StatusBar(s)
        assert bar.mcp_status() == ""

    def test_discovering_shows_spinner(self, monkeypatch):
        """Discovering status → loaded/total + spinner."""
        s = Session(cwd="/tmp", config=Config.from_dict({"mcp": {"test": {"url": "http://x/mcp"}}}))
        bootstrap_features(s)
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
        bootstrap_features(s)
        s.mcp.discovery_status = "discovering"
        s.mcp.tools["a"] = [mcp_tool_info("a", "echo")]
        bar = StatusBar(s)
        monkeypatch.setattr(time, "monotonic", lambda: 0.0)
        assert bar.mcp_status() == "mcp 1/2" + bar.INDEX_SPINNER[0]

    def test_ready_shows_server_count(self, monkeypatch):
        """Ready status → 'MCP N' where N is server count."""
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg()))
        bootstrap_features(s)
        s.mcp.discovery_status = "ready"
        s.mcp.tools["test"] = [mcp_tool_info("test", "echo")]
        bar = StatusBar(s)
        assert bar.mcp_status() == "mcp 1"

    def test_ready_zero_servers(self):
        """Ready with no servers → 'mcp 0'."""
        s = Session(cwd="/tmp", config=Config.from_dict({"mcp": {"test": {"url": "http://x/mcp"}}}))
        bootstrap_features(s)
        s.mcp.discovery_status = "ready"
        bar = StatusBar(s)
        assert bar.mcp_status() == "mcp 0"

    def test_error_shows_error(self):
        """Error status → 'mcp err'."""
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg()))
        bootstrap_features(s)
        s.mcp.discovery_status = "error"
        bar = StatusBar(s)
        assert bar.mcp_status() == "mcp err"

    def test_discovering_statusbar_spinner_animates(self, monkeypatch):
        """Discovering spinner changes over time."""
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg()))
        bootstrap_features(s)
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


# ---------------------------------------------------------------------------
# Library logging — MCP client transport tracebacks must not reach the TUI
# ---------------------------------------------------------------------------
# wizolt's configure_logging quiets the noisy MCP client transport loggers so that
# expected-and-already-surfaced failures (httpx ReadTimeout on a slow server,
# dropped SSE/stdio frames, JSON-RPC parse errors) don't dump full tracebacks
# onto stderr mid-render via logging.lastResort.


class TestTransportLoggingSuppressed:
    def test_transport_loggers_at_critical(self):
        """configure_logging raises the transport loggers out of the ERROR band."""
        import logging

        from wizolt.base import configure_logging

        configure_logging()

        for name in ("mcp.client.streamable_http", "mcp.client.sse", "mcp.client.stdio"):
            logger = logging.getLogger(name)
            assert logger.level == logging.CRITICAL, f"{name} should be CRITICAL, got {logging.getLevelName(logger.level)}"

    def test_post_writer_traceback_does_not_reach_stderr(self, capsys):
        """logger.exception on the streamable_http transport produces no stderr output."""
        import logging

        from wizolt.base import configure_logging

        configure_logging()
        log = logging.getLogger("mcp.client.streamable_http")
        try:
            raise ZeroDivisionError("simulated transport failure")
        except ZeroDivisionError:
            log.exception("Error in post_writer")
        captured = capsys.readouterr()
        assert "Error in post_writer" not in captured.err
        assert "ZeroDivisionError" not in captured.err
        assert captured.err == ""
