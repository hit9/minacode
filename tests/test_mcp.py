"""Tests for nanocode MCP client integration."""
import asyncio
import json
import os
import time
from types import SimpleNamespace

import pytest

import nanocode as n


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def session(tmp_path):
    return n.Session(cwd=str(tmp_path))


def mcp_cfg(**overrides) -> dict:
    """Return a full [mcp.x] config dict for one server."""
    cfg = {
        "mcp": {
            "test": {
                "url": "http://localhost:9999/mcp",
                "enabled": True,
            }
        }
    }
    server = cfg["mcp"]["test"]
    server.update(overrides)
    return cfg


def parse_one(raw: dict) -> n.MCPServerConfig | None:
    """Parse a config dict and return the first server config."""
    config = n.Config.from_dict(raw)
    s = n.Session(cwd="/tmp", config=config)
    configs = s.mcp.parse_configs()
    return configs[0] if configs else None


def mcp_tool_info(server: str, name: str, **kw) -> n.MCPToolInfo:
    """Create an MCPToolInfo suitable for tests."""
    return n.MCPToolInfo(
        server=server,
        name=name,
        description=kw.pop("description", "A test tool."),
        input_schema=kw.pop("input_schema", {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "Input text."}},
            "required": ["text"],
        }),
        annotations=kw.pop("annotations", {}),
        **kw,
    )


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------

class TestConfigParsing:
    def test_parse_basic_server(self):
        """Parse [mcp.x] with url and default enabled=true."""
        cfg = parse_one(mcp_cfg())
        assert cfg is not None
        assert cfg.name == "test"
        assert cfg.url == "http://localhost:9999/mcp"
        assert cfg.enabled is True
        assert cfg.bearer_token_env_var == ""
        assert cfg.error == ""

    def test_parse_with_bearer_token_env_var(self):
        """Parse with bearer_token_env_var set."""
        cfg = parse_one(mcp_cfg(bearer_token_env_var="MY_TOKEN"))
        assert cfg is not None
        assert cfg.bearer_token_env_var == "MY_TOKEN"

    def test_disabled_server(self):
        """Disabled servers are parsed but not discovered."""
        cfg = parse_one(mcp_cfg(enabled=False))
        assert cfg is not None
        assert cfg.enabled is False

    def test_missing_url_on_enabled_server(self):
        """Missing URL on enabled server — server config stores error."""
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
        config = n.Config.from_dict(raw)
        s = n.Session(cwd="/tmp", config=config)
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
        assert "exactly one" in parse_one({"mcp": {"x": {"enabled": True}}}).error

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
        s = n.Session(cwd="/tmp", config=n.Config.from_dict({"mcp": {"x": {"command": "npx", "args": ["srv"]}}}))
        assert isinstance(s.mcp._transport(s.mcp.parse_configs()[0], {}), StdioTransport)
        s = n.Session(cwd="/tmp", config=n.Config.from_dict(mcp_cfg()))
        assert isinstance(s.mcp._transport(s.mcp.parse_configs()[0], {}), StreamableHttpTransport)


# ---------------------------------------------------------------------------
# MCPManager header/auth building
# ---------------------------------------------------------------------------

class TestMCPManagerHeaders:
    def test_bearer_token_success(self, monkeypatch):
        """bearer_token_env_var reads from environment."""
        monkeypatch.setenv("MY_TEST_TOKEN", "secret123")
        raw = mcp_cfg(bearer_token_env_var="MY_TEST_TOKEN")
        s = n.Session(cwd="/tmp", config=n.Config.from_dict(raw))
        config = s.mcp.parse_configs()[0]
        headers = s.mcp._build_mcp_headers(config)
        assert headers == {"Authorization": "Bearer secret123"}

    def test_bearer_token_missing_var(self, monkeypatch):
        """Missing bearer env var returns error string."""
        monkeypatch.delenv("MISSING_TOKEN", raising=False)
        raw = mcp_cfg(bearer_token_env_var="MISSING_TOKEN")
        s = n.Session(cwd="/tmp", config=n.Config.from_dict(raw))
        config = s.mcp.parse_configs()[0]
        result = s.mcp._build_mcp_headers(config)
        assert isinstance(result, str)
        assert "missing" in result.lower()

    def test_env_http_headers_success(self, monkeypatch):
        """env_http_headers reads header values from environment."""
        monkeypatch.setenv("MY_HEADER_VAL", "xyz")
        raw = mcp_cfg(env_http_headers={"X-Custom": "MY_HEADER_VAL"})
        s = n.Session(cwd="/tmp", config=n.Config.from_dict(raw))
        config = s.mcp.parse_configs()[0]
        headers = s.mcp._build_mcp_headers(config)
        assert headers == {"X-Custom": "xyz"}

    def test_authorization_env_http_header_success(self, monkeypatch):
        """Authorization is allowed via env_http_headers when it is the only auth source."""
        monkeypatch.setenv("AUTH_VAL", "Bearer custom")
        raw = mcp_cfg(env_http_headers={"Authorization": "AUTH_VAL"})
        s = n.Session(cwd="/tmp", config=n.Config.from_dict(raw))
        config = s.mcp.parse_configs()[0]
        headers = s.mcp._build_mcp_headers(config)
        assert headers == {"Authorization": "Bearer custom"}

    def test_env_http_headers_missing_var(self, monkeypatch):
        """Missing env_http_headers env var returns error string."""
        monkeypatch.delenv("MISSING_HEADER_VAL", raising=False)
        raw = mcp_cfg(env_http_headers={"X-Custom": "MISSING_HEADER_VAL"})
        s = n.Session(cwd="/tmp", config=n.Config.from_dict(raw))
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
        s = n.Session(cwd="/tmp", config=n.Config.from_dict(raw))
        config = s.mcp.parse_configs()[0]
        result = s.mcp._build_mcp_headers(config)
        assert isinstance(result, str)
        assert "conflicting" in result.lower()

    def test_no_auth_config(self):
        """No auth config produces empty headers dict."""
        raw = mcp_cfg()
        s = n.Session(cwd="/tmp", config=n.Config.from_dict(raw))
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

    def test_oauth_token_store_is_shared_for_manager(self, tmp_path):
        """Token storage keeps one store and one lock per token file path."""
        s = session(tmp_path)
        store = s.mcp.oauth_token_store()
        same_path_store = n.MCPFileTokenStore(store.path)

        assert s.mcp.oauth_token_store() is store
        assert same_path_store.lock is store.lock

    def test_discover_enabled_stale_to_discovering(self, monkeypatch):
        """discover_enabled sets status to discovering then ready."""
        raw = mcp_cfg()
        s = n.Session(cwd="/tmp", config=n.Config.from_dict(raw))

        # Inject a fake _list_tools to avoid real HTTP
        async def fake_list(url, headers):
            return []
        monkeypatch.setattr(s.mcp, "_list_tools", fake_list)

        assert s.mcp.discovery_status == "stale"
        s.mcp.discover_enabled()
        assert s.mcp.discovery_status == "ready"

    def test_discover_enabled_error_sets_status(self, monkeypatch):
        """Failed discovery sets error status."""
        raw = mcp_cfg()
        s = n.Session(cwd="/tmp", config=n.Config.from_dict(raw))

        async def fake_fail(url, headers):
            raise Exception("connection refused")
        monkeypatch.setattr(s.mcp, "_list_tools", fake_fail)

        s.mcp.discover_enabled()
        assert s.mcp.discovery_status == "ready"
        assert s.mcp.server_errors.get("test") is not None

    def test_discover_enabled_skips_missing_bearer_env(self, monkeypatch):
        """Missing bearer_token_env_var skips discovery without an error log."""
        monkeypatch.delenv("MISSING_TOKEN", raising=False)
        raw = mcp_cfg(bearer_token_env_var="MISSING_TOKEN")
        s = n.Session(cwd="/tmp", config=n.Config.from_dict(raw))

        s.mcp.discover_enabled()

        assert "test" not in s.mcp.server_errors
        assert "test" in s.mcp.server_skips
        assert "missing environment variable MISSING_TOKEN" in s.mcp.render_server_status()

    def test_discover_enabled_loads_servers_in_parallel(self, monkeypatch):
        """Multiple enabled servers are discovered in parallel."""
        raw = {
            "mcp": {
                "a": {"url": "http://a/mcp"},
                "b": {"url": "http://b/mcp"},
            }
        }
        s = n.Session(cwd="/tmp", config=n.Config.from_dict(raw))

        async def fake_list(url, headers):
            await asyncio.sleep(0.1)
            return []

        monkeypatch.setattr(s.mcp, "_list_tools", fake_list)
        monkeypatch.setattr(s.mcp, "_list_resources", fake_list)
        started = time.monotonic()
        s.mcp.discover_enabled()
        elapsed = time.monotonic() - started

        assert elapsed < 0.18
        assert s.mcp.discovery_status == "ready"

    def test_tools_are_cached_after_discovery(self, monkeypatch):
        """Listed tools are cached after discovery."""
        raw = mcp_cfg()
        s = n.Session(cwd="/tmp", config=n.Config.from_dict(raw))

        class FakeTool:
            name = "echo"
            description = "Echo input"
            inputSchema = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}
            annotations = None

        async def fake_list(url, headers):
            return [FakeTool()]

        monkeypatch.setattr(s.mcp, "_list_tools", fake_list)
        s.mcp.discover_enabled()

        assert "test" in s.mcp.tools
        assert len(s.mcp.tools["test"]) == 1
        assert s.mcp.tools["test"][0].name == "echo"

    def test_disabled_servers_removed_from_tools(self, monkeypatch):
        """Disabled servers are removed from tools on discovery."""
        raw = {
            "mcp": {
                "enabled_server": {"url": "http://a/mcp", "enabled": True},
                "disabled_server": {"url": "http://b/mcp", "enabled": False},
                "removed_server": {"url": "http://c/mcp", "enabled": True},
            }
        }
        s = n.Session(cwd="/tmp", config=n.Config.from_dict(raw))
        async def fake_list(url, headers):
            return []
        monkeypatch.setattr(s.mcp, "_list_tools", fake_list)

        # Manually add pre-existing stale data
        s.mcp.tools["removed_server"] = [mcp_tool_info("removed_server", "stale")]
        s.mcp.tools["stale_server"] = [mcp_tool_info("stale_server", "old")]

        s.mcp.discover_enabled()

        # stale_server was removed (not in config)
        assert "stale_server" not in s.mcp.tools
        # disabled_server was removed
        assert "disabled_server" not in s.mcp.tools
        # enabled_server exists
        assert "enabled_server" in s.mcp.tools


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
        info = mcp_tool_info("test", "echo", input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        })
        s = session("/tmp")
        line = s.mcp._format_tool_line("test", info)
        assert "text: string" in line

    def test_format_tool_line_requires_args(self):
        """Required args appear before semicolon."""
        info = mcp_tool_info("test", "echo", input_schema={
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "integer"},
            },
            "required": ["a"],
        })
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
        s = n.Session(cwd="/tmp", config=n.Config.from_dict(raw))
        class FakeTool:
            name = "echo"
            description = "Echo"
            inputSchema = {"type": "object", "properties": {"t": {"type": "string"}}, "required": ["t"]}
            annotations = None
        async def fake_list(url, headers):
            return [FakeTool()]
        monkeypatch.setattr(s.mcp, "_list_tools", fake_list)
        s.mcp.discover_enabled()

        idx = s.mcp.render_tools_index()
        assert "--- MCP TOOLS ---" in idx
        assert "[test]" in idx

    def test_index_does_not_include_disabled_servers(self, monkeypatch):
        """Disabled servers are not included in the index."""
        raw = {"mcp": {"test": {"url": "http://x/mcp", "enabled": False}}}
        s = n.Session(cwd="/tmp", config=n.Config.from_dict(raw))
        idx = s.mcp.render_tools_index()
        assert idx == ""


# ---------------------------------------------------------------------------
# MCPManager render_tools_index budget degradation (regression: a verbose
# server must never hide later servers from the model)
# ---------------------------------------------------------------------------

def _index_session(servers):
    """Build a session with the given {server: [(tool_name, n_schema_fields), ...]}."""
    s = n.Session(cwd="/tmp", config=n.Config.from_dict(
        {"mcp": {name: {"url": f"https://{name}/mcp", "enabled": True} for name in servers}}))
    for name, tools in servers.items():
        s.mcp.tools[name] = [
            n.MCPToolInfo(
                server=name, name=tool_name, description="A tool.",
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
        s = _index_session({
            "alpha": [(f"q{i}", 30) for i in range(60)],  # huge: full schemas blow the cap
            "beta": [("beta_tool", 2)],
            "gamma": [("gamma_tool", 2)],
        })
        idx = s.mcp.render_tools_index()
        assert len(idx) <= n.MCPManager.INDEX_TOTAL_LIMIT
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
        s = _index_session({
            "alpha": [(f"q{i}", 25) for i in range(40)],
            "beta": [("beta_a", 3), ("beta_b", 3)],
            "slack": [("post", 3)],
        })
        idx = s.mcp.render_tools_index()
        assert len(idx) <= n.MCPManager.INDEX_TOTAL_LIMIT
        assert "Schemas omitted to fit" in idx
        assert "\n  schema: {" not in idx  # no per-tool schema lines
        for header in ("[alpha]", "[beta]", "[slack]"):
            assert header in idx
        for tool in ("q0", "q39", "beta_a", "beta_b", "post"):
            assert tool in idx

    def test_tier3_names_only_lists_every_tool(self):
        """When even arg summaries overflow, fall back to name-only with all tools listed."""
        s = _index_session({
            "alpha": [(f"q{i}", 30) for i in range(120)],
            "github": [(f"gh{i}", 30) for i in range(40)],
            "jira": [(f"j{i}", 30) for i in range(40)],
        })
        idx = s.mcp.render_tools_index()
        assert len(idx) <= n.MCPManager.INDEX_TOTAL_LIMIT
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

    def test_unconnected_server_listed_as_pending(self):
        """A configured-but-not-yet-connected server (e.g. awaiting OAuth login) is still
        surfaced so the model knows it exists."""
        s = n.Session(cwd="/tmp", config=n.Config.from_dict({"mcp": {
            "github": {"url": "https://g/mcp", "enabled": True},
            "metabase": {"url": "https://m/api/mcp", "auth": "oauth", "enabled": True},
        }}))
        s.mcp.tools["github"] = [n.MCPToolInfo(server="github", name="search", description="Search.",
            input_schema={"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}, annotations={})]
        s.mcp.server_errors["metabase"] = "oauth login required; run /mcp login metabase"
        s.mcp.discovery_status = "ready"
        idx = s.mcp.render_tools_index()
        assert "[github]" in idx
        assert "metabase" in idx
        assert "oauth login required" in idx


# ---------------------------------------------------------------------------
# MCPManager render_server_status & render_tool_listing
# ---------------------------------------------------------------------------

class TestServerStatusRendering:
    def test_render_server_status_no_servers(self):
        """No servers returns placeholder."""
        s = session("/tmp")
        status = s.mcp.render_server_status()
        assert status == "(no MCP servers configured)"

    def test_render_server_status_enabled(self, monkeypatch):
        """Enabled server shows connected and tool count."""
        raw = mcp_cfg()
        s = n.Session(cwd="/tmp", config=n.Config.from_dict(raw))
        class FakeTool:
            name = "echo"
            description = "Echo"
            inputSchema = {"type": "object", "properties": {}, "required": []}
            annotations = None
        async def fake_list(url, headers):
            return [FakeTool()]
        monkeypatch.setattr(s.mcp, "_list_tools", fake_list)
        s.mcp.discover_enabled()

        status = s.mcp.render_server_status()
        assert "test" in status
        assert "connected" in status
        assert "| `test` | connected | 1 |" in status
        # No secrets leaked
        assert "localhost" not in status

    def test_render_server_status_disabled(self):
        """Disabled server shows disabled."""
        raw = {"mcp": {"test": {"url": "http://x/mcp", "enabled": False}}}
        s = n.Session(cwd="/tmp", config=n.Config.from_dict(raw))
        status = s.mcp.render_server_status()
        assert "test" in status
        assert "disabled" in status

    def test_render_tool_listing_all(self, monkeypatch):
        """render_tool_listing shows all servers."""
        raw = mcp_cfg()
        s = n.Session(cwd="/tmp", config=n.Config.from_dict(raw))
        class FakeTool:
            name = "echo"
            description = "Echo input back"
            inputSchema = {"type": "object", "properties": {"t": {"type": "string"}}, "required": ["t"]}
            annotations = None
        async def fake_list(url, headers):
            return [FakeTool()]
        monkeypatch.setattr(s.mcp, "_list_tools", fake_list)
        s.mcp.discover_enabled()

        listing = s.mcp.render_tool_listing()
        assert "### `test`" in listing
        assert "| tool | args | description |" in listing
        assert "echo" in listing

    def test_render_tool_listing_specific_server(self, monkeypatch):
        """render_tool_listing('test') filters to one server."""
        raw = {"mcp": {"a": {"url": "http://a/mcp"}, "b": {"url": "http://b/mcp"}}}
        s = n.Session(cwd="/tmp", config=n.Config.from_dict(raw))
        async def fake_list(url, headers):
            return []
        monkeypatch.setattr(s.mcp, "_list_tools", fake_list)
        s.mcp.discover_enabled()

        listing = s.mcp.render_tool_listing("a")
        assert "### `a`" in listing
        assert "### `b`" not in listing

    def test_render_tool_listing_no_servers(self):
        """No servers returns placeholder."""
        s = session("/tmp")
        assert s.mcp.render_tool_listing() == "(no MCP servers configured)"


# ---------------------------------------------------------------------------
# MCPTool — needs_confirmation
# ---------------------------------------------------------------------------

class TestMCPToolConfirmation:
    def test_describe_does_not_require_confirmation(self):
        """MCP(action='describe') → no confirmation."""
        payload = {"action": "describe", "server": "test", "tool": "echo"}
        tool = n.MCPTool(None, [payload])
        assert tool.needs_confirmation() is False

    def test_call_requires_confirmation(self, monkeypatch):
        """MCP(action='call') on an undiscovered tool → confirmation needed by default."""
        raw = mcp_cfg()
        s = n.Session(cwd="/tmp", config=n.Config.from_dict(raw))
        payload = {"action": "call", "server": "test", "tool": "echo", "arguments": {"text": "hi"}}
        tool = n.MCPTool(s, [payload])
        # No info yet (not discovered) → confirm by default
        assert tool.needs_confirmation() is True

    def test_call_without_annotations_requires_confirmation(self, monkeypatch):
        """A discovered tool with no annotations → confirmation needed by default."""
        raw = mcp_cfg()
        s = n.Session(cwd="/tmp", config=n.Config.from_dict(raw))
        s.mcp.tools["test"] = [mcp_tool_info("test", "echo", annotations={})]
        payload = {"action": "call", "server": "test", "tool": "echo", "arguments": {"text": "hi"}}
        tool = n.MCPTool(s, [payload])
        assert tool.needs_confirmation() is True

    def test_call_with_non_destructive_hint_no_confirmation(self, monkeypatch):
        """destructiveHint=false → no confirmation needed."""
        raw = mcp_cfg()
        s = n.Session(cwd="/tmp", config=n.Config.from_dict(raw))
        s.mcp.tools["test"] = [mcp_tool_info("test", "echo", annotations={"destructiveHint": False})]
        payload = {"action": "call", "server": "test", "tool": "echo", "arguments": {"text": "hi"}}
        tool = n.MCPTool(s, [payload])
        assert tool.needs_confirmation() is False

    def test_call_with_readonly_hint_no_confirmation(self, monkeypatch):
        """readOnlyHint=true → no confirmation needed."""
        raw = mcp_cfg()
        s = n.Session(cwd="/tmp", config=n.Config.from_dict(raw))
        # Pre-populate tools with readOnlyHint
        info = mcp_tool_info("test", "echo", annotations={"readOnlyHint": True})
        s.mcp.tools["test"] = [info]

        payload = {"action": "call", "server": "test", "tool": "echo", "arguments": {"text": "hi"}}
        tool = n.MCPTool(s, [payload])
        assert tool.needs_confirmation() is False

    def test_call_with_destructive_hint_requires_confirmation(self, monkeypatch):
        """destructiveHint=true → confirmation needed."""
        raw = mcp_cfg()
        s = n.Session(cwd="/tmp", config=n.Config.from_dict(raw))
        info = mcp_tool_info("test", "delete", annotations={"destructiveHint": True})
        s.mcp.tools["test"] = [info]

        payload = {"action": "call", "server": "test", "tool": "delete", "arguments": {"id": "1"}}
        tool = n.MCPTool(s, [payload])
        assert tool.needs_confirmation() is True

    def test_invalid_payload_raises_tool_error(self):
        """Non-dict payload raises ToolError."""
        tool = n.MCPTool(None, ["bad"])
        with pytest.raises(n.ToolError, match="named fields"):
            tool.payload()

    def test_payload_parsing(self):
        """payload() returns the raw dict."""
        payload = {"action": "call", "server": "x", "tool": "y"}
        tool = n.MCPTool(None, [payload])
        assert tool.payload() == payload


# ---------------------------------------------------------------------------
# MCPTool short_args
# ---------------------------------------------------------------------------

class TestMCPToolShortArgs:
    def test_short_args_call(self):
        """call action shows 'call server.tool'."""
        payload = {"action": "call", "server": "test", "tool": "echo"}
        tool = n.MCPTool(None, [payload])
        args = tool.short_args()
        assert any("call" in str(a) or "test.echo" in str(a) for a in args)

    def test_short_args_describe(self):
        """describe action shows 'describe server.tool'."""
        payload = {"action": "describe", "server": "test", "tool": "echo"}
        tool = n.MCPTool(None, [payload])
        args = tool.short_args()
        assert any("describe" in str(a) for a in args)


# ---------------------------------------------------------------------------
# StatusBar mcp_status
# ---------------------------------------------------------------------------

class TestStatusBarMCPStatus:
    def test_stale_shows_nothing(self):
        """Stale status → empty string."""
        s = n.Session(cwd="/tmp")
        bar = n.StatusBar(s)
        assert bar.mcp_status() == ""

    def test_discovering_shows_spinner(self, monkeypatch):
        """Discovering status → loaded/total + spinner."""
        s = n.Session(cwd="/tmp")
        s.mcp.discovery_status = "discovering"
        bar = n.StatusBar(s)
        monkeypatch.setattr(n.time, "monotonic", lambda: 0.0)
        status = bar.mcp_status()
        # First spinner char
        assert status == "mcp 0/0" + bar.INDEX_SPINNER[0]

    def test_discovering_shows_loaded_and_total(self, monkeypatch):
        """Discovering status includes loaded and configured server counts."""
        raw = {
            "mcp": {
                "a": {"url": "http://a/mcp"},
                "b": {"url": "http://b/mcp"},
            }
        }
        s = n.Session(cwd="/tmp", config=n.Config.from_dict(raw))
        s.mcp.discovery_status = "discovering"
        s.mcp.tools["a"] = [mcp_tool_info("a", "echo")]
        bar = n.StatusBar(s)
        monkeypatch.setattr(n.time, "monotonic", lambda: 0.0)
        assert bar.mcp_status() == "mcp 1/2" + bar.INDEX_SPINNER[0]

    def test_ready_shows_server_count(self, monkeypatch):
        """Ready status → 'MCP N' where N is server count."""
        s = n.Session(cwd="/tmp")
        s.mcp.discovery_status = "ready"
        s.mcp.tools["test"] = [mcp_tool_info("test", "echo")]
        bar = n.StatusBar(s)
        assert bar.mcp_status() == "mcp 1"

    def test_ready_zero_servers(self):
        """Ready with no servers → 'mcp 0'."""
        s = n.Session(cwd="/tmp")
        s.mcp.discovery_status = "ready"
        bar = n.StatusBar(s)
        assert bar.mcp_status() == "mcp 0"

    def test_error_shows_error(self):
        """Error status → 'mcp err'."""
        s = n.Session(cwd="/tmp")
        s.mcp.discovery_status = "error"
        bar = n.StatusBar(s)
        assert bar.mcp_status() == "mcp err"

    def test_discovering_statusbar_spinner_animates(self, monkeypatch):
        """Discovering spinner changes over time."""
        s = n.Session(cwd="/tmp")
        s.mcp.discovery_status = "discovering"
        bar = n.StatusBar(s)

        monkeypatch.setattr(n.time, "monotonic", lambda: 0.0)
        first = bar.mcp_status()

        monkeypatch.setattr(n.time, "monotonic", lambda: n.StatusBar.INTERVAL)
        second = bar.mcp_status()

        assert first != second


# ---------------------------------------------------------------------------
# ContextManager — MCP context blocks
# ---------------------------------------------------------------------------

class TestMCPContextBlocks:
    def test_mcp_tools_context_empty(self):
        """No MCP tools → empty string."""
        s = n.Session(cwd="/tmp")
        ctx = n.ContextManager(s)
        assert ctx.mcp_tools_context() == ""

    def test_mcp_tools_context_includes_tools(self, monkeypatch):
        """MCP tools present in index."""
        raw = mcp_cfg()
        s = n.Session(cwd="/tmp", config=n.Config.from_dict(raw))
        class FakeTool:
            name = "echo"
            description = "Echo"
            inputSchema = {"type": "object", "properties": {"t": {"type": "string"}}, "required": ["t"]}
            annotations = None
        async def fake_list(url, headers):
            return [FakeTool()]
        monkeypatch.setattr(s.mcp, "_list_tools", fake_list)
        s.mcp.discover_enabled()

        ctx = n.ContextManager(s)
        result = ctx.mcp_tools_context()
        assert "--- MCP TOOLS ---" in result
        assert "[test]" in result

    def test_mcp_describe_result_inline_in_history(self):
        """A describe result renders inline like any tool output, not a tail pointer."""
        s = n.Session(cwd="/tmp")
        runner = n.ToolRunner(s, n.ContextManager(s), output_fn=lambda text: None)
        call = n.ToolCall("c", "MCP", [{"action": "describe", "server": "test", "tool": "echo"}])
        desc = '<MCPDescribe server="test" tool="echo">\n<description>\nEcho input back.</description>\n</MCPDescribe>'

        msg = runner.tool_message(call, "tr.1", desc)
        assert "-> MCP TOOL DETAILS" not in msg
        assert "<MCPDescribe" in msg
        assert "tr.1" in msg

    def test_mcp_in_context_order(self):
        """MCP TOOLS appears after Environment and before FILE STATE; no separate details block."""
        s = n.Session(cwd="/tmp", config=n.Config.from_dict(mcp_cfg()))
        s.mcp.tools["test"] = [mcp_tool_info("test", "echo")]

        ctx = n.ContextManager(s)
        msgs = ctx.model_messages("sys")
        texts = [m["content"] for m in msgs if m.get("role") == "user"]

        env_idx = next(i for i, t in enumerate(texts) if t.startswith("--- Environment ---"))
        mcp_tools_idx = next(i for i, t in enumerate(texts) if t.startswith("--- MCP TOOLS ---"))
        file_state_idx = next(i for i, t in enumerate(texts) if t.startswith("--- FILE STATE ---"))

        assert env_idx < mcp_tools_idx < file_state_idx
        assert not any(t.startswith("--- MCP TOOL DETAILS ---") for t in texts)

    @staticmethod
    def _describe_msg(call_id: str, key: str, tool: str, body: str) -> dict:
        desc = f'<MCPDescribe server="test" tool="{tool}">\n{body}\n</MCPDescribe>'
        return {"role": "tool", "tool_call_id": call_id, "content": f"tool {key} MCP(describe, test, {tool})\noutput:\n{desc}"}

    def test_dedup_collapses_repeated_describe(self):
        """A second describe of the same tool collapses to a pointer at the first; the first stays full."""
        s = n.Session(cwd="/tmp")
        ctx = n.ContextManager(s)
        m1 = self._describe_msg("a", "tr.1", "echo", "schema")
        m2 = self._describe_msg("b", "tr.2", "echo", "schema")

        out = ctx.dedup_mcp_describes([m1, m2])

        assert "<MCPDescribe" in out[0]["content"]       # first kept full
        assert "<MCPDescribe" not in out[1]["content"]    # second collapsed
        assert "repeat describe of test.echo" in out[1]["content"]
        assert "tr.1" in out[1]["content"]                # points back to the first
        assert "tr.2" in out[1]["content"]                # head/recall key preserved
        assert m2["content"].count("<MCPDescribe") == 1   # input not mutated (pure transform)

    def test_dedup_keeps_distinct_tools(self):
        """Different tools each keep their full schema."""
        s = n.Session(cwd="/tmp")
        ctx = n.ContextManager(s)
        out = ctx.dedup_mcp_describes([self._describe_msg("a", "tr.1", "echo", "s1"), self._describe_msg("b", "tr.2", "ping", "s2")])

        assert all("<MCPDescribe" in m["content"] for m in out)

    def test_model_messages_dedups_describe(self):
        """model_messages applies the dedup to sent context without touching stored history."""
        s = n.Session(cwd="/tmp")
        s.messages = [self._describe_msg("a", "tr.1", "echo", "schema"), self._describe_msg("b", "tr.2", "echo", "schema")]
        ctx = n.ContextManager(s)

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
        s = n.Session(cwd="/tmp", config=n.Config.from_dict(raw))
        info = mcp_tool_info("test", "echo")
        s.mcp.tools["test"] = [info]

        result = s.mcp.describe_tool("test", "echo")
        assert "<MCPDescribe server=" in result
        assert "echo" in result

    def test_describe_unknown_tool_raises_error(self):
        """Unknown tool raises ToolError."""
        s = n.Session(cwd="/tmp")
        s.mcp.tools["test"] = [mcp_tool_info("test", "echo")]
        with pytest.raises(n.ToolError, match="not found"):
            s.mcp.describe_tool("test", "missing_tool")

    def test_describe_unknown_server_raises_error(self):
        """Unknown server raises ToolError."""
        s = n.Session(cwd="/tmp")
        with pytest.raises(n.ToolError, match="not found"):
            s.mcp.describe_tool("unknown", "echo")

    def test_describe_refreshes_if_missing(self, monkeypatch):
        """describe rediscover server if tools not cached."""
        raw = mcp_cfg()
        s = n.Session(cwd="/tmp", config=n.Config.from_dict(raw))
        class FakeTool:
            name = "echo"
            description = "Echo"
            inputSchema = {"type": "object", "properties": {}, "required": []}
            annotations = None

        async def fake_list(url, headers):
            return [FakeTool()]
        monkeypatch.setattr(s.mcp, "_list_tools", fake_list)

        result = s.mcp.describe_tool("test", "echo")
        assert "<MCPDescribe" in result


# ---------------------------------------------------------------------------
# MCPManager — call_tool
# ---------------------------------------------------------------------------

class TestCallTool:
    def test_call_unknown_server_raises_error(self):
        """Unknown server raises ToolError."""
        s = n.Session(cwd="/tmp")
        with pytest.raises(n.ToolError, match="not found"):
            s.mcp.call_tool("unknown", "echo", {})

    def test_call_server_with_error_raises(self, monkeypatch):
        """Server with prior error raises ToolError."""
        raw = mcp_cfg()
        s = n.Session(cwd="/tmp", config=n.Config.from_dict(raw))
        s.mcp.server_errors["test"] = "connection failed"

        with pytest.raises(n.ToolError, match="error"):
            s.mcp.call_tool("test", "echo", {})

    def test_call_without_url(self):
        """Server without URL raises ToolError."""
        raw = {"mcp": {"test": {"url": "", "enabled": True}}}
        s = n.Session(cwd="/tmp", config=n.Config.from_dict(raw))
        with pytest.raises(n.ToolError, match="url"):
            s.mcp.call_tool("test", "echo", {})


# ---------------------------------------------------------------------------
# CommandLoop — /mcp commands
# ---------------------------------------------------------------------------

class TestMCPCommands:
    def test_mcp_command_no_args_shows_status(self, monkeypatch):
        """/mcp returns server status."""
        raw = mcp_cfg()
        s = n.Session(cwd="/tmp", config=n.Config.from_dict(raw))
        class FakeTool:
            name = "echo"
            description = "Echo"
            inputSchema = {"type": "object", "properties": {}, "required": []}
            annotations = None
        async def fake_list(url, headers):
            return [FakeTool()]
        monkeypatch.setattr(s.mcp, "_list_tools", fake_list)
        s.mcp.discover_enabled()

        loop = n.CommandLoop(n.Agent(s), input_fn=lambda _: "", output_fn=lambda _: None)
        result = loop.mcp_command("")
        assert "test" in result
        assert "| `test` | connected | 1 |" in result

    def test_mcp_tools_shows_listing(self, monkeypatch):
        """/mcp tools returns tool listing."""
        raw = mcp_cfg()
        s = n.Session(cwd="/tmp", config=n.Config.from_dict(raw))
        class FakeTool:
            name = "echo"
            description = "Echo"
            inputSchema = {"type": "object", "properties": {}, "required": []}
            annotations = None
        async def fake_list(url, headers):
            return [FakeTool()]
        monkeypatch.setattr(s.mcp, "_list_tools", fake_list)
        s.mcp.discover_enabled()

        loop = n.CommandLoop(n.Agent(s), input_fn=lambda _: "", output_fn=lambda _: None)
        result = loop.mcp_command("tools")
        assert "### `test`" in result
        assert "echo" in result

    def test_mcp_login_failure_includes_mcp_url(self, monkeypatch):
        """/mcp login shows a fallback URL when OAuth does not provide one."""
        raw = mcp_cfg(auth="oauth")
        s = n.Session(cwd="/tmp", config=n.Config.from_dict(raw))

        async def fake_login(config, headers, *, interactive=False, notify=None):
            raise RuntimeError("Unexpected content type: text/html")

        monkeypatch.setattr(s.mcp, "_list_oauth_tools", fake_login)
        loop = n.CommandLoop(n.Agent(s), input_fn=lambda _: "", output_fn=lambda _: None)
        result = loop.mcp_command("login test")

        assert "Unexpected content type: text/html" in result
        assert "Open MCP URL: http://localhost:9999/mcp" in result

    def test_mcp_refresh_invokes_discovery(self, monkeypatch):
        """/mcp refresh calls discover_enabled."""
        raw = mcp_cfg()
        s = n.Session(cwd="/tmp", config=n.Config.from_dict(raw))
        async def fake_list(url, headers):
            return []
        monkeypatch.setattr(s.mcp, "_list_tools", fake_list)

        loop = n.CommandLoop(n.Agent(s), input_fn=lambda _: "", output_fn=lambda _: None)
        result = loop.mcp_command("refresh")
        assert s.mcp.discovery_status == "ready"

    def test_mcp_refresh_specific_server(self, monkeypatch):
        """/mcp refresh NAME calls discover_server."""
        calls = []
        original = type("", (), {"discover_server": lambda self, name: calls.append(name)})()

        raw = mcp_cfg()
        s = n.Session(cwd="/tmp", config=n.Config.from_dict(raw))
        monkeypatch.setattr(s.mcp, "discover_server", lambda name: calls.append(name))
        async def fake_list(url, headers):
            return []
        monkeypatch.setattr(s.mcp, "_list_tools", fake_list)

        loop = n.CommandLoop(n.Agent(s), input_fn=lambda _: "", output_fn=lambda _: None)
        loop.mcp_command("refresh test")
        assert "test" in calls

    def test_unknown_mcp_subcommand(self):
        """Bad /mcp subcommand returns error."""
        s = n.Session(cwd="/tmp")
        loop = n.CommandLoop(n.Agent(s), input_fn=lambda _: "", output_fn=lambda _: None)
        result = loop.mcp_command("bad_subcommand")
        assert "Unknown" in result

    def test_mcp_subcommands_reject_extra_args(self):
        """MCP subcommands do not silently ignore extra args."""
        s = n.Session(cwd="/tmp")
        loop = n.CommandLoop(n.Agent(s), input_fn=lambda _: "", output_fn=lambda _: None)

        assert loop.mcp_command("tools a b") == "Usage: /mcp tools [server]"
        assert loop.mcp_command("login a b").startswith("Usage: /mcp login")
        assert loop.mcp_command("logout a b").startswith("Usage: /mcp logout")
        assert loop.mcp_command("refresh a b") == "Usage: /mcp refresh [server]"

    def test_no_mcp_config(self):
        """No MCP config returns message."""
        s = n.Session(cwd="/tmp")
        s.mcp = None
        loop = n.CommandLoop(n.Agent(s), input_fn=lambda _: "", output_fn=lambda _: None)
        result = loop.mcp_command("")
        assert "not configured" in result


# ---------------------------------------------------------------------------
# Tab completion
# ---------------------------------------------------------------------------

class TestMCPTabCompletion:
    def test_mcp_command_completion(self):
        """/mcp  completes with tools, refresh."""
        completer = n.CommandCompleter()
        from prompt_toolkit.document import Document

        doc = Document("/mcp ")
        completions = list(completer.get_completions(doc, None))
        texts = [c.text for c in completions]
        assert "tools" in texts
        assert "refresh" in texts

    def test_mcp_completion_prefix_filtering(self):
        """Prefix filters subcommands."""
        completer = n.CommandCompleter()
        from prompt_toolkit.document import Document

        doc = Document("/mcp r")
        completions = list(completer.get_completions(doc, None))
        texts = [c.text for c in completions]
        assert "tools" not in texts
        assert "refresh" in texts

    def test_mcp_login_completion_uses_oauth_servers(self):
        """/mcp login completes only OAuth server names."""
        completer = n.CommandCompleter(
            mcp_servers=lambda: ("plain", "oauthOne"),
            mcp_oauth_servers=lambda: ("oauthOne",),
        )
        from prompt_toolkit.document import Document

        completions = list(completer.get_completions(Document("/mcp login "), None))
        texts = [c.text for c in completions]
        assert texts == ["oauthOne"]

    def test_mcp_tools_completion_uses_all_servers(self):
        """/mcp tools completes all MCP server names."""
        completer = n.CommandCompleter(
            mcp_servers=lambda: ("plain", "oauthOne"),
            mcp_oauth_servers=lambda: ("oauthOne",),
        )
        from prompt_toolkit.document import Document

        completions = list(completer.get_completions(Document("/mcp tools o"), None))
        texts = [c.text for c in completions]
        assert texts == ["oauthOne"]


# ---------------------------------------------------------------------------
# render_tools_index truncation
# ---------------------------------------------------------------------------

class TestToolIndexTruncation:
    def test_index_truncation_long_block(self, monkeypatch):
        """Long index block is bounded by INDEX_TOTAL_LIMIT."""
        raw = mcp_cfg()
        s = n.Session(cwd="/tmp", config=n.Config.from_dict(raw))

        # Create many tools to exceed budget
        class FakeTool:
            name = "tool"
            description = "Desc"
            inputSchema = {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}
            annotations = None

        many_tools = []
        for i in range(200):
            t = type("FakeTool", (), {
                "name": f"tool{i}",
                "description": "x" * 80,
                "inputSchema": {"type": "object", "properties": {"p": {"type": "string", "description": "x" * 100}}, "required": ["p"]},
                "annotations": None,
            })()
            many_tools.append(t)

        s.mcp.tools["test"] = []
        async def fake_list(url, headers):
            return many_tools
        monkeypatch.setattr(s.mcp, "_list_tools", fake_list)
        s.mcp.discover_enabled()

        idx = s.mcp.render_tools_index()
        assert len(idx) <= n.MCPManager.INDEX_TOTAL_LIMIT + 100
        assert "truncated" in idx  # 200 tools with schemas exceed the budget

    def test_format_tool_line_long_args(self):
        """Long args list is truncated."""
        props = {f"p{i}": {"type": "string"} for i in range(20)}
        required = [f"p{i}" for i in range(20)]
        info = mcp_tool_info("test", "big", input_schema={
            "type": "object",
            "properties": props,
            "required": required,
        })
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
        result = s.mcp.normalize_result([
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
        ])
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
        s = n.Session(cwd="/tmp", config=n.Config.from_dict(raw))

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
        s = n.Session(cwd="/tmp", config=n.Config.from_dict(raw))

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
        s = n.Session(cwd="/tmp", config=n.Config.from_dict(raw))

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
        s = n.Session(cwd="/tmp")
        s.mcp.tools["test"] = [mcp_tool_info("test", "echo")]
        desc = '<MCPDescribe server="test" tool="echo">\n<description>\nEcho back.</description>\n</MCPDescribe>'
        s.store_tool_result("MCP", [{"action": "describe", "server": "test", "tool": "echo"}], desc)
        ctx = n.ContextManager(s)

        key = s.tool_records[0].key
        keep_messages = [{"role": "tool", "content": f"tool {key} MCP(describe)\noutput:\n{desc}"}]
        ctx.prune_tool_records(keep_messages)

        assert [r.key for r in s.tool_records] == [key]

    def test_prune_drops_unreferenced_describe_record(self):
        """With the tail digest gone, an unreferenced describe record prunes like any other."""
        s = n.Session(cwd="/tmp")
        s.mcp.tools["test"] = [mcp_tool_info("test", "echo")]
        desc = '<MCPDescribe server="test" tool="echo">\n<description>\nEcho.</description>\n</MCPDescribe>'
        s.store_tool_result("MCP", [{"action": "describe", "server": "test", "tool": "echo"}], desc)
        ctx = n.ContextManager(s)

        ctx.prune_tool_records([])

        assert s.tool_records == []

    def test_prune_drops_non_mcp_records(self):
        """prune_tool_records drops records not referenced in messages."""
        s = n.Session(cwd="/tmp")
        s.store_tool_result("Find", [], "results from find")
        s.store_tool_result("Read", [], "read output")
        ctx = n.ContextManager(s)

        ctx.prune_tool_records([])

        assert len(s.tool_records) == 0


# ---------------------------------------------------------------------------
# CommandLoop — /mcp tools NAME
# ---------------------------------------------------------------------------


class TestMCPCommandsByName:
    def test_mcp_tools_specific_server(self, monkeypatch):
        "/mcp tools test filters to one server."""
        raw = {"mcp": {"a": {"url": "http://a/mcp"}, "b": {"url": "http://b/mcp"}}}
        s = n.Session(cwd="/tmp", config=n.Config.from_dict(raw))

        async def fake_list(url, headers):
            class T:
                name = "tool"
                description = "A test tool"
                inputSchema = {"type": "object", "properties": {}, "required": []}
                annotations = None
            return [T()]

        monkeypatch.setattr(s.mcp, "_list_tools", fake_list)
        s.mcp.discover_enabled()

        loop = n.CommandLoop(n.Agent(s), input_fn=lambda _: "", output_fn=lambda _: None)
        result = loop.mcp_command("tools a")

        assert "### `a`" in result
        assert "### `b`" not in result
        assert "tool" in result


# ---------------------------------------------------------------------------
# MCPManager — discover_server with nonexistent server
# ---------------------------------------------------------------------------


class TestMCPDiscoverServer:
    def test_discover_nonexistent_server_sets_error(self):
        """discover_server for a server not in config sets server_errors."""
        s = n.Session(cwd="/tmp", config=n.Config.from_dict(mcp_cfg()))
        s.mcp.discover_server("nonexistent")
        assert "nonexistent" in s.mcp.server_errors
        assert "not found" in s.mcp.server_errors["nonexistent"]

    def test_discover_nonexistent_server_removes_tools(self):
        """discover_server for a server not in config clears its stale tools."""
        s = n.Session(cwd="/tmp", config=n.Config.from_dict(mcp_cfg()))
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
        s = n.Session(cwd="/tmp", config=n.Config.from_dict(mcp_cfg()))

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
        s.mcp.discover_enabled()
        return s

    def test_action_schema_includes_resource_actions(self):
        schema = n.MCPTool.params_schema()
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
        s = n.Session(cwd="/tmp", config=n.Config.from_dict(mcp_cfg()))

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
        s.mcp.discover_enabled()
        assert s.mcp.tools["test"]  # tool discovery still succeeded
        assert s.mcp.resources["test"] == []
        assert "test" not in s.mcp.server_errors

    def test_read_resource_dispatch(self, monkeypatch):
        s = self._server_with_resources(monkeypatch, [_fake_resource(uri="docs://a.md")])

        async def fake_read(config, headers, uri):
            return [SimpleNamespace(text="hello " + uri, blob=None)]

        monkeypatch.setattr(s.mcp, "_read_resource", fake_read)
        out = n.MCPTool(s, [{"action": "read_resource", "server": "test", "uri": "docs://a.md"}]).call()
        assert '<MCPResource server="test" uri="docs://a.md">' in out
        assert "hello docs://a.md" in out

    def test_read_resource_requires_uri(self, monkeypatch):
        s = self._server_with_resources(monkeypatch, [])
        with pytest.raises(n.ToolError, match="requires a uri"):
            n.MCPTool(s, [{"action": "read_resource", "server": "test"}]).call()

    def test_read_resource_is_read_only(self, monkeypatch):
        s = self._server_with_resources(monkeypatch, [])
        tool = n.MCPTool(s, [{"action": "read_resource", "server": "test", "uri": "docs://a.md"}])
        assert tool.needs_confirmation() is False

    def test_list_resources_dispatch(self, monkeypatch):
        s = self._server_with_resources(monkeypatch, [_fake_resource(uri="docs://a.md", description="Doc A")])
        out = n.MCPTool(s, [{"action": "list_resources", "server": "test"}]).call()
        assert "docs://a.md" in out and "Doc A" in out

    def test_normalize_resource_blob(self):
        mgr = n.MCPManager.__new__(n.MCPManager)
        out = mgr.normalize_resource([SimpleNamespace(text=None, blob=b"\x00\x01", mimeType="application/pdf")])
        assert "binary" in out and "application/pdf" in out

    def test_action_defaults_to_call_when_omitted(self):
        assert n.MCPTool.resolved_action({"tool": "x", "server": "s"}) == "call"
        assert n.MCPTool.resolved_action({"arguments": {}, "server": "s"}) == "call"
        assert n.MCPTool.resolved_action({"server": "s"}) == ""
        assert n.MCPTool.resolved_action({"action": "describe", "server": "s"}) == "describe"

    def test_omitted_action_invokes_tool(self, monkeypatch):
        s = self._server_with_resources(monkeypatch, [])

        async def fake_call(config, headers, name, arguments):
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="ok " + name)])

        monkeypatch.setattr(s.mcp, "_call_tool", fake_call)
        out = n.MCPTool(s, [{"server": "test", "tool": "query", "arguments": {"q": 1}}]).call()
        assert "ok query" in out

    def test_unknown_action_error_is_actionable(self, monkeypatch):
        s = self._server_with_resources(monkeypatch, [])
        with pytest.raises(n.ToolError, match=r'tool=.search'):
            n.MCPTool(s, [{"action": "search", "server": "test", "arguments": {}}]).call()

    def test_extract_uris_from_description(self):
        text = "See metabase://docs/cq.md for syntax. Also https://x.io/a, and (file://y.txt)."
        assert n.MCPManager._extract_uris(text) == ["metabase://docs/cq.md", "https://x.io/a", "file://y.txt"]

    def test_index_surfaces_description_uris(self, monkeypatch):
        s = n.Session(cwd="/tmp", config=n.Config.from_dict(mcp_cfg()))

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
        s.mcp.discover_enabled()
        idx = s.mcp.render_tools_index()
        # URI survives even though the description is truncated to 80 chars on the main line.
        assert "metabase://docs/construct-query.md" in idx
        assert "refs" in idx

    def test_mention_block_lists_resources(self, monkeypatch):
        s = self._server_with_resources(monkeypatch, [_fake_resource(uri="docs://a.md", description="Doc A")])
        block = s.mcp._mention_block("test", "")
        assert "docs://a.md" in block and "read_resource" in block

    def test_resource_only_server_renders_in_index(self):
        """A connected server with resources but zero tools is listed (not dumped into pending)."""
        s = n.Session(cwd="/tmp", config=n.Config.from_dict(mcp_cfg()))
        s.mcp.tools["test"] = []
        s.mcp.resources["test"] = [n.MCPResourceInfo("test", "docs://guide.md", "guide", "Usage guide", "text/markdown")]
        s.mcp.discovery_status = "ready"
        idx = s.mcp.render_tools_index()
        assert "[test]" in idx
        assert "docs://guide.md" in idx
        assert "not connected" not in idx

    def test_pending_status_connected_but_empty(self):
        """A ready server with neither tools nor resources is reported as connected, not 'not connected'."""
        s = n.Session(cwd="/tmp", config=n.Config.from_dict(mcp_cfg()))
        s.mcp.tools["test"] = []
        s.mcp.discovery_status = "ready"
        assert s.mcp._pending_status("test") == "connected; no tools or resources advertised"

    def _server_with_doc_tool(self, monkeypatch, description, read_calls):
        s = n.Session(cwd="/tmp", config=n.Config.from_dict(mcp_cfg()))

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
        s.mcp.discover_enabled()
        return s

    def test_auto_read_injects_doc_on_first_call(self, monkeypatch):
        reads = []
        s = self._server_with_doc_tool(monkeypatch, "Run. See metabase://docs/cq.md for syntax.", reads)

        async def ok(config, headers, name, arguments):
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="ROWS")])

        monkeypatch.setattr(s.mcp, "_call_tool", ok)
        out1 = n.MCPTool(s, [{"action": "call", "server": "test", "tool": "query", "arguments": {}}]).call()
        assert "MCPAutoResources" in out1 and "GRAMMAR DOC" in out1 and "ROWS" in out1
        # injected once: a second call neither re-reads nor re-injects
        out2 = n.MCPTool(s, [{"action": "call", "server": "test", "tool": "query", "arguments": {}}]).call()
        assert "MCPAutoResources" not in out2
        assert reads == ["metabase://docs/cq.md"]

    def test_auto_read_attaches_doc_to_failed_call(self, monkeypatch):
        reads = []
        s = self._server_with_doc_tool(monkeypatch, "Run. See metabase://docs/cq.md for syntax.", reads)

        async def boom(config, headers, name, arguments):
            raise RuntimeError("Invalid body")

        monkeypatch.setattr(s.mcp, "_call_tool", boom)
        with pytest.raises(n.ToolError) as exc:
            n.MCPTool(s, [{"action": "call", "server": "test", "tool": "query", "arguments": {}}]).call()
        assert "Invalid body" in str(exc.value) and "GRAMMAR DOC" in str(exc.value)

    def test_auto_read_skips_web_links(self, monkeypatch):
        reads = []
        s = self._server_with_doc_tool(monkeypatch, "Run. Docs at https://web.example/guide.", reads)

        async def ok(config, headers, name, arguments):
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="ROWS")])

        monkeypatch.setattr(s.mcp, "_call_tool", ok)
        out = n.MCPTool(s, [{"action": "call", "server": "test", "tool": "query", "arguments": {}}]).call()
        assert "MCPAutoResources" not in out and reads == []


# ---------------------------------------------------------------------------
# Smoke: py_compile
# ---------------------------------------------------------------------------

def test_py_compile():
    """Module compiles without errors."""
    import py_compile
    py_compile.compile("nanocode.py", doraise=True)
