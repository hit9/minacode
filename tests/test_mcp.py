"""Tests for nanocode MCP client integration."""
import json
import os
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
        assert len(line.split(" - ")[-1]) <= 83

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
        assert "enabled" in status
        assert "connected" in status
        assert "tools=1" in status
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
        assert "[test]" in listing
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
        assert "[a]" in listing
        assert "[b]" not in listing

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
        """MCP(action='call') → confirmation needed by default."""
        raw = mcp_cfg()
        s = n.Session(cwd="/tmp", config=n.Config.from_dict(raw))
        payload = {"action": "call", "server": "test", "tool": "echo", "arguments": {"text": "hi"}}
        tool = n.MCPTool(s, [payload])
        # No annotations → default destructiveHint=False → no confirmation
        # Actually need to check: needs_confirmation uses tool_needs_confirmation
        # which returns False if no info found or no destructiveHint
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
        """Discovering status → spinner + 'MCP'."""
        s = n.Session(cwd="/tmp")
        s.mcp.discovery_status = "discovering"
        bar = n.StatusBar(s)
        monkeypatch.setattr(n.time, "monotonic", lambda: 0.0)
        status = bar.mcp_status()
        # First spinner char
        assert status == "mcp" + bar.INDEX_SPINNER[0]

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

    def test_mcp_tool_details_empty(self):
        """No describe records → empty."""
        s = n.Session(cwd="/tmp")
        ctx = n.ContextManager(s)
        assert ctx.mcp_tool_details() == ""

    def test_mcp_tool_details_built_from_records(self):
        """Details rebuilt from tool_records."""
        s = n.Session(cwd="/tmp")
        s.mcp.tools["test"] = [mcp_tool_info("test", "echo")]
        desc = '<MCPDescribe server="test" tool="echo">\n<description>\nEcho input back.</description>\n<arguments>\n- text required string: Input.</arguments>\n</MCPDescribe>'
        s.store_tool_result("MCP", [{"action": "describe", "server": "test", "tool": "echo", "arguments": {}}], desc)

        ctx = n.ContextManager(s)
        details = ctx.mcp_tool_details()
        assert "--- MCP TOOL DETAILS ---" in details
        assert "test.echo" in details

    def test_mcp_tool_details_latest_wins(self):
        """Later describe for same server.tool overrides earlier."""
        s = n.Session(cwd="/tmp")
        s.mcp.tools["test"] = [mcp_tool_info("test", "echo")]

        old = '<MCPDescribe server="test" tool="echo">\n<description>\nOld</description>\n</MCPDescribe>'
        new = '<MCPDescribe server="test" tool="echo">\n<description>\nNew version</description>\n</MCPDescribe>'
        s.store_tool_result("MCP", [{"action": "describe", "server": "test", "tool": "echo"}], old)
        s.store_tool_result("MCP", [{"action": "describe", "server": "test", "tool": "echo"}], new)

        ctx = n.ContextManager(s)
        details = ctx.mcp_tool_details()
        assert "New version" in details
        assert "Old" not in details

    def test_mcp_in_context_order(self):
        """MCP TOOLS appears after ENVIRONMENT, MCP DETAILS before FILE STATE."""
        s = n.Session(cwd="/tmp")
        s.mcp.tools["test"] = [mcp_tool_info("test", "echo")]
        desc = '<MCPDescribe server="test" tool="echo">\n<description>\nOK</description>\n</MCPDescribe>'
        s.store_tool_result("MCP", [{"action": "describe", "server": "test", "tool": "echo"}], desc)

        ctx = n.ContextManager(s)
        msgs = ctx.model_messages("sys")
        texts = [m["content"] for m in msgs if m.get("role") == "user"]

        env_idx = next(i for i, t in enumerate(texts) if t.startswith("--- Environment ---"))
        mcp_tools_idx = next(i for i, t in enumerate(texts) if t.startswith("--- MCP TOOLS ---"))
        mcp_detail_idx = next(i for i, t in enumerate(texts) if t.startswith("--- MCP TOOL DETAILS ---"))
        file_state_idx = next(i for i, t in enumerate(texts) if t.startswith("--- FILE STATE ---"))

        assert env_idx < mcp_tools_idx
        assert mcp_detail_idx < file_state_idx
        assert mcp_tools_idx < mcp_detail_idx


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
        assert "tools=1" in result

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
        assert "[test]" in result
        assert "echo" in result

    def test_mcp_refresh_invokes_discovery(self, monkeypatch):
        """/mcp refresh calls discover_enabled."""
        raw = mcp_cfg()
        s = n.Session(cwd="/tmp", config=n.Config.from_dict(raw))
        async def fake_list(url, headers):
            return []
        monkeypatch.setattr(s.mcp, "_list_tools", fake_list)

        loop = n.CommandLoop(n.Agent(s), input_fn=lambda _: "", output_fn=lambda _: None)
        result = loop.mcp_command("refresh")
        assert s.mcp._discovered is True

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


# ---------------------------------------------------------------------------
# render_tools_index truncation
# ---------------------------------------------------------------------------

class TestToolIndexTruncation:
    def test_index_truncation_long_block(self, monkeypatch):
        """Long index block is truncated at 4000 chars."""
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
        assert len(idx) <= 4100
        if "truncated" not in idx:
            # Index fits within limit
            assert True

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


# ---------------------------------------------------------------------------
# ContextManager — prune_tool_records preserves MCP describe records
# ---------------------------------------------------------------------------


class TestMCPPruning:
    def test_prune_preserves_mcp_describe_records(self):
        """prune_tool_records retains MCP describe records referenced in active_mcp_tool_details."""
        s = n.Session(cwd="/tmp")
        s.mcp.tools["test"] = [mcp_tool_info("test", "echo")]
        desc = '<MCPDescribe server="test" tool="echo">\n<description>\nEcho back.</description>\n<arguments>\n- text string: Input.</arguments>\n</MCPDescribe>'
        s.store_tool_result("MCP", [{"action": "describe", "server": "test", "tool": "echo"}], desc)
        ctx = n.ContextManager(s)

        key = s.tool_records[0].key
        ctx.prune_tool_records([])

        assert len(s.tool_records) == 1
        assert s.tool_records[0].key == key

    def test_prune_drops_non_mcp_records(self):
        """prune_tool_records drops non-MCP records not referenced in messages."""
        s = n.Session(cwd="/tmp")
        s.store_tool_result("Find", [], "results from find")
        s.store_tool_result("Read", [], "read output")
        ctx = n.ContextManager(s)

        ctx.prune_tool_records([])

        assert len(s.tool_records) == 0

    def test_prune_keeps_mcp_describe_and_messages_referenced(self):
        """MCP describe records and message-referenced records are both kept."""
        s = n.Session(cwd="/tmp")
        s.mcp.tools["test"] = [mcp_tool_info("test", "echo")]
        desc = '<MCPDescribe server="test" tool="echo">\n<description>\nEcho.</description>\n</MCPDescribe>'
        s.store_tool_result("MCP", [{"action": "describe", "server": "test", "tool": "echo"}], desc)
        s.store_tool_result("Read", [], "some read")
        ctx = n.ContextManager(s)

        # The Read record is referenced in messages
        read_key = s.tool_records[1].key
        keep_messages = [{"role": "user", "content": f"see {read_key}"}]
        ctx.prune_tool_records(keep_messages)

        keys = {r.key for r in s.tool_records}
        assert len(s.tool_records) == 2
        assert read_key in keys
        assert s.tool_records[0].name == "MCP"


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

        assert "[a]" in result
        assert "[b]" not in result
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
# Smoke: py_compile
# ---------------------------------------------------------------------------

def test_py_compile():
    """Module compiles without errors."""
    import py_compile
    py_compile.compile("nanocode.py", doraise=True)
