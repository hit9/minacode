"""MCP tools as the agent sees them: the tool index and its budget, confirmation, context
blocks, calling, result normalization, and resources."""

import asyncio
from types import SimpleNamespace
from typing import ClassVar

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
from mcp_harness import _fake_resource, mcp_cfg, mcp_tool_info, session

from minacode.base import Config, ToolCall, ToolError
from minacode.context import ContextManager
from minacode.mcp import MCPManager, MCPResourceInfo, MCPToolInfo
from minacode.runner import ToolRunner
from minacode.session import Session
from minacode.tools import MCPTool, Tool


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
            inputSchema: ClassVar[dict] = {"type": "object", "properties": {"t": {"type": "string"}}, "required": ["t"]}
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


class TestToolIndexTruncation:
    def test_index_truncation_long_block(self, monkeypatch):
        """Long index block is bounded by INDEX_TOTAL_LIMIT."""
        raw = mcp_cfg()
        s = Session(cwd="/tmp", config=Config.from_dict(raw))

        # Create many tools to exceed budget
        class FakeTool:
            name = "tool"
            description = "Desc"
            inputSchema: ClassVar[dict] = {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}
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
            inputSchema: ClassVar[dict] = {"type": "object", "properties": {"t": {"type": "string"}}, "required": ["t"]}
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
        """MCP TOOLS appears after Environment; no repeated Memory/details block."""
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg()))
        s.mcp.tools["test"] = [mcp_tool_info("test", "echo")]

        ctx = ContextManager(s)
        msgs = ctx.model_messages("sys")
        texts = [m["content"] for m in msgs if m.get("role") == "user"]

        env_idx = next(i for i, t in enumerate(texts) if t.startswith("--- Environment ---"))
        mcp_tools_idx = next(i for i, t in enumerate(texts) if t.startswith("--- MCP TOOLS ---"))
        assert env_idx < mcp_tools_idx
        assert not any(t.startswith("--- Memory ---") for t in texts)
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


class TestMCPResources:
    def _server_with_resources(self, monkeypatch, resources):
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg()))

        class FakeTool:
            name = "query"
            description = "Run a program."
            inputSchema: ClassVar[dict] = {"type": "object", "properties": {"operations": {"type": "array"}}, "required": ["operations"]}
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
            inputSchema: ClassVar[dict] = {"type": "object", "properties": {}}
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
            inputSchema: ClassVar[dict] = {"type": "object", "properties": {}}
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
            inputSchema: ClassVar[dict] = {"type": "object", "properties": {}}
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
