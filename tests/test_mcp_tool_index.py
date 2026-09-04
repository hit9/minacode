"""mcp tool index (split from tests/test_mcp_tools.py)."""
from typing import ClassVar

from mcp_harness import mcp_cfg, mcp_tool_info, session
from test_mcp_tools import _index_session

from wizolt.config import (
    Config,
)
from wizolt.mcp import MCPManager, MCPToolInfo
from wizolt.session import Session, bootstrap_features
from wizolt.tools import Tool


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

    async def test_index_contains_mcp_tools_header(self, monkeypatch):
        """render_tools_index includes the MCP TOOLS header."""
        raw = mcp_cfg()
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        bootstrap_features(s)

        class FakeTool:
            name = "echo"
            description = "Echo"
            inputSchema: ClassVar[dict] = {"type": "object", "properties": {"t": {"type": "string"}}, "required": ["t"]}
            annotations = None

        async def fake_list(url, headers):
            return [FakeTool()]

        monkeypatch.setattr(s.mcp, "_list_tools", fake_list)
        await s.mcp.discover_auto_async()

        idx = s.mcp.render_tools_index()
        assert "--- MCP TOOLS ---" in idx
        assert "[test]" in idx

    async def test_legacy_enabled_does_not_connect_server(self, monkeypatch):
        raw = {"mcp": {"test": {"url": "http://x/mcp", "enabled": False}}}
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        bootstrap_features(s)
        await s.mcp.discover_auto_async()
        idx = s.mcp.render_tools_index()
        assert idx == ""

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
        bootstrap_features(s)
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
        bootstrap_features(s)

        assert s.mcp.render_tools_index() == ""
        assert "MCP" not in {schema["function"]["name"] for schema in Tool.resolved_schemas(s)}

        s.mcp.tools["test"] = [mcp_tool_info("test", "echo")]
        s.mcp.resources["test"] = []

        assert "[test]" in s.mcp.render_tools_index()
        assert "MCP" in {schema["function"]["name"] for schema in Tool.resolved_schemas(s)}

    def test_disconnect_removes_server_from_model_context(self):
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg()))
        bootstrap_features(s)
        s.mcp.tools["test"] = [mcp_tool_info("test", "echo")]
        s.mcp.resources["test"] = []

        result = s.mcp.disconnect_server("test")

        assert result == "MCP server disconnected: test"
        assert s.mcp.render_tools_index() == ""
        assert "MCP" not in {schema["function"]["name"] for schema in Tool.resolved_schemas(s)}

class TestToolIndexTruncation:
    async def test_index_truncation_long_block(self, monkeypatch):
        """Long index block is bounded by INDEX_TOTAL_LIMIT."""
        raw = mcp_cfg()
        s = Session(cwd="/tmp", config=Config.from_dict(raw))
        bootstrap_features(s)

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
        await s.mcp.discover_auto_async()

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
