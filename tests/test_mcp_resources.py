"""mcp resources (split from tests/test_mcp_tools.py)."""
from types import SimpleNamespace
from typing import ClassVar

import pytest
from mcp_harness import _fake_resource, mcp_cfg, session

from wizolt.base import ToolError
from wizolt.config import (
    Config,
)
from wizolt.mcp import MCPManager, MCPResourceInfo
from wizolt.session import Session, bootstrap_features
from wizolt.tools import MCPTool


class TestMCPResources:
    async def _server_with_resources(self, monkeypatch, resources):
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg()))
        bootstrap_features(s)

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
        await s.mcp.discover_auto()
        return s

    def test_action_schema_includes_resource_actions(self):
        schema = MCPTool.params_schema()
        assert {"call", "describe", "list_resources", "read_resource"} <= set(schema["properties"]["action"]["enum"])
        assert "uri" in schema["properties"]
        assert schema["required"] == ["action", "server"]

    async def test_discovery_populates_resources(self, monkeypatch):
        s = await self._server_with_resources(monkeypatch, [_fake_resource(uri="metabase://docs/construct-query.md")])
        assert [r.uri for r in s.mcp.resources["test"]] == ["metabase://docs/construct-query.md"]

    async def test_index_lists_resources(self, monkeypatch):
        s = await self._server_with_resources(monkeypatch, [_fake_resource(uri="metabase://docs/construct-query.md")])
        idx = s.mcp.render_tools_index()
        assert "metabase://docs/construct-query.md" in idx
        assert "read_resource" in idx

    async def test_resources_best_effort_on_failure(self, monkeypatch):
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg()))
        bootstrap_features(s)

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
        await s.mcp.discover_auto()
        assert s.mcp.tools["test"]  # tool discovery still succeeded
        assert s.mcp.resources["test"] == []
        assert "test" not in s.mcp.server_errors

    async def test_read_resource_dispatch(self, monkeypatch):
        s = await self._server_with_resources(monkeypatch, [_fake_resource(uri="docs://a.md")])

        async def fake_read(config, headers, uri):
            return [SimpleNamespace(text="hello " + uri, blob=None)]

        monkeypatch.setattr(s.mcp, "_read_resource", fake_read)
        out = await MCPTool(s, [{"action": "read_resource", "server": "test", "uri": "docs://a.md"}]).call()
        assert '<MCPResource server="test" uri="docs://a.md">' in out
        assert "hello docs://a.md" in out

    async def test_read_resource_requires_uri(self, monkeypatch):
        s = await self._server_with_resources(monkeypatch, [])
        with pytest.raises(ToolError, match="requires a uri"):
            await MCPTool(s, [{"action": "read_resource", "server": "test"}]).call()

    async def test_read_resource_is_read_only(self, monkeypatch):
        s = await self._server_with_resources(monkeypatch, [])
        tool = MCPTool(s, [{"action": "read_resource", "server": "test", "uri": "docs://a.md"}])
        assert tool.needs_confirmation() is False

    async def test_list_resources_dispatch(self, monkeypatch):
        s = await self._server_with_resources(monkeypatch, [_fake_resource(uri="docs://a.md", description="Doc A")])
        out = await MCPTool(s, [{"action": "list_resources", "server": "test"}]).call()
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

    async def test_omitted_action_invokes_tool(self, monkeypatch):
        s = await self._server_with_resources(monkeypatch, [])

        async def fake_call(config, headers, name, arguments):
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="ok " + name)])

        monkeypatch.setattr(s.mcp, "_call_tool", fake_call)
        out = await MCPTool(s, [{"server": "test", "tool": "query", "arguments": {"q": 1}}]).call()
        assert "ok query" in out

    async def test_unknown_action_error_is_actionable(self, monkeypatch):
        s = await self._server_with_resources(monkeypatch, [])
        with pytest.raises(ToolError, match=r"tool=.search"):
            await MCPTool(s, [{"action": "search", "server": "test", "arguments": {}}]).call()

    def test_extract_uris_from_description(self):
        text = "See metabase://docs/cq.md for syntax. Also https://x.io/a, and (file://y.txt)."
        assert MCPManager._extract_uris(text) == ["metabase://docs/cq.md", "https://x.io/a", "file://y.txt"]

    async def test_index_surfaces_description_uris(self, monkeypatch):
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg()))
        bootstrap_features(s)

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
        await s.mcp.discover_auto()
        idx = s.mcp.render_tools_index()
        # URI survives even though the description is truncated to 80 chars on the main line.
        assert "metabase://docs/construct-query.md" in idx
        assert "refs" in idx

    async def test_mention_block_lists_resources(self, monkeypatch):
        s = await self._server_with_resources(monkeypatch, [_fake_resource(uri="docs://a.md", description="Doc A")])
        block = await s.mcp._mention_block("test", "")
        assert "docs://a.md" in block and "read_resource" in block

    async def test_mention_block_lists_resources_without_tools(self):
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg()))
        bootstrap_features(s)
        s.mcp.tools["test"] = []
        s.mcp.resources["test"] = [MCPResourceInfo("test", "docs://guide.md", "guide", "Usage guide", "text/markdown")]

        block = await s.mcp._mention_block("test", "")

        assert "docs://guide.md" in block
        assert "no tools or resources" not in block

    def test_resource_only_server_renders_in_index(self):
        """A connected server with resources but zero tools is listed (not dumped into pending)."""
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg()))
        bootstrap_features(s)
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
        bootstrap_features(s)
        s.mcp.tools["test"] = []
        s.mcp.discovery_status = "ready"
        assert s.mcp._pending_status("test") == "connected; no tools or resources advertised"

    async def _server_with_doc_tool(self, monkeypatch, description, read_calls):
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg()))
        bootstrap_features(s)

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
        await s.mcp.discover_auto()
        return s

    async def test_auto_read_injects_doc_on_first_call(self, monkeypatch):
        reads = []
        s = await self._server_with_doc_tool(monkeypatch, "Run. See metabase://docs/cq.md for syntax.", reads)

        async def ok(config, headers, name, arguments):
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="ROWS")])

        monkeypatch.setattr(s.mcp, "_call_tool", ok)
        out1 = await MCPTool(s, [{"action": "call", "server": "test", "tool": "query", "arguments": {}}]).call()
        assert "MCPAutoResources" in out1 and "GRAMMAR DOC" in out1 and "ROWS" in out1
        # injected once: a second call neither re-reads nor re-injects
        out2 = await MCPTool(s, [{"action": "call", "server": "test", "tool": "query", "arguments": {}}]).call()
        assert "MCPAutoResources" not in out2
        assert reads == ["metabase://docs/cq.md"]

    async def test_auto_read_attaches_doc_to_failed_call(self, monkeypatch):
        reads = []
        s = await self._server_with_doc_tool(monkeypatch, "Run. See metabase://docs/cq.md for syntax.", reads)

        async def boom(config, headers, name, arguments):
            raise RuntimeError("Invalid body")

        monkeypatch.setattr(s.mcp, "_call_tool", boom)
        with pytest.raises(ToolError) as exc:
            await MCPTool(s, [{"action": "call", "server": "test", "tool": "query", "arguments": {}}]).call()
        assert "Invalid body" in str(exc.value) and "GRAMMAR DOC" in str(exc.value)

    async def test_auto_read_skips_web_links(self, monkeypatch):
        reads = []
        s = await self._server_with_doc_tool(monkeypatch, "Run. Docs at https://web.example/guide.", reads)

        async def ok(config, headers, name, arguments):
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="ROWS")])

        monkeypatch.setattr(s.mcp, "_call_tool", ok)
        out = await MCPTool(s, [{"action": "call", "server": "test", "tool": "query", "arguments": {}}]).call()
        assert "MCPAutoResources" not in out and reads == []

class TestToolOutputSchemaCapture:
    def test_output_schema_is_captured_under_either_spelling(self):
        """The wire field is camelCase and SDK models expose it that way; a hand-built server
        object may carry the Python spelling instead."""
        shape = {"type": "object", "properties": {"ok": {"type": "boolean"}}}

        camel = SimpleNamespace(name="a", description="", inputSchema={}, outputSchema=shape, annotations=None)
        snake = SimpleNamespace(name="b", description="", inputSchema={}, output_schema=shape, annotations=None)

        assert MCPManager.tool_output_schema(camel) == shape
        assert MCPManager.tool_output_schema(snake) == shape

    def test_a_tool_that_declares_nothing_captures_nothing(self):
        bare = SimpleNamespace(name="c", description="", inputSchema={}, annotations=None)

        assert MCPManager.tool_output_schema(bare) == {}
        assert MCPManager.tool_output_schema(SimpleNamespace(outputSchema=None)) == {}
        assert MCPManager.tool_output_schema(SimpleNamespace(outputSchema="not a schema")) == {}

    def test_discovery_carries_the_schema_into_the_cached_tool_info(self, tmp_path):
        s = session(tmp_path)
        shape = {"type": "object", "properties": {"total": {"type": "integer"}}}
        tools = [SimpleNamespace(name="echo", description="d", inputSchema={}, outputSchema=shape, annotations=None)]

        (info,) = s.mcp._tools_info("test", tools)

        assert info.output_schema == shape
