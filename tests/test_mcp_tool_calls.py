"""mcp tool calls (split from tests/test_mcp_tools.py)."""
from typing import ClassVar

import pytest
from mcp_harness import mcp_cfg, mcp_tool_info

from minacode.base import ToolCall, ToolError
from minacode.config import (
    Config,
)
from minacode.context import ContextManager
from minacode.runner import ToolRunner
from minacode.session import Session
from minacode.tools import MCPTool


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

    def test_describe_renders_a_declared_result_shape(self):
        """Without this the only way to learn what a tool returns is to call it, so every
        unfamiliar tool costs an exploratory call before it can be used for real."""
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg()))
        s.mcp.tools["test"] = [
            mcp_tool_info(
                "test",
                "echo",
                output_schema={
                    "type": "object",
                    "properties": {"total": {"type": "integer", "description": "How many matched"}, "items": {"type": "array"}},
                    "required": ["total"],
                },
            )
        ]

        result = s.mcp.describe_tool("test", "echo")

        assert "<returns>" in result
        assert "- total required integer: How many matched" in result
        assert "- items optional array:" in result
        assert "<returns_schema>" in result

    def test_describe_omits_returns_when_the_server_declares_none(self):
        """Most servers declare no outputSchema; they must read exactly as they did before."""
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg()))
        s.mcp.tools["test"] = [mcp_tool_info("test", "echo")]

        result = s.mcp.describe_tool("test", "echo")

        assert "returns" not in result
        assert "<arguments>" in result and "<schema>" in result

    def test_describe_names_a_result_that_is_not_an_object(self):
        """A bare array or scalar has no properties to list, and an empty block would read as
        'returns nothing' rather than 'see the schema'."""
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg()))
        s.mcp.tools["test"] = [mcp_tool_info("test", "echo", output_schema={"type": "array", "items": {"type": "string"}})]

        result = s.mcp.describe_tool("test", "echo")

        assert "(array; see returns_schema below)" in result

    def test_describe_bounds_a_large_result_shape(self, monkeypatch):
        """Result shapes can be far larger than argument lists; the same cap applies."""
        s = Session(cwd="/tmp", config=Config.from_dict(mcp_cfg()))
        monkeypatch.setattr(s.mcp, "DESCRIBE_ARGUMENT_LIMIT", 3)
        props = {f"f{i}": {"type": "string"} for i in range(10)}
        s.mcp.tools["test"] = [mcp_tool_info("test", "echo", output_schema={"type": "object", "properties": props})]

        result = s.mcp.describe_tool("test", "echo")

        assert "- f2 optional string:" in result
        assert "- f3 optional string:" not in result
        assert "... 7 more fields omitted" in result

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
