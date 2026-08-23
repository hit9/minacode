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

from minacode.base import ToolCall, ToolError
from minacode.config import (
    Config,
)
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




# ---------------------------------------------------------------------------
# MCPManager render_tools_index budget degradation (regression: a verbose
# server must never hide later servers from the model)
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# MCPManager render_server_status & render_tool_listing
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# MCPManager — normalize_result
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# MCPTool short_args
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# StatusBar mcp_status
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# MCPManager — describe_tool
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# MCPManager — call_tool
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# CommandLoop — /mcp commands
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# MCPManager — call_tool success path
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# ContextManager — prune_tool_records preserves MCP describe records
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# User scenarios — public commands through model-visible context
# ---------------------------------------------------------------------------


