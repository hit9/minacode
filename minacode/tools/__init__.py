"""minacode tools: the built-in tool set exposed to the model."""

from __future__ import annotations

from minacode.base import ToolArgs
from minacode.tools.ask import AskSpec, AskTool
from minacode.tools.base import Tool
from minacode.tools.delegate import WORKER_TOOLS, DelegateTool
from minacode.tools.files import Edit, EditApplyResult, EditTool, ReadTool, ViewImageTool
from minacode.tools.mcp import MCPTool
from minacode.tools.memory import NextHintsTool, NoteTool, RecallContextTool, RecallTool
from minacode.tools.search import CodeIndex, InspectCodeTool, SearchTool
from minacode.tools.shell import BashTool, JobTool
from minacode.tools.skill import SkillTool
from minacode.tools.toolscript import ToolScript

TOOLS: tuple[type[Tool], ...] = (
    MCPTool,
    ToolScript,
    SkillTool,
    ReadTool,
    ViewImageTool,
    InspectCodeTool,
    SearchTool,
    EditTool,
    BashTool,
    JobTool,
    RecallTool,
    RecallContextTool,
    NoteTool,
    NextHintsTool,
    AskTool,
    DelegateTool,
)
TOOL_REGISTRY: dict[str, type[Tool]] = {tool.NAME: tool for tool in TOOLS}


def drop_nulls(value: object) -> object:
    """Recursively strip explicit nulls. Strict schemas express optional params as nullable, so the
    model may send null for an omitted argument, and in every minacode tool null means "absent"."""
    if isinstance(value, dict):
        return {key: drop_nulls(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [drop_nulls(item) for item in value]
    return value


def tool_payload(name: str, payload: object) -> ToolArgs:
    """Shape a raw provider argument payload into the tool's canonical positional args."""
    if isinstance(payload, dict) and (tool := TOOL_REGISTRY.get(name)):
        cleaned = drop_nulls(payload)
        assert isinstance(cleaned, dict)
        return tool.payload_args(cleaned)
    return [payload]


__all__ = [
    "TOOLS",
    "TOOL_REGISTRY",
    "WORKER_TOOLS",
    "AskSpec",
    "AskTool",
    "BashTool",
    "CodeIndex",
    "DelegateTool",
    "Edit",
    "EditApplyResult",
    "EditTool",
    "InspectCodeTool",
    "JobTool",
    "MCPTool",
    "NextHintsTool",
    "NoteTool",
    "ReadTool",
    "RecallContextTool",
    "RecallTool",
    "SearchTool",
    "SkillTool",
    "Tool",
    "ToolScript",
    "ViewImageTool",
    "drop_nulls",
    "tool_payload",
]
