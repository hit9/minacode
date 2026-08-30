"""wizolt tools: the built-in tool set exposed to the model."""

from __future__ import annotations

from wizolt.base import ToolArgs, drop_nulls
from wizolt.tools.ask import AskSpec, AskTool
from wizolt.tools.base import Tool
from wizolt.tools.delegate import WORKER_TOOLS, DelegateTool
from wizolt.tools.files import Edit, EditApplyResult, EditTool, ReadTool, ViewImageTool
from wizolt.tools.mcp import MCPTool
from wizolt.tools.memory import NextHintsTool, NoteTool, RecallContextTool, RecallTool
from wizolt.tools.search import CodeIndex, InspectCodeTool, SearchTool
from wizolt.tools.shell import BashTool, JobTool
from wizolt.tools.skill import SkillTool
from wizolt.tools.toolscript import ToolScript

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
    "tool_payload",
]
