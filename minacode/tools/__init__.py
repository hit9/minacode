"""minacode tools: the built-in tool set exposed to the model."""

from __future__ import annotations

from minacode.tools.ask import AskSpec, AskTool
from minacode.tools.base import Tool
from minacode.tools.files import Edit, EditApplyResult, EditTool, ReadTool, ViewImageTool
from minacode.tools.memory import NoteTool, RecallContextTool, RecallTool
from minacode.tools.plugin import MCPTool, SkillTool
from minacode.tools.search import CodeIndex, InspectCodeTool, SearchTool
from minacode.tools.shell import BashTool, JobTool

# fmt: off
TOOLS: tuple[type[Tool], ...] = (
    MCPTool, SkillTool, ReadTool, ViewImageTool, InspectCodeTool, SearchTool, EditTool,
    BashTool, JobTool, RecallTool, RecallContextTool, NoteTool, AskTool,
)
# fmt: on
TOOL_REGISTRY: dict[str, type[Tool]] = {tool.NAME: tool for tool in TOOLS}

__all__ = [
    "TOOLS",
    "TOOL_REGISTRY",
    "AskSpec",
    "AskTool",
    "BashTool",
    "CodeIndex",
    "Edit",
    "EditApplyResult",
    "EditTool",
    "InspectCodeTool",
    "JobTool",
    "MCPTool",
    "NoteTool",
    "ReadTool",
    "RecallContextTool",
    "RecallTool",
    "SearchTool",
    "SkillTool",
    "Tool",
    "ViewImageTool",
]
