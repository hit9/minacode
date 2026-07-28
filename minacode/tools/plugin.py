"""Pluggable tools backed by external providers: MCP servers and skills."""

from __future__ import annotations

import json
from typing import ClassVar

from minacode.base import Json, ToolArgs, ToolError
from minacode.tools.base import Tool


class MCPTool(Tool):
    NAME = "MCP"
    DESCRIPTION = "Call/describe external MCP server tools, and list/read MCP resources"
    MUTATES = True

    @classmethod
    def params_schema(cls) -> Json:
        # fmt: off
        return cls.object_schema({
            "action": {"type": "string", "enum": ["call", "describe", "list_resources", "read_resource"], "description": '"call" invokes a tool; "describe" returns a tool\'s schema; "list_resources" lists a server\'s resources; "read_resource" reads one by uri'},
            "server": {"type": "string", "description": "MCP server name from config"},
            "tool": {"type": "string", "description": "Remote MCP tool name (required for call/describe)"},
            "arguments": {"type": "object", "description": "Arguments for the remote tool (required for call)"},
            "uri": {"type": "string", "description": "Resource URI (required for read_resource), e.g. scheme://path"},
        }, ["action", "server"])
        # fmt: on

    def payload(self) -> Json:
        return self.single_dict_arg("MCP requires named fields")

    ACTIONS: ClassVar[tuple[str, ...]] = ("call", "describe", "list_resources", "read_resource")

    @classmethod
    def resolved_action(cls, payload: Json) -> str:
        """Effective action; defaults to "call" when omitted but the envelope looks like an invocation."""
        action = str(payload.get("action") or "").strip()
        if action:
            return action
        if payload.get("tool") or payload.get("arguments") is not None:
            return "call"
        return action

    def needs_confirmation(self) -> bool:
        payload = self.payload()
        if self.resolved_action(payload) != "call":
            return False
        if self.session.mcp is None:
            return False
        return self.session.mcp.tool_needs_confirmation(str(payload.get("server") or ""), str(payload.get("tool") or ""))

    def short_args(self) -> list[str]:
        payload = self.payload()
        action = self.resolved_action(payload)
        server = str(payload.get("server") or "")
        tool_name = str(payload.get("tool") or "")
        target = (server + " " + str(payload.get("uri") or "")).strip() if action == "read_resource" else (server + "." + tool_name).strip(".")
        parts = [part for part in (action, target) if part]
        arguments = payload.get("arguments")
        if action == "call" and isinstance(arguments, dict) and arguments:
            parts.append(self.format_call_args(arguments))
        return parts

    @staticmethod
    def format_call_args(arguments: Json) -> str:
        rendered = ", ".join(f"{key}={Tool.compact(value, 60)}" for key, value in arguments.items())
        return "(" + rendered + ")"

    def call(self) -> str:
        payload = self.payload()
        action = self.resolved_action(payload)
        server = payload.get("server", "")
        tool_name = payload.get("tool", "")
        arguments = payload.get("arguments", {})
        if action == "call" and not isinstance(arguments, dict):
            raise ToolError("MCP arguments must be an object")

        mcp = self.session.mcp
        if mcp is None:
            raise ToolError("MCP not configured")

        if action == "describe":
            return mcp.describe_tool(server, tool_name)
        if action == "call":
            prefix = mcp.auto_read_prefix(server, tool_name)
            try:
                output = mcp.call_tool(server, tool_name, arguments)
            except ToolError as error:
                raise ToolError(f"{error}\n\n{prefix}" if prefix else str(error)) from error
            return prefix + output if prefix else output
        if action == "list_resources":
            return mcp.list_resources(server)
        if action == "read_resource":
            return mcp.read_resource(server, str(payload.get("uri") or ""))
        raise ToolError(
            f"unknown MCP action {action!r}. Valid actions: {', '.join(self.ACTIONS)}. "
            f'To invoke a remote tool named {action!r}, use action="call", tool={action!r}.'
        )


class SkillTool(Tool):
    NAME = "Skill"
    DESCRIPTION = "Load a skill's full instructions by name (skills are listed in the SKILLS section). Follow the returned steps, running any bundled scripts it references via Bash."

    # fmt: off
    @classmethod
    def params_schema(cls) -> Json:
        return cls.object_schema({"name": {"type": "string", "description": "Skill name from the SKILLS section"}}, ["name"])
    # fmt: on

    @classmethod
    def payload_args(cls, payload: Json) -> ToolArgs:
        return [payload.get("name", "")]

    def call(self) -> str:
        (name,) = self.strings(min_count=1, max_count=1)
        library = self.session.skills
        skill = library.get(name) if library else None
        if skill is None:
            available = ", ".join(item.name for item in library.all()) if library else ""
            raise ToolError(f"unknown skill {name!r}" + (f"; available: {available}" if available else "; no skills are installed"))
        assert library is not None
        return f"<Skill name={json.dumps(skill.name)}>\n{library.expand(skill)}\n</Skill>"


# fmt: off
