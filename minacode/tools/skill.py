"""Skill tool: loading an installed skill's full instructions on demand."""

from __future__ import annotations

import json

from minacode.base import Json, ToolArgs, ToolError
from minacode.tools.base import Tool


class SkillTool(Tool):
    NAME = "Skill"
    DESCRIPTION = "Load a skill's full instructions by name (skills are listed in the SKILLS section). Follow the returned steps, running any bundled scripts it references via Bash."

    @classmethod
    def params_schema(cls) -> Json:
        return cls.object_schema({"name": {"type": "string", "description": "Skill name from the SKILLS section"}}, ["name"])

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
