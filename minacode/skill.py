"""minacode skills: Markdown instruction packs loaded on demand."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from minacode.session import Session


@dataclass
class Skill:
    name: str
    description: str
    body: str
    dir: str
    source: str  # "builtin", "user", or "project"


class SkillLibrary:
    """Skills discovered from builtin, user, and project skill directories.

    Each skill is a Markdown file with `name`/`description` frontmatter; the index (name + description)
    rides the cache-stable prefix so the model knows what exists, and the full body is pulled into the
    conversation only when the model calls Skill(name) or the user references it with `$name` or
    `@skill:name`."""

    FRONTMATTER = re.compile(r"^---\n(.*?)\n---\n?(.*)$", re.DOTALL)
    META_LINE = re.compile(r"^([A-Za-z0-9_-]+):[ \t]*(.*)$", re.MULTILINE)
    # Both legacy $name and canonical @skill:name forms name a skill; $ keeps its existing
    # negative lookbehind, @skill: the same word-boundary rule as every other mention.
    MENTION_PATTERN = re.compile(r"(?<![A-Za-z0-9_])(?:\$|@skill:)([A-Za-z0-9_-]+)")

    def __init__(self, skills: dict[str, Skill]):
        self.skills = skills

    @classmethod
    def load(cls, session: Session) -> SkillLibrary:
        skills: dict[str, Skill] = {}
        # Later roots override earlier ones: projects can customize user skills, and users can
        # customize the read-only skills shipped with minacode.
        builtin_skills = os.path.join(os.path.dirname(__file__), "builtin_skills")
        project_skills = os.path.join(session.cwd, ".minacode", "skills")
        if not os.path.isdir(project_skills):
            legacy_skills = os.path.join(session.cwd, ".nanocode", "skills")
            if os.path.isdir(legacy_skills):
                project_skills = legacy_skills
        for root, source in (
            (builtin_skills, "builtin"),
            (session.data_path("skills"), "user"),
            (project_skills, "project"),
        ):
            if not os.path.isdir(root):
                continue
            for entry in sorted(os.listdir(root)):
                skill = cls.parse(os.path.join(root, entry, "SKILL.md"), entry, source)
                if skill is not None:
                    skills[skill.name] = skill
        return cls(skills)

    @classmethod
    def parse(cls, path: str, folder: str, source: str) -> Skill | None:
        try:
            with open(path, encoding="utf-8") as handle:
                text = handle.read()
        except OSError:
            return None
        # Normalize BOM and CRLF/CR so the frontmatter regex (which keys on "\n") matches files
        # authored on any platform; we only read two simple scalars, so this stays regex-light.
        text = text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
        match = cls.FRONTMATTER.match(text)
        meta, body = (match.group(1), match.group(2)) if match else ("", text)
        fields = {key: cls.scalar(value) for key, value in cls.META_LINE.findall(meta)}
        name = fields.get("name") or folder.strip()
        if not name:
            return None
        return Skill(name, fields.get("description", ""), body.strip(), os.path.dirname(path), source)

    @staticmethod
    def scalar(value: str) -> str:
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        return value.strip()

    def all(self) -> list[Skill]:
        return sorted(self.skills.values(), key=lambda skill: skill.name)

    def get(self, name: str) -> Skill | None:
        if name in self.skills:
            return self.skills[name]
        resolved = {key.lower(): key for key in self.skills}.get(name.lower())
        return self.skills.get(resolved) if resolved else None

    def expand(self, skill: Skill) -> str:
        return skill.body.replace("{skill_dir}", skill.dir).replace("${SKILL_DIR}", skill.dir)

    def index(self) -> str:
        if not self.skills:
            return ""
        rows = [f"- {skill.name}: {skill.description or '(no description)'}" for skill in self.all()]
        return "\n".join(["--- SKILLS ---", "Use Skill(name) to load a skill's full instructions when its description fits the task.", "", *rows])

    def resolve_mentions(self, text: str) -> str:
        seen: set[str] = set()
        blocks: list[str] = []
        for raw in self.MENTION_PATTERN.findall(text):
            skill = self.get(raw)
            if skill is None or skill.name in seen:
                continue
            seen.add(skill.name)
            blocks.append(f"[{skill.name}] {skill.description}\n{self.expand(skill)}")
        if not blocks:
            return ""
        header = ["--- SKILL MENTIONS ---", "The user explicitly referenced these skills; follow their instructions unless clearly irrelevant.", ""]
        return "\n".join(header + blocks).strip()
