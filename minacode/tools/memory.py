"""Memory tools: history recall, context recall, and durable notes."""

from __future__ import annotations

import json
import re
from typing import ClassVar, cast

from minacode.base import Json, ToolArgs, ToolError
from minacode.session import AgentState, HistorySegment, PlanItem
from minacode.tools.base import Tool


class RecallTool(Tool):
    NAME = "Recall"
    _KEY_RE: ClassVar[re.Pattern] = re.compile(r"tr\.\d+")
    DESCRIPTION = "Recall stored non-Recall tool results by tr.N key; ranges slice output lines to control context."
    STORES_RESULT = False

    @classmethod
    def params_schema(cls) -> Json:
        # fmt: off
        return cls.object_schema({
            "keys": {"type": "array", "items": {"type": "string", "pattern": "^tr\\.\\d+$"}, "minItems": 1, "description": 'Stored result keys to recall, e.g. ["tr.3","tr.5"]'},
            "ranges": {"type": "array", "items": cls.RANGE_SCHEMA, "minItems": 1, "description": "Optional 0-based [start,end] output-line slices to limit recalled context"},
        }, ["keys"])
        # fmt: on

    @classmethod
    def payload_args(cls, payload: Json) -> ToolArgs:
        return [{"keys": payload.get("keys", []), **({"ranges": payload["ranges"]} if "ranges" in payload else {})}]

    def call(self) -> str:
        requests = self.requests()
        if not requests:
            raise ToolError("Recall requires at least one key")
        chunks = ["<RecallToolResult>"]
        for key, ranges in requests:
            value = self.session.tool_results.get(key)
            if value is None:
                chunks.append(f"* {key}: missing")
                continue
            chunks.append(f"<Result key={json.dumps(key)}>")
            chunks.append(self.slice(value, ranges).rstrip())
            chunks.append("</Result>")
        chunks.append("</RecallToolResult>")
        return "\n".join(chunks)

    def short_args(self) -> list[str]:
        rows = []
        for key, ranges in self.requests():
            row = key
            if ranges:
                row += " " + ",".join(f"{start}:{end}" for start, end in ranges)
            rows.append(row)
        return ["; ".join(rows)]

    def requests(self) -> list[tuple[str, tuple[tuple[int, int], ...]]]:
        payload = self.single_dict_arg("Recall requires keys")
        if unexpected := sorted(set(payload) - {"keys", "ranges"}):
            raise ToolError("Recall unexpected field: " + ", ".join(unexpected))
        raw_keys = payload.get("keys")
        if not isinstance(raw_keys, list) or not raw_keys:
            raise ToolError("Recall requires keys")
        ranges = self.parse_ranges(payload)
        keys = []
        for item in raw_keys:
            key = str(item).strip()
            if not self._KEY_RE.fullmatch(key):
                raise ToolError("Recall key must look like tr.N")
            keys.append(key)
        return list(dict.fromkeys((key, ranges) for key in keys))

    def parse_ranges(self, payload: Json) -> tuple[tuple[int, int], ...]:
        raw = payload.get("ranges")
        if raw is None:
            return ()
        if not isinstance(raw, list) or not raw:
            raise ToolError("Recall ranges must be a non-empty array")
        return tuple(self.line_range(item, "Recall range") for item in raw)

    @staticmethod
    def slice(value: str, ranges: tuple[tuple[int, int], ...]) -> str:
        if not ranges:
            return value
        lines = value.splitlines()
        return "\n".join("\n".join(lines[start:end]) for start, end in ranges if end > start)


class RecallContextTool(Tool):
    NAME = "RecallContext"
    _KEY_RE: ClassVar[re.Pattern] = re.compile(r"seg\.\d+")
    DESCRIPTION = "Recall stored compacted-conversation excerpts by seg.N key, or regex-search their titles and text; query alternation A|B|C is allowed."
    DEFAULT_LIMIT = 20
    MAX_LIMIT = 100
    MAX_QUERY_LENGTH = 500
    EXAMPLE = (
        'Recall one segment. Example: {"keys":["seg.1"]}',
        'Search all segments. Example: {"query":"cache prefix|task memory","limit":10}',
    )
    STORES_RESULT = False

    @classmethod
    def params_schema(cls) -> Json:
        # fmt: off
        return cls.object_schema({
            "keys": {"type": "array", "items": {"type": "string", "pattern": "^seg\\.\\d+$"}, "minItems": 1, "description": "Segment keys to retrieve, or to restrict query search"},
            "query": {"type": "string", "maxLength": cls.MAX_QUERY_LENGTH, "description": "Case-insensitive regex over segment titles and text; A|B|C is allowed"},
            "case_sensitive": {"type": "boolean", "description": "Make query matching case-sensitive; default false"},
            "limit": {"type": "integer", "minimum": 1, "maximum": cls.MAX_LIMIT, "description": f"Maximum matching lines to return; default {cls.DEFAULT_LIMIT}"},
        })
        # fmt: on

    @classmethod
    def payload_args(cls, payload: Json) -> ToolArgs:
        return [payload]

    def call(self) -> str:
        request = self.request()
        if request["query"]:
            return self.search(request)
        keys = request["keys"]
        segments = {segment.key: segment for segment in self.session.history}
        chunks = ["<RecallContextResult>"]
        for key in keys:
            segment = segments.get(key)
            if segment is None:
                chunks.append(f"* {key}: missing")
                continue
            chunks.append(f"<Segment key={json.dumps(key)} title={json.dumps(segment.title)}>")
            chunks.append(segment.text.rstrip())
            chunks.append("</Segment>")
        chunks.append("</RecallContextResult>")
        return "\n".join(chunks)

    def short_args(self) -> list[str]:
        request = self.request()
        if request["query"]:
            scope = " in " + ",".join(request["keys"]) if request["keys"] else ""
            return [json.dumps(request["query"], ensure_ascii=False) + scope]
        return ["; ".join(request["keys"])]

    def request(self) -> Json:
        payload = self.single_dict_arg("RecallContext requires keys or query")
        if unexpected := sorted(set(payload) - {"keys", "query", "case_sensitive", "limit"}):
            raise ToolError("RecallContext unexpected field: " + ", ".join(unexpected))
        raw_keys = payload.get("keys")
        if raw_keys is not None and (not isinstance(raw_keys, list) or not raw_keys):
            raise ToolError("RecallContext keys must be a non-empty array")
        keys = []
        for item in raw_keys or []:
            key = str(item).strip()
            if not self._KEY_RE.fullmatch(key):
                raise ToolError("RecallContext key must look like seg.N")
            keys.append(key)
        query = payload.get("query")
        if query is not None and (not isinstance(query, str) or not query.strip()):
            raise ToolError("RecallContext query must be a non-empty regex")
        query = query.strip() if isinstance(query, str) else ""
        if len(query) > self.MAX_QUERY_LENGTH:
            raise ToolError(f"RecallContext query must be at most {self.MAX_QUERY_LENGTH} characters")
        case_sensitive = payload.get("case_sensitive", False)
        if not isinstance(case_sensitive, bool):
            raise ToolError("RecallContext case_sensitive must be boolean")
        limit = payload.get("limit", self.DEFAULT_LIMIT)
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1 or limit > self.MAX_LIMIT:
            raise ToolError(f"RecallContext limit must be 1..{self.MAX_LIMIT}")
        if not keys and not query:
            raise ToolError("RecallContext requires keys or query")
        if not query and ("case_sensitive" in payload or "limit" in payload):
            raise ToolError("RecallContext case_sensitive and limit require query")
        return {"keys": list(dict.fromkeys(keys)), "query": query, "case_sensitive": case_sensitive, "limit": limit}

    def search(self, request: Json) -> str:
        regex = self.compile_regex(request["query"], case_sensitive=request["case_sensitive"])
        by_key = {segment.key: segment for segment in self.session.history}
        segments = [by_key[key] for key in request["keys"] if key in by_key] if request["keys"] else self.session.history
        rows = []
        for segment in segments:
            if regex.search(segment.title):
                rows.append(self.match_row(segment, "title", segment.title))
            for line_number, line in enumerate(segment.text.splitlines(), 1):
                if regex.search(line):
                    rows.append(self.match_row(segment, str(line_number), line))
                if len(rows) >= request["limit"]:
                    break
            if len(rows) >= request["limit"]:
                break
        rows = rows[: request["limit"]]
        header = f"<RecallContextSearchResult query={json.dumps(request['query'])} matches={len(rows)}>"
        return "\n".join([header, *rows, "</RecallContextSearchResult>"])

    @staticmethod
    def match_row(segment: HistorySegment, location: str, text: str) -> str:
        return f"- {segment.key} {location} title={json.dumps(segment.title)}: {Tool.compact(text, 300)}"


class NoteTool(Tool):
    NAME = "Note"
    DESCRIPTION = (
        "Maintain durable working notes; "
        "set_goal, replace_plan, and set_check replace current values, append_known appends, replace_known replaces all known facts. "
        "Plan items are objects with status todo|doing|done|blocked and text."
    )
    STORES_RESULT = False

    @classmethod
    def params_schema(cls) -> Json:
        # fmt: off
        plan_item = cls.object_schema({
            "status": {"type": "string", "enum": list(PlanItem.STATUSES), "description": "todo|doing|done|blocked"},
            "text": {"type": "string", "description": "Plan step description"},
        }, ["status", "text"])
        return cls.object_schema({
            "set_goal": {"type": "string", "description": "Replace the current goal"},
            "replace_plan": {"type": "array", "items": plan_item, "description": "Replace the plan with these status/text items"},
            "append_known": {"type": "array", "items": {"type": "string"}, "description": "Append these facts to known"},
            "replace_known": {"type": "array", "items": {"type": "string"}, "description": "Replace all known facts with these"},
            "set_check": {"type": "string", "description": "Replace the success/verification criteria"},
        })
        # fmt: on

    def call(self) -> str:
        data = self.single_dict_arg("Note requires named fields")
        if unexpected := sorted(set(data) - {"set_goal", "replace_plan", "append_known", "replace_known", "set_check"}):
            raise ToolError("Note unexpected field: " + ", ".join(unexpected))
        changed = []
        goal = self.session.state.goal
        plan = list(self.session.state.plan)
        known = list(self.session.state.known)
        check = self.session.state.check
        if "set_goal" in data:
            goal = str(data["set_goal"]).strip()
            changed.append("set_goal")
        if "set_check" in data:
            check = str(data["set_check"]).strip()
            changed.append("set_check")
        if "replace_plan" in data:
            if not isinstance(data["replace_plan"], list):
                raise ToolError('Note replace_plan must be an array of plan items, e.g. {"replace_plan":[{"status":"doing","text":"inspect"}]}')
            for item in data["replace_plan"]:
                if isinstance(item, dict):
                    if str(item.get("status") or "").strip().lower() not in PlanItem.STATUSES:
                        raise ToolError("Note replace_plan status must be one of: " + ", ".join(PlanItem.STATUSES))
                    if not str(item.get("text") or "").strip():
                        raise ToolError("Note replace_plan text is required")
            plan = cast(list[PlanItem | Json | str], AgentState.plan_items(data["replace_plan"]))
            changed.append("replace_plan")
        if "append_known" in data:
            if not isinstance(data["append_known"], list):
                raise ToolError('Note append_known must be an array of strings, e.g. {"append_known":["tests use pytest"]}')
            known = list(dict.fromkeys([*known, *(str(item).strip() for item in data["append_known"] if str(item).strip())]))
            changed.append("append_known")
        if "replace_known" in data:
            if not isinstance(data["replace_known"], list):
                raise ToolError('Note replace_known must be an array of strings, e.g. {"replace_known":["fact"]}')
            known = [str(item).strip() for item in data["replace_known"] if str(item).strip()]
            changed.append("replace_known")
        if not changed:
            raise ToolError("Note requires set_goal, replace_plan, append_known, replace_known, or set_check")
        self.session.state.goal = goal
        self.session.state.plan = plan
        self.session.state.known = known
        self.session.state.check = check
        return "Updated memory: " + ", ".join(changed)

    def short_args(self) -> list[str]:
        data = self.args[0] if self.args and isinstance(self.args[0], dict) else {}
        lines = []
        if goal := str(data.get("set_goal") or "").strip():
            lines.append("goal: " + Tool.compact(goal, 120))
        if check := str(data.get("set_check") or "").strip():
            lines.append("check: " + Tool.compact(check, 120))
        if isinstance(data.get("replace_plan"), list):
            lines.extend(["plan:", *(f"  {row}" for row in AgentState.plan_rows_for(data["replace_plan"], status=True, style="symbol") if row != "- (empty)")])
        if isinstance(data.get("append_known"), list):
            known = [Tool.compact(item, 120) for item in data["append_known"] if str(item).strip() and str(item).strip() not in self.session.state.known]
            if known:
                lines.extend(["known:", *(f"  + {item}" for item in known)])
        if isinstance(data.get("replace_known"), list):
            known = [Tool.compact(item, 120) for item in data["replace_known"] if str(item).strip()]
            if known:
                lines.extend(["known:", *(f"  {item}" for item in known)])
        return ["\n".join(lines) or "{}"]
