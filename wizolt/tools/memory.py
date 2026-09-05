"""Memory tools: history recall, context recall, and durable notes."""

from __future__ import annotations

import json
import re
from typing import ClassVar, cast

from wizolt.base import Json, ToolArgs, ToolError
from wizolt.session import AgentState, HistorySegment, PlanItem
from wizolt.tools.base import Tool


class RecallTool(Tool):
    NAME = "Recall"
    _KEY_RE: ClassVar[re.Pattern] = re.compile(r"tr\.\d+")
    DESCRIPTION = (
        "Recall stored tool output by tr.N, optionally slicing inclusive output lines. A bounded_output marker names a file containing any truncated remainder."
    )
    STORES_RESULT = False

    @classmethod
    def params_schema(cls) -> Json:
        # fmt: off
        return cls.object_schema({
            "keys": {"type": "array", "items": {"type": "string", "pattern": "^tr\\.\\d+$"}, "minItems": 1, "description": 'Stored result keys to recall, e.g. ["tr.3","tr.5"]'},
            "ranges": {"type": "array", "items": cls.RANGE_SCHEMA, "minItems": 1, "description": "Optional 1-based [start,end] output-line slices, inclusive of both ends, to limit recalled context"},
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
        # Ranges are 1-based and inclusive, like every other line range the model works with.
        return "\n".join("\n".join(lines[max(start, 1) - 1 : end]) for start, end in ranges if end >= max(start, 1))


class RecallContextTool(Tool):
    NAME = "RecallContext"
    _KEY_RE: ClassVar[re.Pattern] = re.compile(r"seg\.\d+")
    DESCRIPTION = "List compacted history segments, retrieve them by seg.N key, or regex-search their titles and text; query alternation A|B|C is allowed."
    DEFAULT_LIMIT = 20
    MAX_LIMIT = 100
    MAX_QUERY_LENGTH = 500
    EXAMPLE = (
        'Retrieve segments. Example: {"action":"get","keys":["seg.1","seg.2"]}',
        'Search segments. Example: {"action":"search","query":"cache prefix|task memory","limit":10}',
    )
    STORES_RESULT = False

    @classmethod
    def params_schema(cls) -> Json:
        # fmt: off
        return cls.object_schema({
            "action": {"type": "string", "enum": ["list", "get", "search"], "description": "Operation"},
            "keys": {"type": "array", "items": {"type": "string", "pattern": "^seg\\.\\d+$"}, "minItems": 1, "description": "Keys to get or search within"},
            "query": {"type": "string", "maxLength": cls.MAX_QUERY_LENGTH, "description": "Regex over segment titles and text"},
            "case_sensitive": {"type": "boolean", "description": "Case-sensitive search; default false"},
            "limit": {"type": "integer", "minimum": 1, "maximum": cls.MAX_LIMIT, "description": f"Result limit; default {cls.DEFAULT_LIMIT}"},
            "before": {"type": "string", "pattern": "^seg\\.\\d+$", "description": "List segments older than this key"},
        })
        # fmt: on

    @classmethod
    def payload_args(cls, payload: Json) -> ToolArgs:
        return [payload]

    def call(self) -> str:
        request = self.request()
        if request["action"] == "list":
            return self.list_segments(request)
        if request["action"] == "search":
            return self.search(request)
        keys = request["keys"]
        segments = {segment.key: segment for segment in self.session.history}
        chunks = ["<RecallContextResult>"]
        for key in keys:
            segment = segments.get(key)
            if segment is None:
                chunks.append(f"* {key}: {self.missing_reason(key)}")
                continue
            chunks.append(f"<Segment key={json.dumps(key)} title={json.dumps(segment.title)}>")
            # The checkpoint summary as it stood at this compaction, which is not the one in
            # context: each compaction folds the previous summary into the next, so the live one
            # has been through a pass per compaction, while this copy has been through exactly
            # one. Empty for a segment trimmed without a model reply.
            if segment.summary:
                chunks.append(f"<SummaryAtCompaction>\n{segment.summary.rstrip()}\n</SummaryAtCompaction>")
            chunks.append(segment.text.rstrip())
            chunks.append("</Segment>")
        chunks.append("</RecallContextResult>")
        return "\n".join(chunks)

    def short_args(self) -> list[str]:
        request = self.request()
        if request["action"] == "list":
            suffix = f" before {request['before']}" if request["before"] else ""
            return [f"list {request['limit']}{suffix}"]
        if request["action"] == "search":
            scope = " in " + ",".join(request["keys"]) if request["keys"] else ""
            return [json.dumps(request["query"], ensure_ascii=False) + scope]
        return ["; ".join(request["keys"])]

    def request(self) -> Json:
        payload = {key: value for key, value in self.single_dict_arg("RecallContext requires an action, keys, or query").items() if value is not None}
        if unexpected := sorted(set(payload) - {"action", "keys", "query", "case_sensitive", "limit", "before"}):
            raise ToolError("RecallContext unexpected field: " + ", ".join(unexpected))
        if payload.get("keys") == []:
            payload.pop("keys")
        if isinstance(payload.get("query"), str) and not payload["query"].strip():
            payload.pop("query")
        action = payload.get("action")
        if action is None:
            action = "search" if payload.get("query") is not None else "get" if payload.get("keys") is not None else "list"
        if action not in {"list", "get", "search"}:
            raise ToolError("RecallContext action must be list, get, or search")
        if action != "search" and payload.get("case_sensitive") is False:
            payload.pop("case_sensitive")
        if action == "get" and payload.get("limit") == self.DEFAULT_LIMIT:
            payload.pop("limit")
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
        before = str(payload.get("before") or "").strip()
        if before and not self._KEY_RE.fullmatch(before):
            raise ToolError("RecallContext before must look like seg.N")
        if action == "list":
            if keys or query or "case_sensitive" in payload:
                raise ToolError("RecallContext list accepts only limit and before")
        elif action == "get":
            if not keys:
                raise ToolError("RecallContext get requires keys")
            if query or "case_sensitive" in payload or "limit" in payload or before:
                raise ToolError("RecallContext get accepts only keys")
        else:
            if not query:
                raise ToolError("RecallContext search requires query")
            if before:
                raise ToolError("RecallContext before is only valid for list")
        return {
            "action": action,
            "keys": list(dict.fromkeys(keys)),
            "query": query,
            "case_sensitive": case_sensitive,
            "limit": limit,
            "before": before,
        }

    def missing_reason(self, key: str) -> str:
        """Why a key resolved to nothing: dropped by the retention bound, or never issued.

        Only the retained window is recallable, so a key older than it is gone for good. Saying so
        stops the retry that "missing" invites, and points at what is still there instead."""
        oldest = self.session.history[0].key if self.session.history else ""
        if oldest and int(key.split(".", 1)[1]) < int(oldest.split(".", 1)[1]):
            return f"dropped; only the newest {len(self.session.history)} segments are kept, from {oldest}"
        return "missing"

    def list_segments(self, request: Json) -> str:
        segments = list(reversed(self.session.history))
        if request["before"]:
            before_number = int(str(request["before"]).split(".", 1)[1])
            segments = [segment for segment in segments if int(segment.key.split(".", 1)[1]) < before_number]
        selected = segments[: request["limit"]]
        result: Json = {
            "segments": [{"key": segment.key, "title": segment.title} for segment in selected],
            "total": len(self.session.history),
            "returned": len(selected),
        }
        if len(segments) > len(selected) and selected:
            result["next_before"] = selected[-1].key
        return json.dumps(result, ensure_ascii=False)

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
    DESCRIPTION = "Durable goal, plan, facts, and checks across context compaction. Use early and keep current for non-trivial work. Replacement fields replace; append_known adds facts."
    STORES_RESULT = False
    MUTATES = True

    def needs_confirmation(self) -> bool:
        # MUTATES serializes Note with other state edits; working-note changes do not need user
        # confirmation because they are reversible session metadata, not workspace writes.
        return False

    @classmethod
    def params_schema(cls) -> Json:
        # fmt: off
        plan_item = cls.object_schema({
            "status": {"type": "string", "enum": list(PlanItem.STATUSES), "description": "todo|doing|done|blocked"},
            "text": {"type": "string", "description": "Plan step description"},
        }, ["status", "text"])
        return cls.object_schema({
            "action": {"type": "string", "enum": ["view", "update"], "description": "Operation"},
            "fields": {"type": "array", "items": {"type": "string", "enum": ["goal", "plan", "known", "check"]}, "minItems": 1, "description": "Fields to view; default all"},
            "set_goal": {"type": "string", "description": "Replace or clear goal"},
            "replace_plan": {"type": "array", "items": plan_item, "description": "Replace plan"},
            "append_known": {"type": "array", "items": {"type": "string"}, "description": "Append facts"},
            "replace_known": {"type": "array", "items": {"type": "string"}, "description": "Replace all facts"},
            "set_check": {"type": "string", "description": "Replace or clear verification criteria"},
        })
        # fmt: on

    def call(self) -> str:
        data = self.data()
        mutation_fields = {"set_goal", "replace_plan", "append_known", "replace_known", "set_check"}
        if unexpected := sorted(set(data) - {"action", "fields", *mutation_fields}):
            raise ToolError("Note unexpected field: " + ", ".join(unexpected))
        action = data.get("action")
        if action is None:
            action = "update" if mutation_fields.intersection(data) else "view"
        if action not in {"view", "update"}:
            raise ToolError("Note action must be view or update")
        if action == "view":
            if mutation_fields.intersection(data):
                raise ToolError("Note view does not accept update fields")
            return self.view(data)
        if not mutation_fields.intersection(data):
            raise ToolError("Note update requires set_goal, replace_plan, append_known, replace_known, or set_check")

        goal = self.session.state.goal
        plan = list(self.session.state.plan)
        known = list(self.session.state.known)
        check = self.session.state.check
        if "set_goal" in data:
            goal = str(data["set_goal"]).strip()
        if "set_check" in data:
            check = str(data["set_check"]).strip()
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
        if "append_known" in data:
            if not isinstance(data["append_known"], list):
                raise ToolError('Note append_known must be an array of strings, e.g. {"append_known":["tests use pytest"]}')
            known = list(dict.fromkeys([*known, *(str(item).strip() for item in data["append_known"] if str(item).strip())]))
        if "replace_known" in data:
            if not isinstance(data["replace_known"], list):
                raise ToolError('Note replace_known must be an array of strings, e.g. {"replace_known":["fact"]}')
            known = [str(item).strip() for item in data["replace_known"] if str(item).strip()]

        before = (self.session.state.goal, list(self.session.state.plan), list(self.session.state.known), self.session.state.check)
        self.session.state.goal = goal
        self.session.state.plan = plan
        self.session.state.known = known
        self.session.state.check = check
        after = (goal, plan, known, check)
        names = ("goal", "plan", "known", "check")
        changed = [name for name, old, new in zip(names, before, after, strict=True) if old != new]
        known_added = len(set(known) - set(before[2]))
        return json.dumps({"ok": True, "changed": changed, "known_added": known_added}, ensure_ascii=False)

    def view(self, data: Json) -> str:
        raw_fields = data.get("fields", ["goal", "plan", "known", "check"])
        if not isinstance(raw_fields, list) or not raw_fields:
            raise ToolError("Note fields must be a non-empty array")
        fields = list(dict.fromkeys(str(field) for field in raw_fields))
        if invalid := [field for field in fields if field not in {"goal", "plan", "known", "check"}]:
            raise ToolError("Note fields must contain only goal, plan, known, check: " + ", ".join(invalid))
        state = self.session.state
        values: Json = {
            "goal": state.goal,
            "plan": [{"status": item.status, "text": item.text} for item in AgentState.plan_items(state.plan)],
            "known": list(state.known),
            "check": state.check,
        }
        return json.dumps({field: values[field] for field in fields}, ensure_ascii=False)

    def data(self) -> Json:
        data = {key: value for key, value in self.single_dict_arg("Note requires named fields").items() if value is not None}
        if data.get("fields") == []:
            data.pop("fields")
        return data

    def short_args(self) -> list[str]:
        data = self.data()
        if data.get("action") == "view" or not any(key in data for key in ("set_goal", "replace_plan", "append_known", "replace_known", "set_check")):
            fields = data.get("fields")
            return ["view " + (", ".join(str(field) for field in fields) if isinstance(fields, list) else "all")]
        lines = []
        if "set_goal" in data:
            goal = str(data["set_goal"] or "").strip()
            lines.append("goal: " + (Tool.compact(goal, 120) if goal else "(cleared)"))
        if "set_check" in data:
            check = str(data["set_check"] or "").strip()
            lines.append("check: " + (Tool.compact(check, 120) if check else "(cleared)"))
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


class NextHintsTool(Tool):
    NAME = "NextHints"
    DESCRIPTION = (
        "Offer 2-3 useful one-line follow-up inputs after completed work. Use sparingly and only when natural next steps exist. "
        "Call once in the same response as the final answer; that response ends the turn."
    )
    STORES_RESULT = False
    SILENT = True
    MAX_HINTS: ClassVar[int] = 4
    MAX_LEN: ClassVar[int] = 48

    @classmethod
    def params_schema(cls) -> Json:
        # fmt: off
        return cls.object_schema({
            "inputs": {"type": "array", "items": {"type": "string"}, "description": 'Short one-line next-step prompts to offer, e.g. ["run the tests", "show the diff"]'},
        }, ["inputs"])
        # fmt: on

    def call(self) -> str:
        data = self.single_dict_arg("NextHints requires named fields")
        if unexpected := sorted(set(data) - {"inputs"}):
            raise ToolError("NextHints unexpected field: " + ", ".join(unexpected))
        raw = data.get("inputs")
        if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
            raise ToolError('NextHints inputs must be an array of strings, e.g. {"inputs":["run the tests"]}')
        hints = list(dict.fromkeys(Tool.compact(item, self.MAX_LEN) for item in raw if item.strip()))[: self.MAX_HINTS]
        if not hints:
            raise ToolError("NextHints inputs must contain at least one non-empty string")
        # Merge rather than replace: a batch of several NextHints calls offers every suggestion,
        # not just the last call's.
        self.session.add_quick_hints(hints, limit=self.MAX_HINTS)
        return f"Offered {len(hints)} quick input(s)"

    def short_args(self) -> list[str]:
        data = self.args[0] if self.args and isinstance(self.args[0], dict) else {}
        inputs = data.get("inputs")
        if isinstance(inputs, list):
            items = [Tool.compact(item, 120) for item in inputs if str(item).strip()]
            if items:
                return ["inputs: " + ", ".join(f'"{item}"' for item in items)]
        return ["{}"]
