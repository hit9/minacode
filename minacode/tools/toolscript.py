"""ToolScript: batch describe of MCP tool shapes, and scripted nested MCP calls."""

from __future__ import annotations

import builtins
import contextlib
import io
import json
import linecache
import re
import sys
import time
import traceback
from typing import TYPE_CHECKING

from minacode.base import Json, ToolCall, ToolError
from minacode.tools.base import Tool

if TYPE_CHECKING:
    from minacode.runner import ToolRunner

# The fake filename scripts are compiled under: linecache keeps the source visible in tracebacks.
SCRIPT_FILENAME = "<toolscript>"
# Pure script execution budget (nested tool calls are paused out of it). Best-effort, not a kill.
SCRIPT_TIME_LIMIT = 60.0
PREVIEW_LIMIT = 2000
CALLS_LINE_LIMIT = 200
_RESULT_KEY_RE = re.compile(r"^tool (tr\.\d+)")


class _ScriptTimeBudget:
    """Accumulate wall time across `line` events of <toolscript> frames only.

    Nested calls pause the clock: a human confirmation inside call() must not count against the
    script's pure execution budget. The tracer is installed with sys.settrace and only touches the
    calling thread, so MCP's own event-loop threads are unaffected.
    """

    def __init__(self, limit: float):
        self.limit = limit
        self.elapsed = 0.0
        self._last: float | None = None
        self._paused = False

    def tracer(self):
        def trace(frame, event, arg):
            if frame.f_code.co_filename != SCRIPT_FILENAME:
                return trace
            if event == "line":
                self._check(time.monotonic())
            return trace

        return trace

    def _check(self, now: float) -> None:
        if self._paused:
            self._last = now
            return
        if self._last is not None:
            self.elapsed += now - self._last
        self._last = now
        if self.elapsed > self.limit:
            raise ToolError(f"script exceeded {self.limit:g}s of execution time (excluding nested calls)")

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False
        self._last = None


class ToolScript(Tool):
    NAME = "ToolScript"
    DESCRIPTION = (
        "Batch-query tool shapes for scripting, or run a Python script that calls tools: "
        'action="call" executes code where call(name, {...}) performs nested tool invocations with '
        "normal confirmation and logging, and only printed output returns. Use only for 4+ "
        'consecutive same-shape calls; for single tools use MCP(action="describe"). Built-in tools '
        'are scriptable with format="text"; Delegate/Job/ToolScript are not.'
    )
    MUTATES = True
    runner: ToolRunner | None = None  # injected by ToolRunner.call_tool; the runner owns the confirm wiring

    @classmethod
    def params_schema(cls) -> Json:
        # fmt: off
        return cls.object_schema({
            "action": {"type": "string", "enum": ["describe", "call"], "description": '"describe" batch-queries tool return shapes; "call" runs a Python script that invokes tools'},
            "tools": {
                "type": "array",
                "items": {"type": "string", "description": 'a built-in tool name like "Read", or an MCP tool as "server.tool"'},
                "minItems": 1,
                "description": 'Tools to describe, e.g. ["Read", "server.tool", ...]',
            },
            "code": {"type": "string", "description": 'Python source for action="call"; nested tool invocations go through call(name, {...}, format="text"|"json")'},
        }, ["tools"])
        # fmt: on

    @staticmethod
    def resolved_action(payload: Json) -> str:
        return str(payload.get("action") or "").strip() or "call"

    def payload(self) -> Json:
        return self.single_dict_arg("ToolScript requires named fields")

    def needs_confirmation(self) -> bool:
        return self.resolved_action(self.payload()) == "call"

    def preview(self) -> str:
        """The full script, so a confirmation block can show the code. Truncated beyond PREVIEW_LIMIT."""
        try:
            code = str(self.payload().get("code") or "")
        except ToolError:
            return self.NAME
        if not code:
            return self.NAME
        return code if len(code) <= PREVIEW_LIMIT else code[: PREVIEW_LIMIT - 3] + "..."

    def short_args(self) -> list[str]:
        """A short display identity: first code line plus total length (Bash-like one-liner)."""
        payload = self.payload()
        action = self.resolved_action(payload)
        if action != "call":
            return [action, str(payload.get("tools") or "")]
        code = str(payload.get("code") or "")
        first = " ".join(code.strip().splitlines()[:1]) if code.strip() else ""
        label = first if len(first) <= 80 else first[:77] + "..."
        return ["call", f"{label} ({len(code)} chars)"]

    def call(self) -> str:
        payload = self.single_dict_arg("ToolScript requires named fields")
        action = self.resolved_action(payload)
        if action == "describe":
            return self._describe(payload)
        if action != "call":
            raise ToolError(f"unknown ToolScript action {action!r}. Valid actions: describe, call")
        code = payload.get("code")
        if not isinstance(code, str) or not code.strip():
            raise ToolError("ToolScript call requires a non-empty code string")
        return self._run_script(code)

    def _describe(self, payload: Json) -> str:
        raw = payload.get("tools")
        if not isinstance(raw, list) or not raw or not all(isinstance(item, str) and item for item in raw):
            raise ToolError('ToolScript tools must be a non-empty list of tool names or "server.tool" strings')
        return "\n\n".join(self._describe_entry(entry) for entry in raw)

    @staticmethod
    def _split_name(entry: str) -> tuple[str, str]:
        """Split "server.tool" (optionally prefixed with "MCP:") into (server, tool)."""
        name = entry.removeprefix("MCP:")
        server, _, tool = name.partition(".")
        return server, tool

    def _describe_entry(self, raw_entry: str) -> str:
        from minacode.tools import TOOL_REGISTRY  # local import: the registry is built on top of every tool

        if raw_entry in TOOL_REGISTRY:
            return self._describe_builtin(TOOL_REGISTRY[raw_entry])
        mcp = self.session.mcp
        if mcp is None:
            return f"{raw_entry}: MCP not configured"
        server, tool = self._split_name(raw_entry)
        if mcp.find_config(server) is None:
            if not tool:
                return f'{raw_entry}: expected "server.tool" format'
            return f"{raw_entry}: MCP server '{server}' not found"
        if not tool:
            return f'{raw_entry}: expected "server.tool" format'
        try:
            text, info = mcp.describe_tool_block(server, tool)
        except ToolError as error:
            return f"{raw_entry}: {error}"
        gate = "json:    yes" if info.output_schema else "json:    unknown"
        return text + "\n" + gate

    @staticmethod
    def _describe_builtin(tool_class: type[Tool]) -> str:
        """One compact block per built-in tool: name, parameter essentials, and the json gate.
        Built-ins have no structured return yet, so the gate is always "no" (format="json" refuses)."""
        lines = [tool_class.NAME]
        schema = tool_class.params_schema()
        if isinstance(schema, dict):
            props = schema.get("properties")
            required = set(schema.get("required") or [])
            if isinstance(props, dict):
                args = []
                for prop_name, prop in props.items():
                    if not isinstance(prop, dict):
                        continue
                    ptype = str(prop.get("type") or "any")
                    args.append(f"{prop_name}  {'required ' if prop_name in required else ''}{ptype}")
                if args:
                    lines.append("  args:    " + ", ".join(args))
        lines.append("json:    no")
        return "\n".join(lines)

    def _run_script(self, code: str) -> str:
        runner = getattr(self, "runner", None)
        if runner is None:
            raise ToolError("ToolScript requires a tool runner")

        budget = _ScriptTimeBudget(SCRIPT_TIME_LIMIT)
        keys: list[str] = []

        def call_fn(name, args=None, format="text"):
            return self._nested_call(runner, budget, keys, name, args, format)

        compiled = compile(code, SCRIPT_FILENAME, "exec")
        linecache.cache[SCRIPT_FILENAME] = (len(code), None, code.splitlines(True), SCRIPT_FILENAME)

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        previous_trace = sys.gettrace()
        failed = False
        error_text = ""
        try:
            sys.settrace(budget.tracer())
            try:
                with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                    exec(  # noqa: S102 - ToolScript is the sanctioned script executor; not a sandbox, the outer confirmation is the boundary
                        compiled,
                        {"__name__": "__toolscript__", "__builtins__": builtins, "call": call_fn},
                    )
            except Exception:  # noqa: BLE001 - script failures become a failed envelope, not a ToolScript crash.
                failed = True
                error_text = traceback.format_exc()
            finally:
                sys.settrace(previous_trace)
        finally:
            linecache.cache.pop(SCRIPT_FILENAME, None)

        return self._envelope(failed, keys, stdout_buf.getvalue(), stderr_buf.getvalue(), error_text)

    def _nested_call(self, runner: ToolRunner, budget: _ScriptTimeBudget, keys: list[str], name, args, format) -> Json | str:
        if name == "ToolScript":
            raise ToolError('call("ToolScript", ...) is not allowed')
        if name in ("Delegate", "Job"):
            raise ToolError(f"{name} is not scriptable")
        if name == "MCP":
            if not isinstance(args, dict):
                raise ToolError('call("MCP", ...) requires {"server": str, "tool": str, "arguments": dict}')
            server, tool, arguments = args.get("server"), args.get("tool"), args.get("arguments")
            if not isinstance(server, str) or not server or not isinstance(tool, str) or not tool or not isinstance(arguments, dict):
                raise ToolError('call("MCP", ...) requires {"server": str, "tool": str, "arguments": dict}')
            if self.session.mcp is None:
                raise ToolError("MCP not configured")
            payload: Json = {"action": "call", "server": server, "tool": tool, "arguments": arguments}
            if format == "json":
                payload["format"] = "json"
            call = ToolCall(f"toolscript.{len(keys) + 1}", "MCP", [payload])
        else:
            from minacode.tools import TOOL_REGISTRY  # local import: the registry is built on top of every tool

            tool_class = TOOL_REGISTRY.get(name)
            if tool_class is None:
                raise ToolError(f'unknown tool "{name}"')
            if format == "json":
                raise ToolError(f'{name} does not support format="json"; use format="text"')
            if not isinstance(args, dict):
                raise ToolError(f'call("{name}", ...) requires named arguments')
            from minacode.model import ModelClient  # local import: model.py imports the tool registry

            try:
                call = ToolCall(f"toolscript.{len(keys) + 1}", name, ModelClient.tool_payload(name, args))
            except ToolError as error:
                raise ToolError(f"{name}: {error}") from error
        budget.pause()
        try:
            status, message, _observation = self._run_nested(runner, call)
        finally:
            budget.resume()
        if status == "refused":
            raise ToolError("nested call refused by user")
        if status == "failed":
            raise ToolError(message)
        key = self._result_key(message)
        if key:
            keys.append(key)
            full = self.session.tool_results.get(key, message)
        else:
            full = message
        if format == "json":
            try:
                return json.loads(full)
            except (json.JSONDecodeError, ValueError):
                raise ToolError(f'MCP returned text that is not JSON for tool "{tool}"')
        return full

    def _run_nested(self, runner: ToolRunner, call: ToolCall) -> tuple[str, str, object | None]:
        """Run one nested call through the runner. Edits go through a single-element plan so a nested
        Edit behaves exactly like a top-level single Edit (anchor planning, stale checks, write-time
        verification) instead of a plan-less EditTool.call()."""
        if call.name == "Edit":
            from minacode.runner import EditBatchPlan  # local import: runner.py imports the tool registry

            plan = EditBatchPlan(self.session).build([call])
            return runner.run_one(call, planned_edit=plan.planned.get(call.id), plan_error=plan.errors.get(call.id, ""))
        return runner.run_one(call)

    @staticmethod
    def _result_key(message: str) -> str:
        match = _RESULT_KEY_RE.match(message)
        return match.group(1) if match else ""

    @staticmethod
    def _format_keys(keys: list[str]) -> str:
        """Compress consecutive tr.N keys into ranges, bounded to CALLS_LINE_LIMIT chars."""
        if not keys:
            return "0"

        def num(key: str) -> int:
            return int(key.split(".", 1)[1])

        ranges: list[str] = []
        start = prev = keys[0]
        for key in keys[1:]:
            if num(key) == num(prev) + 1:
                prev = key
                continue
            ranges.append(start if start == prev else f"{start}-{prev}")
            start = prev = key
        ranges.append(start if start == prev else f"{start}-{prev}")
        text = "[" + ", ".join(ranges) + "]"
        if len(text) <= CALLS_LINE_LIMIT:
            return f"{len(keys)} {text}"
        return f"... +{len(keys)} keys"

    @staticmethod
    def _envelope(failed: bool, keys: list[str], stdout: str, stderr: str, error_text: str) -> str:
        lines = ["ToolScript failed" if failed else "ToolScript ok"]
        lines.append("calls: " + ToolScript._format_keys(keys))
        if stdout:
            lines.extend(["stdout:", stdout.rstrip()])
        if stderr:
            lines.extend(["stderr:", stderr.rstrip()])
        if failed:
            lines.extend(["error:", error_text.rstrip()])
        return "\n".join(lines)
