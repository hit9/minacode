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
    from minacode.mcp import MCPManager
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
        "Batch-query MCP tool shapes for scripting, or run a Python script that calls MCP tools: "
        'action="call" executes code where call("MCP", {...}) performs nested MCP invocations with '
        "normal confirmation and logging, and only printed output returns. Use only for 4+ "
        'consecutive same-shape calls; for single tools use MCP(action="describe"). Built-in tools '
        "are not scriptable yet."
    )
    MUTATES = True
    runner: ToolRunner | None = None  # injected by ToolRunner.call_tool; the runner owns the confirm wiring

    @classmethod
    def params_schema(cls) -> Json:
        # fmt: off
        return cls.object_schema({
            "action": {"type": "string", "enum": ["describe", "call"], "description": '"describe" batch-queries MCP tool return shapes; "call" runs a Python script that invokes MCP tools'},
            "tools": {
                "type": "array",
                "items": {"type": "string", "description": 'MCP tool as "server.tool" — the name used in the MCP tools index'},
                "minItems": 1,
                "description": 'MCP tools to describe, e.g. ["server.tool", ...]',
            },
            "code": {"type": "string", "description": 'Python source for action="call"; nested MCP invocations go through call("MCP", {"server", "tool", "arguments"}, format="text"|"json")'},
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
        if self.session.mcp is None:
            raise ToolError("MCP not configured")
        return self._run_script(code)

    def _describe(self, payload: Json) -> str:
        raw = payload.get("tools")
        if not isinstance(raw, list) or not raw or not all(isinstance(item, str) and item for item in raw):
            raise ToolError('ToolScript tools must be a non-empty list of "server.tool" strings')
        mcp = self.session.mcp
        if mcp is None:
            raise ToolError("MCP not configured")
        return "\n\n".join(self._describe_entry(mcp, entry) for entry in raw)

    @staticmethod
    def _split_name(entry: str) -> tuple[str, str]:
        """Split "server.tool" (optionally prefixed with "MCP:") into (server, tool)."""
        name = entry.removeprefix("MCP:")
        server, _, tool = name.partition(".")
        return server, tool

    def _describe_entry(self, mcp: MCPManager, entry: str) -> str:
        from minacode.tools import TOOL_REGISTRY  # local import: the registry is built on top of every tool

        server, tool = self._split_name(entry)
        if mcp.find_config(server) is None:
            if server in TOOL_REGISTRY:
                return f"{entry}: {server} is not scriptable yet"
            if not tool:
                return f'{entry}: expected "server.tool" format'
            return f"{entry}: MCP server '{server}' not found"
        if not tool:
            return f'{entry}: expected "server.tool" format'
        try:
            text, info = mcp.describe_tool_block(server, tool)
        except ToolError as error:
            return f"{entry}: {error}"
        gate = "json:    yes" if info.output_schema else "json:    unknown"
        return text + "\n" + gate

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
        if name != "MCP":
            from minacode.tools import TOOL_REGISTRY  # local import: the registry is built on top of every tool

            if name in TOOL_REGISTRY:
                raise ToolError(f"{name} is not scriptable yet")
            raise ToolError(f'unknown tool "{name}"')
        if not isinstance(args, dict):
            raise ToolError('call("MCP", ...) requires {"server": str, "tool": str, "arguments": dict}')
        server, tool, arguments = args.get("server"), args.get("tool"), args.get("arguments")
        if not isinstance(server, str) or not server or not isinstance(tool, str) or not tool or not isinstance(arguments, dict):
            raise ToolError('call("MCP", ...) requires {"server": str, "tool": str, "arguments": dict}')
        payload: Json = {"action": "call", "server": server, "tool": tool, "arguments": arguments}
        if format == "json":
            payload["format"] = "json"
        call = ToolCall(f"toolscript.{len(keys) + 1}", "MCP", [payload])
        budget.pause()
        try:
            status, message, _observation = runner.run_one(call)
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
