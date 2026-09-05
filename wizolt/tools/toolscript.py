"""ToolScript: batch describe of MCP tool shapes, and scripted nested MCP calls."""

from __future__ import annotations

import builtins
import contextlib
import io
import json
import linecache
import re
import sys
import threading
import time
import traceback
from collections.abc import Callable
from typing import TYPE_CHECKING

from wizolt.base import ApprovalView, Json, ToolCall, ToolError
from wizolt.tools.base import Tool

if TYPE_CHECKING:
    from wizolt.runner import ToolRunner

# The fake filename scripts are compiled under: linecache keeps the source visible in tracebacks.
SCRIPT_FILENAME = "<toolscript>"
# Pure script execution budget (nested tool calls are paused out of it). Best-effort, not a kill.
SCRIPT_TIME_LIMIT = 60.0
CALLS_LINE_LIMIT = 200
_RESULT_KEY_RE = re.compile(r"^tool (tr\.\d+)")


class ScriptCancelled(BaseException):
    """The turn was cancelled while the script was running.

    BaseException, not Exception: a script may wrap its own work in `except Exception` -- catching
    a per-item failure is exactly what call_many exists to encourage -- and a cancellation caught
    there would let the script keep making calls after the turn was told to stop."""


class _ScriptTimeBudget:
    """Accumulate wall time across `line` events of <toolscript> frames only, and carry the
    cooperative cancellation token the same tracer checks.

    Nested calls pause the clock: a human confirmation inside call() must not count against the
    script's pure execution budget. The tracer is installed with sys.settrace and only touches the
    calling thread, so MCP's own event-loop threads are unaffected.

    Cancellation rides the same hook because it needs the same property: a script that never calls
    a tool again -- a long pure loop -- still executes lines, and that is the only place the worker
    can be reached from outside without injecting an exception into a thread.
    """

    def __init__(self, limit: float):
        self.limit = limit
        self.elapsed = 0.0
        self._last: float | None = None
        self._paused = False
        self._cancelled = threading.Event()

    def cancel(self) -> None:
        """Set the token. Idempotent, and safe from the loop thread: the worker reads it next line."""
        self._cancelled.set()

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def raise_if_cancelled(self) -> None:
        if self._cancelled.is_set():
            raise ScriptCancelled

    def tracer(self):
        def trace(frame, event, arg):
            # None for a foreign frame, not `trace`: what a global trace call returns becomes that
            # frame's local trace, so returning the tracer here would fire a Python callback on
            # every line of every tool the script reaches -- Read, Search, the MCP transport --
            # to immediately return. Script frames are still offered at their own call event.
            if frame.f_code.co_filename != SCRIPT_FILENAME:
                return None
            if event == "line":
                self.raise_if_cancelled()
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


class _StdoutCapture:
    """The script's own stdout/stderr, captured -- and stepped aside for nested calls.

    Only what the script itself prints returns to the model, so its writes are captured. But a
    nested call is not the script writing: it logs to the terminal and may prompt for
    confirmation, and both go through sys.stdout on the headless path. Capturing those would post
    the log into the model's result and hide the prompt, so `paused()` restores the real streams
    for exactly as long as the nested call runs."""

    def __init__(self, stdout_buf: io.StringIO, stderr_buf: io.StringIO):
        self.stdout_buf = stdout_buf
        self.stderr_buf = stderr_buf
        # Whatever was current when capture began, not sys.__stdout__: an outer redirect (the test
        # harness, a caller capturing the run) owns these streams, and stepping aside must hand
        # them back to that owner rather than punch through to the process's original terminal.
        self.outer_stdout = sys.stdout
        self.outer_stderr = sys.stderr

    @contextlib.contextmanager
    def active(self):
        self.outer_stdout, self.outer_stderr = sys.stdout, sys.stderr
        with contextlib.redirect_stdout(self.stdout_buf), contextlib.redirect_stderr(self.stderr_buf):
            yield

    @contextlib.contextmanager
    def paused(self):
        with contextlib.redirect_stdout(self.outer_stdout), contextlib.redirect_stderr(self.outer_stderr):
            yield


class ToolScript(Tool):
    NAME = "ToolScript"
    DESCRIPTION = (
        "Describe tool shapes or run Python that invokes tools through call(name, args) and prints only the derived result. "
        "Use for 4+ same-shape calls when individual outputs are unnecessary; otherwise emit normal tool calls. "
        "Use call_many([(name, args), ...]) for independent calls; it preserves order and returns failures as values. "
        "call raises on failure. MCP calls may use format='json'; built-ins use text. Do not start threads."
    )
    EXAMPLE = (
        'Aggregate many same-shape calls into one line. Example: {"action":"call","code":"hits = 0\\nfor path in (\\"a.py\\", \\"b.py\\", \\"c.py\\", \\"d.py\\"):\\n    hits += call(\\"Search\\", {\\"pattern\\": \\"TODO\\", \\"path\\": path}).count(\\"TODO\\")\\nprint(hits)"}',
        'Learn call shapes before scripting them. Example: {"action":"describe","tools":["Read","server.tool"]}',
    )
    MUTATES = True
    runner: ToolRunner | None = None  # injected by ToolRunner.call_tool; the runner owns the confirm wiring
    # Injected by the runner for the length of one script: publishes this run's cancellation token,
    # so the loop can set it without reaching into the worker.
    on_budget: Callable[[_ScriptTimeBudget], None] | None = None

    @classmethod
    def params_schema(cls) -> Json:
        # fmt: off
        return cls.object_schema({
            "action": {"type": "string", "enum": ["describe", "call"], "description": "Describe tool shapes or run code"},
            "tools": {
                "type": "array",
                "items": {"type": "string", "description": 'a built-in tool name like "Read", or an MCP tool as "server.tool"'},
                "minItems": 1,
                "description": "Tools to describe",
            },
            "code": {"type": "string", "description": "Python for action=call; invoke tools only through call or call_many"},
        }, ["action"])
        # fmt: on

    @staticmethod
    def resolved_action(payload: Json) -> str:
        return str(payload.get("action") or "").strip() or "call"

    def payload(self) -> Json:
        return self.single_dict_arg("ToolScript requires named fields")

    def needs_confirmation(self) -> bool:
        return self.resolved_action(self.payload()) == "call"

    def script(self) -> str:
        """The script source of a `call`, or "" when there is none (a describe, a malformed call)."""
        try:
            payload = self.payload()
        except ToolError:
            return ""
        if self.resolved_action(payload) != "call":
            return ""
        return str(payload.get("code") or "")

    def approval_view(self) -> ApprovalView | None:
        """The whole script, lexed as Python: what `v` opens at the confirmation prompt, what the
        approval block shows a clipped excerpt of, and what Ctrl-O reopens afterwards -- the last
        being the only way to read it under yolo, where nothing stops to ask."""
        code = self.script()
        if not code.strip():
            return None
        return ApprovalView("script", code, "python", [("lines", str(len(code.splitlines()))), ("chars", str(len(code)))])

    def short_args(self) -> list[str]:
        """A short display identity: the script's size, not its first line.

        The first line of a script is usually setup (`rows = []`) and says nothing about what the
        script does; the body is one keypress away in the viewer, and the approval block shows its
        opening lines, so the log line only owes the reader the scale of what ran."""
        payload = self.payload()
        action = self.resolved_action(payload)
        if action != "call":
            return [action, str(payload.get("tools") or "")]
        code = str(payload.get("code") or "")
        lines = len(code.splitlines())
        return ["call", f"{lines} line{'' if lines == 1 else 's'} ({len(code)} chars)"]

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
        from wizolt.tools import TOOL_REGISTRY  # local import: the registry is built on top of every tool

        if raw_entry in TOOL_REGISTRY:
            if self.session.tool_names and raw_entry not in self.session.tool_names:
                return f"{raw_entry}: not available in this session"
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
        if self.on_budget is not None:
            self.on_budget(budget)
        keys: list[str] = []

        def call_fn(name, args=None, format="text"):
            return self._nested_call(runner, budget, keys, name, args, format, capture)

        def call_many_fn(calls, format="text"):
            return self._nested_call_many(runner, budget, keys, calls, format, capture)

        compiled = compile(code, SCRIPT_FILENAME, "exec")
        linecache.cache[SCRIPT_FILENAME] = (len(code), None, code.splitlines(True), SCRIPT_FILENAME)

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        capture = _StdoutCapture(stdout_buf, stderr_buf)
        previous_trace = sys.gettrace()
        script_status = getattr(runner, "script_status", None)
        failed = False
        error_text = ""
        try:
            if script_status is not None:
                script_status(True, code)
            sys.settrace(budget.tracer())
            try:
                # Everything the nested calls log is printed one level deeper, so the batch reads as
                # what it is -- calls made by this script -- instead of as calls the model made
                # itself. The indent is the whole signal; the nested lines keep their usual shape.
                with runner.nested(), capture.active():
                    exec(  # noqa: S102 - ToolScript is the sanctioned script executor; not a sandbox, the outer confirmation is the boundary
                        compiled,
                        {"__name__": "__toolscript__", "__builtins__": builtins, "call": call_fn, "call_many": call_many_fn},
                    )
            except (ScriptCancelled, KeyboardInterrupt):
                # Cancelling the turn is not the script failing. It has to keep travelling:
                # swallowing it here would report a failed script and carry on making calls.
                raise
            except BaseException:  # noqa: BLE001 - script failures become a failed envelope, not a ToolScript crash.
                # BaseException, not Exception: `sys.exit()` is an ordinary idiom in written-to-be-
                # standalone Python, and a model writes it without thinking. As SystemExit it flew
                # past this handler, past run_one (which catches Exception), and out of the agent
                # loop -- one line of a script could end the session.
                failed = True
                error_text = traceback.format_exc()
            finally:
                sys.settrace(previous_trace)
        finally:
            linecache.cache.pop(SCRIPT_FILENAME, None)
            if script_status is not None:
                script_status(False, "")

        return self._envelope(failed, keys, stdout_buf.getvalue(), stderr_buf.getvalue(), error_text)

    def _mcp_target(self, name: str) -> tuple[str, str] | None:
        """The (server, tool) a "server.tool" name resolves to, or None when it names no MCP tool.

        The tool listing spells MCP tools "server.tool" and `describe` takes them in that form, so
        that is the form a script reaches for first; without this it failed as an unknown tool and
        the script had to be rewritten into the call("MCP", {...}) shape."""
        from wizolt.tools import TOOL_REGISTRY  # local import: the registry is built on top of every tool

        # isinstance first: `name` is whatever the script passed, and `"." in 123` raises a
        # TypeError that would surface as a traceback instead of the plain "unknown tool" below.
        if not isinstance(name, str) or name in TOOL_REGISTRY or "." not in name:
            return None
        mcp = self.session.mcp
        if mcp is None:
            return None
        server, tool = self._split_name(name)
        return (server, tool) if tool and mcp.find_config(server) is not None else None

    def _nested_call(self, runner: ToolRunner, budget: _ScriptTimeBudget, keys: list[str], name, args, format, capture: _StdoutCapture) -> Json | str:
        call, tool = self._build_call(len(keys) + 1, name, args, format)
        # The capture steps aside for the same reason the clock pauses: what the nested call logs
        # belongs on the terminal, not in the script's stdout. Left in place, every nested call
        # line was swallowed into the buffer and handed back to the model as the script's own
        # output, and a headless confirmation prompt -- input() writes to sys.stdout -- went with
        # it, stopping the run at a prompt nobody could see.
        budget.pause()
        try:
            with capture.paused():
                status, message, _ = self._run_nested(runner, call)
        finally:
            budget.resume()
        if status == "refused":
            raise ToolError("nested call refused by user")
        if status == "failed":
            raise ToolError(message)
        return self._nested_result(keys, message, format, tool)

    def _build_call(self, index: int, name, args, format) -> tuple[ToolCall, str]:
        """Validate one scripted invocation and turn it into a ToolCall.

        Returns the call and, for MCP, the tool's own name — the only thing the caller still needs
        it for is naming the tool when its reply does not parse as JSON.
        """
        if name == "ToolScript":
            raise ToolError('call("ToolScript", ...) is not allowed')
        if name in ("Delegate", "Job"):
            raise ToolError(f"{name} is not scriptable")
        target = self._mcp_target(name)
        if target is not None:
            # Rewritten into the canonical form rather than handled apart, so a "server.tool" call
            # goes through exactly the same validation, confirmation, and logging as call("MCP", ...).
            if args is not None and not isinstance(args, dict):
                raise ToolError(f'call("{name}", ...) requires named arguments')
            name, args = "MCP", {"server": target[0], "tool": target[1], "arguments": args or {}}
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
            return ToolCall(f"toolscript.{index}", "MCP", [payload]), tool
        else:
            from wizolt.tools import TOOL_REGISTRY  # local import: the registry is built on top of every tool

            tool_class = TOOL_REGISTRY.get(name)
            if tool_class is None:
                if isinstance(name, str) and "." in name:
                    # The name is spelled like an MCP tool but resolved to nothing: say which half
                    # is wrong, so the script is not rewritten into a shape that was never the problem.
                    raise ToolError(f'unknown tool "{name}": no MCP server named "{name.split(".", 1)[0]}" is configured')
                raise ToolError(f'unknown tool "{name}"')
            if format == "json":
                raise ToolError(f'{name} does not support format="json"; use format="text"')
            if not isinstance(args, dict):
                raise ToolError(f'call("{name}", ...) requires named arguments')
            from wizolt.tools import tool_payload  # local import: the registry is built on top of every tool

            try:
                return ToolCall(f"toolscript.{index}", name, tool_payload(name, args)), ""
            except ToolError as error:
                raise ToolError(f"{name}: {error}") from error

    def _nested_result(self, keys: list[str], message: str, format, tool: str) -> Json | str:
        """The value a completed nested call hands back to the script."""
        key = self._result_key(message)
        if key:
            keys.append(key)
            full = self.session.tool_results.get(key, message)
        else:
            # A tool the runner does not retain (Recall, RecallContext, Note) has no tr.N to read
            # the result back from, and `message` is the model-facing envelope: a header line, then
            # `output:`, then the text. The script asked for the result, so hand it the result.
            full = self._message_body(message)
        if format == "json":
            try:
                return json.loads(full)
            except (json.JSONDecodeError, ValueError):
                raise ToolError(f'MCP returned text that is not JSON for tool "{tool}"')
        return full

    def _nested_call_many(
        self, runner: ToolRunner, budget: _ScriptTimeBudget, keys: list[str], entries, format, capture: _StdoutCapture
    ) -> list[Json | str | ToolError]:
        """Run a list of (name, args) invocations, returning one value per entry in the order given.

        Every entry is validated before any of them runs, so a malformed list is a script error
        rather than a half-executed batch. A call that fails comes back as the ToolError it raised
        instead of ending the script: handing over a batch is only worth doing if one bad item does
        not cost the others, which a loop of `call()` buys only with a try/except per item. A
        refusal still raises — a user declining a call is a decision about the run, not a datum.
        """
        if isinstance(entries, (str, bytes)) or not isinstance(entries, (list, tuple)):
            raise ToolError("call_many(...) requires a list of (name, args) pairs")
        prepared: list[tuple[ToolCall, str]] = []
        for position, entry in enumerate(entries):
            if isinstance(entry, (str, bytes)) or not isinstance(entry, (list, tuple)) or len(entry) != 2:
                raise ToolError(f"call_many(...) entry {position} is not a (name, args) pair")
            prepared.append(self._build_call(len(keys) + 1 + position, entry[0], entry[1], format))
        if not prepared:
            return []
        budget.pause()
        try:
            with capture.paused():
                outcomes = self._run_nested_many(runner, [call for call, _ in prepared])
        finally:
            budget.resume()
        results: list[Json | str | ToolError] = []
        for (status, message), (_, tool) in zip(outcomes, prepared):
            if status == "refused":
                raise ToolError("nested call refused by user")
            if status == "failed":
                results.append(ToolError(message))
                continue
            try:
                results.append(self._nested_result(keys, message, format, tool))
            except ToolError as error:
                results.append(error)
        return results

    def _run_nested_many(self, runner: ToolRunner, calls: list[ToolCall]) -> list[tuple[str, str]]:
        """Hand a whole prepared batch to the runner and wait, on this worker, for its answers.

        The segmentation, concurrency cap, and original-order publication are the runner's, applied
        on its loop: only calls that neither mutate state nor stop for the user may overlap, so a
        write or a confirmation never runs beside a concurrent read, and the log reads as the script
        was written rather than as the workers happened to finish."""

        return runner.nested_calls(calls, batch=True)

    def _run_nested(self, runner: ToolRunner, call: ToolCall) -> tuple[str, str, object | None]:
        """Run one nested call through the runner, over the same gateway. Edit planning happens on
        the runner's side, so a nested Edit behaves exactly like a top-level single Edit."""

        status, message = runner.nested_calls([call], batch=False)[0]
        return status, message, None

    @staticmethod
    def _message_body(message: str) -> str:
        """The output half of a tool message, or the whole thing when it carries no `output:` line."""
        head, separator, body = message.partition("\noutput:\n")
        return body if separator else head

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
