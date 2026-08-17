"""ToolScript: stage 1 describe + stage 2 scripted nested MCP calls (black-box)."""

from types import SimpleNamespace

import pytest
from mcp_harness import mcp_cfg, mcp_tool_info

from minacode.base import ToolCall, ToolError
from minacode.config import Config
from minacode.context import ContextManager
from minacode.runner import ToolRunner
from minacode.session import Session
from minacode.tools import MCPTool, ReadTool, Tool, ToolScript

OUTPUT_SHAPE = {"type": "object", "properties": {"ok": {"type": "boolean"}}}


def _mcp_session(tmp_path):
    """A session with one configured MCP server 'test', populated in memory (no network)."""
    s = Session(cwd=str(tmp_path), config=Config.from_dict(mcp_cfg()))
    s.mcp.tools["test"] = []
    s.mcp.resources["test"] = []
    return s


def _runner(s, input_fn=None):
    return ToolRunner(
        s,
        ContextManager(s),
        input_fn=input_fn or (lambda prompt: "y"),
        output_fn=lambda text: None,
    )


def _describe(s, tools):
    return ToolScript(s, [{"action": "describe", "tools": tools}]).call()


def _run_script(s, code, input_fn=None):
    runner = _runner(s, input_fn=input_fn)
    (message,) = runner.run([ToolCall("ts1", "ToolScript", [{"action": "call", "code": code}])])
    return str(message["content"])


# ---------------------------------------------------------------------------
# Stage 1: describe
# ---------------------------------------------------------------------------


class TestJsonGate:
    def test_declared_output_schema_gates_yes(self, tmp_path):
        s = _mcp_session(tmp_path)
        s.mcp.tools["test"] = [mcp_tool_info("test", "echo", output_schema=OUTPUT_SHAPE)]
        out = _describe(s, ["test.echo"])
        assert "json:    yes" in out
        assert "json:    unknown" not in out

    def test_undeclared_output_schema_gates_unknown(self, tmp_path):
        s = _mcp_session(tmp_path)
        s.mcp.tools["test"] = [mcp_tool_info("test", "echo")]
        out = _describe(s, ["test.echo"])
        assert "json:    unknown" in out
        assert "json:    yes" not in out


class TestRenderingReuse:
    def test_success_block_is_mcp_describe_plus_json_gate(self, tmp_path):
        """The describe block is exactly MCP(describe)'s rendering with the json line appended."""
        s = _mcp_session(tmp_path)
        s.mcp.tools["test"] = [mcp_tool_info("test", "echo", output_schema=OUTPUT_SHAPE)]
        describe = MCPTool(s, [{"action": "describe", "server": "test", "tool": "echo"}]).call()
        assert _describe(s, ["test.echo"]) == describe + "\njson:    yes"


class TestEntryErrors:
    def test_unknown_server_is_error_entry_and_others_render(self, tmp_path):
        s = _mcp_session(tmp_path)
        s.mcp.tools["test"] = [mcp_tool_info("test", "echo")]
        out = _describe(s, ["ghost.tool", "test.echo"])
        assert "ghost.tool" in out
        assert "MCP server 'ghost' not found" in out
        assert '<MCPDescribe server="test" tool="echo">' in out
        assert "json:    unknown" in out

    def test_unknown_tool_is_error_entry_and_others_render(self, tmp_path):
        s = _mcp_session(tmp_path)
        s.mcp.tools["test"] = [mcp_tool_info("test", "echo")]
        out = _describe(s, ["test.nope", "test.echo"])
        assert "test.nope" in out
        assert "MCP tool 'nope' not found on server 'test'" in out
        assert '<MCPDescribe server="test" tool="echo">' in out

    def test_builtin_tool_describes_compact_block(self, tmp_path):
        s = _mcp_session(tmp_path)
        out = _describe(s, ["Read"])
        assert "Read\n" in out
        assert "  args:    path  string, ranges  array, files  array" in out
        assert "json:    no" in out
        assert "<MCPDescribe" not in out


class TestMcpPrefix:
    def test_mcp_prefix_and_plain_names_are_equivalent(self, tmp_path):
        s = _mcp_session(tmp_path)
        s.mcp.tools["test"] = [mcp_tool_info("test", "echo", output_schema=OUTPUT_SHAPE)]
        plain = _describe(s, ["test.echo"])
        prefixed = _describe(s, ["MCP:test.echo"])
        assert plain == prefixed


class TestMixedBatch:
    def test_mixed_batch_reports_each_entry(self, tmp_path):
        s = _mcp_session(tmp_path)
        s.mcp.tools["test"] = [
            mcp_tool_info("test", "echo"),
            mcp_tool_info("test", "lookup", output_schema=OUTPUT_SHAPE),
        ]
        out = _describe(s, ["Read", "test.lookup", "ghost.tool", "test.echo"])
        assert "Read\n" in out and "json:    no" in out
        assert "MCP server 'ghost' not found" in out
        assert out.count("<MCPDescribe") == 2
        assert '<MCPDescribe server="test" tool="lookup">' in out
        assert '<MCPDescribe server="test" tool="echo">' in out
        assert "json:    yes" in out
        assert "json:    unknown" in out


class TestActionValidation:
    def test_unknown_action_is_error(self, tmp_path):
        s = _mcp_session(tmp_path)
        with pytest.raises(ToolError, match="unknown ToolScript action"):
            ToolScript(s, [{"action": "bogus"}]).call()

    def test_missing_action_defaults_to_call_and_requires_code(self, tmp_path):
        s = _mcp_session(tmp_path)
        with pytest.raises(ToolError, match="requires a non-empty code"):
            ToolScript(s, [{"tools": ["test.echo"]}]).call()

    def test_no_mcp_describe_reports_mcp_entries_only(self, tmp_path):
        s = Session(cwd=str(tmp_path))
        s.mcp = None
        out = ToolScript(s, [{"action": "describe", "tools": ["Read", "test.echo"]}]).call()
        assert "Read\n" in out and "json:    no" in out
        assert "test.echo: MCP not configured" in out


class TestRegistration:
    def test_registered_in_resolved_schemas_with_mcp(self, tmp_path):
        s = _mcp_session(tmp_path)
        s.mcp.tools["test"] = [mcp_tool_info("test", "echo")]
        schemas = {schema["function"]["name"]: schema["function"] for schema in Tool.resolved_schemas(s)}
        params = schemas["ToolScript"]["parameters"]
        assert set(params["properties"]) == {"action", "tools", "code"}
        assert params["properties"]["action"]["enum"] == ["describe", "call"]
        assert params["required"] == ["tools"]
        assert params["properties"]["tools"]["minItems"] == 1

    def test_toolscript_always_in_schemas(self, tmp_path):
        s = Session(cwd=str(tmp_path))
        names = {schema["function"]["name"] for schema in Tool.resolved_schemas(s)}
        assert "ToolScript" in names
        assert "MCP" not in names


# ---------------------------------------------------------------------------
# Stage 2: scripted nested calls
# ---------------------------------------------------------------------------


class TestNestedCalls:
    def test_message_conservation(self, tmp_path, monkeypatch):
        """N nested calls produce no extra tool messages; results land in session.tool_results."""
        s = _mcp_session(tmp_path)
        s.mcp.tools["test"] = [mcp_tool_info("test", "echo")]

        async def fake_call(config, headers, name, arguments):
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="ok " + str(arguments))])

        monkeypatch.setattr(s.mcp, "_call_tool", fake_call)
        code = 'for i in range(3):\n    call("MCP", {"server": "test", "tool": "echo", "arguments": {"text": str(i)}})\nprint("done")\n'
        runner = _runner(s)
        messages = runner.run(
            [
                ToolCall("ts1", "ToolScript", [{"action": "call", "code": code}]),
                ToolCall("m1", "MCP", [{"action": "describe", "server": "test", "tool": "echo"}]),
            ]
        )
        assert len(messages) == 2
        content = str(messages[0]["content"])
        assert "ToolScript ok" in content
        assert "calls: 3 [tr.1-tr.3]" in content
        assert "done" in content
        assert "<MCPCall" not in content  # nested outputs never enter the message stream
        assert "<MCPDescribe" in str(messages[1]["content"])  # the batch's own next call still runs
        for key in ("tr.1", "tr.2", "tr.3"):
            assert "<MCPCall" in s.tool_results[key]

    def test_stdout_and_stderr_captured(self, tmp_path):
        s = _mcp_session(tmp_path)
        code = 'print("out-line")\nimport sys\nprint("err-line", file=sys.stderr)\n'
        content = _run_script(s, code)
        assert "ToolScript ok" in content
        assert "calls: 0" in content
        assert "stdout:" in content and "out-line" in content
        assert "stderr:" in content and "err-line" in content

    def test_refused_nested_call_aborts_script_but_batch_continues(self, tmp_path):
        s = _mcp_session(tmp_path)
        s.mcp.tools["test"] = [mcp_tool_info("test", "echo")]
        code = 'call("MCP", {"server": "test", "tool": "echo", "arguments": {}})\nprint("after")\n'
        answers = iter(["y", "n"])
        runner = ToolRunner(s, ContextManager(s), input_fn=lambda prompt: next(answers), output_fn=lambda text: None)
        messages = runner.run(
            [
                ToolCall("ts1", "ToolScript", [{"action": "call", "code": code}]),
                ToolCall("m1", "MCP", [{"action": "describe", "server": "test", "tool": "echo"}]),
            ]
        )
        assert len(messages) == 2
        content = str(messages[0]["content"])
        assert "ToolScript failed" in content
        assert "nested call refused by user" in content
        assert "after" not in content  # the script aborted before its print
        assert "<MCPDescribe" in str(messages[1]["content"])  # the outer batch kept going

    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            ('call("ToolScript", {})\n', 'call("ToolScript", ...) is not allowed'),
            ('call("Delegate", {})\n', "Delegate is not scriptable"),
            ('call("Job", {})\n', "Job is not scriptable"),
            ('call("Reed", {})\n', 'unknown tool "Reed"'),
        ],
    )
    def test_forbidden_call_names(self, tmp_path, code, expected):
        s = _mcp_session(tmp_path)
        content = _run_script(s, code)
        assert expected in content


class TestJsonFormat:
    def test_declared_schema_with_structured_content_returns_dict(self, tmp_path, monkeypatch):
        s = _mcp_session(tmp_path)
        s.mcp.tools["test"] = [mcp_tool_info("test", "lookup", output_schema={"type": "object"})]

        async def fake_call(config, headers, name, arguments):
            return SimpleNamespace(content=[], structuredContent={"answer": 42})

        monkeypatch.setattr(s.mcp, "_call_tool", fake_call)
        code = 'd = call("MCP", {"server": "test", "tool": "lookup", "arguments": {}}, format="json")\nprint(d)\n'
        content = _run_script(s, code)
        assert "ToolScript ok" in content
        assert "'answer': 42" in content

    def test_undeclared_schema_with_json_text_returns_dict(self, tmp_path, monkeypatch):
        s = _mcp_session(tmp_path)
        s.mcp.tools["test"] = [mcp_tool_info("test", "echo")]

        async def fake_call(config, headers, name, arguments):
            return SimpleNamespace(content=[SimpleNamespace(type="text", text='{"a": 1}')])

        monkeypatch.setattr(s.mcp, "_call_tool", fake_call)
        code = 'd = call("MCP", {"server": "test", "tool": "echo", "arguments": {}}, format="json")\nprint(d)\n'
        content = _run_script(s, code)
        assert "ToolScript ok" in content
        assert "'a': 1" in content

    def test_undeclared_schema_with_non_json_text_errors(self, tmp_path, monkeypatch):
        s = _mcp_session(tmp_path)
        s.mcp.tools["test"] = [mcp_tool_info("test", "echo")]

        async def fake_call(config, headers, name, arguments):
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="hello")])

        monkeypatch.setattr(s.mcp, "_call_tool", fake_call)
        code = 'call("MCP", {"server": "test", "tool": "echo", "arguments": {}}, format="json")\n'
        content = _run_script(s, code)
        assert "ToolScript failed" in content
        assert 'MCP returned text that is not JSON for tool "echo"' in content

    def test_declared_schema_without_structured_content_errors(self, tmp_path, monkeypatch):
        s = _mcp_session(tmp_path)
        s.mcp.tools["test"] = [mcp_tool_info("test", "lookup", output_schema={"type": "object"})]

        async def fake_call(config, headers, name, arguments):
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="hello")])

        monkeypatch.setattr(s.mcp, "_call_tool", fake_call)
        code = 'call("MCP", {"server": "test", "tool": "lookup", "arguments": {}}, format="json")\n'
        content = _run_script(s, code)
        assert "ToolScript failed" in content
        assert 'server declared outputSchema but no structuredContent for tool "lookup"' in content


class TestScriptFailures:
    def test_script_exception_shows_source_line(self, tmp_path):
        s = _mcp_session(tmp_path)
        code = 'x = 1\nraise ValueError("boom")\n'
        content = _run_script(s, code)
        assert "ToolScript failed" in content
        assert "ValueError: boom" in content
        assert 'raise ValueError("boom")' in content  # source line via linecache
        assert "x = 1" in content

    def test_infinite_loop_hits_time_budget(self, tmp_path, monkeypatch):
        import minacode.tools.toolscript as toolscript_module

        monkeypatch.setattr(toolscript_module, "SCRIPT_TIME_LIMIT", 0.1)
        s = _mcp_session(tmp_path)
        content = _run_script(s, "while True:\n    pass\n")
        assert "ToolScript failed" in content
        assert "exceeded" in content

    def test_huge_stdout_is_bounded(self, tmp_path):
        s = _mcp_session(tmp_path)
        content = _run_script(s, 'print("x" * 100_000)\n')
        assert len(content) < 50_000
        assert "<bounded_output" in content


class TestGate:
    def test_needs_confirmation_and_parallel_safety(self, tmp_path):
        s = _mcp_session(tmp_path)
        assert ToolScript(s, [{"action": "call", "code": "print(1)"}]).needs_confirmation() is True
        assert ToolScript(s, [{"action": "describe", "tools": ["test.echo"]}]).needs_confirmation() is False
        runner = _runner(s)
        assert runner.parallel_safe(ToolCall("t1", "ToolScript", [{"action": "call", "code": "print(1)"}])) is False
        assert runner.parallel_safe(ToolCall("t2", "ToolScript", [{"action": "describe", "tools": ["test.echo"]}])) is False

    def test_mcp_params_schema_has_no_format_key(self):
        assert "format" not in MCPTool.params_schema()["properties"]

    def test_yolo_skips_nested_confirmation(self, tmp_path, monkeypatch):
        s = _mcp_session(tmp_path)
        s.settings.yolo = True
        s.mcp.tools["test"] = [mcp_tool_info("test", "echo")]

        async def fake_call(config, headers, name, arguments):
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="ok")])

        monkeypatch.setattr(s.mcp, "_call_tool", fake_call)
        code = 'call("MCP", {"server": "test", "tool": "echo", "arguments": {}})\nprint("ok")\n'

        def no_prompt(prompt):
            raise AssertionError("input_fn must not be called under yolo")

        runner = ToolRunner(s, ContextManager(s), input_fn=no_prompt, output_fn=lambda text: None)
        (message,) = runner.run([ToolCall("ts1", "ToolScript", [{"action": "call", "code": code}])])
        content = str(message["content"])
        assert "ToolScript ok" in content
        assert "ok" in content


class TestConfirmationBlockShowsScript:
    def test_confirm_block_contains_full_script(self, tmp_path):
        """The confirmation block shows the script body: the user approves code, not a label."""
        s = _mcp_session(tmp_path)
        runner = _runner(s)
        code = 'for i in range(3):\n    print(call("MCP", {"server": "test", "tool": "echo", "arguments": {"i": i}}))\n'
        args = [{"action": "call", "code": code}]
        tool = ToolScript(s, args)
        block = runner.approval_display(ToolCall("ts-1", "ToolScript", args), tool, "confirm")
        assert block.has_children
        lines = [line for line, _ in block.walk()]
        assert any(line.label == "preview" for line in lines)
        assert any('tool": "echo"' in line.text for line in lines)
        assert any("for i in range(3):" in line.text for line in lines)


# ---------------------------------------------------------------------------
# Stage 3: built-in tools are scriptable (format="text")
# ---------------------------------------------------------------------------


class TestNestedBuiltinCalls:
    def test_nested_read_returns_text_and_stores_result(self, tmp_path):
        (tmp_path / "f.txt").write_text("hello\n", encoding="utf-8")
        s = _mcp_session(tmp_path)
        s.mcp.tools["test"] = [mcp_tool_info("test", "echo")]
        code = 't = call("Read", {"path": "f.txt"})\nprint(t)\n'
        runner = _runner(s)
        messages = runner.run(
            [
                ToolCall("ts1", "ToolScript", [{"action": "call", "code": code}]),
                ToolCall("m1", "MCP", [{"action": "describe", "server": "test", "tool": "echo"}]),
            ]
        )
        assert len(messages) == 2  # nested calls add no tool messages
        content = str(messages[0]["content"])
        assert "ToolScript ok" in content
        assert "calls: 1 [tr.1]" in content
        assert "hello" in content
        assert "<Read path=\"f.txt\">" in content
        assert "<Read path=\"f.txt\">" in s.tool_results["tr.1"]
        assert "<MCPDescribe" in str(messages[1]["content"])

    def test_nested_read_rejects_json_format(self, tmp_path):
        (tmp_path / "f.txt").write_text("x\n", encoding="utf-8")
        s = _mcp_session(tmp_path)
        code = 'call("Read", {"path": "f.txt"}, format="json")\n'
        content = _run_script(s, code)
        assert "ToolScript failed" in content
        assert 'Read does not support format="json"; use format="text"' in content

    def test_nested_read_without_mcp(self, tmp_path):
        (tmp_path / "f.txt").write_text("hi\n", encoding="utf-8")
        s = Session(cwd=str(tmp_path))
        code = 'print(call("Read", {"path": "f.txt"}))\n'
        content = _run_script(s, code)
        assert "ToolScript ok" in content
        assert "hi" in content

    def test_nested_mcp_without_config_fails(self, tmp_path):
        s = Session(cwd=str(tmp_path))
        s.mcp = None
        code = 'call("MCP", {"server": "test", "tool": "echo", "arguments": {}})\n'
        content = _run_script(s, code)
        assert "ToolScript failed" in content
        assert "MCP not configured" in content

    def test_nested_bash_readonly_runs_without_prompt(self, tmp_path):
        s = _mcp_session(tmp_path)
        code = 'out = call("Bash", {"command": "echo hi"})\nprint(out)\n'
        content = _run_script(s, code)
        assert "ToolScript ok" in content
        assert "hi" in content

    def test_nested_bash_refused_aborts_script(self, tmp_path):
        s = _mcp_session(tmp_path)
        code = 'call("Bash", {"command": "mkdir sub"})\nprint("after")\n'
        answers = iter(["y", "n"])
        runner = ToolRunner(s, ContextManager(s), input_fn=lambda prompt: next(answers), output_fn=lambda text: None)
        (message,) = runner.run([ToolCall("ts1", "ToolScript", [{"action": "call", "code": code}])])
        content = str(message["content"])
        assert "ToolScript failed" in content
        assert "nested call refused by user" in content
        assert "after" not in content
        assert not (tmp_path / "sub").exists()

    def test_yolo_skips_nested_bash_confirmation(self, tmp_path):
        s = _mcp_session(tmp_path)
        s.settings.yolo = True
        code = 'call("Bash", {"command": "mkdir made"})\nprint("done")\n'

        def no_prompt(prompt):
            raise AssertionError("input_fn must not be called under yolo")

        runner = ToolRunner(s, ContextManager(s), input_fn=no_prompt, output_fn=lambda text: None)
        (message,) = runner.run([ToolCall("ts1", "ToolScript", [{"action": "call", "code": code}])])
        content = str(message["content"])
        assert "ToolScript ok" in content
        assert (tmp_path / "made").is_dir()

    def test_nested_builtin_message_conservation(self, tmp_path):
        (tmp_path / "f.txt").write_text("x\n", encoding="utf-8")
        s = _mcp_session(tmp_path)
        s.mcp.tools["test"] = [mcp_tool_info("test", "echo")]
        code = 'for i in range(3):\n    call("Read", {"path": "f.txt"})\nprint("done")\n'
        runner = _runner(s)
        messages = runner.run(
            [
                ToolCall("ts1", "ToolScript", [{"action": "call", "code": code}]),
                ToolCall("m1", "MCP", [{"action": "describe", "server": "test", "tool": "echo"}]),
            ]
        )
        assert len(messages) == 2
        content = str(messages[0]["content"])
        assert "calls: 3 [tr.1-tr.3]" in content
        assert "done" in content
        for key in ("tr.1", "tr.2", "tr.3"):
            assert "<Read path=\"f.txt\">" in s.tool_results[key]

    def test_nested_args_conversion_error_names_tool(self, tmp_path):
        s = _mcp_session(tmp_path)
        code = 'call("Bash", {"comand": "echo hi"})\n'
        content = _run_script(s, code)
        assert "ToolScript failed" in content
        assert "Bash: Bash command must be non-empty" in content


class TestNestedEdit:
    def test_nested_edit_applies(self, tmp_path):
        path = tmp_path / "code.txt"
        path.write_text("a\nb\nc\n", encoding="utf-8")
        s = _mcp_session(tmp_path)
        start = ReadTool.anchor(1, "b\n")
        code = f'call("Edit", {{"path": "code.txt", "edits": [{{"op": "replace", "start": "{start}", "end": "{start}", "content": "B\\n"}}]}})\nprint("edited")\n'
        content = _run_script(s, code)
        assert "ToolScript ok" in content
        assert path.read_text(encoding="utf-8") == "a\nB\nc\n"

    def test_nested_edit_stale_anchor_reports_error(self, tmp_path):
        path = tmp_path / "code.txt"
        path.write_text("a\nb\nc\n", encoding="utf-8")
        s = _mcp_session(tmp_path)
        bad = ReadTool.anchor(1, "wrong\n")
        code = f'call("Edit", {{"path": "code.txt", "edits": [{{"op": "replace", "start": "{bad}", "end": "{bad}", "content": "B\\n"}}]}})\nprint("after")\n'
        content = _run_script(s, code)
        assert "ToolScript failed" in content
        assert "stale anchor" in content
        assert "after" not in content
        assert path.read_text(encoding="utf-8") == "a\nb\nc\n"
