"""ToolScript: stage 1 describe + stage 2 scripted nested MCP calls (black-box)."""

from types import SimpleNamespace

import pytest
from mcp_harness import mcp_cfg, mcp_tool_info

from wizolt.base import LogEdge, LogRole, ToolCall, ToolError
from wizolt.config import Config
from wizolt.context import ContextManager
from wizolt.render import UiPrinter
from wizolt.runner import ToolRunner
from wizolt.session import Session, bootstrap_features
from wizolt.tools import MCPTool, ReadTool, Tool, ToolScript, toolblocks, tooloutput

OUTPUT_SHAPE = {"type": "object", "properties": {"ok": {"type": "boolean"}}}


def _mcp_session(tmp_path):
    """A session with one configured MCP server 'test', populated in memory (no network)."""
    s = Session(cwd=str(tmp_path), config=Config.from_dict(mcp_cfg()))
    bootstrap_features(s)
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
        bootstrap_features(s)
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
        # `action` is the discriminator and the only field every call has: requiring the
        # describe-only `tools` made every script emit a list it has no use for -- and under a
        # strict-tools provider, made running a script at all a schema violation.
        assert params["required"] == ["action"]
        assert params["properties"]["tools"]["minItems"] == 1

    def test_toolscript_always_in_schemas(self, tmp_path):
        s = Session(cwd=str(tmp_path))
        bootstrap_features(s)
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

    def test_dotted_name_calls_the_mcp_tool(self, tmp_path, monkeypatch):
        """call("server.tool", {...}) is the call("MCP", {...}) form -- same name the listing shows."""
        s = _mcp_session(tmp_path)
        s.mcp.tools["test"] = [mcp_tool_info("test", "echo")]

        async def fake_call(config, headers, name, arguments):
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="ok " + str(arguments))])

        monkeypatch.setattr(s.mcp, "_call_tool", fake_call)
        content = _run_script(s, 'call("test.echo", {"text": "hi"})\nprint("done")\n')
        assert "ToolScript ok" in content
        assert "calls: 1 [tr.1]" in content
        assert "<MCPCall" in s.tool_results["tr.1"]
        assert "'text': 'hi'" in s.tool_results["tr.1"]

    def test_dotted_name_on_unknown_server_says_so(self, tmp_path):
        content = _run_script(_mcp_session(tmp_path), 'call("ghost.echo", {})\n')
        assert "ToolScript failed" in content
        assert 'unknown tool "ghost.echo": no MCP server named "ghost" is configured' in content

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

    def test_declared_schema_with_an_empty_payload_returns_it(self, tmp_path, monkeypatch):
        """`{}` and `[]` are payloads a search that matched nothing legitimately returns. Reported
        as a missing payload, every such call raised instead of handing the script its answer."""
        s = _mcp_session(tmp_path)
        s.mcp.tools["test"] = [
            mcp_tool_info("test", "empty_object", output_schema={"type": "object"}),
            mcp_tool_info("test", "empty_list", output_schema={"type": "array"}),
        ]

        async def fake_call(config, headers, name, arguments):
            return SimpleNamespace(content=[], structuredContent=[] if name == "empty_list" else {})

        monkeypatch.setattr(s.mcp, "_call_tool", fake_call)
        code = (
            'a = call("MCP", {"server": "test", "tool": "empty_object", "arguments": {}}, format="json")\n'
            'b = call("MCP", {"server": "test", "tool": "empty_list", "arguments": {}}, format="json")\n'
            "print(repr(a), repr(b))\n"
        )
        content = _run_script(s, code)
        assert "ToolScript ok" in content
        assert "{} []" in content

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
        # The call line names the script's size, not its first line: the body is in the viewer.
        assert "ToolScript call 2 lines (31 chars)" in content

    def test_infinite_loop_hits_time_budget(self, tmp_path, monkeypatch):
        import wizolt.tools.toolscript as toolscript_module

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
    def test_confirm_block_contains_script_excerpt(self, tmp_path):
        """The confirmation block shows the script body: the user approves code, not a label."""
        s = _mcp_session(tmp_path)
        code = 'for i in range(3):\n    print(call("MCP", {"server": "test", "tool": "echo", "arguments": {"i": i}}))\n'
        args = [{"action": "call", "code": code}]
        tool = ToolScript(s, args)
        block = toolblocks.approval_display(s, ToolCall("ts-1", "ToolScript", args), tool, "confirm")
        assert block.has_children
        lines = [line for line, _ in block.walk()]
        assert any(line.label == "script" for line in lines)
        assert any('tool": "echo"' in line.text for line in lines)
        assert any("for i in range(3):" in line.text for line in lines)
        # Code lines are lexed as one block by the renderer, so they carry the lexer name.
        assert {line.syntax for line in lines if line.role is LogRole.CODE} == {"python"}
        # No action row was declared, so the block spells out the typed keys, `v` included.
        assert lines[-1].text == "Y/Enter approve · n refuse · v view script · else reason"

    def test_confirm_block_clips_long_script_and_says_how_much_is_hidden(self, tmp_path):
        """A long script is clipped in the transcript; the whole body stays one keypress away."""
        s = _mcp_session(tmp_path)
        code = "\n".join(f"x{index} = {index}" for index in range(30))
        args = [{"action": "call", "code": code}]
        block = toolblocks.approval_display(s, ToolCall("ts-1", "ToolScript", args), ToolScript(s, args), "confirm")
        lines = [line for line, _ in block.walk()]
        code_lines = [line for line in lines if line.role is LogRole.CODE]
        assert len(code_lines) == toolblocks.VIEW_EXCERPT_LINES
        assert code_lines[0].text == "x0 = 0"
        assert lines[-1].text.startswith("… +20 more lines · ")

    def test_view_action_is_offered_and_opens_the_whole_script(self, tmp_path):
        """`v` at the prompt opens the untruncated script, and the prompt re-asks afterwards."""
        s = _mcp_session(tmp_path)
        runner = _runner(s)
        code = "\n".join(f"x{index} = {index}" for index in range(30))
        tool = ToolScript(s, [{"action": "call", "code": code}])
        assert ("View script", "v") in toolblocks.approval_actions(tool, False)
        views = []
        runner.text_viewer = views.append
        replies = iter(["v", "y"])
        runner.input_fn = lambda _prompt: next(replies)
        confirmed, _ = runner.confirm(ToolCall("ts-1", "ToolScript", tool.args), tool)
        assert confirmed
        assert [view.label for view in views] == ["script"]
        assert views[0].text == code and views[0].lexer == "python"

    def test_describe_has_nothing_to_view(self, tmp_path):
        """A describe commits to no code, so no viewer is offered for it."""
        s = _mcp_session(tmp_path)
        tool = ToolScript(s, [{"action": "describe", "tools": ["Read"]}])
        assert tool.approval_view() is None
        assert ("View script", "v") not in toolblocks.approval_actions(tool, False)


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
        assert '<Read path="f.txt">' in content
        assert '<Read path="f.txt">' in s.tool_results["tr.1"]
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
        bootstrap_features(s)
        code = 'print(call("Read", {"path": "f.txt"}))\n'
        content = _run_script(s, code)
        assert "ToolScript ok" in content
        assert "hi" in content

    def test_nested_mcp_without_config_fails(self, tmp_path):
        s = Session(cwd=str(tmp_path))
        bootstrap_features(s)
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
            assert '<Read path="f.txt">' in s.tool_results[key]

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
        code = (
            f'call("Edit", {{"path": "code.txt", "edits": [{{"op": "replace", "start": "{start}", "end": "{start}", "content": "B\\n"}}]}})\nprint("edited")\n'
        )
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


# ---------------------------------------------------------------------------
# Log shape: nested calls indent under the script, and the finish line closes it
# ---------------------------------------------------------------------------


class TestScriptLogShape:
    def _blocks(self, s, code, **kwargs):
        blocks = []
        runner = _runner(s)
        runner.output_fn = blocks.append
        runner.run([ToolCall("ts1", "ToolScript", [{"action": "call", "code": code}])], **kwargs)
        return blocks

    def test_nested_calls_are_indented_under_the_script(self, tmp_path):
        """A nested call is logged as what it is -- a call this script made -- not as a top-level
        one the model asked for. The indent is the whole signal, so it has to be there."""
        (tmp_path / "f.txt").write_text("hello\n", encoding="utf-8")
        s = _mcp_session(tmp_path)
        blocks = self._blocks(s, 'print(call("Read", {"path": "f.txt"}))\n')
        levels = {line.text: level for block in blocks for line, level in block.walk()}
        nested = next(level for text, level in levels.items() if text.startswith("f.txt"))
        top = next(level for text, level in levels.items() if text.startswith("call 1 line"))
        assert nested > top

    def test_nested_calls_carry_the_tree_gutter(self, tmp_path):
        """Indent alone reads as an ordinary call sitting further right. The nested roots continue
        the enclosing call's `|`, so the script, its calls, and its result are one bracket -- and
        the excerpt above them must not close the tree with an END edge first."""
        (tmp_path / "f.txt").write_text("hello\n", encoding="utf-8")
        s = _mcp_session(tmp_path)
        blocks = self._blocks(s, 'print(call("Read", {"path": "f.txt"}))\n')
        roots = [line for block in blocks for line, _ in block.walk() if line.label == "Read"]
        assert roots and all(line.edge is LogEdge.CONTINUE for line in roots)
        excerpt = [line for block in blocks for line, _ in block.walk() if line.role in (LogRole.CODE, LogRole.META)]
        assert not any(line.edge is LogEdge.END for line in excerpt[: len(excerpt) - 1])

    def test_the_rail_survives_a_nested_call_that_logs_children(self, tmp_path):
        """The rail used to stop at the nested root, so a call with a block under it (a Bash
        preview, an Edit diff, an error) punched a hole in the bracket exactly where it was
        tallest. Every rendered row of the nested region carries it now, in one column."""
        s = _mcp_session(tmp_path)
        blocks = self._blocks(s, 'print(call("Bash", {"command": "printf one; printf two"}))\n')
        rows = [row for block in blocks for row in "".join(text for _, text in UiPrinter(output_fn=lambda _text: None).log_segments(block)).splitlines()]
        start = next(index for index, row in enumerate(rows) if row.endswith("Bash printf one; printf two"))
        end = next(index for index, row in enumerate(rows) if "calls 1" in row)  # the script's own result line closes the region
        nested = rows[start:end]
        assert len(nested) > 1, nested  # the call plus the block it logged under itself
        assert all(row.startswith("    │ ") for row in nested), nested

    def test_nesting_depth_is_restored_after_a_failed_script(self, tmp_path):
        """A script that raises must not leave the rest of the session permanently indented."""
        s = _mcp_session(tmp_path)
        runner = _runner(s)
        runner.run([ToolCall("ts1", "ToolScript", [{"action": "call", "code": 'raise ValueError("boom")\n'}])])
        assert runner.nesting == 0

    def test_finish_line_summarizes_the_script_run(self, tmp_path):
        """The block closes with what the script did: how many calls, and what it printed --
        the printed output being all that reaches the model."""
        (tmp_path / "f.txt").write_text("hello\n", encoding="utf-8")
        s = _mcp_session(tmp_path)
        blocks = self._blocks(s, 'call("Read", {"path": "f.txt"})\nprint("counted 1")\n')
        lines = [line for block in blocks for line, _ in block.walk()]
        assert any(line.label.startswith("calls 1") and line.text == "Ctrl-O for more" for line in lines)
        assert any(line.text == "counted 1" for line in lines)

    def test_describe_has_no_script_summary(self, tmp_path):
        """A describe returns tool shapes, not a script envelope, so there is nothing to count."""
        s = _mcp_session(tmp_path)
        block = toolblocks.finish_display(s, ToolCall("ts1", "ToolScript", [{"action": "describe", "tools": ["Read"]}]), "tr.1", "Read\njson:    no", failed=False)
        assert not any(line.label.startswith("calls") for line, _ in block.walk())

    def test_result_fields_parse_the_envelope(self, tmp_path):
        envelope = "ToolScript ok\ncalls: 3 [tr.1-3]\nstdout:\nfirst\nsecond\nstderr:\nnoise\n"
        assert tooloutput.toolscript_result_fields(envelope) == ("3", "first\nsecond", "")
        assert tooloutput.toolscript_result_fields("ToolScript ok\ncalls: ... +120 keys\n") == ("120", "", "")
        assert tooloutput.toolscript_result_fields("Read\njson:    no") is None
        # A failed script keeps the line that names what went wrong; the frames are in the viewer.
        failed = (
            'ToolScript failed\ncalls: 1 [tr.1]\nstdout:\npartial\nerror:\nTraceback (most recent call last):\n  File "<toolscript>", line 2\nValueError: boom'
        )
        assert tooloutput.toolscript_result_fields(failed) == ("1", "partial", "ValueError: boom")


class TestScriptCannotEndTheSession:
    def test_sys_exit_becomes_a_failed_envelope(self, tmp_path):
        """`sys.exit()` is ordinary in standalone Python and a model writes it without thinking.
        As a SystemExit it flew past this tool, past run_one, and out of the agent loop: one line
        of a script ended the session. It is a script failure like any other."""
        s = _mcp_session(tmp_path)
        content = _run_script(s, 'print("before")\nimport sys\nsys.exit(1)\nprint("after")\n')
        assert "ToolScript failed" in content
        assert "SystemExit" in content
        assert "before" in content and "after" not in content

    def test_keyboard_interrupt_still_travels(self, tmp_path):
        """Ctrl-C is the user cancelling the turn, not the script failing; swallowing it would
        report a failed script and carry on."""
        s = _mcp_session(tmp_path)
        runner = _runner(s)
        with pytest.raises(KeyboardInterrupt):
            runner.run([ToolCall("ts1", "ToolScript", [{"action": "call", "code": "raise KeyboardInterrupt\n"}])])


class TestFailedScriptLooksFailed:
    def test_the_result_line_reports_the_failure_and_the_error(self, tmp_path):
        """A script that raised returns its envelope normally, so the call itself did not fail and
        nothing else in the block says otherwise. Without this, a script that died on call 2 of 40
        read exactly like one that finished."""
        s = _mcp_session(tmp_path)
        runner = _runner(s)
        blocks = []
        runner.output_fn = blocks.append
        runner.run([ToolCall("ts1", "ToolScript", [{"action": "call", "code": 'print("partial")\nraise ValueError("boom")\n'}])])
        lines = [line for block in blocks for line, _ in block.walk()]
        head = next(line for line in lines if line.label.startswith(("calls", "failed")))
        assert head.label.startswith("failed · calls 0")
        assert head.role is LogRole.ERROR
        assert any("ValueError: boom" in line.text for line in lines)


class TestNestedCallsDoNotStealTheScriptStdout:
    def test_nested_logging_reaches_the_terminal_not_the_model(self, tmp_path):
        """The capture is for what the script prints. A nested call's log is not that: swallowed,
        it was posted back as the script's own output, and on the headless path the confirmation
        prompt -- input() writes to sys.stdout -- went the same way, stopping the run at a prompt
        nobody could see."""
        import contextlib
        import io

        (tmp_path / "f.txt").write_text("hello\n", encoding="utf-8")
        s = _mcp_session(tmp_path)
        runner = ToolRunner(s, ContextManager(s), input_fn=lambda prompt: "y", output_fn=print)  # the headless default
        terminal = io.StringIO()
        with contextlib.redirect_stdout(terminal):
            (message,) = runner.run(
                [ToolCall("ts1", "ToolScript", [{"action": "call", "code": 't = call("Read", {"path": "f.txt"})\nprint("script says hi")\n'}])]
            )

        content = str(message["content"])
        assert "Read f.txt" in terminal.getvalue()  # the nested call was logged where the user is
        assert content.split("stdout:")[1].strip() == "script says hi"  # and nowhere near the result

    def test_the_outer_stream_owner_gets_it_back(self, tmp_path):
        """Stepping aside restores whatever was current when capture began, not the process's
        original stdout: an outer redirect owns those streams and must keep owning them."""
        import contextlib
        import io

        (tmp_path / "f.txt").write_text("hello\n", encoding="utf-8")
        s = _mcp_session(tmp_path)
        runner = ToolRunner(s, ContextManager(s), input_fn=lambda prompt: "y", output_fn=print)
        outer = io.StringIO()
        with contextlib.redirect_stdout(outer):
            runner.run([ToolCall("ts1", "ToolScript", [{"action": "call", "code": 'call("Read", {"path": "f.txt"})\n'}])])

        assert "Read f.txt" in outer.getvalue()


def test_non_string_tool_name_stays_a_plain_tool_error(tmp_path):
    """`name` is whatever the script passed. The dotted-name checks read it as text, so a number
    has to be turned away before them or it surfaces as a TypeError traceback."""
    content = _run_script(_mcp_session(tmp_path), "call(123, {})\n")
    assert "ToolScript failed" in content
    assert 'unknown tool "123"' in content
    assert "TypeError" not in content


class TestWhitelistGating:
    """A session tool whitelist gates nested calls and describe entries, not just schemas."""

    def test_nested_call_rejects_tool_outside_whitelist(self, tmp_path):
        s = _mcp_session(tmp_path)
        s.tool_names = ("Read", "Search", "ToolScript")
        content = _run_script(s, 'call("Bash", {"command": "echo hi"})\n')
        assert "ToolScript failed" in content
        assert "Bash is not available in this session" in content

    def test_nested_call_inside_whitelist_still_runs(self, tmp_path):
        (tmp_path / "a.txt").write_text("hello\n", encoding="utf-8")
        s = _mcp_session(tmp_path)
        s.tool_names = ("Read", "ToolScript")
        content = _run_script(s, 'print(call("Read", {"path": "a.txt"}))\n')
        assert "ToolScript ok" in content
        assert "hello" in content

    def test_describe_gates_schema_of_tool_outside_whitelist(self, tmp_path):
        s = _mcp_session(tmp_path)
        s.tool_names = ("Read", "Search", "ToolScript")
        out = _describe(s, ["Bash", "Read"])
        assert "Bash: not available in this session" in out
        assert "Read\n" in out
        assert "  args:    path  string" in out
