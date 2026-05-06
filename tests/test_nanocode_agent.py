import json

import nanocode
from nanocode import Agent, KnownItem, ParsedToolCall, RecentToolCallResultBuffer, Session, ToolCallEvent, ToolCallExecution, VerificationStatus


def test_agent_tool_results_go_to_recent_area_and_logs_not_conversation(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)

    latest = agent.execute_tool_calls(
        [
            {
                "name": "Read",
                "intention": "read sample",
                "args": ["sample.txt", "0", "1"],
            }
        ]
    )

    event = session.conversation[-1]
    assert isinstance(event, ToolCallEvent)
    assert "alpha" in latest
    assert "alpha" not in event.format()
    assert event.result_file
    assert event.outcome == "success"
    assert "<outcome>success</outcome>" in event.format()
    assert nanocode.ToolCallRunner.TOOL_RESULTS_DIR in event.result_file
    assert "alpha" in (tmp_path / event.result_file).read_text(encoding="utf-8")

    prompt = agent.build_user_prompt()
    assert "Recent_Tool_Call_Results" in prompt
    assert "alpha" in prompt
    assert "alpha" in agent.build_user_prompt()

    agent.apply_response(
        {
            "actions": [
                {
                    "type": "tool_summary",
                    "tool": "Read",
                    "intention": "read sample",
                    "outcome": "success",
                    "summary": "Read sample.txt line 1.",
                    "key_evidence": ["sample.txt:1 alpha"],
                    "known_facts": [{"fact": "sample.txt line 1 is alpha.", "details": ["sample.txt:1 alpha"]}],
                    "result_file": event.result_file,
                    "needs_raw_read": False,
                }
            ]
        }
    )

    assert "Read sample.txt line 1." in event.summary
    assert event.key_details == ["sample.txt:1 alpha"]
    assert "sample.txt:1 alpha" in event.format()
    assert "<key_details>\n    <detail>sample.txt:1 alpha</detail>\n  </key_details>" in event.format()
    assert "alpha\n  </content>" not in event.format()
    assert session.current.known == [KnownItem(fact="sample.txt line 1 is alpha.", details=["sample.txt:1 alpha"])]
    assert "  Known\n" in agent.state_updater.latest_report
    assert "sample.txt line 1 is alpha." in agent.state_updater.latest_report


def test_recent_tool_call_result_buffer_keeps_last_batch_and_trims_older_blocks():
    def execution(name: str, output: str) -> ToolCallExecution:
        return ToolCallExecution(
            call=ParsedToolCall(name="Read", intention="read " + name, args=[name]),
            outcome="success",
            output=output,
            result_file=name + ".log",
            result_file_lines=1,
        )

    keep_buffer = RecentToolCallResultBuffer(older_char_budget=1000)
    keep_buffer.record([execution("first", "first-output-token")])
    keep_buffer.record([execution("second", "second-output-token")])
    keep_prompt = keep_buffer.format()
    assert "second-output-token" in keep_prompt.split("</last_batch>", 1)[0]
    assert "first-output-token" in keep_prompt.split("<older_buffer", 1)[1]

    trim_buffer = RecentToolCallResultBuffer(older_char_budget=1)
    trim_buffer.record([execution("first", "first-output-token")])
    trim_buffer.record([execution("second", "second-output-token")])
    trim_prompt = trim_buffer.format()
    assert "second-output-token" in trim_prompt
    assert "first-output-token" not in trim_prompt


def test_agent_user_prompt_has_no_blackboard_section(tmp_path):
    prompt = Agent(Session(cwd=str(tmp_path))).build_user_prompt()

    assert "Blackboard_Keys" not in prompt
    assert "Blackboard" not in prompt


def test_agent_request_calls_chat_completions_and_parses_json(tmp_path, monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps(
                {
                    "choices": [{"message": {"content": json.dumps({"type": "message", "text": "ok"})}}],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["authorization"] = request.headers["Authorization"]
        return FakeResponse()

    monkeypatch.setattr(nanocode.urllib.request, "urlopen", fake_urlopen)
    session = Session(cwd=str(tmp_path), api_url="https://example.test/v1", api_key="key", model="model", model_timeout=12, stream=False)
    session.prompt_price_per_1m_tokens = 1.0
    session.completion_price_per_1m_tokens = 2.0

    response = Agent(session).request("system", "user")

    assert response == {"actions": [{"type": "message", "text": "ok"}]}
    assert captured["url"] == "https://example.test/v1/chat/completions"
    assert captured["timeout"] == 12
    assert captured["authorization"] == "Bearer key"
    assert captured["payload"]["model"] == "model"
    assert captured["payload"]["messages"] == [{"role": "system", "content": "system"}, {"role": "user", "content": "user"}]
    assert "response_format" not in captured["payload"]
    assert "reasoning_effort" not in captured["payload"]
    assert "reasoning" not in captured["payload"]
    assert session.last_prompt_tokens == 2
    assert session.last_completion_tokens == 3
    assert session.last_total_tokens == 5
    assert abs(session.last_cost_usd - 0.000008) < 1e-12
    assert abs(session.session_cost_usd - 0.000008) < 1e-12


def test_agent_request_streams_and_reports_completed_actions(tmp_path, monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def __iter__(self):
            chunks = [
                '{"type":"tool","name":"Read",',
                '"intention":"read sample","args":["sample.txt"]}__END_ACTION__',
                '{"type":"message","text":"done"}__END_ACTION__',
            ]
            for chunk in chunks:
                yield ("data: " + json.dumps({"choices": [{"delta": {"content": chunk}}]}) + "\n").encode("utf-8")
            yield (
                "data: "
                + json.dumps({"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}})
                + "\n"
            ).encode("utf-8")
            yield b"data: [DONE]\n"

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr(nanocode.urllib.request, "urlopen", fake_urlopen)
    session = Session(cwd=str(tmp_path), api_url="https://example.test/v1", api_key="key", model="model")
    actions = []

    response = Agent(session).request("system", "user", on_action=actions.append)

    assert captured["payload"]["stream"] is True
    assert captured["payload"]["stream_options"] == {"include_usage": True}
    assert response["actions"] == [
        {"type": "tool", "name": "Read", "intention": "read sample", "args": ["sample.txt"]},
        {"type": "message", "text": "done"},
    ]
    assert actions == response["actions"]
    assert session.last_prompt_tokens == 2
    assert session.last_completion_tokens == 3
    assert session.last_total_tokens == 5
    assert session.session_total_tokens == 5


def test_agent_run_previews_streamed_tool_action_before_execution_report(tmp_path, monkeypatch):
    (tmp_path / "sample.txt").write_text("alpha\n", encoding="utf-8")
    captured_payloads = []
    responses = [
        [
            '{"type":"tool","name":"Read",',
            '"intention":"read sample","args":["sample.txt","0","1"]}__END_ACTION__',
        ],
        [
            '{"type":"tool_summary","tool":"Read","intention":"read sample","outcome":"success",',
            '"summary":"Read sample.txt.","key_evidence":["alpha"],"known_facts":null,',
            '"result_file":null,"needs_raw_read":false}__END_ACTION__',
            '{"type":"message","text":"done"}__END_ACTION__',
        ],
    ]

    class FakeResponse:
        def __init__(self, chunks):
            self.chunks = chunks

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def __iter__(self):
            for chunk in self.chunks:
                yield ("data: " + json.dumps({"choices": [{"delta": {"content": chunk}}]}) + "\n").encode("utf-8")
            yield b"data: [DONE]\n"

    def fake_urlopen(request, timeout):
        captured_payloads.append(json.loads(request.data.decode("utf-8")))
        return FakeResponse(responses.pop(0))

    monkeypatch.setattr(nanocode.urllib.request, "urlopen", fake_urlopen)
    session = Session(cwd=str(tmp_path), api_url="https://example.test/v1", api_key="key", model="model")
    messages = []

    response = Agent(session).run("read sample", on_message=messages.append)

    assert response["actions"][-1] == {"type": "message", "text": "done"}
    assert len(captured_payloads) == 2
    assert [payload["stream"] for payload in captured_payloads] == [True, True]
    assert messages[0] == "Queued: Read sample.txt:0-1 - read sample"
    assert any(message.startswith("Tool Calls") for message in messages[1:])
    assert messages[-1] == "done"


def test_agent_stream_preview_summarizes_long_tool_arguments(tmp_path):
    agent = Agent(Session(cwd=str(tmp_path)))

    bash_preview = agent._format_stream_action_preview(
        {
            "type": "tool",
            "name": "Bash",
            "intention": "Create a test file for fingerprint experiments.",
            "args": ["cat <<EOF > test_fingerprint.txt\nLine 1: Alpha\nLine 2: Beta\nEOF"],
        }
    )
    replace_preview = agent._format_stream_action_preview(
        {
            "type": "tool",
            "name": "ReplaceRange",
            "intention": "Test a valid ReplaceRange to ensure baseline works.",
            "args": [
                "test_fingerprint.txt",
                "1",
                "4",
                "5743477810356510368",
                "Line 2: Beta Updated\nLine 3: Gamma Updated\nLine 4: Delta Updated",
            ],
        }
    )

    assert bash_preview == "Queued: Bash - Create a test file for fingerprint experiments."
    assert replace_preview == "Queued: ReplaceRange test_fingerprint.txt:1-4 - Test a valid ReplaceRange to ensure baseline works."


def test_agent_request_uses_openrouter_reasoning_payload(tmp_path, monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps({"choices": [{"message": {"content": json.dumps({"type": "message", "text": "ok"})}}], "usage": {}}).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr(nanocode.urllib.request, "urlopen", fake_urlopen)
    session = Session(
        cwd=str(tmp_path),
        api_url="https://openrouter.ai/api/v1",
        api_key="key",
        model="model",
        reasoning_effort="high",
        stream=False,
    )

    Agent(session).request("system", "user")

    assert captured["payload"]["reasoning"] == {"effort": "high"}
    assert "reasoning_effort" not in captured["payload"]


def test_agent_request_writes_debug_prompt(tmp_path, monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps({"choices": [{"message": {"content": json.dumps({"type": "message", "text": "ok"})}}], "usage": {}}).encode("utf-8")

    monkeypatch.setattr(nanocode.urllib.request, "urlopen", lambda request, timeout: FakeResponse())
    session = Session(
        cwd=str(tmp_path),
        api_url="https://example.test/v1",
        api_key="key",
        model="model",
        model_timeout=12,
        debug=True,
        stream=False,
    )

    response = Agent(session).request("system prompt", "user prompt")

    files = list((tmp_path / ".nanocode" / "debug").glob("*-0001-main.txt"))
    assert response["actions"][0]["text"] == "ok"
    assert len(files) == 1
    content = files[0].read_text(encoding="utf-8")
    assert content == "--- system message 1 ---\nsystem prompt\n\n--- user message 2 ---\nuser prompt\n"
    assert "model:" not in content
    assert "extra_params:" not in content
    assert "key" not in content


def test_agent_request_accepts_json_fenced_model_content(tmp_path, monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps(
                {
                    "choices": [{"message": {"content": '```json\n{"type":"message","text":"ok"}\n__END_ACTION__\n```'}}],
                    "usage": {},
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        return FakeResponse()

    monkeypatch.setattr(nanocode.urllib.request, "urlopen", fake_urlopen)
    session = Session(cwd=str(tmp_path), api_url="https://example.test/v1", api_key="key", model="model", stream=False)

    response = Agent(session).request("system", "user")

    assert response == {"actions": [{"type": "message", "text": "ok"}]}


def test_agent_request_accepts_leaked_think_tags_before_json(tmp_path):
    client = Agent(Session(cwd=str(tmp_path))).model_client

    assert client._parse_model_content('</think>{"type":"message","text":"ok"}\n__END_ACTION__') == {
        "actions": [{"type": "message", "text": "ok"}],
    }
    assert client._parse_model_content('<think>reasoning</think>\n{"type":"message","text":"ok"}\n__END_ACTION__') == {
        "actions": [{"type": "message", "text": "ok"}],
    }


def test_agent_request_accepts_pretty_action_frames_and_marker_variants(tmp_path):
    client = Agent(Session(cwd=str(tmp_path))).model_client

    response = client._parse_model_content(
        '{\n  "type": "message",\n  "text": "ok"\n}\n**END_ACTION**\n{"type":"goal","text":"next"}\nEND_ACTION'
    )

    assert response == {"actions": [{"type": "message", "text": "ok"}, {"type": "goal", "text": "next"}]}


def test_agent_request_accepts_inline_action_frame_markers(tmp_path):
    client = Agent(Session(cwd=str(tmp_path))).model_client

    response = client._parse_model_content('{"type":"message","text":"ok"}__END_ACTION__{"type":"goal","text":"next"}__END_ACTION__')

    assert response == {"actions": [{"type": "message", "text": "ok"}, {"type": "goal", "text": "next"}]}


def test_agent_request_ignores_bad_action_frames_when_other_actions_are_valid(tmp_path):
    client = Agent(Session(cwd=str(tmp_path))).model_client

    response = client._parse_model_content('plain answer\n__END_ACTION__\n{"type":"message","text":"ok"}\n__END_ACTION__')

    assert response["actions"] == [{"type": "message", "text": "ok"}]
    assert response["_format_frame_errors"] == ["frame 1: expected JSON object action"]


def test_agent_request_rejects_native_tool_call_syntax(tmp_path):
    client = Agent(Session(cwd=str(tmp_path))).model_client

    response = client._parse_model_content('<tool_call>Read("nanocode.py", 0, 100)')

    assert response["actions"] == []
    assert "Native tool_call syntax is not supported" in response["_format_error"]
    assert '"name":"Read"' in response["_format_error"]
    assert '"args":["nanocode.py","0","100"]' in response["_format_error"]


def test_agent_request_wraps_non_json_model_content_as_format_error(tmp_path, monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "plain answer"}}], "usage": {}}).encode("utf-8")

    def fake_urlopen(request, timeout):
        return FakeResponse()

    monkeypatch.setattr(nanocode.urllib.request, "urlopen", fake_urlopen)
    session = Session(cwd=str(tmp_path), api_url="https://example.test/v1", api_key="key", model="model", stream=False)

    response = Agent(session).request("system", "user")

    assert response["actions"] == []
    assert "expected at least one valid action frame" in response["_format_error"]
    assert "plain answer" in response["_format_error"]


def test_agent_request_wraps_missing_message_content_as_format_error(tmp_path, monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": None},
                        }
                    ],
                    "usage": {},
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        return FakeResponse()

    monkeypatch.setattr(nanocode.urllib.request, "urlopen", fake_urlopen)
    session = Session(cwd=str(tmp_path), api_url="https://example.test/v1", api_key="key", model="model", stream=False)

    response = Agent(session).request("system", "user")

    assert response["actions"] == []
    assert "expected one JSON object" in response["_format_error"]
    assert "API response missing message content" in response["_format_error"]


def test_agent_keeps_known_items_structured_in_current_and_prompt(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)

    agent.apply_response(
        {
            "actions": [
                {
                    "type": "known",
                    "items": [
                        {"fact": "Search only supports rg and Python fallback.", "details": ["grep was removed"]},
                        {"fact": "Search only supports rg and Python fallback.", "details": ["duplicate ignored"]},
                    ],
                }
            ]
        }
    )

    assert session.current.known == [KnownItem(fact="Search only supports rg and Python fallback.", details=["grep was removed"])]

    prompt = agent.build_user_prompt()
    assert "<KnownItem>" in prompt
    assert "<fact>Search only supports rg and Python fallback.</fact>" in prompt
    assert "  <details>\n    <detail>grep was removed</detail>\n  </details>" in prompt
    assert "duplicate ignored" not in prompt


def test_agent_system_prompt_guides_known_and_tool_summaries(tmp_path):
    prompt = Agent(Session(cwd=str(tmp_path))).build_system_prompt()

    assert "Rules:" in prompt
    assert "{ __tools__ }" not in prompt
    assert "- Read(filepath" in prompt
    assert "Use one OR search for related symbols: A|B|C or 3+ plain args" in prompt
    assert "Options: path=FILE, context=N|N, glob=*.py or bare glob." in prompt
    assert "Output exactly one known action every turn." in prompt
    assert "tool_summary for fresh tool results." in prompt
    assert "Fresh tool results: summarize all first; each tool_summary needs known_facts." in prompt
    assert "Known: items=[] means no new durable facts" in prompt
    assert '"known_facts": null | [{"fact": "string", "details": null | ["string"]}]' in prompt
    assert "Order:" in prompt
    assert "goal if needed." in prompt
    assert "known always." in prompt
    assert "Blackboard" not in prompt
    assert "Current_Context" not in prompt
    assert '{"type": "context"' not in prompt


def test_agent_state_report_only_includes_real_plan_and_known_changes(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)

    response = {
        "actions": [
            {"type": "plan", "mode": "replace", "items": [{"id": "p1", "text": "Inspect file", "status": "todo"}]},
            {"type": "known", "items": [{"fact": "Search uses rg.", "details": ["Python fallback exists"]}]},
        ]
    }

    agent.apply_response(response)

    assert "State Updated | VERIFY:idle" in agent.state_updater.latest_report
    assert "  Plan\n" in agent.state_updater.latest_report
    assert "    1. [○ todo] Inspect file" in agent.state_updater.latest_report
    assert "  Known\n" in agent.state_updater.latest_report
    assert "    1. Search uses rg. | Python fallback exists" in agent.state_updater.latest_report

    agent.apply_response(response)

    assert agent.state_updater.latest_report == ""


def test_agent_resets_verification_when_goal_changes(tmp_path):
    session = Session(cwd=str(tmp_path))
    session.current.goal = "old goal"
    session.current.verification.goal = "old goal"
    session.current.verification.status = VerificationStatus.DONE
    session.current.verification.method = "old check"
    session.current.verification.evidence = "old evidence"
    agent = Agent(session)

    agent.apply_response({"actions": [{"type": "goal", "text": "new goal"}]})

    assert session.current.verification.goal == ""
    assert session.current.verification.status == VerificationStatus.IDLE
    assert session.current.verification.method == ""
    assert session.current.verification.evidence == ""

    agent.apply_response({"actions": [{"type": "verify", "method": "run tests", "status": "pending", "evidence": None}]})

    assert session.current.verification.goal == "new goal"
    assert session.current.verification.status == VerificationStatus.REQUIRED
    assert session.current.verification.method == "run tests"
    assert session.current.verification.evidence == ""
    assert "<goal>new goal</goal>" in agent.build_user_prompt()


def test_agent_execute_tool_calls_requests_confirmation_for_edit_tools(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("old\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    confirmations = []

    latest = agent.execute_tool_calls(
        [{"name": "Edit", "intention": "edit sample", "args": ["sample.txt", "old", "new"]}],
        confirm=lambda call, tool: confirmations.append((call.executed, tool.display())) or False,
    )

    assert confirmations
    assert confirmations[0][0] == 'Edit("sample.txt", "old", "new")'
    assert "-old" in confirmations[0][1]
    assert "+new" in confirmations[0][1]
    assert "Cancelled: user refused" in latest
    assert path.read_text(encoding="utf-8") == "old\n"


def test_agent_execute_tool_calls_records_refusal_reason(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("old\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)

    latest = agent.execute_tool_calls(
        [{"name": "Edit", "intention": "edit sample", "args": ["sample.txt", "old", "new"]}],
        confirm=lambda call, tool: "please inspect tests first",
    )

    assert "Cancelled: user refused: please inspect tests first" in latest
    assert path.read_text(encoding="utf-8") == "old\n"
    event = agent.latest_tool_call_events[0]
    assert event.outcome == "failure"
    log_path = tmp_path / event.result_file
    assert "Cancelled: user refused: please inspect tests first" in log_path.read_text(encoding="utf-8")
    assert "please inspect tests first" in agent.build_user_prompt()


def test_agent_execute_tool_calls_rejects_failed_preview_before_confirmation(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("old\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    confirmations = []

    latest = agent.execute_tool_calls(
        [{"name": "ReplaceRange", "intention": "edit stale range", "args": ["sample.txt", "0", "1", "bad", "new"]}],
        confirm=lambda call, tool: confirmations.append((call.executed, tool.display())) or True,
    )

    assert confirmations == []
    assert "ToolCallError: preview unavailable: fingerprint mismatch" in latest
    assert path.read_text(encoding="utf-8") == "old\n"


def test_agent_execute_tool_calls_logs_malformed_tool_call(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)

    latest = agent.execute_tool_calls([{"intention": "bad call", "args": []}])

    assert "ToolCallError: tool call missing name" in latest
    event = agent.latest_tool_call_events[0]
    assert event.outcome == "failure"
    assert event.executed.startswith("InvalidToolCall(")
    log_path = tmp_path / event.result_file
    assert "ToolCallError: tool call missing name" in log_path.read_text(encoding="utf-8")


def test_agent_execute_tool_calls_shows_auto_approval_in_yolo_mode(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("old\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path), yolo=True)
    agent = Agent(session)
    confirmations = []
    auto_approvals = []

    latest = agent.execute_tool_calls(
        [{"name": "Edit", "intention": "edit sample", "args": ["sample.txt", "old", "new"]}],
        confirm=lambda call, tool: confirmations.append(call.executed) or False,
        on_auto_approve=lambda call, tool: auto_approvals.append((call.executed, tool.display())),
    )

    assert confirmations == []
    assert auto_approvals
    assert auto_approvals[0][0] == 'Edit("sample.txt", "old", "new")'
    assert "-old" in auto_approvals[0][1]
    assert "+new" in auto_approvals[0][1]
    assert "outcome>success" in latest
    assert path.read_text(encoding="utf-8") == "new\n"


def test_agent_run_loops_tool_results_into_next_model_prompt(tmp_path):
    (tmp_path / "sample.txt").write_text("alpha\n", encoding="utf-8")

    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {
                    "actions": [{"type": "tool", "name": "Read", "intention": "read sample", "args": ["sample.txt", "0", "1"]}]
                },
                {
                    "actions": [
                        {
                            "type": "tool_summary",
                            "tool": "Read",
                            "intention": "read sample",
                            "outcome": "success",
                            "summary": "Read sample.txt and found alpha.",
                            "key_evidence": ["alpha"],
                            "result_file": None,
                            "needs_raw_read": False,
                        },
                        {"type": "message", "text": "done"},
                    ],
                },
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    fake_client = FakeModelClient()
    agent.model_client = fake_client

    messages = []
    response = agent.run("read sample", on_message=messages.append)

    assert response["actions"][-1]["text"] == "done"
    assert messages[0].startswith("Tool Calls\n")
    assert '1. [success] Read("sample.txt", "0", "1")' in messages[0]
    assert "why: read sample" in messages[0]
    assert "log: .nanocode/tool_results/" in messages[0]
    assert messages[-1] == "done"
    assert "alpha" not in fake_client.user_prompts[0]
    assert "alpha" in fake_client.user_prompts[1]
    assert "alpha" in agent.recent_tool_call_results.format()
    assert session.current.known == [KnownItem(fact="Read sample.txt and found alpha.", details=["alpha"])]
    assert session.current.user_input == "read sample"
    assert session.current.goal_reached is True


def test_agent_run_keeps_tool_results_when_format_retry_happens(tmp_path):
    (tmp_path / "sample.txt").write_text("alpha\n", encoding="utf-8")

    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {"actions": [{"type": "tool", "name": "Read", "intention": "read sample", "args": ["sample.txt", "0", "1"]}]},
                {"_format_error": "Invalid model output: plain answer", "actions": []},
                {
                    "actions": [
                        {
                            "type": "tool_summary",
                            "tool": "Read",
                            "intention": "read sample",
                            "outcome": "success",
                            "summary": "Read sample.txt.",
                            "key_evidence": ["alpha"],
                            "known_facts": None,
                            "result_file": None,
                            "needs_raw_read": False,
                        },
                        {"type": "message", "text": "done"},
                    ]
                },
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    agent.model_client = FakeModelClient()

    response = agent.run("read sample")

    assert response["actions"][-1]["text"] == "done"
    assert "alpha" in agent.model_client.user_prompts[1]
    assert "alpha" in agent.model_client.user_prompts[2]
    assert "Invalid model output: plain answer" in agent.model_client.user_prompts[2]
    assert "alpha" in agent.recent_tool_call_results.format()


def test_agent_run_does_not_gate_when_tool_summary_does_not_update_known(tmp_path):
    (tmp_path / "sample.txt").write_text("alpha\n", encoding="utf-8")

    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {"actions": [{"type": "tool", "name": "Read", "intention": "read sample", "args": ["sample.txt", "0", "1"]}]},
                {
                    "actions": [
                        {
                            "type": "tool_summary",
                            "tool": "Read",
                            "intention": "read sample",
                            "outcome": "success",
                            "summary": "Read sample.txt.",
                            "key_evidence": None,
                            "result_file": None,
                            "needs_raw_read": False,
                        },
                        {"type": "message", "text": "done too early"},
                    ],
                },
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    agent.model_client = FakeModelClient()
    messages = []

    response = agent.run("read sample", on_message=messages.append)

    assert response["actions"][-1]["text"] == "done too early"
    assert all("Known_Gate:" not in prompt for prompt in agent.model_client.user_prompts)
    assert "Retrying: Known was not reviewed after tool results." not in messages
    assert "done too early" in messages
    assert session.current.known == []


def test_agent_run_does_not_gate_when_tool_results_are_not_reviewed_for_known(tmp_path):
    (tmp_path / "sample.txt").write_text("alpha\n", encoding="utf-8")

    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {"actions": [{"type": "tool", "name": "Read", "intention": "read sample", "args": ["sample.txt", "0", "1"]}]},
                {"actions": [{"type": "message", "text": "done too early"}]},
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    agent.model_client = FakeModelClient()
    messages = []

    response = agent.run("read sample", on_message=messages.append)

    assert response["actions"][-1]["text"] == "done too early"
    assert all("Known_Gate:" not in prompt for prompt in agent.model_client.user_prompts)
    assert "Retrying: Known was not reviewed after tool results." not in messages
    assert "done too early" in messages
    assert len(agent.model_client.user_prompts) == 2
    assert agent.latest_tool_call_events[0].summary == ""


def test_agent_summary_gate_allows_failure_summary_without_key_evidence(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    agent.execute_tool_calls(
        [{"name": "Read", "intention": "read missing", "args": ["missing.txt"]}],
    )
    event = agent.latest_tool_call_events[0]

    agent.state_updater.apply_tool_call_summaries(
        {
            "actions": [
                {
                    "type": "tool_summary",
                    "tool": "Read",
                    "intention": "read missing",
                    "outcome": "failure",
                    "summary": "Read failed.",
                    "key_evidence": None,
                    "result_file": event.result_file,
                    "needs_raw_read": False,
                }
            ]
        }
    )

    gate = agent._format_tool_summary_gate([])

    assert gate == ""


def test_agent_summary_gate_allows_large_success_summary_without_key_evidence(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    event = ToolCallEvent(
        intent="list many lines",
        executed='Bash("printf many lines")',
        outcome="success",
        summary="outcome: success\nsummary: printed many lines",
        result_file=".nanocode/tool_results/result.log",
    )
    agent.tool_runner.latest_events = [event]
    agent.tool_runner.latest_executions = [
        ToolCallExecution(
            call=ParsedToolCall(name="Bash", intention="list many lines", args=["printf many lines"]),
            outcome="success",
            output="\n".join("line " + str(index) for index in range(45)),
            result_file=event.result_file,
            result_file_lines=45,
        )
    ]

    gate = agent._format_tool_summary_gate([])

    assert gate == ""


def test_tool_result_file_read_does_not_create_conversation_event_or_new_log(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    result_dir = tmp_path / ".nanocode" / "tool_results"
    result_dir.mkdir(parents=True)
    result_file = result_dir / "result.log"
    result_file.write_text(
        "\n".join(
            [
                "<Tool_Call_Result_Log>",
                "  <tool>ListDir</tool>",
                "  <raw_result>",
                "<ListDirToolResult>",
                "* (file): nanocode.py",
                "</ListDirToolResult>",
                "  </raw_result>",
                "</Tool_Call_Result_Log>",
            ]
        ),
        encoding="utf-8",
    )

    latest = agent.execute_tool_calls([{"name": "Read", "intention": "read old result log", "args": [".nanocode/tool_results/result.log"]}])

    assert "nanocode.py" in latest
    assert agent.latest_tool_call_events == []
    assert session.conversation == []
    assert sorted(path.name for path in result_dir.iterdir()) == ["result.log"]
    assert "source: .nanocode/tool_results/result.log" in agent.tool_runner.format_latest_report()


def test_agent_summary_gate_blocks_needs_raw_read_until_result_log_is_read(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    event = ToolCallEvent(
        intent="inspect large result",
        executed='Bash("pytest")',
        outcome="success",
        summary="outcome: success\nsummary: output needs inspection\nneeds_raw_read: true",
        result_file=".nanocode/tool_results/result.log",
        key_details=[],
        needs_raw_read=True,
    )
    agent.tool_runner.latest_events = [event]
    agent.tool_runner.latest_executions = [
        ToolCallExecution(
            call=ParsedToolCall(name="Bash", intention="inspect large result", args=["pytest"]),
            outcome="success",
            output="large output",
            result_file=event.result_file,
            result_file_lines=100,
        )
    ]

    gate = agent._format_tool_summary_gate([])

    assert "Needs raw read:" in gate
    assert "Read(.nanocode/tool_results/result.log)" in gate
    assert agent._format_tool_summary_gate([{"type": "tool", "name": "Read", "intention": "read result log", "args": [event.result_file]}]) == ""
    assert agent._format_tool_summary_gate([{"type": "tool", "name": "Read", "intention": "read result log", "args": [str(tmp_path / event.result_file)]}]) == ""


def test_agent_run_continues_when_no_tool_calls_and_goal_not_reached(tmp_path):
    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {"actions": [{"type": "goal", "text": "answer"}]},
                {"actions": [{"type": "message", "text": "done"}]},
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    agent.model_client = FakeModelClient()
    messages = []

    response = agent.run("answer", on_message=messages.append)

    assert response["actions"][-1]["text"] == "done"
    assert len(agent.model_client.user_prompts) == 2
    assert "No tool actions and no message action" in agent.model_client.user_prompts[1]
    assert "Continuing: goal is not complete yet." in messages


def test_agent_run_enforces_verification_gate_before_completion(tmp_path):
    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {
                    "actions": [
                        {"type": "goal", "text": "change file"},
                        {"type": "verify", "method": "run tests", "status": "pending", "evidence": None},
                    ],
                },
                {
                    "actions": [
                        {"type": "verify", "method": "run tests", "status": "passed", "evidence": "tests passed"},
                        {"type": "message", "text": "done"},
                    ],
                },
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    agent.model_client = FakeModelClient()
    messages = []

    response = agent.run("change file", on_message=messages.append)

    assert response["actions"][-1]["text"] == "done"
    assert len(agent.model_client.user_prompts) == 2
    assert "Verification_Gate: required before completion." in agent.model_client.user_prompts[1]
    assert session.current.verification.status == VerificationStatus.DONE
    assert session.current.verification.evidence == "tests passed"
    assert "Retrying: verification is required before completion." in messages


def test_agent_run_retries_format_error_in_recent_tool_results(tmp_path):
    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {"_format_error": "Invalid model output: plain answer", "actions": []},
                {"actions": [{"type": "message", "text": "done"}]},
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    agent.model_client = FakeModelClient()
    messages = []

    response = agent.run("answer", on_message=messages.append)

    assert response["actions"][-1]["text"] == "done"
    assert "Invalid model output: plain answer" in agent.model_client.user_prompts[1]
    assert "<Agent_Feedback>" in agent.model_client.user_prompts[1]
    assert "<Recent_Tool_Call_Results>\n(empty)\n</Recent_Tool_Call_Results>" in agent.model_client.user_prompts[1]
    assert messages == ["Retrying: model returned invalid output: plain answer", "done"]


def test_agent_run_rejects_extra_top_level_response_keys(tmp_path):
    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {"actions": [], "message_to_user": "old protocol"},
                {"actions": [{"type": "message", "text": "done"}]},
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    agent.model_client = FakeModelClient()

    response = agent.run("answer")

    assert response["actions"][-1]["text"] == "done"
    assert "unexpected top-level keys: message_to_user" in agent.model_client.user_prompts[1]


def test_agent_run_only_shows_ignored_action_frame_errors_in_debug(tmp_path):
    class FakeModelClient:
        def __init__(self):
            self.responses = [
                {
                    "actions": [{"type": "message", "text": "done"}],
                    "_format_frame_errors": ["frame 1: expected JSON object action"],
                }
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    agent.model_client = FakeModelClient()
    messages = []

    agent.run("answer", on_message=messages.append)

    assert messages == ["done"]

    debug_session = Session(cwd=str(tmp_path), debug=True)
    debug_agent = Agent(debug_session)
    debug_agent.model_client = FakeModelClient()
    debug_messages = []

    debug_agent.run("answer", on_message=debug_messages.append)

    assert debug_messages == ["Format_Warning: ignored invalid action frame(s).\n- frame 1: expected JSON object action", "done"]


def test_agent_run_shows_debug_gate_details_when_debug_enabled(tmp_path):
    class FakeModelClient:
        def __init__(self):
            self.responses = [
                {"_format_error": "Invalid model output: plain answer", "_format_bad_output": "plain answer", "actions": []},
                {"actions": [{"type": "message", "text": "done"}]},
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path), debug=True)
    agent = Agent(session)
    agent.model_client = FakeModelClient()
    messages = []

    agent.run("answer", on_message=messages.append)

    assert messages[0] == "Format_Gate: retrying model response. Invalid model output: plain answer\nFull bad output:\nplain answer"


def test_agent_run_stops_after_repeated_format_errors(tmp_path):
    class FakeModelClient:
        def __init__(self):
            self.calls = 0

        def request(self, system_prompt, user_prompt, *, activity="main"):
            self.calls += 1
            return {"_format_error": "Invalid model output: missing content", "actions": []}

    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    agent.model_client = FakeModelClient()
    messages = []

    try:
        agent.run("answer", on_message=messages.append)
    except nanocode.LLMError as error:
        message = str(error)
    else:
        raise AssertionError("expected LLMError")

    assert agent.model_client.calls == Agent.MAX_CONSECUTIVE_FORMAT_ERRORS
    assert "model returned invalid output 3 times in a row" in message
    assert messages[-1] == "Stopped: model returned invalid output 3 times in a row."
