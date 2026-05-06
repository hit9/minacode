import json

import nanocode
from nanocode import Agent, KnownItem, Session, VerificationStatus


def test_agent_tool_results_go_to_last_tool_calls_without_conversation_or_log(tmp_path):
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

    assert "alpha" in latest
    assert "<result_file>" not in latest
    assert session.conversation == []
    assert not (tmp_path / ".nanocode" / "tool_results").exists()


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
            '{"type":"goal","text":"read sample","complete":true}__END_ACTION__',
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


def test_agent_keeps_known_items_structured_in_current(tmp_path):
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


def test_agent_ignores_known_items_without_fact_or_detail_keys(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)

    agent.apply_response(
        {
            "actions": [
                {
                    "type": "known",
                    "items": [
                        {"fact": "", "details": ["parser.notes"]},
                        {"fact": "Parser notes exist.", "details": []},
                        {"fact": "Whitespace details are ignored.", "details": ["   "]},
                        {"fact": "Parser notes were captured.", "details": [" parser.notes "]},
                    ],
                }
            ]
        }
    )

    assert session.current.known == [KnownItem(fact="Parser notes were captured.", details=["parser.notes"])]


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

    agent.apply_response({"actions": [{"type": "goal", "text": "new goal", "complete": False}]})

    assert session.current.goal_reached is False
    assert session.current.verification.goal == ""
    assert session.current.verification.status == VerificationStatus.IDLE
    assert session.current.verification.method == ""
    assert session.current.verification.evidence == ""

    agent.apply_response({"actions": [{"type": "verify", "method": "run tests", "status": "pending", "evidence": None}]})

    assert session.current.verification.goal == "new goal"
    assert session.current.verification.status == VerificationStatus.REQUIRED
    assert session.current.verification.method == "run tests"
    assert session.current.verification.evidence == ""

    agent.apply_response({"actions": [{"type": "goal", "text": "new goal", "complete": True}]})

    assert session.current.goal_reached is True


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
    assert session.conversation == []
    assert not (tmp_path / ".nanocode" / "tool_results").exists()


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


def test_agent_execute_tool_calls_returns_malformed_tool_call_error(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)

    latest = agent.execute_tool_calls([{"intention": "bad call", "args": []}])

    assert "ToolCallError: tool call missing name" in latest
    assert "InvalidToolCall(" in latest
    assert session.conversation == []
    assert not (tmp_path / ".nanocode" / "tool_results").exists()


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
                            "type": "known",
                            "items": [{"fact": "Read sample.txt and found alpha.", "details": ["alpha"]}],
                        },
                        {"type": "goal", "text": "read sample", "complete": True},
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
    assert "result:" not in messages[0]
    assert "log:" not in messages[0]
    assert messages[-1] == "done"
    assert "alpha" not in fake_client.user_prompts[0]
    assert "alpha" in fake_client.user_prompts[1]
    assert "alpha" in agent.last_tool_calls
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
                        {"type": "goal", "text": "read sample", "complete": True},
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
    assert len(agent.model_client.user_prompts) == 3
    assert "alpha" in agent.last_tool_calls


def test_agent_run_does_not_gate_when_tool_results_are_not_reviewed_for_known(tmp_path):
    (tmp_path / "sample.txt").write_text("alpha\n", encoding="utf-8")

    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {"actions": [{"type": "tool", "name": "Read", "intention": "read sample", "args": ["sample.txt", "0", "1"]}]},
                {"actions": [{"type": "goal", "text": "read sample", "complete": True}, {"type": "message", "text": "done too early"}]},
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
    assert "Retrying: Known was not reviewed after tool results." not in messages
    assert "done too early" in messages
    assert len(agent.model_client.user_prompts) == 2


def test_agent_run_continues_when_no_tool_calls_and_goal_not_reached(tmp_path):
    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {"actions": [{"type": "goal", "text": "answer", "complete": False}]},
                {"actions": [{"type": "goal", "text": "answer", "complete": True}, {"type": "message", "text": "done"}]},
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
    assert "Continuing: goal is not complete yet." not in messages
    assert any(message.startswith("State Updated") for message in messages)


def test_agent_run_does_not_report_continuation_for_action_only_turn(tmp_path):
    class FakeModelClient:
        def __init__(self):
            self.responses = [
                {"actions": [{"type": "known", "items": []}]},
                {"actions": [{"type": "goal", "text": "answer", "complete": True}, {"type": "message", "text": "done"}]},
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    agent.model_client = FakeModelClient()
    messages = []

    response = agent.run("answer", on_message=messages.append)

    assert response["actions"][-1]["text"] == "done"
    assert "Continuing: goal is not complete yet." not in messages


def test_agent_run_reports_continuation_only_when_no_actions(tmp_path):
    class FakeModelClient:
        def __init__(self):
            self.responses = [
                {"actions": []},
                {"actions": [{"type": "goal", "text": "answer", "complete": True}, {"type": "message", "text": "done"}]},
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    agent.model_client = FakeModelClient()
    messages = []

    response = agent.run("answer", on_message=messages.append)

    assert response["actions"][-1]["text"] == "done"
    assert "Continuing: goal is not complete yet." in messages


def test_agent_run_enforces_verification_gate_before_completion(tmp_path):
    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {
                    "actions": [
                        {"type": "goal", "text": "change file", "complete": False},
                        {"type": "verify", "method": "run tests", "status": "pending", "evidence": None},
                    ],
                },
                {
                    "actions": [
                        {"type": "verify", "method": "run tests", "status": "passed", "evidence": "tests passed"},
                        {"type": "goal", "text": "change file", "complete": True},
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
    assert session.current.verification.status == VerificationStatus.DONE
    assert session.current.verification.evidence == "tests passed"
    assert "Retrying: verification is required before completion." in messages


def test_agent_run_retries_format_error_with_last_tool_calls(tmp_path):
    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {"_format_error": "Invalid model output: plain answer", "actions": []},
                {"actions": [{"type": "goal", "text": "answer", "complete": True}, {"type": "message", "text": "done"}]},
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
    assert messages[0] == "Retrying: model returned invalid output: plain answer"
    assert messages[-1] == "done"


def test_agent_feedback_accumulates_errors_until_goal_complete(tmp_path):
    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {"_format_error": "Invalid model output: plain answer", "actions": []},
                {"actions": [{"type": "goal", "text": "answer", "complete": False}]},
                {"actions": [{"type": "goal", "text": "answer", "complete": True}, {"type": "message", "text": "done"}]},
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    agent.model_client = FakeModelClient()

    response = agent.run("answer")

    assert response["actions"][-1]["text"] == "done"
    assert len(agent.model_client.user_prompts) == 3
    assert "model returned invalid output" in agent.model_client.user_prompts[1]
    assert "Rule: return valid JSON action frames only." in agent.model_client.user_prompts[1]
    assert "model returned invalid output" in agent.model_client.user_prompts[2]
    assert agent.agent_feedback_errors == []


def test_agent_feedback_records_message_before_goal_complete(tmp_path):
    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {"actions": [{"type": "message", "text": "progress"}]},
                {"actions": [{"type": "goal", "text": "answer", "complete": True}, {"type": "message", "text": "done"}]},
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    agent.model_client = FakeModelClient()

    response = agent.run("answer")

    assert response["actions"][-1]["text"] == "done"
    assert "message before goal.complete=true" in agent.model_client.user_prompts[1]
    assert agent.agent_feedback_errors == []


def test_agent_feedback_clears_on_keyboard_interrupt(tmp_path):
    class FakeModelClient:
        def __init__(self):
            self.responses = [
                {"_format_error": "Invalid model output: plain answer", "actions": []},
                KeyboardInterrupt(),
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            response = self.responses.pop(0)
            if isinstance(response, KeyboardInterrupt):
                raise response
            return response

    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    agent.model_client = FakeModelClient()

    try:
        agent.run("answer")
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("expected KeyboardInterrupt")

    assert agent.agent_feedback_errors == []
    assert session.current.goal_reached is False


def test_agent_run_rejects_extra_top_level_response_keys(tmp_path):
    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {"actions": [], "message_to_user": "old protocol"},
                {"actions": [{"type": "goal", "text": "answer", "complete": True}, {"type": "message", "text": "done"}]},
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    agent.model_client = FakeModelClient()

    response = agent.run("answer")

    assert response["actions"][-1]["text"] == "done"
    assert len(agent.model_client.user_prompts) == 2


def test_agent_run_only_shows_ignored_action_frame_errors_in_debug(tmp_path):
    class FakeModelClient:
        def __init__(self):
            self.responses = [
                {
                    "actions": [{"type": "goal", "text": "answer", "complete": True}, {"type": "message", "text": "done"}],
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

    assert "Format_Warning:" not in "\n".join(messages)
    assert messages[-1] == "done"

    debug_session = Session(cwd=str(tmp_path), debug=True)
    debug_agent = Agent(debug_session)
    debug_agent.model_client = FakeModelClient()
    debug_messages = []

    debug_agent.run("answer", on_message=debug_messages.append)

    assert debug_messages[0] == "Format_Warning: ignored invalid action frame(s).\n- frame 1: expected JSON object action"
    assert debug_messages[-1] == "done"


def test_agent_run_shows_debug_gate_details_when_debug_enabled(tmp_path):
    class FakeModelClient:
        def __init__(self):
            self.responses = [
                {"_format_error": "Invalid model output: plain answer", "_format_bad_output": "plain answer", "actions": []},
                {"actions": [{"type": "goal", "text": "answer", "complete": True}, {"type": "message", "text": "done"}]},
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
