import json

import nanocode
from nanocode import Agent, CurrentContextItem, KnownItem, Session, ToolCallEvent, VerificationStatus


def test_agent_tool_results_go_to_latest_area_and_logs_not_conversation(tmp_path):
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
    assert "Latest_Tool_Call_Results" in prompt
    assert "alpha" in prompt
    assert "alpha" not in agent.build_user_prompt()

    agent.apply_response(
        {
            "last_tool_calls_summaries": [
                {
                    "tool": "Read",
                    "intention": "read sample",
                    "outcome": "success",
                    "summary": "Read sample.txt line 1.",
                    "key_evidence": ["sample.txt:1 alpha"],
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
                    "choices": [{"message": {"content": json.dumps({"message_to_user": "ok", "tool_calls": None})}}],
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
    session = Session(cwd=str(tmp_path), api_url="https://example.test/v1", api_key="key", model="model", model_timeout=12)

    response = Agent(session).request("system", "user")

    assert response == {"message_to_user": "ok", "tool_calls": None}
    assert captured["url"] == "https://example.test/v1/chat/completions"
    assert captured["timeout"] == 12
    assert captured["authorization"] == "Bearer key"
    assert captured["payload"]["model"] == "model"
    assert captured["payload"]["messages"] == [{"role": "system", "content": "system"}, {"role": "user", "content": "user"}]
    assert "reasoning_effort" not in captured["payload"]
    assert "reasoning" not in captured["payload"]
    assert session.last_prompt_tokens == 2
    assert session.last_completion_tokens == 3
    assert session.last_total_tokens == 5


def test_agent_request_uses_openrouter_reasoning_payload(tmp_path, monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": json.dumps({"message_to_user": "ok"})}}], "usage": {}}
            ).encode("utf-8")

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
            return json.dumps(
                {"choices": [{"message": {"content": json.dumps({"message_to_user": "ok"})}}], "usage": {}}
            ).encode("utf-8")

    monkeypatch.setattr(nanocode.urllib.request, "urlopen", lambda request, timeout: FakeResponse())
    session = Session(
        cwd=str(tmp_path),
        api_url="https://example.test/v1",
        api_key="key",
        model="model",
        model_timeout=12,
        debug=True,
    )

    response = Agent(session).request("system prompt", "user prompt")

    files = list((tmp_path / ".nanocode" / "debug").glob("*-0001-main.txt"))
    assert response["message_to_user"] == "ok"
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
                    "choices": [
                        {
                            "message": {
                                "content": "```json\n{\"message_to_user\": \"ok\", \"tool_calls\": null}\n```"
                            }
                        }
                    ],
                    "usage": {},
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        return FakeResponse()

    monkeypatch.setattr(nanocode.urllib.request, "urlopen", fake_urlopen)
    session = Session(cwd=str(tmp_path), api_url="https://example.test/v1", api_key="key", model="model")

    response = Agent(session).request("system", "user")

    assert response == {"message_to_user": "ok", "tool_calls": None}


def test_agent_request_wraps_non_json_model_content_as_format_error(tmp_path, monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": "plain answer"}}], "usage": {}}
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        return FakeResponse()

    monkeypatch.setattr(nanocode.urllib.request, "urlopen", fake_urlopen)
    session = Session(cwd=str(tmp_path), api_url="https://example.test/v1", api_key="key", model="model")

    response = Agent(session).request("system", "user")

    assert response["tool_calls"] is None
    assert "expected one JSON object" in response["_format_error"]
    assert "plain answer" in response["_format_error"]


def test_agent_keeps_known_items_structured_in_current_and_prompt(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)

    agent.apply_response(
        {
            "known_append": [
                {"fact": "Search only supports rg and Python fallback.", "details": ["grep was removed"]},
                {"fact": "Search only supports rg and Python fallback.", "details": ["duplicate ignored"]},
            ]
        }
    )

    assert session.current.known == [
        KnownItem(fact="Search only supports rg and Python fallback.", details=["grep was removed"])
    ]

    prompt = agent.build_user_prompt()
    assert "<KnownItem>" in prompt
    assert "<fact>Search only supports rg and Python fallback.</fact>" in prompt
    assert "  <details>\n    <detail>grep was removed</detail>\n  </details>" in prompt
    assert "duplicate ignored" not in prompt


def test_agent_keeps_current_context_separate_from_known(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)

    agent.apply_response(
        {
            "current_context_update": {
                "mode": "append",
                "items": [
                    {"note": "pytest failed in tests/test_nanocode_agent.py", "details": ["exit code 1"]},
                    {"note": "pytest failed in tests/test_nanocode_agent.py", "details": ["updated failure"]},
                ],
            }
        }
    )

    assert session.current.known == []
    assert session.current.current_context == [
        CurrentContextItem(note="pytest failed in tests/test_nanocode_agent.py", details=["updated failure"])
    ]

    assert "  Context\n" in agent.state_updater.latest_report
    assert "    1. pytest failed in tests/test_nanocode_agent.py | updated failure" in agent.state_updater.latest_report


def test_agent_clears_current_context_when_goal_changes(tmp_path):
    session = Session(cwd=str(tmp_path))
    session.current.goal = "old goal"
    session.current.current_context = [CurrentContextItem(note="old dirty state")]
    agent = Agent(session)

    agent.apply_response({"goal_update": "new goal", "goal_reached": False})

    assert session.current.current_context == []
    assert "  Goal    new goal" in agent.state_updater.latest_report
    assert "  Context\n" in agent.state_updater.latest_report
    assert "    (empty)" in agent.state_updater.latest_report


def test_agent_keeps_last_fifty_current_context_items(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)

    agent.apply_response(
        {
            "current_context_update": {
                "mode": "append",
                "items": [{"note": "note " + str(index)} for index in range(55)],
            }
        }
    )

    assert len(session.current.current_context) == 50
    assert session.current.current_context[0].note == "note 5"
    assert session.current.current_context[-1].note == "note 54"


def test_agent_state_report_only_includes_real_plan_and_known_changes(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)

    response = {
        "plan_update": {"mode": "replace", "items": [{"id": "p1", "text": "Inspect file", "status": "todo"}]},
        "known_append": [{"fact": "Search uses rg.", "details": ["Python fallback exists"]}],
    }

    agent.apply_response(response)

    assert "State Updated | VERIFY:idle" in agent.state_updater.latest_report
    assert "  Plan\n" in agent.state_updater.latest_report
    assert "    1. [todo] Inspect file" in agent.state_updater.latest_report
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

    agent.apply_response({"goal_update": "new goal", "goal_reached": False})

    assert session.current.verification.goal == ""
    assert session.current.verification.status == VerificationStatus.IDLE
    assert session.current.verification.method == ""
    assert session.current.verification.evidence == ""

    agent.apply_response({"verification": {"method": "run tests", "status": "pending", "evidence": None}})

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
                    "tool_calls": [
                        {"name": "Read", "intention": "read sample", "args": ["sample.txt", "0", "1"]}
                    ]
                },
                {
                    "last_tool_calls_summaries": [
                        {
                            "tool": "Read",
                            "intention": "read sample",
                            "outcome": "success",
                            "summary": "Read sample.txt and found alpha.",
                            "key_evidence": ["alpha"],
                            "result_file": None,
                            "needs_raw_read": False,
                        }
                    ],
                    "goal_reached": True,
                    "message_to_user": "done",
                    "tool_calls": None,
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

    assert response["message_to_user"] == "done"
    assert messages[0].startswith("Tool Calls\n")
    assert "1. [success] Read(\"sample.txt\", \"0\", \"1\")" in messages[0]
    assert "why: read sample" in messages[0]
    assert "log: .nanocode/tool_results/" in messages[0]
    assert messages[-1] == "done"
    assert "alpha" not in fake_client.user_prompts[0]
    assert "alpha" in fake_client.user_prompts[1]
    assert agent.latest_tool_call_results == ""
    assert session.current.user_input == "read sample"
    assert session.current.goal_reached is True


def test_agent_run_requires_latest_tool_summaries_before_continuing(tmp_path):
    (tmp_path / "sample.txt").write_text("alpha\n", encoding="utf-8")

    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {
                    "tool_calls": [
                        {"name": "Read", "intention": "read sample", "args": ["sample.txt", "0", "1"]}
                    ]
                },
                {"goal_reached": True, "message_to_user": "premature", "tool_calls": None},
                {
                    "last_tool_calls_summaries": [
                        {
                            "tool": "Read",
                            "intention": "read sample",
                            "outcome": "success",
                            "summary": "Read sample.txt and found alpha.",
                            "key_evidence": ["alpha"],
                            "result_file": None,
                            "needs_raw_read": False,
                        }
                    ],
                    "goal_reached": True,
                    "message_to_user": "done",
                    "tool_calls": None,
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

    assert response["message_to_user"] == "done"
    assert "premature" not in messages
    assert all("premature" not in item.format() for item in session.conversation)
    assert len(agent.model_client.user_prompts) == 3
    assert "Tool_Summary_Gate: summarize every latest tool result" in agent.model_client.user_prompts[2]
    assert "Read sample.txt and found alpha." in agent.latest_tool_call_events[0].summary


def test_agent_run_continues_when_no_tool_calls_and_goal_not_reached(tmp_path):
    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {"goal_update": "answer", "goal_reached": False, "message_to_user": None, "tool_calls": None},
                {"goal_reached": True, "message_to_user": "done", "tool_calls": None},
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    agent.model_client = FakeModelClient()

    response = agent.run("answer")

    assert response["message_to_user"] == "done"
    assert len(agent.model_client.user_prompts) == 2
    assert "No tool calls and goal not reached" in agent.model_client.user_prompts[1]


def test_agent_run_enforces_verification_gate_before_completion(tmp_path):
    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {
                    "goal_update": "change file",
                    "goal_reached": True,
                    "verification": {"method": "run tests", "status": "pending", "evidence": None},
                    "tool_calls": None,
                },
                {
                    "goal_reached": True,
                    "verification": {"method": "run tests", "status": "passed", "evidence": "tests passed"},
                    "message_to_user": "done",
                    "tool_calls": None,
                },
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    agent.model_client = FakeModelClient()

    response = agent.run("change file")

    assert response["message_to_user"] == "done"
    assert len(agent.model_client.user_prompts) == 2
    assert "Verification_Gate: required before completion." in agent.model_client.user_prompts[1]
    assert session.current.verification.status == VerificationStatus.DONE
    assert session.current.verification.evidence == "tests passed"


def test_agent_run_retries_format_error_in_latest_tool_results(tmp_path):
    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {"_format_error": "Invalid model output: plain answer", "tool_calls": None},
                {"goal_reached": True, "message_to_user": "done", "tool_calls": None},
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    agent.model_client = FakeModelClient()
    messages = []

    response = agent.run("answer", on_message=messages.append)

    assert response["message_to_user"] == "done"
    assert "Invalid model output: plain answer" in agent.model_client.user_prompts[1]
    assert messages == ["done"]


def test_agent_system_prompt_forbids_non_json_answers(tmp_path):
    prompt = Agent(Session(cwd=str(tmp_path))).build_system_prompt()

    assert "Never answer outside JSON" in prompt
    assert "message_to_user" in prompt
