import json

import nanocode
from nanocode import MainAgent, LLMError, ParsedToolCall, Session, VerificationStatus


def _verify_passed_action():
    return {"type": "verify", "method": "unit", "status": "passed", "context": "checked"}


def _final_actions(goal="answer", message="done"):
    return [
        _verify_passed_action(),
        {"type": "goal", "text": goal, "complete": True, "message_for_complete": message},
    ]


def _make_edit_agent(session: Session) -> nanocode.EditAgent:
    parent_agent = MainAgent(session)
    return nanocode.EditAgent(
        parent_session=session,
        parent_blackboard=parent_agent.blackboard,
        goal="edit",
        scope=[],
    )


def test_agent_tool_results_go_to_recent_tool_calls_and_store(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)

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
    assert '<ToolCall ok key="tr.1">' in latest
    assert 'call: Read("sample.txt", "0", "1")' in latest
    assert "why: read sample" in latest
    assert "result:\n<ReadToolResult>" in latest
    assert session.tool_result_store["tr.1"].value.startswith("<ReadToolResult>")
    assert "alpha" in session.tool_result_store["tr.1"].value
    assert session.tool_result_store["tr.1"].log_path.startswith(".nanocode/tool_results/")
    assert session.tool_result_store["tr.1"].original_chars > 0
    assert session.tool_result_store["tr.1"].original_lines > 0
    assert session.tool_result_store["tr.1"].excerpted is False
    assert (tmp_path / session.tool_result_store["tr.1"].log_path).read_text(encoding="utf-8") == session.tool_result_store["tr.1"].value
    assert session.conversation == []
    assert (tmp_path / ".nanocode" / "tool_results").exists()


def test_explore_agent_cli_uses_compact_tool_report(tmp_path):
    (tmp_path / "sample.txt").write_text("alpha\n", encoding="utf-8")

    class FakeModelClient:
        def __init__(self):
            self.responses = [
                {
                    "actions": [
                        {"type": "goal", "text": "find sample", "complete": False},
                        {"type": "tool", "name": "Read", "intention": "read sample", "args": ["sample.txt", "0,1"]},
                    ]
                },
                {"actions": [{"type": "deliver", "targets": [{"path": "sample.txt", "area": "line 1", "reason": "found"}], "known": []}]},
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            return self.responses.pop(0)

    parent_session = Session(cwd=str(tmp_path))
    parent_agent = MainAgent(parent_session)
    explorer = nanocode.ExploreAgent(parent_session=parent_session, parent_blackboard=parent_agent.blackboard, goal="find sample", scope=["sample.txt"])
    explorer.model_client = FakeModelClient()
    messages = []

    explorer.run(on_message=messages.append)

    assert messages == ['[success] Read("sample.txt", "0,1")']


def test_agent_formats_explore_done_targets_on_separate_lines(tmp_path):
    agent = MainAgent(Session(cwd=str(tmp_path)))

    message = agent._format_explore_done(
        nanocode.ExploreReport(
            targets=[
                {"path": "producer.py", "line_range": "440-460", "area": "pipeline integration"},
                {"path": "detector.py", "line_range": "186-206", "area": "page type detection"},
            ],
            known=[],
            verification=nanocode.Verification(),
        )
    )

    assert message == "Explore done: 2 target(s)\n  1. producer.py:440-460 pipeline integration\n  2. detector.py:186-206 page type detection"


def test_agent_dedupes_same_batch_readonly_tool_calls_keeping_latest(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)

    latest = agent.execute_tool_calls(
        [
            {"name": "Read", "intention": "first read", "args": ["sample.txt", "0,1"]},
            {"name": "Read", "intention": "second read", "args": ["sample.txt", "0,1"]},
        ]
    )

    assert len(agent.tool_runner.latest_executions) == 1
    assert agent.tool_runner.latest_executions[0].call.intention == "second read"
    assert list(session.tool_result_store) == ["tr.1"]
    assert "second read" in session.tool_result_store["tr.1"].description
    assert "first read" not in latest


def test_agent_does_not_dedupe_same_batch_edit_tool_calls(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("old\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    agent = _make_edit_agent(session)

    agent.execute_tool_calls(
        [
            {"name": "Edit", "intention": "first edit", "args": ["sample.txt", "old", "new"]},
            {"name": "Edit", "intention": "second edit", "args": ["sample.txt", "old", "new"]},
        ],
        confirm=lambda call, tool: True,
    )

    assert len(agent.tool_runner.latest_executions) == 2
    assert [execution.outcome for execution in agent.tool_runner.latest_executions] == ["success", "failure"]
    assert list(agent.runtime.tool_result_store) == ["tr.1", "tr.2"]
    assert path.read_text(encoding="utf-8") == "new\n"


def test_agent_tool_results_are_bounded_and_logged(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("H" * 5000 + "M" * 5000 + "T" * 5000 + "\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)

    latest = agent.execute_tool_calls([{"name": "Read", "intention": "read large sample", "args": ["sample.txt", "0", "1"]}])

    item = session.tool_result_store["tr.1"]
    assert item.excerpted is True
    assert len(item.value) <= nanocode.MAX_TOOL_OUTPUT_CHARS
    assert "excerpted: true" in item.value
    assert "original_lines: " + str(item.original_lines) in item.value
    assert "original_chars: " + str(item.original_chars) in item.value
    assert "full_log: " + item.log_path in item.value
    assert "H" * 50 in item.value
    assert "M" * 50 in item.value
    assert "T" * 50 in item.value
    assert "[tool result excerpt]" in latest
    assert (tmp_path / item.log_path).read_text(encoding="utf-8").startswith("<ReadToolResult>")


def test_agent_keeps_latest_batch_and_recent_tool_calls(tmp_path):
    for name in ["one.txt", "two.txt", "three.txt", "four.txt"]:
        (tmp_path / name).write_text(name + "\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)
    agent.RECENT_TOOL_CALLS = 2

    for name in ["one.txt", "two.txt", "three.txt", "four.txt"]:
        agent.execute_tool_calls([{"name": "Read", "intention": "read " + name, "args": [name, "0", "1"]}])

    assert "four.txt" in agent.latest_tool_batch
    assert "four.txt" not in agent.recent_tool_calls
    assert "one.txt" not in agent.recent_tool_calls
    assert "two.txt" in agent.recent_tool_calls
    assert "three.txt" in agent.recent_tool_calls
    assert len(agent.recent_tool_call_blocks) == 2
    context = agent._format_recent_tool_call_context()
    assert "one.txt" not in context
    assert context.index("two.txt") < context.index("three.txt") < context.index("four.txt")


def test_agent_recent_tool_calls_respects_char_budget(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)
    agent.RECENT_TOOL_CALL_CHARS = 80

    agent._append_recent_tool_call_blocks(["old call " + "x" * 40])
    agent._append_recent_tool_call_blocks(["new call " + "y" * 40])

    assert "old call" not in agent.recent_tool_calls
    assert "new call" in agent.recent_tool_calls
    assert len(agent.recent_tool_call_blocks) == 1


def test_tool_result_store_keeps_latest_256_items(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)

    for index in range(257):
        agent.tool_runner._store_tool_result(ParsedToolCall(name="Read", intention="", args=[str(index)]), "success", "output " + str(index))

    assert len(session.tool_result_store) == 256
    assert list(session.tool_result_store)[:2] == ["tr.2", "tr.3"]
    assert list(session.tool_result_store)[-1] == "tr.257"
    assert session.tool_result_counter == 257


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

    response = MainAgent(session).request("system", "user")

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
    assert abs(session.last_cost - 0.000008) < 1e-12
    assert abs(session.session_cost - 0.000008) < 1e-12


def test_agent_request_uses_worker_model_config_for_explore_activity(tmp_path, monkeypatch):
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
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(nanocode.urllib.request, "urlopen", fake_urlopen)
    session = Session(cwd=str(tmp_path), api_url="https://openrouter.ai/api/v1", api_key="key", model="main", stream=False)
    session.worker_model_config = nanocode.ModelConfig(
        model="worker",
        temperature=0.2,
        reasoning=True,
        reasoning_effort="low",
        stream=False,
        timeout=7,
        prompt_price_per_1m_tokens=3.0,
        completion_price_per_1m_tokens=4.0,
    )

    response = MainAgent(session).model_client.request("system", "user", activity="explore")

    assert response == {"actions": [{"type": "message", "text": "ok"}]}
    assert captured["payload"]["model"] == "worker"
    assert captured["payload"]["temperature"] == 0.2
    assert captured["payload"]["reasoning"] == {"effort": "low"}
    assert captured["timeout"] == 7
    assert abs(session.last_cost - 0.000018) < 1e-12


def test_agent_request_retries_model_timeout(tmp_path, monkeypatch):
    class FakeModelClient:
        def __init__(self):
            self.calls = 0

        def request(self, system_prompt, user_prompt, *, activity="main"):
            self.calls += 1
            if self.calls <= 3:
                raise LLMError("request model timeout")
            return {"actions": [{"type": "message", "text": "ok"}]}

    sleeps = []
    monkeypatch.setattr(nanocode.time, "sleep", sleeps.append)
    agent = MainAgent(Session(cwd=str(tmp_path)))
    agent.model_client = FakeModelClient()

    response = agent.request("system", "user")

    assert response["actions"][0]["text"] == "ok"
    assert agent.model_client.calls == 4
    assert agent.session.turn_model_calls == 4
    assert sleeps == [3, 10, 20]


def test_agent_request_reports_model_timeout_retries(tmp_path, monkeypatch):
    class FakeModelClient:
        def __init__(self):
            self.calls = 0

        def request(self, system_prompt, user_prompt, *, activity="main"):
            self.calls += 1
            if self.calls <= 2:
                raise LLMError("request model timeout")
            return {"actions": [{"type": "message", "text": "ok"}]}

    sleeps = []
    messages = []
    monkeypatch.setattr(nanocode.time, "sleep", sleeps.append)
    agent = MainAgent(Session(cwd=str(tmp_path)))
    agent.model_client = FakeModelClient()

    response = agent.request("system", "user", on_message=messages.append)

    assert response["actions"][0]["text"] == "ok"
    assert agent.model_client.calls == 3
    assert agent.session.turn_model_calls == 3
    assert sleeps == [3, 10]
    assert messages == [
        "Retrying: request model timeout; retry 1/6 in 3s.",
        "Retrying: request model timeout; retry 2/6 in 10s.",
    ]


def test_agent_request_stops_after_model_timeout_retries(tmp_path, monkeypatch):
    class FakeModelClient:
        def __init__(self):
            self.calls = 0

        def request(self, system_prompt, user_prompt, *, activity="main"):
            self.calls += 1
            raise LLMError("request model timeout")

    sleeps = []
    monkeypatch.setattr(nanocode.time, "sleep", sleeps.append)
    agent = MainAgent(Session(cwd=str(tmp_path)))
    agent.model_client = FakeModelClient()

    try:
        agent.request("system", "user")
    except LLMError as error:
        assert str(error) == "request model timeout"
    else:
        raise AssertionError("expected LLMError")

    assert agent.model_client.calls == 7
    assert agent.session.turn_model_calls == 7
    assert sleeps == [3, 10, 20, 30, 60, 120]


def test_agent_request_does_not_retry_other_llm_errors(tmp_path, monkeypatch):
    class FakeModelClient:
        def __init__(self):
            self.calls = 0

        def request(self, system_prompt, user_prompt, *, activity="main"):
            self.calls += 1
            raise LLMError("API request failed")

    sleeps = []
    monkeypatch.setattr(nanocode.time, "sleep", sleeps.append)
    agent = MainAgent(Session(cwd=str(tmp_path)))
    agent.model_client = FakeModelClient()

    try:
        agent.request("system", "user")
    except LLMError as error:
        assert str(error) == "API request failed"
    else:
        raise AssertionError("expected LLMError")

    assert agent.model_client.calls == 1
    assert agent.session.turn_model_calls == 1
    assert sleeps == []


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

    response = MainAgent(session).request("system", "user", on_action=actions.append)

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


def test_agent_request_stream_hard_timeout_becomes_model_timeout(tmp_path, monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def __iter__(self):
            nanocode.signal.raise_signal(nanocode.signal.SIGALRM)
            yield b""

    sleeps = []
    monkeypatch.setattr(nanocode.urllib.request, "urlopen", lambda request, timeout: FakeResponse())
    monkeypatch.setattr(nanocode.time, "sleep", sleeps.append)
    session = Session(cwd=str(tmp_path), api_url="https://example.test/v1", api_key="key", model="model", model_timeout=12)

    try:
        MainAgent(session).request("system", "user")
    except LLMError as error:
        assert str(error) == "request model timeout"
    else:
        raise AssertionError("expected LLMError")

    assert session.current_model_call_started_at == 0.0
    assert sleeps == [3, 10, 20, 30, 60, 120]


def test_agent_run_previews_streamed_tool_action_before_execution_report(tmp_path, monkeypatch):
    (tmp_path / "sample.txt").write_text("alpha\n", encoding="utf-8")
    (tmp_path / "other.txt").write_text("beta\n", encoding="utf-8")
    captured_payloads = []
    responses = [
        [
            '{"type":"tool","name":"Read",',
            '"intention":"read sample","args":["sample.txt","0","1"]}__END_ACTION__',
            '{"type":"tool","name":"Read",',
            '"intention":"read other","args":["other.txt","0","1"]}__END_ACTION__',
        ],
        [
            '{"type":"verify","method":"unit","status":"passed","context":"checked"}__END_ACTION__',
            '{"type":"goal","text":"read sample","complete":true,"message_for_complete":"done"}__END_ACTION__',
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
    agent = MainAgent(session)
    messages = []

    response = agent.run("read sample", on_message=messages.append)

    assert response["actions"][-1] == {"type": "goal", "text": "read sample", "complete": True, "message_for_complete": "done"}
    assert len(captured_payloads) == 2
    assert [payload["stream"] for payload in captured_payloads] == [True, True]
    assert messages[0] == "Queued: Read Read"
    assert sum(message.startswith("Queued:") for message in messages) == 1
    assert any(message.startswith("Tool Calls") for message in messages[1:])
    assert messages[-1] == "done"


def test_agent_stream_preview_summarizes_long_tool_arguments(tmp_path):
    agent = MainAgent(Session(cwd=str(tmp_path)))

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

    assert bash_preview == "Bash"
    assert replace_preview == "ReplaceRange"


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

    MainAgent(session).request("system", "user")

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

    response = MainAgent(session).request("system prompt", "user prompt")

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

    response = MainAgent(session).request("system", "user")

    assert response == {"actions": [{"type": "message", "text": "ok"}]}


def test_agent_request_accepts_leaked_think_tags_before_json(tmp_path):
    client = MainAgent(Session(cwd=str(tmp_path))).model_client

    assert client._parse_model_content('</think>{"type":"message","text":"ok"}\n__END_ACTION__') == {
        "actions": [{"type": "message", "text": "ok"}],
    }
    assert client._parse_model_content('<think>reasoning</think>\n{"type":"message","text":"ok"}\n__END_ACTION__') == {
        "actions": [{"type": "message", "text": "ok"}],
    }


def test_agent_request_accepts_pretty_action_frames_and_marker_variants(tmp_path):
    client = MainAgent(Session(cwd=str(tmp_path))).model_client

    response = client._parse_model_content(
        '{\n  "type": "message",\n  "text": "ok"\n}\n**END_ACTION**\n{"type":"goal","text":"next"}\nEND_ACTION'
    )

    assert response == {"actions": [{"type": "message", "text": "ok"}, {"type": "goal", "text": "next"}]}


def test_agent_request_accepts_inline_action_frame_markers(tmp_path):
    client = MainAgent(Session(cwd=str(tmp_path))).model_client

    response = client._parse_model_content('{"type":"message","text":"ok"}__END_ACTION__{"type":"goal","text":"next"}__END_ACTION__')

    assert response == {"actions": [{"type": "message", "text": "ok"}, {"type": "goal", "text": "next"}]}


def test_agent_request_accepts_single_unmarked_json_action(tmp_path):
    client = MainAgent(Session(cwd=str(tmp_path))).model_client

    response = client._parse_model_content('{"type":"message","text":"ok"}')

    assert response == {"actions": [{"type": "message", "text": "ok"}]}


def test_agent_request_ignores_bad_action_frames_when_other_actions_are_valid(tmp_path):
    client = MainAgent(Session(cwd=str(tmp_path))).model_client

    response = client._parse_model_content('plain answer\n__END_ACTION__\n{"type":"message","text":"ok"}\n__END_ACTION__')

    assert response["actions"] == [{"type": "message", "text": "ok"}]
    assert response["_format_frame_errors"] == ["frame 1: expected JSON object action"]


def test_agent_request_rejects_native_tool_call_syntax(tmp_path):
    client = MainAgent(Session(cwd=str(tmp_path))).model_client

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

    response = MainAgent(session).request("system", "user")

    assert response["actions"] == []
    assert "expected one JSON action object or action frames ending with __END_ACTION__" in response["_format_error"]
    assert "plain answer" in response["_format_error"]


def test_agent_request_rejects_unmarked_json_action_array(tmp_path):
    client = MainAgent(Session(cwd=str(tmp_path))).model_client

    response = client._parse_model_content('[{"type":"message","text":"ok"}]')

    assert response["actions"] == []
    assert "expected JSON object action" in response["_format_error"]


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

    response = MainAgent(session).request("system", "user")

    assert response["actions"] == []
    assert "expected one JSON object" in response["_format_error"]
    assert "API response missing message content" in response["_format_error"]


def test_agent_keeps_known_items_structured_in_current(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)

    agent.apply_response(
        {
            "actions": [
                {
                    "type": "known",
                    "items": [
                        "Search only supports rg and Python fallback.",
                        "Search only supports rg and Python fallback.",
                    ],
                }
            ]
        }
    )

    assert agent.blackboard.known == ["Search only supports rg and Python fallback."]


def test_agent_dedupes_exact_known_facts(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)

    agent.apply_response(
        {
            "actions": [
                {
                    "type": "known",
                    "items": [
                        "Preview logic exists in _format_stream_action_preview.",
                        "Preview logic exists in _format_stream_action_preview.",
                        "Preview logic exists in _format_stream_action_preview!",
                    ],
                }
            ]
        }
    )

    assert agent.blackboard.known == [
        "Preview logic exists in _format_stream_action_preview.",
        "Preview logic exists in _format_stream_action_preview!",
    ]


def test_agent_keeps_latest_50_known_items(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)

    agent.apply_response({"actions": [{"type": "known", "items": ["fact " + str(index) for index in range(51)]}]})

    assert len(agent.blackboard.known) == 50
    assert agent.blackboard.known[0] == "fact 1"
    assert agent.blackboard.known[-1] == "fact 50"


def test_main_agent_applies_project_knowledge_and_saves(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)

    agent.apply_response(
        {
            "actions": [
                {
                    "type": "learn",
                    "summary": "Single-file CLI coding assistant.",
                    "structure": ["nanocode.py contains the CLI and agent loop."],
                    "architecture": ["MainAgent delegates uncertain code discovery to ExploreAgent."],
                    "workflows": ["Run pytest for verification."],
                    "conventions": ["Use JSON action frames."],
                }
            ]
        }
    )

    data = json.loads((tmp_path / ".nanocode" / "project_knowledge.json").read_text(encoding="utf-8"))
    assert session.project_knowledge.summary == "Single-file CLI coding assistant."
    assert data["summary"] == "Single-file CLI coding assistant."
    assert data["structure"] == ["nanocode.py contains the CLI and agent loop."]
    assert "  Project_Knowledge\n" in agent.state_updater.latest_report
    assert "structure: 1 item(s)" in agent.state_updater.latest_report


def test_project_knowledge_dedupes_and_keeps_latest_30_items(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)

    agent.apply_response(
        {
            "actions": [
                {
                    "type": "learn",
                    "structure": ["item " + str(index) for index in range(31)] + ["item 30"],
                }
            ]
        }
    )

    assert len(session.project_knowledge.structure) == 30
    assert session.project_knowledge.structure[0] == "item 1"
    assert session.project_knowledge.structure[-1] == "item 30"


def test_project_knowledge_can_correct_and_delete_existing_items(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)
    agent.apply_response(
        {
            "actions": [
                {
                    "type": "learn",
                    "structure": ["old structure", "remove me"],
                    "architecture": ["old architecture"],
                }
            ]
        }
    )

    agent.apply_response(
        {
            "actions": [
                {
                    "type": "learn",
                    "corrections": [
                        {"field": "structure", "old": "old structure", "new": "new structure"},
                        {"field": "structure", "old": "remove me", "new": None},
                        {"field": "architecture", "old": "old architecture", "new": "new architecture"},
                    ],
                }
            ]
        }
    )

    assert session.project_knowledge.structure == ["new structure"]
    assert session.project_knowledge.architecture == ["new architecture"]


def test_explore_agent_does_not_apply_project_knowledge(tmp_path):
    session = Session(cwd=str(tmp_path))
    parent_agent = MainAgent(session)
    explorer = nanocode.ExploreAgent(parent_session=session, parent_blackboard=parent_agent.blackboard, goal="inspect", scope=[])

    explorer.apply_response(
        {
            "actions": [
                {
                    "type": "learn",
                    "summary": "Should be ignored.",
                    "structure": ["Should not be saved."],
                }
            ]
        }
    )

    assert session.project_knowledge.is_empty()
    assert not (tmp_path / ".nanocode" / "project_knowledge.json").exists()


def test_prompt_includes_project_knowledge(tmp_path):
    session = Session(cwd=str(tmp_path))
    session.project_knowledge.apply(
        {
            "summary": "Single-file app.",
            "structure": ["nanocode.py is the main file."],
            "architecture": ["BaseAgent owns the common agent loop."],
        }
    )
    agent = MainAgent(session)

    prompt = agent.build_user_prompt()

    assert "<Project_Knowledge>" in prompt
    assert "Summary:\nSingle-file app." in prompt
    assert "nanocode.py is the main file." in prompt
    assert '"corrections"' in agent.build_system_prompt()


def test_agent_ignores_known_items_without_fact(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)

    agent.apply_response(
        {
            "actions": [
                {
                    "type": "known",
                    "items": [
                        "",
                        "Parser notes exist.",
                        "Parser notes were captured.",
                    ],
                }
            ]
        }
    )

    assert agent.blackboard.known == [
        "Parser notes exist.",
        "Parser notes were captured.",
    ]


def test_agent_state_report_only_includes_real_plan_and_known_changes(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)

    response = {
        "actions": [
            {"type": "plan", "mode": "replace", "items": [{"id": "p1", "text": "Inspect file", "status": "todo"}]},
            {"type": "known", "items": ["Search uses rg."]},
        ]
    }

    agent.apply_response(response)

    assert "State Updated | VERIFY:idle" in agent.state_updater.latest_report
    assert "  Plan\n" in agent.state_updater.latest_report
    assert "    1. [○ todo] Inspect file" in agent.state_updater.latest_report
    assert "  Known\n" in agent.state_updater.latest_report
    assert "    1. Search uses rg." in agent.state_updater.latest_report

    agent.apply_response(response)

    assert agent.state_updater.latest_report == ""


def test_agent_resets_verification_when_goal_changes(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)
    agent.blackboard.goal = "old goal"
    agent.blackboard.verification.goal = "old goal"
    agent.blackboard.verification.status = VerificationStatus.DONE
    agent.blackboard.verification.method = "old check"
    agent.blackboard.verification.context = "old context"

    agent.apply_response({"actions": [{"type": "goal", "text": "new goal", "complete": False}]})

    assert agent.blackboard.goal_reached is False
    assert agent.blackboard.verification.goal == ""
    assert agent.blackboard.verification.status == VerificationStatus.IDLE
    assert agent.blackboard.verification.method == ""
    assert agent.blackboard.verification.context == ""

    agent.apply_response({"actions": [{"type": "verify", "method": "run tests", "status": "pending", "context": None}]})

    assert agent.blackboard.verification.goal == "new goal"
    assert agent.blackboard.verification.status == VerificationStatus.REQUIRED
    assert agent.blackboard.verification.method == "run tests"
    assert agent.blackboard.verification.context == ""

    agent.apply_response({"actions": [{"type": "goal", "text": "new goal", "complete": True}]})

    assert agent.blackboard.goal_reached is True


def test_agent_execute_tool_calls_requests_confirmation_for_edit_tools(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("old\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    agent = _make_edit_agent(session)
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
    agent = _make_edit_agent(session)

    latest = agent.execute_tool_calls(
        [{"name": "Edit", "intention": "edit sample", "args": ["sample.txt", "old", "new"]}],
        confirm=lambda call, tool: "please inspect tests first",
    )

    assert "Cancelled: user refused: please inspect tests first" in latest
    assert path.read_text(encoding="utf-8") == "old\n"
    assert session.conversation == []
    assert (tmp_path / ".nanocode" / "tool_results").exists()


def test_agent_execute_tool_calls_rejects_failed_preview_before_confirmation(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("old\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    agent = _make_edit_agent(session)
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
    agent = MainAgent(session)

    latest = agent.execute_tool_calls([{"intention": "bad call", "args": []}])

    assert "ToolCallError: tool call missing name" in latest
    assert "InvalidToolCall(" in latest
    assert session.conversation == []
    assert (tmp_path / ".nanocode" / "tool_results").exists()


def test_agent_execute_tool_calls_records_arg_errors_in_feedback(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)

    latest = agent.execute_tool_calls([{"name": "Read", "intention": "bad range", "args": ["sample.txt", "bad", "1"]}])

    assert "ToolCallError: invalid start: should be an integer" in latest
    assert agent.agent_feedback_errors == [
        'Error: tool call args invalid: Read("sample.txt", "bad", "1") -> ToolCallError: invalid start: should be an integer. Rule: use the tool signature exactly.'
    ]


def test_agent_execute_tool_calls_does_not_record_runtime_errors_in_feedback(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)

    latest = agent.execute_tool_calls([{"name": "Read", "intention": "missing file", "args": ["missing.txt", "0", "1"]}])

    assert "ToolCallError: " in latest
    assert agent.agent_feedback_errors == []


def test_main_agent_rejects_search_tool(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)

    latest = agent.execute_tool_calls([{"name": "Search", "intention": "find symbol", "args": ["class Foo"]}])

    system_prompt = agent.build_system_prompt()
    assert "Search(" not in system_prompt
    assert "Read(" in system_prompt
    assert "tool not allowed for this agent: Search" in latest


def test_explore_agent_rejects_edit_tools(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("old\n", encoding="utf-8")
    parent_session = Session(cwd=str(tmp_path))
    parent_agent = MainAgent(parent_session)
    explorer = nanocode.ExploreAgent(
        parent_session=parent_session,
        parent_blackboard=parent_agent.blackboard,
        goal="find relevant target",
        scope=["sample.txt"],
    )

    latest = explorer.execute_tool_calls([{"name": "Edit", "intention": "try edit", "args": ["sample.txt", "old", "new"]}])

    system_prompt = explorer.build_system_prompt()
    assert "Read(" in system_prompt
    assert "Search(" in system_prompt
    assert "Bash(" in system_prompt
    assert "Edit(" not in system_prompt
    assert "ReplaceRange(" not in system_prompt
    assert "ApplyPatch(" not in system_prompt
    assert "tool not allowed for this agent: Edit" in latest
    assert path.read_text(encoding="utf-8") == "old\n"
    assert explorer.session is parent_session
    assert parent_session.tool_result_store == {}
    assert list(explorer.runtime.tool_result_store) == ["tr.1"]


def test_verify_agent_rejects_edit_tools(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("old\n", encoding="utf-8")
    parent_session = Session(cwd=str(tmp_path))
    parent_agent = MainAgent(parent_session)
    verifier = nanocode.VerifyAgent(
        parent_session=parent_session,
        parent_blackboard=parent_agent.blackboard,
        goal="verify change",
        scope=["sample.txt"],
    )

    latest = verifier.execute_tool_calls([{"name": "Edit", "intention": "try edit", "args": ["sample.txt", "old", "new"]}])

    system_prompt = verifier.build_system_prompt()
    assert "Read(" in system_prompt
    assert "Search(" in system_prompt
    assert "Bash(" in system_prompt
    assert "Edit(" not in system_prompt
    assert "ReplaceRange(" not in system_prompt
    assert "ApplyPatch(" not in system_prompt
    assert "tool not allowed for this agent: Edit" in latest
    assert path.read_text(encoding="utf-8") == "old\n"
    assert verifier.session is parent_session
    assert parent_session.tool_result_store == {}
    assert list(verifier.runtime.tool_result_store) == ["tr.1"]


def test_edit_agent_rejects_bash_and_allows_edit_tools(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("old\n", encoding="utf-8")
    parent_session = Session(cwd=str(tmp_path))
    parent_agent = MainAgent(parent_session)
    editor = nanocode.EditAgent(
        parent_session=parent_session,
        parent_blackboard=parent_agent.blackboard,
        goal="edit sample",
        scope=["target: sample.txt"],
    )

    latest = editor.execute_tool_calls([{"name": "Bash", "intention": "try shell", "args": ["printf nope"]}])

    system_prompt = editor.build_system_prompt()
    assert "Read(" in system_prompt
    assert "Search(" in system_prompt
    assert "Git(" in system_prompt
    assert "Edit(" in system_prompt
    assert "ReplaceRange(" in system_prompt
    assert "ApplyPatch(" in system_prompt
    assert "Bash(" not in system_prompt
    assert "prefer small target ranges over whole files" in system_prompt
    assert "one-sentence summary, at most 3 checks" in system_prompt
    assert "tool not allowed for this agent: Bash" in latest
    assert path.read_text(encoding="utf-8") == "old\n"
    assert editor.session is parent_session
    assert parent_session.tool_result_store == {}
    assert list(editor.runtime.tool_result_store) == ["tr.1"]


def test_worker_prompt_receives_compact_handoff_context(tmp_path):
    parent_session = Session(cwd=str(tmp_path))
    parent_agent = MainAgent(parent_session)
    handoff_context = nanocode.AgentReportHistory(
        explored=["target sample.py:1,3 | parser | parser target"],
        edited=["changed | sample.py | renamed message to progress"],
        verified=["failed | pytest | assertion failed | issue: old prompt expected"],
    )
    editor = nanocode.EditAgent(
        parent_session=parent_session,
        parent_blackboard=parent_agent.blackboard,
        goal="edit sample",
        scope=["target: sample.py"],
        handoff_context=handoff_context,
    )

    prompt = editor.build_user_prompt()

    assert "<Handoff_Context>" in prompt
    assert "<explored>" in prompt
    assert "target sample.py:1,3 | parser | parser target" in prompt
    assert "<edited>" in prompt
    assert "changed | sample.py | renamed message to progress" in prompt
    assert "<verified>" in prompt
    assert "failed | pytest | assertion failed" in prompt
    assert "<Edit_Goal>" in prompt
    assert prompt.index("<Handoff_Context>") < prompt.index("<Edit_Goal>")


def test_explore_agent_keeps_tool_results_local_and_delivers(tmp_path):
    (tmp_path / "sample.txt").write_text("alpha\n", encoding="utf-8")
    parent_session = Session(cwd=str(tmp_path))
    parent_agent = MainAgent(parent_session)
    parent_agent.blackboard.known = ["MainAgent knows sample.txt exists."]

    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {"actions": [{"type": "tool", "name": "Read", "intention": "read sample", "args": ["sample.txt", "0", "1"]}]},
                {
                    "actions": [
                        {"type": "known", "items": ["sample.txt contains alpha."]},
                        {"type": "verify", "method": "read", "status": "passed", "context": "target found"},
                        {
                            "type": "deliver",
                            "targets": [
                                {
                                    "path": "sample.txt",
                                    "area": "line 1",
                                    "line_range": "0,1",
                                    "context": "alpha",
                                    "reason": "contains alpha",
                                }
                            ],
                            "known": ["relevant target is sample.txt line 1."],
                        },
                    ]
                },
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    explorer = nanocode.ExploreAgent(
        parent_session=parent_session,
        parent_blackboard=parent_agent.blackboard,
        goal="find relevant target",
        scope=["sample.txt"],
    )
    explorer.model_client = FakeModelClient()

    report = explorer.run()

    assert report.targets == [
        {
            "path": "sample.txt",
            "area": "line 1",
            "line_range": "0,1",
            "context": "alpha",
            "reason": "contains alpha",
        }
    ]
    assert report.known == ["sample.txt contains alpha.", "relevant target is sample.txt line 1."]
    assert report.verification.status == VerificationStatus.DONE
    assert explorer.session is parent_session
    assert parent_session.tool_result_store == {}
    assert list(explorer.runtime.tool_result_store) == ["tr.1"]
    assert "MainAgent knows sample.txt exists." in explorer.model_client.user_prompts[0]
    assert "alpha" in explorer.model_client.user_prompts[1]


def test_explore_agent_goal_changes_do_not_clear_parent_range_fingerprints(tmp_path):
    parent_session = Session(cwd=str(tmp_path))
    parent_session.range_fingerprints.remember(filepath=str(tmp_path / "sample.txt"), start=0, end=1, content="alpha\n")
    parent_agent = MainAgent(parent_session)
    explorer = nanocode.ExploreAgent(
        parent_session=parent_session,
        parent_blackboard=parent_agent.blackboard,
        goal="find relevant target",
        scope=["sample.txt"],
    )

    explorer.apply_response({"actions": [{"type": "goal", "text": "refined target", "complete": False}]})

    assert len(parent_session.range_fingerprints) == 1


def test_agent_execute_tool_calls_shows_auto_approval_in_yolo_mode(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("old\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path), yolo=True)
    agent = _make_edit_agent(session)
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
    assert "<ToolCall ok" in latest
    assert path.read_text(encoding="utf-8") == "new\n"
    assert agent.blackboard.verification_required is False


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
                            "items": ["Read sample.txt and found alpha."],
                        },
                        _verify_passed_action(),
                        {"type": "goal", "text": "read sample", "complete": True, "message_for_complete": "done"},
                    ],
                },
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)
    fake_client = FakeModelClient()
    agent.model_client = fake_client

    messages = []
    response = agent.run("read sample", on_message=messages.append)

    assert response["actions"][-1]["message_for_complete"] == "done"
    assert messages[0].startswith("Tool Calls\n")
    assert '1. [success] Read("sample.txt", "0", "1")' in messages[0]
    assert "     tr.1 | why: read sample" in messages[0]
    assert "log: .nanocode/tool_results/" not in messages[0]
    assert messages[-1] == "done"
    assert "alpha" not in fake_client.user_prompts[0]
    assert "alpha" in fake_client.user_prompts[1]
    assert "tr.1" in fake_client.user_prompts[1]
    assert agent.latest_tool_batch == ""
    assert agent.recent_tool_calls == ""
    assert agent.blackboard.known == ["Read sample.txt and found alpha."]
    assert agent.blackboard.user_input == "read sample"
    assert agent.blackboard.goal == ""
    assert agent.blackboard.plan == []
    assert agent.blackboard.verification.status == VerificationStatus.IDLE
    assert agent.blackboard.goal_reached is False
    assert agent.blackboard.verification_required is False


def test_agent_run_allows_readonly_answer_without_verification(tmp_path):
    (tmp_path / "sample.txt").write_text("alpha\n", encoding="utf-8")

    class FakeModelClient:
        def __init__(self):
            self.responses = [
                {"actions": [{"type": "tool", "name": "Read", "intention": "read sample", "args": ["sample.txt", "0", "1"]}]},
                {
                    "actions": [
                        {"type": "goal", "text": "answer sample", "complete": True, "message_for_complete": "sample contains alpha"},
                    ],
                },
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            return self.responses.pop(0)

    agent = MainAgent(Session(cwd=str(tmp_path)))
    agent.model_client = FakeModelClient()
    messages = []

    response = agent.run("answer sample", on_message=messages.append)

    assert response["actions"][-1]["message_for_complete"] == "sample contains alpha"
    assert "Retrying: verification must pass before completion." not in messages
    assert messages[-1] == "sample contains alpha"


def test_agent_run_feeds_explore_report_into_next_prompt(tmp_path):
    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {
                    "actions": [
                        {"type": "goal", "text": "relevant target", "complete": False},
                        {"type": "explore", "goal": "find target", "scope": ["sample.txt"], "reason": "target uncertain"},
                    ]
                },
                {"actions": _final_actions("relevant target")},
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    class FakeExploreAgent:
        def run(self, *, confirm=None, on_auto_approve=None, on_message=None):
            if on_message is not None:
                on_message("Tool Calls\n  1. [success] Read(\"sample.txt\", \"0\", \"1\")\n     tr.1 | why: inspect sample")
            return nanocode.ExploreReport(
                targets=[{"path": "sample.txt", "area": "line 1", "reason": "target found"}],
                known=["sample.txt line 1 is the relevant target."],
                verification=nanocode.Verification(status=VerificationStatus.DONE, method="explore", context="target found"),
            )

    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)
    agent.model_client = FakeModelClient()
    agent._make_explore_agent = lambda *, goal, scope: FakeExploreAgent()
    messages = []

    response = agent.run("relevant target", on_message=messages.append)

    assert response["actions"][-1]["message_for_complete"] == "done"
    assert len(agent.model_client.user_prompts) == 2
    assert "<Agent_Reports>" in agent.model_client.user_prompts[1]
    assert "<Explore_History>" in agent.model_client.user_prompts[1]
    assert "sample.txt line 1 is the relevant target." in agent.model_client.user_prompts[1]
    assert session.tool_result_store == {}
    assert agent.recent_tool_calls == ""
    assert any(message.startswith("[explore] Tool Calls") for message in messages)
    assert messages[-1] == "done"


def test_agent_run_hands_edit_to_edit_agent_and_requires_verification(tmp_path):
    edit_calls = []
    verify_calls = []

    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {
                    "actions": [
                        {"type": "goal", "text": "change sample", "complete": False},
                        {
                            "type": "edit",
                            "goal": "change sample text",
                            "targets": [{"path": "sample.txt", "area": "line 1", "line_range": "0,1", "context": "old", "reason": "line needs update"}],
                            "constraints": ["preserve newline"],
                            "self_check": ["read back line 1"],
                        },
                    ]
                },
                {"actions": [{"type": "goal", "text": "change sample", "complete": True, "message_for_complete": "done"}]},
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    class FakeEditAgent:
        def __init__(self, *, goal, scope):
            self.goal = goal
            self.scope = scope

        def run(self, *, confirm=None, on_auto_approve=None, on_message=None):
            edit_calls.append((self.goal, self.scope))
            if on_message is not None:
                on_message('Tool Calls\n  1. [success] Edit("sample.txt", "old", "new")\n     why: update sample')
            return nanocode.EditReport(status="changed", summary="sample changed", changed_files=["sample.txt"], checks=["read back line 1"])

    class FakeVerifyAgent:
        def run(self, *, confirm=None, on_auto_approve=None, on_message=None):
            verify_calls.append(True)
            return nanocode.VerifyReport(status="passed", method="review", summary="edit verified")

    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)
    agent.model_client = FakeModelClient()
    agent._make_edit_agent = lambda *, goal, scope: FakeEditAgent(goal=goal, scope=scope)
    agent._make_verify_agent = lambda *, goal, scope: FakeVerifyAgent()
    messages = []

    response = agent.run("change sample", on_message=messages.append)

    assert response["actions"][-1]["message_for_complete"] == "done"
    assert edit_calls == [
        (
            "change sample text",
            [
                "target: sample.txt line 1 line_range=0,1",
                "target_context: old",
                "target_reason: line needs update",
                "constraint: preserve newline",
                "self_check: read back line 1",
            ],
        )
    ]
    assert verify_calls == [True]
    assert "<EditReport>" in agent.model_client.user_prompts[1]
    assert "sample changed" in agent.model_client.user_prompts[1]
    assert "Editing: change sample text" in messages
    assert any(message.startswith("[edit] Tool Calls") for message in messages)
    assert "Edit done: changed\n  sample changed" in messages
    assert "Verify done: passed | review\n  edit verified" in messages
    assert messages[-1] == "done"


def test_agent_report_history_keeps_explore_edit_and_verify_reports(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)
    agent.agent_reports.explore.append(nanocode.ExploreReport(targets=[], known=["found target"], verification=nanocode.Verification()).format())
    agent.agent_reports.edit.append(nanocode.EditReport(status="changed", summary="edited target").format())
    agent.agent_reports.verify.append(nanocode.VerifyReport(status="passed", method="review", summary="verified target").format())

    prompt = agent.build_user_prompt()

    assert "<Explore_History>" in prompt
    assert "found target" in prompt
    assert "<Edit_History>" in prompt
    assert "edited target" in prompt
    assert "<Verify_History>" in prompt
    assert "verified target" in prompt


def test_agent_run_keeps_tool_results_when_format_retry_happens(tmp_path):
    (tmp_path / "sample.txt").write_text("alpha\n", encoding="utf-8")

    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {"actions": [{"type": "tool", "name": "Read", "intention": "read sample", "args": ["sample.txt", "0", "1"]}]},
                {"_format_error": "Invalid model output: plain answer", "actions": []},
                {"actions": _final_actions("read sample")},
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)
    agent.model_client = FakeModelClient()

    response = agent.run("read sample")

    assert response["actions"][-1]["message_for_complete"] == "done"
    assert len(agent.model_client.user_prompts) == 3
    assert agent.latest_tool_batch == ""
    assert agent.recent_tool_calls == ""


def test_agent_run_trims_tool_result_store_when_goal_completes(tmp_path):
    for index in range(51):
        (tmp_path / f"sample-{index}.txt").write_text(f"line {index}\n", encoding="utf-8")

    class FakeModelClient:
        def __init__(self):
            self.responses = [
                {
                    "actions": [
                        {"type": "tool", "name": "Read", "intention": f"read {index}", "args": [f"sample-{index}.txt", "0", "1"]}
                        for index in range(51)
                    ]
                },
                {"actions": _final_actions("read samples")},
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)
    agent.blackboard.goal = "answer"
    agent.blackboard.plan = [nanocode.PlanItem(text="try answer")]
    agent.blackboard.known = ["keep this fact"]
    agent.blackboard.verification.status = VerificationStatus.REQUIRED
    agent.latest_tool_batch = "old tool call"
    agent.latest_tool_call_blocks = ["old tool call"]
    agent.model_client = FakeModelClient()

    agent.run("read samples")

    assert len(session.tool_result_store) == 50
    assert list(session.tool_result_store)[:2] == ["tr.2", "tr.3"]
    assert list(session.tool_result_store)[-1] == "tr.51"
    assert session.tool_result_counter == 51
    assert agent.blackboard.goal == ""
    assert agent.blackboard.plan == []
    assert agent.blackboard.known == ["keep this fact"]
    assert agent.blackboard.verification.status == VerificationStatus.IDLE
    assert agent.blackboard.goal_reached is False


def test_agent_run_does_not_gate_when_tool_results_are_not_reviewed_for_known(tmp_path):
    (tmp_path / "sample.txt").write_text("alpha\n", encoding="utf-8")

    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {"actions": [{"type": "tool", "name": "Read", "intention": "read sample", "args": ["sample.txt", "0", "1"]}]},
                {"actions": _final_actions("read sample", "done too early")},
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)
    agent.model_client = FakeModelClient()
    messages = []

    response = agent.run("read sample", on_message=messages.append)

    assert response["actions"][-1]["message_for_complete"] == "done too early"
    assert "Retrying: Known was not reviewed after tool results." not in messages
    assert "done too early" in messages
    assert len(agent.model_client.user_prompts) == 2


def test_agent_run_continues_when_no_tool_calls_and_goal_not_reached(tmp_path):
    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {"actions": [{"type": "goal", "text": "answer", "complete": False}]},
                {"actions": _final_actions()},
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)
    agent.model_client = FakeModelClient()
    messages = []

    response = agent.run("answer", on_message=messages.append)

    assert response["actions"][-1]["message_for_complete"] == "done"
    assert len(agent.model_client.user_prompts) == 2
    assert "Continuing: goal is not complete yet." not in messages
    assert any(message.startswith("State Updated") for message in messages)


def test_agent_run_stops_after_chat_action(tmp_path):
    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []

        def request(self, system_prompt, user_prompt, *, activity="main"):
            self.user_prompts.append(user_prompt)
            return {"actions": [{"type": "chat", "text": "你好"}]}

    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)
    agent.model_client = FakeModelClient()
    messages = []

    response = agent.run("你好", on_message=messages.append)

    assert response["actions"] == [{"type": "chat", "text": "你好"}]
    assert messages == ["你好"]
    assert len(agent.model_client.user_prompts) == 1


def test_agent_run_does_not_report_continuation_for_action_only_turn(tmp_path):
    class FakeModelClient:
        def __init__(self):
            self.responses = [
                {"actions": [{"type": "known", "items": []}]},
                {"actions": _final_actions()},
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)
    agent.model_client = FakeModelClient()
    messages = []

    response = agent.run("answer", on_message=messages.append)

    assert response["actions"][-1]["message_for_complete"] == "done"
    assert "Continuing: goal is not complete yet." not in messages


def test_agent_run_reports_continuation_only_when_no_actions(tmp_path):
    class FakeModelClient:
        def __init__(self):
            self.responses = [
                {"actions": []},
                {"actions": _final_actions()},
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)
    agent.model_client = FakeModelClient()
    messages = []

    response = agent.run("answer", on_message=messages.append)

    assert response["actions"][-1]["message_for_complete"] == "done"
    assert "Continuing: assistant must set current task's goal." in messages


def test_agent_run_enforces_verification_gate_before_completion(tmp_path):
    (tmp_path / "sample.txt").write_text("old\n", encoding="utf-8")
    verify_confirm_callbacks = []

    class FakeEditAgent:
        def run(self, *, confirm=None, on_auto_approve=None, on_message=None):
            (tmp_path / "sample.txt").write_text("new\n", encoding="utf-8")
            return nanocode.EditReport(status="changed", summary="sample changed", changed_files=["sample.txt"])

    class FakeVerifyAgent:
        def run(self, *, confirm=None, on_auto_approve=None, on_message=None):
            verify_confirm_callbacks.append(confirm)
            if on_message is not None:
                on_message('Tool Calls\n  1. [success] Git("diff", "--", "sample.txt")\n     why: inspect diff')
            return nanocode.VerifyReport(status="passed", method="git diff", summary="diff matches goal", evidence=["sample.txt changed"])

    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {
                    "actions": [
                        {"type": "goal", "text": "change file", "complete": False},
                        {
                            "type": "edit",
                            "goal": "edit sample",
                            "targets": [{"path": "sample.txt", "area": "line 1"}],
                            "constraints": [],
                            "self_check": [],
                        },
                    ],
                },
                {
                    "actions": [
                        {"type": "goal", "text": "change file done", "complete": True, "message_for_complete": "done"},
                    ],
                },
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)
    agent.model_client = FakeModelClient()
    agent._make_edit_agent = lambda *, goal, scope: FakeEditAgent()
    agent._make_verify_agent = lambda *, goal, scope: FakeVerifyAgent()
    messages = []

    response = agent.run("change file", confirm=lambda call, tool: True, on_message=messages.append)

    assert response["actions"][-1]["message_for_complete"] == "done"
    assert len(agent.model_client.user_prompts) == 2
    assert len(verify_confirm_callbacks) == 1
    assert verify_confirm_callbacks[0] is not None
    assert agent.blackboard.verification.status == VerificationStatus.IDLE
    assert agent.blackboard.verification.context == ""
    assert "Verifying: change file done" in messages
    assert any(message.startswith("[verify] Tool Calls") for message in messages)
    assert "Verify done: passed | git diff\n  diff matches goal" in messages
    assert (tmp_path / "sample.txt").read_text(encoding="utf-8") == "new\n"


def test_agent_run_feeds_failed_verify_report_into_next_prompt(tmp_path):
    (tmp_path / "sample.txt").write_text("old\n", encoding="utf-8")
    handoff_prompts = []

    class FakeEditAgent:
        def __init__(self):
            self.contents = ["bad\n", "new\n"]

        def run(self, *, confirm=None, on_auto_approve=None, on_message=None):
            (tmp_path / "sample.txt").write_text(self.contents.pop(0), encoding="utf-8")
            return nanocode.EditReport(status="changed", summary="sample changed", changed_files=["sample.txt"])

    class FakeVerifyAgent:
        def __init__(self):
            self.reports = [
                nanocode.VerifyReport(status="failed", method="unit", summary="assertion failed", issues=["sample still wrong"]),
                nanocode.VerifyReport(status="passed", method="unit", summary="tests passed", evidence=["sample fixed"]),
            ]

        def run(self, *, confirm=None, on_auto_approve=None, on_message=None):
            return self.reports.pop(0)

    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {
                    "actions": [
                        {"type": "goal", "text": "change file", "complete": False},
                        {
                            "type": "edit",
                            "goal": "edit sample badly",
                            "targets": [{"path": "sample.txt", "area": "line 1"}],
                            "constraints": [],
                            "self_check": [],
                        },
                    ],
                },
                {"actions": [{"type": "goal", "text": "change file", "complete": True, "message_for_complete": "done"}]},
                {
                    "actions": [
                        {
                            "type": "edit",
                            "goal": "fix sample",
                            "targets": [{"path": "sample.txt", "area": "line 1"}],
                            "constraints": [],
                            "self_check": [],
                        },
                    ],
                },
                {"actions": [{"type": "goal", "text": "change file", "complete": True, "message_for_complete": "done"}]},
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)
    agent.model_client = FakeModelClient()
    editor = FakeEditAgent()
    verifier = FakeVerifyAgent()

    def make_edit_agent(*, goal, scope):
        real_editor = nanocode.EditAgent(
            parent_session=session,
            parent_blackboard=agent.blackboard,
            goal=goal,
            scope=scope,
            handoff_context=agent._handoff_context_snapshot(),
        )
        handoff_prompts.append(real_editor.build_user_prompt())
        return editor

    agent._make_edit_agent = make_edit_agent
    agent._make_verify_agent = lambda *, goal, scope: verifier

    response = agent.run("change file", confirm=lambda call, tool: True)

    assert response["actions"][-1]["message_for_complete"] == "done"
    assert "<Agent_Reports>" in agent.model_client.user_prompts[2]
    assert "<Verify_History>" in agent.model_client.user_prompts[2]
    assert "assertion failed" in agent.model_client.user_prompts[2]
    assert "<verified>" in handoff_prompts[1]
    assert "failed | unit | assertion failed | issue: sample still wrong" in handoff_prompts[1]
    assert (tmp_path / "sample.txt").read_text(encoding="utf-8") == "new\n"


def test_agent_run_hands_pending_verification_to_verify_agent(tmp_path):
    verifier_calls = []

    class FakeVerifyAgent:
        def __init__(self, *, goal, scope):
            self.goal = goal
            self.scope = scope

        def run(self, *, confirm=None, on_auto_approve=None, on_message=None):
            verifier_calls.append((self.goal, self.scope))
            return nanocode.VerifyReport(status="passed", method="manual check", summary="checked")

    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {
                    "actions": [
                        {"type": "goal", "text": "answer", "complete": False},
                        {"type": "verify", "method": "manual check", "status": "pending", "context": "check answer"},
                    ],
                },
                {
                    "actions": [
                        {"type": "goal", "text": "answer", "complete": True, "message_for_complete": "done"},
                    ],
                },
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    agent = MainAgent(Session(cwd=str(tmp_path)))
    agent.model_client = FakeModelClient()
    agent._make_verify_agent = lambda *, goal, scope: FakeVerifyAgent(goal=goal, scope=scope)
    messages = []

    response = agent.run("answer", on_message=messages.append)

    assert response["actions"][-1]["message_for_complete"] == "done"
    assert verifier_calls
    assert verifier_calls[0][0] == "manual check"
    assert "verification target: manual check" in verifier_calls[0][1]
    assert "verification context: check answer" in verifier_calls[0][1]
    assert "Verifying: manual check" in messages
    assert "<Agent_Reports>" in agent.model_client.user_prompts[1]
    assert "<Verify_History>" in agent.model_client.user_prompts[1]


def test_agent_run_retries_when_verification_done_without_goal_complete(tmp_path):
    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {
                    "actions": [
                        {"type": "goal", "text": "change file", "complete": False},
                        {"type": "verify", "method": "run tests", "status": "passed", "context": "tests passed"},
                    ],
                },
                {"actions": _final_actions("change file")},
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)
    agent.model_client = FakeModelClient()
    messages = []

    response = agent.run("change file", on_message=messages.append)

    assert response["actions"][-1]["message_for_complete"] == "done"
    assert len(agent.model_client.user_prompts) == 2
    assert "Retrying: verification is done but goal is not complete." in messages
    assert "verification is done but goal.complete is not true" in agent.model_client.user_prompts[1]
    assert "goal complete=true with message_for_complete" in agent.model_client.user_prompts[1]
    assert agent.blackboard.verification.status == VerificationStatus.IDLE


def test_agent_run_retries_when_goal_complete_has_no_message(tmp_path):
    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {"actions": [{"type": "goal", "text": "answer", "complete": True}]},
                {"actions": _final_actions()},
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)
    agent.model_client = FakeModelClient()
    messages = []

    response = agent.run("answer", on_message=messages.append)

    assert response["actions"][-1]["message_for_complete"] == "done"
    assert len(agent.model_client.user_prompts) == 2
    assert "Retrying: goal is complete but message_for_complete is missing." in messages
    assert "goal.complete=true without message_for_complete" in agent.model_client.user_prompts[1]
    assert agent.agent_feedback_errors == []
    assert agent.blackboard.goal_reached is False


def test_agent_run_retries_format_error_with_recent_tool_calls(tmp_path):
    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {"_format_error": "Invalid model output: plain answer", "actions": []},
                {"actions": _final_actions()},
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)
    agent.model_client = FakeModelClient()
    messages = []

    response = agent.run("answer", on_message=messages.append)

    assert response["actions"][-1]["message_for_complete"] == "done"
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
                {"actions": _final_actions()},
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)
    agent.model_client = FakeModelClient()

    response = agent.run("answer")

    assert response["actions"][-1]["message_for_complete"] == "done"
    assert len(agent.model_client.user_prompts) == 3
    assert "model returned invalid output" in agent.model_client.user_prompts[1]
    assert "Rule: return valid JSON action frames only." in agent.model_client.user_prompts[1]
    assert "model returned invalid output" in agent.model_client.user_prompts[2]
    assert agent.agent_feedback_errors == []


def test_agent_allows_progress_message_before_goal_complete(tmp_path):
    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {"actions": [{"type": "progress", "text": "progress"}]},
                {"actions": _final_actions()},
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)
    agent.model_client = FakeModelClient()

    messages = []
    response = agent.run("answer", on_message=messages.append)

    assert response["actions"][-1]["message_for_complete"] == "done"
    assert messages[0] == "progress"
    assert messages[-1] == "done"
    assert "progress" not in [item.content for item in session.conversation]
    assert agent.agent_feedback_errors == []


def test_agent_shows_progress_with_tool_action_without_storing_it(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\n", encoding="utf-8")

    class FakeModelClient:
        def __init__(self):
            self.responses = [
                {
                    "actions": [
                        {"type": "progress", "text": "reading sample"},
                        {"type": "tool", "name": "Read", "intention": "read sample", "args": ["sample.txt"]},
                    ]
                },
                {"actions": _final_actions()},
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)
    agent.model_client = FakeModelClient()

    messages = []
    response = agent.run("answer", on_message=messages.append)

    assert response["actions"][-1]["message_for_complete"] == "done"
    assert messages[0] == "reading sample"
    assert "reading sample" not in [item.content for item in session.conversation]


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
    agent = MainAgent(session)
    agent.blackboard.goal = "answer"
    agent.blackboard.plan = [nanocode.PlanItem(text="try answer")]
    agent.blackboard.known = ["keep this fact"]
    agent.blackboard.verification.status = VerificationStatus.REQUIRED
    agent.latest_tool_batch = "old tool call"
    agent.latest_tool_call_blocks = ["old tool call"]
    agent.model_client = FakeModelClient()

    try:
        agent.run("answer")
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("expected KeyboardInterrupt")

    assert agent.agent_feedback_errors == []
    assert agent.latest_tool_batch == ""
    assert agent.recent_tool_calls == ""
    assert agent.blackboard.goal == ""
    assert agent.blackboard.plan == []
    assert agent.blackboard.known == ["keep this fact"]
    assert agent.blackboard.verification.status == VerificationStatus.IDLE
    assert agent.blackboard.goal_reached is False


def test_agent_run_rejects_extra_top_level_response_keys(tmp_path):
    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {"actions": [], "message_to_user": "old protocol"},
                {"actions": _final_actions()},
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)
    agent.model_client = FakeModelClient()

    response = agent.run("answer")

    assert response["actions"][-1]["message_for_complete"] == "done"
    assert len(agent.model_client.user_prompts) == 2


def test_agent_run_only_shows_ignored_action_frame_errors_in_debug(tmp_path):
    class FakeModelClient:
        def __init__(self):
            self.responses = [
                {
                    "actions": _final_actions(),
                    "_format_frame_errors": ["frame 1: expected JSON object action"],
                }
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)
    agent.model_client = FakeModelClient()
    messages = []

    agent.run("answer", on_message=messages.append)

    assert "Format_Warning:" not in "\n".join(messages)
    assert messages[-1] == "done"

    debug_session = Session(cwd=str(tmp_path), debug=True)
    debug_agent = MainAgent(debug_session)
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
                {"actions": _final_actions()},
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path), debug=True)
    agent = MainAgent(session)
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
    agent = MainAgent(session)
    agent.model_client = FakeModelClient()
    messages = []

    try:
        agent.run("answer", on_message=messages.append)
    except nanocode.LLMError as error:
        message = str(error)
    else:
        raise AssertionError("expected LLMError")

    assert agent.model_client.calls == MainAgent.MAX_CONSECUTIVE_FORMAT_ERRORS
    assert "model returned invalid output 3 times in a row" in message
    assert messages[-1] == "Stopped: model returned invalid output 3 times in a row."


def test_agent_run_no_retry_when_goal_complete_has_message_for_complete(tmp_path):
    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {"actions": [{"type": "goal", "text": "answer", "complete": True, "message_for_complete": "Task completed successfully"}]},
                {"actions": _final_actions()},
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)
    agent.model_client = FakeModelClient()
    messages = []

    response = agent.run("answer", on_message=messages.append)

    assert response["actions"][0]["message_for_complete"] == "Task completed successfully"
    assert len(agent.model_client.user_prompts) == 1
    assert "Task completed successfully" in messages
    assert "Retrying: goal is complete but message_for_complete is missing." not in " ".join(messages)

def test_agent_run_retries_when_goal_complete_has_empty_message_for_complete(tmp_path):
    """Empty string message_for_complete is falsy, so retry should still happen."""
    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {"actions": [{"type": "goal", "text": "answer", "complete": True, "message_for_complete": ""}]},
                {"actions": _final_actions()},
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)
    agent.model_client = FakeModelClient()
    messages = []

    response = agent.run("answer", on_message=messages.append)

    assert response["actions"][-1]["message_for_complete"] == "done"
    assert len(agent.model_client.user_prompts) == 2
    assert "Retrying: goal is complete but message_for_complete is missing." in messages
    assert agent.agent_feedback_errors == []


def test_agent_run_uses_message_for_complete_even_when_progress_actions_exist(tmp_path):
    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {
                    "actions": [
                        {"type": "goal", "text": "answer", "complete": True, "message_for_complete": "fallback message"},
                        {"type": "progress", "text": "explicit progress"},
                    ]
                },
                {"actions": _final_actions()},
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)
    agent.model_client = FakeModelClient()
    messages = []

    response = agent.run("answer", on_message=messages.append)

    assert response["actions"][0]["message_for_complete"] == "fallback message"
    assert "explicit progress" in messages
    assert messages[-1] == "fallback message"
    assert len(agent.model_client.user_prompts) == 1
    assert "explicit progress" not in [item.content for item in session.conversation]


def test_agent_run_ignores_message_for_complete_when_goal_not_complete(tmp_path):
    """message_for_complete should be ignored when complete=false."""
    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {"actions": [{"type": "goal", "text": "answer", "complete": False, "message_for_complete": "should be ignored"}]},
                {"actions": [{"type": "progress", "text": "done without goal"}]},
                {"actions": _final_actions()},
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)
    agent.model_client = FakeModelClient()
    messages = []

    response = agent.run("answer", on_message=messages.append)

    assert response["actions"][-1]["message_for_complete"] == "done"
    assert len(agent.model_client.user_prompts) == 3
    assert "should be ignored" not in messages
    assert agent.agent_feedback_errors == []
