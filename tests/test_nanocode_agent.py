import json

import nanocode
from nanocode import Agent, LLMError, ParsedToolCall, Session, VerificationStatus


def _verify_passed_action():
    return {"type": "verify", "method": "unit", "status": "passed", "context": "checked"}


def _final_actions(goal="answer", message="done"):
    return [
        _verify_passed_action(),
        {"type": "goal", "text": goal, "complete": True},
        {"type": "message", "text": message},
    ]


def test_agent_tool_results_go_to_recent_tool_calls_and_store(tmp_path):
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


def test_agent_dedupes_same_batch_readonly_tool_calls_keeping_latest(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)

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
    agent = Agent(session)

    agent.execute_tool_calls(
        [
            {"name": "Edit", "intention": "first edit", "args": ["sample.txt", "old", "new"]},
            {"name": "Edit", "intention": "second edit", "args": ["sample.txt", "old", "new"]},
        ],
        confirm=lambda call, tool: True,
    )

    assert len(agent.tool_runner.latest_executions) == 2
    assert [execution.outcome for execution in agent.tool_runner.latest_executions] == ["success", "failure"]
    assert list(session.tool_result_store) == ["tr.1", "tr.2"]
    assert path.read_text(encoding="utf-8") == "new\n"


def test_agent_tool_results_are_bounded_and_logged(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("H" * 5000 + "M" * 5000 + "T" * 5000 + "\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)

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
    agent = Agent(session)
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
    agent = Agent(session)
    agent.RECENT_TOOL_CALL_CHARS = 80

    agent._append_recent_tool_call_blocks(["old call " + "x" * 40])
    agent._append_recent_tool_call_blocks(["new call " + "y" * 40])

    assert "old call" not in agent.recent_tool_calls
    assert "new call" in agent.recent_tool_calls
    assert len(agent.recent_tool_call_blocks) == 1


def test_tool_result_store_keeps_latest_256_items(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)

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
    agent = Agent(Session(cwd=str(tmp_path)))
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
    agent = Agent(Session(cwd=str(tmp_path)))
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
    agent = Agent(Session(cwd=str(tmp_path)))
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
    agent = Agent(Session(cwd=str(tmp_path)))
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
        Agent(session).request("system", "user")
    except LLMError as error:
        assert str(error) == "request model timeout"
    else:
        raise AssertionError("expected LLMError")

    assert session.current_model_call_started_at == 0.0
    assert sleeps == [3, 10, 20, 30, 60, 120]


def test_agent_run_previews_streamed_tool_action_before_execution_report(tmp_path, monkeypatch):
    (tmp_path / "sample.txt").write_text("alpha\n", encoding="utf-8")
    captured_payloads = []
    responses = [
        [
            '{"type":"tool","name":"Read",',
            '"intention":"read sample","args":["sample.txt","0","1"]}__END_ACTION__',
        ],
        [
            '{"type":"verify","method":"unit","status":"passed","context":"checked"}__END_ACTION__',
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
    agent = Agent(session)

    class FakeProjectMapClient:
        def request(self, system_prompt, user_prompt, *, activity="main"):
            return {"items": []}

    agent.project_map_extractor.model_client = FakeProjectMapClient()
    messages = []

    response = agent.run("read sample", on_message=messages.append)

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


def test_agent_request_accepts_single_unmarked_json_action(tmp_path):
    client = Agent(Session(cwd=str(tmp_path))).model_client

    response = client._parse_model_content('{"type":"message","text":"ok"}')

    assert response == {"actions": [{"type": "message", "text": "ok"}]}


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
    assert "expected one JSON action object or action frames ending with __END_ACTION__" in response["_format_error"]
    assert "plain answer" in response["_format_error"]


def test_agent_request_rejects_unmarked_json_action_array(tmp_path):
    client = Agent(Session(cwd=str(tmp_path))).model_client

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
                        "Search only supports rg and Python fallback.",
                        "Search only supports rg and Python fallback.",
                    ],
                }
            ]
        }
    )

    assert session.current.known == ["Search only supports rg and Python fallback."]


def test_agent_dedupes_exact_known_facts(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)

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

    assert session.current.known == [
        "Preview logic exists in _format_stream_action_preview.",
        "Preview logic exists in _format_stream_action_preview!",
    ]


def test_agent_keeps_latest_50_known_items(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)

    agent.apply_response({"actions": [{"type": "known", "items": ["fact " + str(index) for index in range(51)]}]})

    assert len(session.current.known) == 50
    assert session.current.known[0] == "fact 1"
    assert session.current.known[-1] == "fact 50"


def test_agent_ignores_known_items_without_fact(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)

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

    assert session.current.known == [
        "Parser notes exist.",
        "Parser notes were captured.",
    ]


def test_agent_ignores_project_map_actions_from_main_response(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)

    agent.apply_response(
        {
            "actions": [
                {
                    "type": "project_map",
                    "items": [
                        "",
                        "nanocode is a single-file Python CLI.",
                        "nanocode is a single-file Python CLI.",
                        "Tests live in tests/.",
                    ],
                }
            ]
        }
    )

    assert session.project_map == []
    assert session.current.known == []
    assert "Project_Map" not in agent.state_updater.latest_report


def test_project_map_updater_keeps_latest_30_project_map_items(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)

    agent.project_map_extractor._apply({"items": ["map " + str(index) for index in range(31)]})

    assert len(session.project_map) == 30
    assert session.project_map[0] == "map 1"
    assert session.project_map[-1] == "map 30"


def test_project_map_updater_supports_patch_operations(tmp_path):
    session = Session(cwd=str(tmp_path))
    session.project_map = ["old architecture", "remove me", "keep me"]
    agent = Agent(session)

    changed = agent.project_map_extractor._apply(
        {
            "mode": "patch",
            "items": [
                {"op": "update", "index": 1, "text": "updated architecture"},
                {"op": "delete", "old_text": "remove me"},
                {"op": "append", "text": "new stable fact"},
            ],
        }
    )

    assert changed == 3
    assert session.project_map == ["updated architecture", "keep me", "new stable fact"]


def test_project_map_extractor_stores_patch_text_not_raw_dict(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)

    changed = agent.project_map_extractor._apply({"items": [{"op": "append", "index": None, "old_text": None, "text": "stable fact"}]})

    assert changed == 1
    assert session.project_map == ["stable fact"]


def test_agent_learns_project_map_from_recent_tool_context(tmp_path):
    class FakeProjectMapClient:
        def __init__(self):
            self.requests = []

        def request(self, system_prompt, user_prompt, *, activity="main"):
            self.requests.append((system_prompt, user_prompt, activity))
            return {
                "items": [
                    "nanocode.py contains the agent loop.",
                    "",
                    "nanocode.py contains the agent loop.",
                ]
            }

    session = Session(cwd=str(tmp_path))
    session.project_map = ["Tests live in tests/."]
    agent = Agent(session)
    fake_client = FakeProjectMapClient()
    agent.project_map_extractor.model_client = fake_client

    added = agent.learn_project_map('Tool Calls\n  1. ok Read("nanocode.py", "0,20")')

    assert added == 1
    assert session.project_map == [
        "Tests live in tests/.",
        "nanocode.py contains the agent loop.",
    ]
    assert fake_client.requests[0][2] == "project_map"
    assert 'Read("nanocode.py", "0,20")' in fake_client.requests[0][1]


def test_agent_state_report_only_includes_real_plan_and_known_changes(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)

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
    session.current.goal = "old goal"
    session.current.verification.goal = "old goal"
    session.current.verification.status = VerificationStatus.DONE
    session.current.verification.method = "old check"
    session.current.verification.context = "old context"
    agent = Agent(session)

    agent.apply_response({"actions": [{"type": "goal", "text": "new goal", "complete": False}]})

    assert session.current.goal_reached is False
    assert session.current.verification.goal == ""
    assert session.current.verification.status == VerificationStatus.IDLE
    assert session.current.verification.method == ""
    assert session.current.verification.context == ""

    agent.apply_response({"actions": [{"type": "verify", "method": "run tests", "status": "pending", "context": None}]})

    assert session.current.verification.goal == "new goal"
    assert session.current.verification.status == VerificationStatus.REQUIRED
    assert session.current.verification.method == "run tests"
    assert session.current.verification.context == ""

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
    assert (tmp_path / ".nanocode" / "tool_results").exists()


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
    assert (tmp_path / ".nanocode" / "tool_results").exists()


def test_agent_execute_tool_calls_records_arg_errors_in_feedback(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)

    latest = agent.execute_tool_calls([{"name": "Read", "intention": "bad range", "args": ["sample.txt", "bad", "1"]}])

    assert "ToolCallError: invalid start: should be an integer" in latest
    assert agent.agent_feedback_errors == [
        'Error: tool call args invalid: Read("sample.txt", "bad", "1") -> ToolCallError: invalid start: should be an integer. Rule: use the tool signature exactly.'
    ]


def test_agent_execute_tool_calls_does_not_record_runtime_errors_in_feedback(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)

    latest = agent.execute_tool_calls([{"name": "Read", "intention": "missing file", "args": ["missing.txt", "0", "1"]}])

    assert "ToolCallError: " in latest
    assert agent.agent_feedback_errors == []


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
    assert "<ToolCall ok" in latest
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
                            "items": ["Read sample.txt and found alpha."],
                        },
                        _verify_passed_action(),
                        {"type": "goal", "text": "read sample", "complete": True},
                        {"type": "message", "text": "done"},
                    ],
                },
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    class FakeProjectMapClient:
        def __init__(self):
            self.requests = []

        def request(self, system_prompt, user_prompt, *, activity="main"):
            self.requests.append((system_prompt, user_prompt, activity))
            return {"items": ["ReadTool can read fixture files during tests."]}

    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    fake_client = FakeModelClient()
    fake_project_map_client = FakeProjectMapClient()
    agent.model_client = fake_client
    agent.project_map_extractor.model_client = fake_project_map_client

    messages = []
    response = agent.run("read sample", on_message=messages.append)

    assert response["actions"][-1]["text"] == "done"
    assert messages[0].startswith("Tool Calls\n")
    assert '1. ok Read("sample.txt", "0", "1")' in messages[0]
    assert "     tr.1 | why: read sample" in messages[0]
    assert "log: .nanocode/tool_results/" not in messages[0]
    assert messages[-2] == "done"
    assert messages[-1] == "Project_Map updated: 1 change(s)"
    assert "alpha" not in fake_client.user_prompts[0]
    assert "alpha" in fake_client.user_prompts[1]
    assert "tr.1" in fake_client.user_prompts[1]
    assert agent.latest_tool_batch == ""
    assert agent.recent_tool_calls == ""
    assert session.current.known == ["Read sample.txt and found alpha."]
    assert session.current.user_input == "read sample"
    assert session.current.goal == ""
    assert session.current.plan == []
    assert session.current.verification.status == VerificationStatus.IDLE
    assert session.current.goal_reached is False
    assert session.project_map == ["ReadTool can read fixture files during tests."]
    assert fake_project_map_client.requests[0][2] == "project_map"
    assert "alpha" in fake_project_map_client.requests[0][1]


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
    agent = Agent(session)
    agent.model_client = FakeModelClient()

    response = agent.run("read sample")

    assert response["actions"][-1]["text"] == "done"
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
    session.current.goal = "answer"
    session.current.plan = [nanocode.PlanItem(text="try answer")]
    session.current.known = ["keep this fact"]
    session.current.verification.status = VerificationStatus.REQUIRED
    agent = Agent(session)
    agent.latest_tool_batch = "old tool call"
    agent.latest_tool_call_blocks = ["old tool call"]
    agent.model_client = FakeModelClient()

    agent.run("read samples")

    assert len(session.tool_result_store) == 50
    assert list(session.tool_result_store)[:2] == ["tr.2", "tr.3"]
    assert list(session.tool_result_store)[-1] == "tr.51"
    assert session.tool_result_counter == 51
    assert session.current.goal == ""
    assert session.current.plan == []
    assert session.current.known == ["keep this fact"]
    assert session.current.verification.status == VerificationStatus.IDLE
    assert session.current.goal_reached is False


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
                {"actions": _final_actions()},
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


def test_agent_run_stops_after_chat_action(tmp_path):
    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []

        def request(self, system_prompt, user_prompt, *, activity="main"):
            self.user_prompts.append(user_prompt)
            return {"actions": [{"type": "chat", "text": "你好"}]}

    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
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
                {"actions": _final_actions()},
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    agent.model_client = FakeModelClient()
    messages = []

    response = agent.run("answer", on_message=messages.append)

    assert response["actions"][-1]["text"] == "done"
    assert "Continuing: assistant must set current task's goal." in messages


def test_agent_run_enforces_verification_gate_before_completion(tmp_path):
    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {
                    "actions": [
                        {"type": "goal", "text": "change file", "complete": False},
                        {"type": "verify", "method": "run tests", "status": "pending", "context": None},
                    ],
                },
                {
                    "actions": [
                        {"type": "verify", "method": "run tests", "status": "passed", "context": "tests passed"},
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
    assert session.current.verification.status == VerificationStatus.IDLE
    assert session.current.verification.context == ""
    assert "Retrying: verification is required before completion." in messages


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
    agent = Agent(session)
    agent.model_client = FakeModelClient()
    messages = []

    response = agent.run("change file", on_message=messages.append)

    assert response["actions"][-1]["text"] == "done"
    assert len(agent.model_client.user_prompts) == 2
    assert "Retrying: verification is done but goal is not complete." in messages
    assert "verification is done but goal.complete is not true" in agent.model_client.user_prompts[1]
    assert "goal complete=true" in agent.model_client.user_prompts[1]
    assert session.current.verification.status == VerificationStatus.IDLE


def test_agent_run_retries_when_goal_complete_has_no_message(tmp_path):
    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {"actions": [{"type": "goal", "text": "answer", "complete": True}]},
                {"actions": [{"type": "message", "text": "done without goal"}]},
                {"actions": _final_actions()},
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
    assert len(agent.model_client.user_prompts) == 3
    assert "Retrying: goal is complete but no message provided." in messages
    assert "done without goal" in messages
    assert "goal.complete=true without a message" in agent.model_client.user_prompts[1]
    assert "message before goal.complete=true" in agent.model_client.user_prompts[2]
    assert agent.agent_feedback_errors == []
    assert session.current.goal_reached is False


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
                {"actions": _final_actions()},
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
                {"actions": _final_actions()},
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
    session.current.goal = "answer"
    session.current.plan = [nanocode.PlanItem(text="try answer")]
    session.current.known = ["keep this fact"]
    session.current.verification.status = VerificationStatus.REQUIRED
    agent = Agent(session)
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
    assert session.current.goal == ""
    assert session.current.plan == []
    assert session.current.known == ["keep this fact"]
    assert session.current.verification.status == VerificationStatus.IDLE
    assert session.current.goal_reached is False


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
                    "actions": _final_actions(),
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
                {"actions": _final_actions()},
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
    agent = Agent(session)
    agent.model_client = FakeModelClient()
    messages = []

    response = agent.run("answer", on_message=messages.append)

    assert response["actions"][-1]["text"] == "done"
    assert len(agent.model_client.user_prompts) == 2
    assert "Task completed successfully" in messages
    assert "Retrying: goal is complete but no message provided." not in " ".join(messages)

def test_agent_run_retries_when_goal_complete_has_empty_message_for_complete(tmp_path):
    """Empty string message_for_complete is falsy, so retry should still happen."""
    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {"actions": [{"type": "goal", "text": "answer", "complete": True, "message_for_complete": ""}]},
                {"actions": [{"type": "message", "text": "done without goal"}]},
                {"actions": _final_actions()},
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
    assert len(agent.model_client.user_prompts) == 3
    assert "Retrying: goal is complete but no message provided." in messages
    assert agent.agent_feedback_errors == []


def test_agent_run_ignores_message_for_complete_when_message_actions_exist(tmp_path):
    """message_for_complete fallback only triggers when no message actions are present."""
    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {
                    "actions": [
                        {"type": "goal", "text": "answer", "complete": True, "message_for_complete": "fallback message"},
                        {"type": "message", "text": "explicit message"},
                    ]
                },
                {"actions": _final_actions()},
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


def test_agent_run_ignores_message_for_complete_when_goal_not_complete(tmp_path):
    """message_for_complete should be ignored when complete=false."""
    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {"actions": [{"type": "goal", "text": "answer", "complete": False, "message_for_complete": "should be ignored"}]},
                {"actions": [{"type": "message", "text": "done without goal"}]},
                {"actions": _final_actions()},
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
    assert len(agent.model_client.user_prompts) == 3
    assert "should be ignored" not in messages
    assert agent.agent_feedback_errors == []
