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


def _observe_actions(fact="observed latest result"):
    return [{"type": "observe", "known": [fact]}]


def _seed_plan(agent, goal="test goal"):
    agent.blackboard.goal = goal
    agent.blackboard.plan = [nanocode.PlanItem(text="test plan")]


def _blocks_text(blocks):
    return "\n".join(blocks)


def _session(
    tmp_path,
    *,
    api_url: str = "",
    api_key: str = "",
    model: str = "",
    stream: bool | None = None,
    timeout: int | None = None,
    first_token_timeout: int | None = None,
    reasoning_effort: str = "",
    yolo: bool = False,
    debug: bool = False,
    explore_max_turns: int = 12,
) -> Session:
    main_model: dict[str, object] = {"model": model}
    if stream is not None:
        main_model["stream"] = stream
    if timeout is not None:
        main_model["timeout"] = timeout
    if first_token_timeout is not None:
        main_model["first_token_timeout"] = first_token_timeout
    if reasoning_effort:
        main_model["reasoning_effort"] = reasoning_effort
    data = {
        "api": {"url": api_url, "key": api_key},
        "main_model": main_model,
        "explore_agent": {"max_turns": explore_max_turns},
    }
    return Session(cwd=str(tmp_path), config=nanocode.Config.from_dict(data), settings=nanocode.RuntimeSettings.from_dict(data, yolo=yolo, debug=debug))


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
    assert "- ok | Read sample.txt 0:1" in latest
    assert "why: read sample" in latest
    assert "result_key: tr.1" in latest
    assert "output:\n<ReadToolResult>" in latest
    assert session.state.tool_result_store["tr.1"].value.startswith("<ReadToolResult>")
    assert "alpha" in session.state.tool_result_store["tr.1"].value
    assert session.state.tool_result_store["tr.1"].log_path.startswith(".nanocode/tool_results/")
    assert session.state.tool_result_store["tr.1"].original_chars > 0
    assert session.state.tool_result_store["tr.1"].original_lines > 0
    assert session.state.tool_result_store["tr.1"].excerpted is False
    assert (tmp_path / session.state.tool_result_store["tr.1"].log_path).read_text(encoding="utf-8") == session.state.tool_result_store["tr.1"].value
    assert session.state.conversation == []
    assert (tmp_path / ".nanocode" / "tool_results").exists()


def test_agent_accepts_lowercase_tool_name_without_prompting_it(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)

    latest = agent.execute_tool_calls(
        [
            {
                "name": "read",
                "intention": "read sample",
                "args": ["sample.txt", "0", "1"],
            }
        ]
    )

    assert "alpha" in latest
    assert "- ok | Read sample.txt 0:1" in latest
    assert agent.tool_runner.latest_executions[0].call.name == "Read"


def test_explore_agent_cli_uses_compact_tool_report(tmp_path):
    (tmp_path / "sample.txt").write_text("alpha\n", encoding="utf-8")

    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {
                    "actions": [
                        {"type": "tool", "name": "Read", "intention": "read sample", "args": ["sample.txt", "0,1"]},
                    ]
                },
                {
                    "actions": [
                        {
                            "type": "deliver",
                            "targets": [{"path": "sample.txt", "area": "line 1", "reason": "found"}],
                            "known": ["sample.txt contains alpha."],
                            "issues": [],
                        }
                    ]
                },
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    parent_session = Session(cwd=str(tmp_path))
    parent_agent = MainAgent(parent_session)
    explorer = nanocode.ExploreAgent(parent_session=parent_session, parent_blackboard=parent_agent.blackboard, goal="find sample", scope=["sample.txt"])
    explorer.model_client = FakeModelClient()
    messages = []

    explorer.run(on_message=messages.append)

    assert messages == ["[success] Read sample.txt 0,1"]


def test_agent_formats_explore_done_targets_on_separate_lines(tmp_path):
    agent = MainAgent(Session(cwd=str(tmp_path)))

    message = agent._format_explore_done(
        nanocode.ExploreReport(
            targets=[
                {"path": "producer.py", "line_range": "440-460", "area": "pipeline integration"},
                {"path": "detector.py", "line_range": "186-206", "area": "page type detection"},
            ],
            known=[],
        )
    )

    assert message == "Explore done: 2 target(s)\n  1. producer.py:440-460 pipeline integration\n  2. detector.py:186-206 page type detection"


def test_explore_report_formats_and_briefs_issues():
    report = nanocode.ExploreReport(
        targets=[],
        known=[],
        issues=["handoff goal asks for analysis, not location"],
    )

    formatted = report.format()

    assert "issues:" in formatted
    assert "handoff goal asks for analysis, not location" in formatted
    assert report.brief() == ["issue: handoff goal asks for analysis, not location"]


def test_worker_report_history_uses_worker_reports_heading():
    history = nanocode.WorkerReportHistory(verify=["Verify Report: passed"], verified=["verify: passed"])

    formatted = history.format()

    assert "Worker Reports:" in formatted
    assert "Verify Report: passed" in formatted
    assert "<Agent_Reports>" not in formatted


def test_worker_report_history_prunes_old_items():
    history = nanocode.WorkerReportHistory(
        explore=[f"explore {index}" for index in range(4)],
        verify=[f"verify {index}" for index in range(4)],
        explored=[f"explored {index}" for index in range(4)],
        verified=[f"verified {index}" for index in range(4)],
    )

    history.prune(2)

    assert history.explore == ["explore 2", "explore 3"]
    assert history.verify == ["verify 2", "verify 3"]
    assert history.explored == ["explored 2", "explored 3"]
    assert history.verified == ["verified 2", "verified 3"]


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
    assert list(session.state.tool_result_store) == ["tr.1"]
    assert "second read" in session.state.tool_result_store["tr.1"].description
    assert "first read" not in latest


def test_agent_does_not_dedupe_nonconsecutive_same_batch_readonly_tool_calls(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\nbeta\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)

    agent.execute_tool_calls(
        [
            {"name": "Read", "intention": "first read", "args": ["sample.txt", "0,1"]},
            {"name": "Read", "intention": "middle read", "args": ["sample.txt", "1,2"]},
            {"name": "Read", "intention": "second read", "args": ["sample.txt", "0,1"]},
        ]
    )

    assert [execution.call.intention for execution in agent.tool_runner.latest_executions] == ["first read", "middle read", "second read"]
    assert list(session.state.tool_result_store) == ["tr.1", "tr.2", "tr.3"]


def test_agent_merges_adjacent_recall_calls(tmp_path):
    session = Session(cwd=str(tmp_path))
    session.state.tool_result_store["tr.1"] = nanocode.ToolResultItem(description="success Read a", value="alpha")
    session.state.tool_result_store["tr.2"] = nanocode.ToolResultItem(description="success Read b", value="beta")
    agent = MainAgent(session)

    agent.execute_tool_calls(
        [
            {"name": "Recall", "intention": "recall first", "args": ["tr.1"]},
            {"name": "Recall", "intention": "recall second", "args": ["tr.2"]},
            {"name": "Recall", "intention": "recall duplicated", "args": ["tr.1", "tr.2"]},
        ]
    )

    assert len(agent.tool_runner.latest_executions) == 1
    assert agent.tool_runner.latest_executions[0].call.args == ["tr.1", "tr.2"]


def test_worker_reuses_repeated_readonly_tool_results_across_turns(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\n", encoding="utf-8")
    parent_session = Session(cwd=str(tmp_path))
    parent_agent = MainAgent(parent_session)
    explorer = nanocode.ExploreAgent(parent_session=parent_session, parent_blackboard=parent_agent.blackboard, goal="inspect sample", scope=["sample.txt"])

    explorer.execute_tool_calls([{"name": "Read", "intention": "first read", "args": ["sample.txt", "0,1"]}])
    explorer.execute_tool_calls([{"name": "Read", "intention": "repeat read", "args": ["sample.txt", "0,1"]}])

    assert list(explorer.runtime.tool_result_store) == ["tr.1"]
    assert explorer.tool_runner.latest_executions[0].result_key == "tr.1"
    assert "alpha" in explorer.tool_runner.latest_executions[0].output


def test_worker_does_not_reuse_nonconsecutive_readonly_tool_results(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\n", encoding="utf-8")
    parent_session = Session(cwd=str(tmp_path))
    parent_agent = MainAgent(parent_session)
    explorer = nanocode.ExploreAgent(parent_session=parent_session, parent_blackboard=parent_agent.blackboard, goal="inspect sample", scope=["sample.txt"])

    explorer.execute_tool_calls([{"name": "Read", "intention": "first read", "args": ["sample.txt", "0,1"]}])
    explorer.execute_tool_calls([{"name": "LineCount", "intention": "count lines", "args": ["sample.txt"]}])
    explorer.execute_tool_calls([{"name": "Read", "intention": "second read", "args": ["sample.txt", "0,1"]}])

    assert list(explorer.runtime.tool_result_store) == ["tr.1", "tr.2", "tr.3"]
    assert explorer.tool_runner.latest_executions[0].result_key == "tr.3"


def test_worker_prompts_do_not_include_parent_verification_state(tmp_path):
    parent_session = Session(cwd=str(tmp_path))
    parent_agent = MainAgent(parent_session)
    parent_agent.blackboard.verification.status = VerificationStatus.DONE
    parent_agent.blackboard.verification.context = "parent verification should stay private"

    explorer = nanocode.ExploreAgent(parent_session=parent_session, parent_blackboard=parent_agent.blackboard, goal="inspect sample", scope=[])
    verifier = nanocode.VerifyAgent(parent_session=parent_session, parent_blackboard=parent_agent.blackboard, goal="verify sample", scope=[])

    assert "### Verification State" not in explorer.build_user_prompt()
    assert "parent verification should stay private" not in explorer.build_user_prompt()
    assert "### Verification State" not in verifier.build_user_prompt()
    assert "parent verification should stay private" not in verifier.build_user_prompt()


def test_agent_does_not_dedupe_same_batch_edit_tool_calls(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("old\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)

    agent.execute_tool_calls(
        [
            {"name": "Edit", "intention": "first edit", "args": ["sample.txt", "old", "new"]},
            {"name": "Edit", "intention": "second edit", "args": ["sample.txt", "old", "new"]},
        ],
        confirm=lambda call, tool: True,
    )

    assert len(agent.tool_runner.latest_executions) == 2
    assert [execution.outcome for execution in agent.tool_runner.latest_executions] == ["success", "failure"]
    assert list(session.state.tool_result_store) == ["tr.1", "tr.2"]
    assert path.read_text(encoding="utf-8") == "new\n"


def test_agent_tool_results_are_bounded_and_logged(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("H" * 5000 + "M" * 5000 + "T" * 5000 + "\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)

    latest = agent.execute_tool_calls([{"name": "Read", "intention": "read large sample", "args": ["sample.txt", "0", "1"]}])

    item = session.state.tool_result_store["tr.1"]
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

    latest = _blocks_text(agent.latest_tool_call_blocks)
    recent = _blocks_text(agent.recent_tool_call_blocks)
    assert "four.txt" in latest
    assert "four.txt" not in recent
    assert "one.txt" not in recent
    assert "two.txt" in recent
    assert "three.txt" in recent
    assert "<ReadToolResult>" in latest
    assert "<ReadToolResult>" not in recent
    assert "output_summary:" in recent
    assert "Recall(" not in recent
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

    recent = _blocks_text(agent.recent_tool_call_blocks)
    assert "old call" not in recent
    assert "new call" in recent
    assert len(agent.recent_tool_call_blocks) == 1


def test_tool_result_store_keeps_latest_256_items(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)

    for index in range(257):
        agent.tool_runner._store_tool_result(ParsedToolCall(name="Read", intention="", args=[str(index)]), "success", "output " + str(index))

    assert len(session.state.tool_result_store) == 256
    assert list(session.state.tool_result_store)[:2] == ["tr.2", "tr.3"]
    assert list(session.state.tool_result_store)[-1] == "tr.257"
    assert session.state.tool_result_counter == 257


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
    session = _session(tmp_path, api_url="https://example.test/v1", api_key="key", model="model", timeout=12, stream=False)

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
    assert session.state.last_prompt_tokens == 2
    assert session.state.last_completion_tokens == 3
    assert session.state.last_total_tokens == 5


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
    session = _session(tmp_path, api_url="https://openrouter.ai/api/v1", api_key="key", model="main", stream=False)
    session.config.worker_model = nanocode.ModelConfig(
        model="worker",
        temperature=0.2,
        reasoning=True,
        reasoning_effort="low",
        stream=False,
        timeout=7,
    )

    response = MainAgent(session).model_client.request("system", "user", activity="explore")

    assert response == {"actions": [{"type": "message", "text": "ok"}]}
    assert captured["payload"]["model"] == "worker"
    assert captured["payload"]["temperature"] == 0.2
    assert captured["payload"]["reasoning"] == {"effort": "low"}
    assert captured["timeout"] == 7


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
    assert agent.session.state.turn_model_calls == 4
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
    assert agent.session.state.turn_model_calls == 3
    assert sleeps == [3, 10]
    assert messages == [
        "Retrying: request model timeout; retry 1/6 in 3s.",
        "Retrying: request model timeout; retry 2/6 in 10s.",
    ]


def test_agent_gate_reports_only_on_second_retry_in_non_debug(tmp_path):
    agent = MainAgent(Session(cwd=str(tmp_path)))
    messages = []

    agent._report_gate(messages.append, "Retrying: sample gate.", "Sample_Gate: first")
    agent._report_gate(messages.append, "Retrying: sample gate.", "Sample_Gate: second")
    agent._report_gate(messages.append, "Retrying: sample gate.", "Sample_Gate: third")

    assert messages == ["Retrying: sample gate."]


def test_agent_gate_reports_immediately_in_debug(tmp_path):
    agent = MainAgent(_session(tmp_path, debug=True))
    messages = []

    agent._report_gate(messages.append, "Retrying: sample gate.", "Sample_Gate: debug")

    assert messages == ["Sample_Gate: debug"]


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
    assert agent.session.state.turn_model_calls == 7
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
    assert agent.session.state.turn_model_calls == 1
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
    session = _session(tmp_path, api_url="https://example.test/v1", api_key="key", model="model")

    response = MainAgent(session).request("system", "user")

    assert captured["payload"]["stream"] is True
    assert captured["payload"]["stream_options"] == {"include_usage": True}
    assert response["actions"] == [
        {"type": "tool", "name": "Read", "intention": "read sample", "args": ["sample.txt"]},
        {"type": "message", "text": "done"},
    ]
    assert session.state.last_prompt_tokens == 2
    assert session.state.last_completion_tokens == 3
    assert session.state.last_total_tokens == 5
    assert session.state.session_total_tokens == 5


def test_agent_request_stream_uses_first_token_timeout_until_content(tmp_path, monkeypatch):
    timers = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def __iter__(self):
            yield ("data: " + json.dumps({"choices": [{"delta": {"role": "assistant"}}]}) + "\n").encode("utf-8")
            yield ("data: " + json.dumps({"choices": [{"delta": {"content": '{"type":"message","text":"ok"}__END_ACTION__'}}]}) + "\n").encode("utf-8")
            yield b"data: [DONE]\n"

    monkeypatch.setattr(nanocode.urllib.request, "urlopen", lambda request, timeout: FakeResponse())
    monkeypatch.setattr(nanocode.signal, "setitimer", lambda timer, seconds: timers.append(seconds))
    session = _session(tmp_path, api_url="https://example.test/v1", api_key="key", model="model", timeout=90, first_token_timeout=4)

    response = MainAgent(session).request("system", "user")

    assert response["actions"][0]["text"] == "ok"
    assert timers[0] == 90
    assert 4 in timers
    assert timers[-1] == 0


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
    session = _session(tmp_path, api_url="https://example.test/v1", api_key="key", model="model", timeout=12)

    try:
        MainAgent(session).request("system", "user")
    except LLMError as error:
        assert str(error) == "request model timeout"
    else:
        raise AssertionError("expected LLMError")

    assert session.state.current_model_call_started_at == 0.0
    assert sleeps == [3, 10, 20, 30, 60, 120]


def test_agent_run_reports_streamed_tool_actions_after_execution(tmp_path, monkeypatch):
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
    session = _session(tmp_path, api_url="https://example.test/v1", api_key="key", model="model")
    agent = MainAgent(session)
    _seed_plan(agent, "read sample")
    messages = []

    response = agent.run("read sample", on_message=messages.append)

    assert response["actions"][-1] == {"type": "goal", "text": "read sample", "complete": True, "message_for_complete": "done"}
    assert len(captured_payloads) == 2
    assert [payload["stream"] for payload in captured_payloads] == [True, True]
    assert messages[0].startswith("[success] Read sample.txt 0:1")
    assert "why:" not in messages[0]
    assert messages[-1] == "done"


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
    session = _session(tmp_path, api_url="https://openrouter.ai/api/v1", api_key="key", model="model", reasoning_effort="high", stream=False)

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
    session = _session(tmp_path, api_url="https://example.test/v1", api_key="key", model="model", timeout=12, debug=True, stream=False)

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
    session = _session(tmp_path, api_url="https://example.test/v1", api_key="key", model="model", stream=False)

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
    assert '"args":["nanocode.py","0,100"]' in response["_format_error"]


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
    session = _session(tmp_path, api_url="https://example.test/v1", api_key="key", model="model", stream=False)

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
    session = _session(tmp_path, api_url="https://example.test/v1", api_key="key", model="model", stream=False)

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
                    "type": "plan",
                    "mode": "patch",
                    "items": [],
                    "known": [
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
                    "type": "plan",
                    "mode": "patch",
                    "items": [],
                    "known": [
                        "Preview logic exists in _preview_segments.",
                        "Preview logic exists in _preview_segments.",
                        "Preview logic exists in _preview_segments!",
                    ],
                }
            ]
        }
    )

    assert agent.blackboard.known == [
        "Preview logic exists in _preview_segments.",
        "Preview logic exists in _preview_segments!",
    ]


def test_agent_keeps_latest_500_known_items(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)

    agent.apply_response({"actions": [{"type": "plan", "mode": "patch", "items": [], "known": ["fact " + str(index) for index in range(501)]}]})

    assert len(agent.blackboard.known) == 500
    assert agent.blackboard.known[0] == "fact 1"
    assert agent.blackboard.known[-1] == "fact 500"


def test_main_agent_applies_user_rule_and_saves(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)

    agent.apply_response({"actions": [{"type": "user_rule", "text": "Prompt-only changes do not need tests."}]})

    content = (tmp_path / ".nanocode" / "user_rules.md").read_text(encoding="utf-8")
    assert session.state.user_rules.format() == "# User Rules\n\n- Prompt-only changes do not need tests."
    assert content == "# User Rules\n\n- Prompt-only changes do not need tests.\n"
    assert "  User_Rules    updated" in agent.state_updater.latest_report


def test_user_rules_deduplicate(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)

    agent.apply_response(
        {
            "actions": [
                {"type": "user_rule", "text": "Prompt-only changes do not need tests."},
                {"type": "user_rule", "text": "- Prompt-only changes do not need tests."},
            ]
        }
    )

    assert session.state.user_rules.format() == "# User Rules\n\n- Prompt-only changes do not need tests."


def test_main_agent_user_rule_finishes_with_message(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)

    class FakeModelClient:
        def __init__(self):
            self.calls = 0

        def request(self, system_prompt, user_prompt, *, activity="main"):
            self.calls += 1
            return {
                "actions": [
                    {
                        "type": "user_rule",
                        "text": "Prompt-only changes do not need tests.",
                        "message": "记住了。",
                    }
                ]
            }

    fake_client = FakeModelClient()
    agent.model_client = fake_client
    messages = []

    response = agent.run("记住：prompt 改动不用测试", on_message=messages.append)

    assert fake_client.calls == 1
    assert response["actions"][0]["type"] == "user_rule"
    assert session.state.user_rules.format() == "# User Rules\n\n- Prompt-only changes do not need tests."
    assert any(message.startswith("State Updated") for message in messages)
    assert session.state.conversation[-1].content == "记住了。"


def test_agent_ignores_known_items_without_fact(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)

    agent.apply_response(
        {
            "actions": [
                {
                    "type": "plan",
                    "mode": "patch",
                    "items": [],
                    "known": [
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
            {
                "type": "plan",
                "mode": "replace",
                "items": [{"id": "p1", "text": "Inspect file", "status": "todo"}],
                "known": ["Search uses rg."],
            },
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


def test_agent_ignores_empty_plan_replace(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)
    agent.blackboard.plan = [nanocode.PlanItem(id="p1", text="Inspect file", status=nanocode.PlanStatus.TODO)]

    agent.apply_response({"actions": [{"type": "plan", "mode": "replace", "items": []}]})

    assert [item.text for item in agent.blackboard.plan] == ["Inspect file"]
    assert agent.state_updater.latest_report == ""


def test_agent_applies_start_action_to_goal_and_plan(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)

    agent.apply_response(
        {
            "actions": [
                {
                    "type": "start",
                    "goal": "change map",
                    "plan": [
                        {"id": "p1", "text": "Find map code", "status": "doing", "context": "need location"},
                        {"id": "p2", "text": "Edit map size", "status": "todo"},
                    ],
                }
            ]
        }
    )

    assert agent.blackboard.goal == "change map"
    assert agent.blackboard.goal_reached is False
    assert [item.text for item in agent.blackboard.plan] == ["Find map code", "Edit map size"]
    assert agent.blackboard.plan[0].status == nanocode.PlanStatus.DOING
    assert "  Goal    change map" in agent.state_updater.latest_report
    assert "  Plan\n" in agent.state_updater.latest_report


def test_agent_state_report_shows_goal_for_restarted_task_even_when_text_matches(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)
    agent.blackboard.goal = "change map"

    agent.apply_response(
        {
            "actions": [
                {
                    "type": "start",
                    "goal": "change map",
                    "plan": [{"id": "p1", "text": "Find map code", "status": "doing"}],
                }
            ]
        }
    )

    assert "  Goal    change map" in agent.state_updater.latest_report
    assert "  Plan\n" in agent.state_updater.latest_report


def test_agent_applies_response_language_from_start_action(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)

    agent.apply_response(
        {
            "actions": [
                {
                    "type": "start",
                    "goal": "change map",
                    "response_language": "zh-cn",
                    "plan": [{"id": "p1", "text": "Find map code", "status": "doing"}],
                }
            ]
        }
    )

    assert session.state.response_language_tag == "zh-CN"


def test_agent_resets_verification_when_goal_changes(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)
    agent.blackboard.goal = "old goal"
    agent.blackboard.verification.goal = "old goal"
    agent.blackboard.verification.status = VerificationStatus.DONE
    agent.blackboard.verification.kind = "test"
    agent.blackboard.verification.method = "old check"
    agent.blackboard.verification.criteria = ["old criterion"]
    agent.blackboard.verification.context = "old context"

    agent.apply_response({"actions": [{"type": "goal", "text": "new goal", "complete": False}]})

    assert agent.blackboard.goal_reached is False
    assert agent.blackboard.verification.goal == ""
    assert agent.blackboard.verification.status == VerificationStatus.IDLE
    assert agent.blackboard.verification.kind == ""
    assert agent.blackboard.verification.method == ""
    assert agent.blackboard.verification.criteria == []
    assert agent.blackboard.verification.context == ""

    agent.apply_response(
        {"actions": [{"type": "verify", "kind": "test", "method": "run tests", "criteria": ["tests pass"], "status": "pending", "context": None}]}
    )

    assert agent.blackboard.verification.goal == "new goal"
    assert agent.blackboard.verification.status == VerificationStatus.REQUIRED
    assert agent.blackboard.verification.kind == "test"
    assert agent.blackboard.verification.method == "run tests"
    assert agent.blackboard.verification.criteria == ["tests pass"]
    assert agent.blackboard.verification.context == ""

    agent.apply_response({"actions": [{"type": "goal", "text": "new goal", "complete": True}]})

    assert agent.blackboard.goal_reached is True


def test_agent_accepts_combined_pending_verification_kind(tmp_path):
    agent = MainAgent(Session(cwd=str(tmp_path)))

    assert (
        agent._pending_verification_error(
            [
                {
                    "type": "verify",
                    "kind": "syntax_check+test",
                    "method": "check edit",
                    "criteria": ["syntax passes", "tests pass"],
                    "status": "pending",
                }
            ]
        )
        == ""
    )

    for kind in ["syntax_check+", "+test", "syntax_check+unknown"]:
        assert (
            agent._pending_verification_error(
                [
                    {
                        "type": "verify",
                        "kind": kind,
                        "method": "check edit",
                        "criteria": ["check passes"],
                        "status": "pending",
                    }
                ]
            )
            == "missing or invalid kind"
        )


def test_agent_execute_tool_calls_requests_confirmation_for_edit_tools(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("old\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)
    confirmations = []

    latest = agent.execute_tool_calls(
        [{"name": "Edit", "intention": "edit sample", "args": ["sample.txt", "old", "new"]}],
        confirm=lambda call, tool: confirmations.append((call.executed, tool.preview())) or False,
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
    agent = MainAgent(session)

    latest = agent.execute_tool_calls(
        [{"name": "Edit", "intention": "edit sample", "args": ["sample.txt", "old", "new"]}],
        confirm=lambda call, tool: "please inspect tests first",
    )

    assert "Cancelled: user refused: please inspect tests first" in latest
    assert path.read_text(encoding="utf-8") == "old\n"
    assert session.state.conversation == []
    assert (tmp_path / ".nanocode" / "tool_results").exists()


def test_agent_execute_tool_calls_rejects_failed_preview_before_confirmation(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("old\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)
    confirmations = []

    latest = agent.execute_tool_calls(
        [{"name": "ReplaceRange", "intention": "edit stale range", "args": ["sample.txt", "0", "1", "bad", "", "", "new"]}],
        confirm=lambda call, tool: confirmations.append((call.executed, tool.preview())) or True,
    )

    assert confirmations == []
    assert "ToolCallError: preview unavailable: fingerprint mismatch" in latest
    assert path.read_text(encoding="utf-8") == "old\n"


def test_agent_execute_tool_calls_returns_malformed_tool_call_error(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)

    latest = agent.execute_tool_calls([{"intention": "bad call", "args": []}])

    assert "ToolCallError: tool action missing required field: name" in latest
    assert '{"type":"tool","name":"Read","intention":"...","args":["path"]}' in latest
    assert "InvalidToolCall" in latest
    assert "bad call" not in latest
    assert session.state.conversation == []
    assert (tmp_path / ".nanocode" / "tool_results").exists()


def test_agent_execute_tool_calls_records_arg_errors_in_feedback(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)

    latest = agent.execute_tool_calls([{"name": "Read", "intention": "bad range", "args": ["sample.txt", "bad", "1"]}])

    assert "ToolCallError: invalid start: should be an integer" in latest
    assert agent.agent_feedback_errors == [
        'Error: tool call args invalid: Read("sample.txt", "bad", "1") -> ToolCallError: invalid start: should be an integer. Rule: use the tool signature exactly.'
    ]


def test_agent_execute_bash_does_not_require_verification(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)

    agent.execute_tool_calls([{"name": "Bash", "intention": "run command", "args": ["true"]}], confirm=lambda call, tool: True)

    assert agent.blackboard.verification_required is False


def test_agent_marks_nonzero_bash_exit_as_failed_tool_call(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)

    latest = agent.execute_tool_calls([{"name": "Bash", "intention": "run failing command", "args": ["exit 7"]}], confirm=lambda call, tool: True)

    assert agent.tool_runner.latest_executions[0].outcome == "failure"
    assert "fail | Bash exit 7" in latest
    assert "* exit_code: 7" in agent.tool_runner.latest_executions[0].output


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

    assert "tool not allowed for this agent: Edit" in latest
    assert path.read_text(encoding="utf-8") == "old\n"
    assert explorer.session is parent_session
    assert parent_session.state.tool_result_store == {}
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

    assert "tool not allowed for this agent: Edit" in latest
    assert path.read_text(encoding="utf-8") == "old\n"
    assert verifier.session is parent_session
    assert parent_session.state.tool_result_store == {}
    assert list(verifier.runtime.tool_result_store) == ["tr.1"]


def test_explore_agent_requires_known_after_tool_results(tmp_path):
    (tmp_path / "sample.txt").write_text("alpha\n", encoding="utf-8")
    parent_session = Session(cwd=str(tmp_path))
    parent_agent = MainAgent(parent_session)
    explorer = nanocode.ExploreAgent(parent_session=parent_session, parent_blackboard=parent_agent.blackboard, goal="find sample", scope=["sample.txt"])

    explorer.execute_tool_calls([{"name": "Read", "intention": "read sample", "args": ["sample.txt", "0,1"]}])

    assert "Use only Recent Tool Calls" in explorer.build_system_prompt()
    assert '"type": "tool"' not in explorer.build_system_prompt()

    missing_known = explorer.handle_response(
        {"actions": [{"type": "deliver", "targets": [{"path": "sample.txt", "area": "line 1", "reason": "found"}], "issues": []}]}
    )

    assert missing_known.done is False
    assert explorer.blackboard.known == []
    assert any("latest results were not recorded" in error for error in explorer.agent_feedback_errors)

    still_searching = explorer.handle_response(
        {"actions": [{"type": "tool", "name": "Search", "intention": "keep searching", "args": ["alpha"], "known": ["sample.txt contains alpha."]}]}
    )

    assert still_searching.done is False
    assert explorer.blackboard.known == []
    assert any("Invalid action(s): tool" in error for error in explorer.agent_feedback_errors)

    observed = explorer.handle_response({"actions": [{"type": "observe", "known": ["sample.txt contains alpha."], "next": "deliver sample target"}]})

    assert observed.done is False
    assert explorer.blackboard.known == ["sample.txt contains alpha."]
    assert explorer.mode == nanocode.AgentMode.ACT

    explorer.execute_tool_calls([{"name": "LineCount", "intention": "count sample lines", "args": ["sample.txt"]}])

    delivered = explorer.handle_response(
        {
            "actions": [
                {
                    "type": "deliver",
                    "targets": [{"path": "sample.txt", "area": "line 1", "reason": "found"}],
                    "known": ["sample.txt contains alpha."],
                    "issues": [],
                }
            ]
        }
    )

    assert delivered.done is True
    assert isinstance(delivered.value, nanocode.ExploreReport)
    assert delivered.value.known == ["sample.txt contains alpha."]


def test_explore_agent_rejects_observe_outside_observation_turn(tmp_path):
    parent_session = Session(cwd=str(tmp_path))
    parent_agent = MainAgent(parent_session)
    explorer = nanocode.ExploreAgent(parent_session=parent_session, parent_blackboard=parent_agent.blackboard, goal="find sample", scope=["sample.txt"])

    result = explorer.handle_response({"actions": [{"type": "observe", "known": ["sample fact"], "next": "read sample"}]})

    assert result.done is False
    assert any("Invalid action(s): observe" in error for error in explorer.agent_feedback_errors)


def test_explore_agent_rejects_deliver_outside_observation_turn(tmp_path):
    parent_session = Session(cwd=str(tmp_path))
    parent_agent = MainAgent(parent_session)
    explorer = nanocode.ExploreAgent(parent_session=parent_session, parent_blackboard=parent_agent.blackboard, goal="find sample", scope=["sample.txt"])

    result = explorer.handle_response(
        {"actions": [{"type": "deliver", "targets": [{"path": "sample.txt", "area": "line 1", "reason": "found"}], "known": ["sample fact"]}]}
    )

    assert result.done is False
    assert any("Invalid action(s): deliver" in error for error in explorer.agent_feedback_errors)


def test_verify_agent_requires_known_after_tool_results(tmp_path):
    (tmp_path / "sample.txt").write_text("alpha\n", encoding="utf-8")
    parent_session = Session(cwd=str(tmp_path))
    parent_agent = MainAgent(parent_session)
    verifier = nanocode.VerifyAgent(parent_session=parent_session, parent_blackboard=parent_agent.blackboard, goal="verify sample", scope=["sample.txt"])

    verifier.execute_tool_calls([{"name": "Read", "intention": "read sample", "args": ["sample.txt", "0,1"]}])

    assert "Use only Recent Tool Calls" in verifier.build_system_prompt()
    assert '"type": "tool"' not in verifier.build_system_prompt()

    missing_known = verifier.handle_response(
        {"actions": [{"type": "deliver", "status": "passed", "method": "read", "summary": "sample has alpha", "evidence": ["alpha"]}]}
    )

    assert missing_known.done is False
    assert verifier.blackboard.known == []
    assert any("latest results were not recorded" in error for error in verifier.agent_feedback_errors)

    observed = verifier.handle_response({"actions": [{"type": "observe", "known": ["sample.txt contains alpha."], "next": "deliver verdict"}]})

    assert observed.done is False
    assert verifier.blackboard.known == ["sample.txt contains alpha."]
    assert verifier.mode == nanocode.AgentMode.ACT

    delivered = verifier.handle_response(
        {
            "actions": [
                {
                    "type": "deliver",
                    "status": "passed",
                    "method": "read",
                    "summary": "sample has alpha",
                    "evidence": ["alpha"],
                    "known": ["sample.txt contains alpha."],
                }
            ]
        }
    )

    assert delivered.done is True
    assert isinstance(delivered.value, nanocode.VerifyReport)
    assert verifier.blackboard.known == ["sample.txt contains alpha."]


def test_verify_agent_rejects_repeating_failed_process_command(tmp_path):
    parent_session = Session(cwd=str(tmp_path))
    parent_agent = MainAgent(parent_session)
    verifier = nanocode.VerifyAgent(
        parent_session=parent_session,
        parent_blackboard=parent_agent.blackboard,
        goal="run tests",
        scope=["kind: test", "target: make test", "expect: exit code 0"],
    )
    messages = []

    verifier.execute_tool_calls([{"name": "Bash", "intention": "run tests", "args": ["exit 7"]}], confirm=lambda call, tool: True)
    result = verifier.handle_response(
        {"actions": [{"type": "tool", "name": "Bash", "intention": "run tests again", "args": ["exit 7"], "known": ["exit 7 command failed."]}]},
        confirm=lambda call, tool: True,
        on_message=messages.append,
    )

    assert result.done is False
    assert list(verifier.runtime.tool_result_store) == ["tr.1"]
    assert any("previous verification command already failed" in error for error in verifier.agent_feedback_errors)

    delivered = verifier.handle_response(
        {
            "actions": [
                {
                    "type": "deliver",
                    "status": "failed",
                    "method": "Bash exit 7",
                    "summary": "command failed",
                    "evidence": ["exit_code: 7"],
                    "issues": ["tests failed"],
                    "next_steps": [],
                }
            ]
        },
        on_message=messages.append,
    )

    assert delivered.done is True
    assert isinstance(delivered.value, nanocode.VerifyReport)
    assert delivered.value.status == "failed"


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
                            "known": ["sample.txt contains alpha.", "relevant target is sample.txt line 1."],
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
    assert explorer.session is parent_session
    assert parent_session.state.tool_result_store == {}
    assert list(explorer.runtime.tool_result_store) == ["tr.1"]
    assert len(explorer.model_client.user_prompts) == 2


def test_explore_agent_rejects_repeated_tool_call_and_delivers(tmp_path):
    (tmp_path / "sample.txt").write_text("alpha\n", encoding="utf-8")
    parent_session = Session(cwd=str(tmp_path))
    parent_agent = MainAgent(parent_session)

    class FakeModelClient:
        def __init__(self):
            self.responses = [
                {"actions": [{"type": "tool", "name": "Read", "intention": "read sample", "args": ["sample.txt", "0,1"]}]},
                {"actions": [{"type": "observe", "known": ["sample.txt contains alpha."], "next": "deliver or avoid repeat"}]},
                {"actions": [{"type": "tool", "name": "Read", "intention": "repeat read", "args": ["sample.txt", "0,1"], "known": ["sample.txt contains alpha."]}]},
                {
                    "actions": [
                        {
                            "type": "deliver",
                            "targets": [
                                {
                                    "path": "sample.txt",
                                    "area": "line 1",
                                    "line_range": "0,1",
                                    "context": "alpha",
                                    "reason": "already read",
                                }
                            ],
                            "known": ["sample.txt contains alpha."],
                        }
                    ]
                },
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            return self.responses.pop(0)

    explorer = nanocode.ExploreAgent(
        parent_session=parent_session,
        parent_blackboard=parent_agent.blackboard,
        goal="find sample",
        scope=["sample.txt"],
    )
    explorer.model_client = FakeModelClient()

    report = explorer.run()

    assert report.targets[0]["path"] == "sample.txt"
    assert list(explorer.runtime.tool_result_store) == ["tr.1"]
    assert any("repeated explore tool call" in error for error in explorer.agent_feedback_errors)


def test_explore_agent_uses_observe_turn_after_tool_results(tmp_path):
    (tmp_path / "sample.txt").write_text("alpha\n", encoding="utf-8")
    parent_session = _session(tmp_path, explore_max_turns=2)
    parent_agent = MainAgent(parent_session)

    class FakeModelClient:
        def __init__(self):
            self.system_prompts = []
            self.user_prompts = []
            self.responses = [
                {"actions": [{"type": "tool", "name": "Read", "intention": "read sample", "args": ["sample.txt", "0,1"]}]},
                {
                    "actions": [
                        {
                            "type": "deliver",
                            "targets": [
                                {
                                    "path": "sample.txt",
                                    "area": "line 1",
                                    "line_range": "0,1",
                                    "context": "alpha",
                                    "reason": "found before step limit",
                                }
                            ],
                            "known": ["sample.txt contains alpha."],
                        }
                    ]
                },
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            self.system_prompts.append(system_prompt)
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    explorer = nanocode.ExploreAgent(
        parent_session=parent_session,
        parent_blackboard=parent_agent.blackboard,
        goal="find sample",
        scope=["sample.txt"],
    )
    explorer.model_client = FakeModelClient()

    report = explorer.run()

    assert report.targets[0]["path"] == "sample.txt"
    assert report.known == ["sample.txt contains alpha."]
    assert len(explorer.model_client.user_prompts) == 2


def test_explore_agent_goal_changes_do_not_clear_parent_range_fingerprints(tmp_path):
    parent_session = Session(cwd=str(tmp_path))
    parent_session.state.range_fingerprints.remember(filepath=str(tmp_path / "sample.txt"), start=0, end=1, content="alpha\n")
    parent_agent = MainAgent(parent_session)
    explorer = nanocode.ExploreAgent(
        parent_session=parent_session,
        parent_blackboard=parent_agent.blackboard,
        goal="find relevant target",
        scope=["sample.txt"],
    )

    explorer.apply_response({"actions": [{"type": "goal", "text": "refined target", "complete": False}]})

    assert len(parent_session.state.range_fingerprints) == 1


def test_agent_execute_tool_calls_shows_auto_approval_in_yolo_mode(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("old\n", encoding="utf-8")
    session = _session(tmp_path, yolo=True)
    agent = MainAgent(session)
    confirmations = []
    auto_approvals = []

    latest = agent.execute_tool_calls(
        [{"name": "Edit", "intention": "edit sample", "args": ["sample.txt", "old", "new"]}],
        confirm=lambda call, tool: confirmations.append(call.executed) or False,
        on_auto_approve=lambda call, tool: auto_approvals.append((call.executed, tool.preview())),
    )

    assert confirmations == []
    assert auto_approvals
    assert auto_approvals[0][0] == 'Edit("sample.txt", "old", "new")'
    assert "-old" in auto_approvals[0][1]
    assert "+new" in auto_approvals[0][1]
    assert latest.startswith("- ok")
    assert path.read_text(encoding="utf-8") == "new\n"
    assert agent.blackboard.verification_required is True


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
                            "type": "verify",
                            "method": "unit",
                            "status": "passed",
                            "context": "checked",
                            "known": ["Read sample.txt and found alpha."],
                        },
                        {"type": "goal", "text": "read sample", "complete": True, "message_for_complete": "done"},
                    ],
                },
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)
    _seed_plan(agent, "read sample")
    fake_client = FakeModelClient()
    agent.model_client = fake_client

    messages = []
    response = agent.run("read sample", on_message=messages.append)

    assert response["actions"][-1]["message_for_complete"] == "done"
    assert messages[0].startswith("[success] Read sample.txt 0:1")
    assert "tr.1" not in messages[0]
    assert "why:" not in messages[0]
    assert "log: .nanocode/tool_results/" not in messages[0]
    assert messages[-1] == "done"
    assert len(fake_client.user_prompts) == 2
    assert "Read sample.txt 0:1" in _blocks_text(agent.latest_tool_call_blocks)
    assert agent.recent_tool_call_blocks == []
    assert agent.blackboard.known == ["Read sample.txt and found alpha."]
    assert agent.blackboard.user_input == "read sample"
    assert agent.blackboard.goal == "read sample"
    assert agent.blackboard.plan == [nanocode.PlanItem(text="test plan")]
    assert agent.blackboard.verification.status == VerificationStatus.DONE
    assert agent.blackboard.goal_reached is False
    assert agent.blackboard.verification_required is False


def test_agent_run_allows_readonly_answer_without_verification(tmp_path):
    (tmp_path / "sample.txt").write_text("alpha\n", encoding="utf-8")

    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {"actions": [{"type": "tool", "name": "Read", "intention": "read sample", "args": ["sample.txt", "0", "1"]}]},
                {
                    "actions": [
                        {"type": "goal", "text": "answer sample", "complete": True, "message_for_complete": "sample contains alpha"},
                    ],
                },
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    agent = MainAgent(Session(cwd=str(tmp_path)))
    agent.model_client = FakeModelClient()
    messages = []

    response = agent.run("answer sample", on_message=messages.append)

    assert response["actions"][-1]["message_for_complete"] == "sample contains alpha"
    assert "Retrying: verification must pass before completion." not in messages
    assert messages[-1] == "sample contains alpha"


def test_agent_run_executes_explore_and_completes(tmp_path):
    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {
                    "actions": [
                        {"type": "goal", "text": "relevant target", "complete": False},
                        {
                            "type": "explore",
                            "kind": "file",
                            "goal": "find target",
                            "scope": ["sample.txt"],
                            "constraints": ["return exact path and line range"],
                            "reason": "target uncertain",
                            "context": "Main saw sample mentioned",
                        },
                    ]
                },
                {"actions": _final_actions("relevant target")},
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    class FakeExploreAgent:
        def __init__(self, *, goal, scope):
            self.goal = goal
            self.scope = scope

        def run(self, *, confirm=None, on_auto_approve=None, on_message=None):
            assert self.scope == ["kind: file", "sample.txt", "constraint: return exact path and line range", "main_context: Main saw sample mentioned"]
            if on_message is not None:
                on_message("[success] Read(\"sample.txt\", \"0\", \"1\")")
            return nanocode.ExploreReport(
                targets=[{"path": "sample.txt", "area": "line 1", "reason": "target found"}],
                known=["sample.txt line 1 is the relevant target."],
            )

    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)
    _seed_plan(agent, "relevant target")
    agent.model_client = FakeModelClient()
    agent._make_explore_agent = lambda *, goal, scope: FakeExploreAgent(goal=goal, scope=scope)
    messages = []

    response = agent.run("relevant target", on_message=messages.append)

    assert response["actions"][-1]["message_for_complete"] == "done"
    assert len(agent.model_client.user_prompts) == 2
    assert session.state.tool_result_store == {}
    assert agent.recent_tool_call_blocks == []
    assert '[explore] [success] Read("sample.txt", "0", "1")' in messages
    assert messages[-1] == "done"


def test_agent_run_retries_explore_without_kind_or_constraints(tmp_path):
    class FakeExploreAgent:
        def run(self, *, confirm=None, on_auto_approve=None, on_message=None):
            raise AssertionError("invalid explore handoff should not start ExploreAgent")

    class FakeModelClient:
        def __init__(self):
            self.responses = [
                {
                    "actions": [
                        {"type": "goal", "text": "relevant target", "complete": False},
                        {"type": "explore", "goal": "find target", "scope": ["sample.txt"], "reason": "target uncertain"},
                    ]
                },
                {
                    "actions": [
                        {"type": "explore", "goal": "find target", "scope": ["sample.txt"], "reason": "target uncertain"},
                    ]
                },
                {"actions": _final_actions("relevant target")},
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)
    _seed_plan(agent, "relevant target")
    agent.model_client = FakeModelClient()
    agent._make_explore_agent = lambda *, goal, scope: FakeExploreAgent()
    messages = []

    response = agent.run("relevant target", on_message=messages.append)

    assert response["actions"][-1]["message_for_complete"] == "done"
    assert "Retrying: explore handoff needs kind and constraints." in messages


def test_agent_run_retries_explore_with_generic_goal(tmp_path):
    class FakeExploreAgent:
        def run(self, *, confirm=None, on_auto_approve=None, on_message=None):
            raise AssertionError("generic explore handoff should not start ExploreAgent")

    class FakeModelClient:
        def __init__(self):
            self.responses = [
                {
                    "actions": [
                        {"type": "goal", "text": "support lowercase tool names", "complete": False},
                        {
                            "type": "explore",
                            "kind": "symbol",
                            "goal": "locate concrete code targets only",
                            "scope": ["tool name"],
                            "constraints": ["find parser and dispatcher"],
                            "reason": "target uncertain",
                        },
                    ]
                },
                {
                    "actions": [
                        {
                            "type": "explore",
                            "kind": "symbol",
                            "goal": "locate concrete code targets only",
                            "scope": ["tool name"],
                            "constraints": ["find parser and dispatcher"],
                            "reason": "target uncertain",
                        },
                    ]
                },
                {"actions": _final_actions("support lowercase tool names")},
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)
    _seed_plan(agent, "support lowercase tool names")
    agent.model_client = FakeModelClient()
    agent._make_explore_agent = lambda *, goal, scope: FakeExploreAgent()
    messages = []

    assert "too generic" in agent._explore_actions_error(
        [
            {
                "type": "explore",
                "kind": "symbol",
                "goal": "locate concrete code targets only",
                "scope": ["tool name"],
                "constraints": ["find parser and dispatcher"],
            }
        ]
    )

    response = agent.run("support lowercase tool names", on_message=messages.append)

    assert response["actions"][-1]["message_for_complete"] == "done"
    assert "Retrying: explore handoff needs kind and constraints." in messages


def test_agent_run_executes_edit_tool_and_requires_verification(tmp_path):
    (tmp_path / "sample.txt").write_text("old\n", encoding="utf-8")
    verify_calls = []

    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {
                    "actions": [
                        {"type": "goal", "text": "change sample", "complete": False},
                        {
                            "type": "tool",
                            "name": "Edit",
                            "intention": "change sample text",
                            "args": ["sample.txt", "old", "new"],
                        },
                    ]
                },
                {"actions": [{"type": "goal", "text": "change sample", "complete": True, "message_for_complete": "done"}]},
                {"actions": [{"type": "goal", "text": "change sample", "complete": True, "message_for_complete": "done"}]},
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    class FakeVerifyAgent:
        def run(self, *, confirm=None, on_auto_approve=None, on_message=None):
            verify_calls.append(True)
            return nanocode.VerifyReport(status="passed", method="review", summary="edit verified")

    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)
    _seed_plan(agent, "change sample")
    agent.model_client = FakeModelClient()
    agent._make_verify_agent = lambda *, goal, scope: FakeVerifyAgent()
    messages = []

    response = agent.run("change sample", confirm=lambda call, tool: True, on_message=messages.append)

    assert response["actions"][-1]["message_for_complete"] == "done"
    assert verify_calls == [True]
    assert any(message.startswith("[success] Edit sample.txt") for message in messages)
    assert "Verify done: passed | review\n  edit verified" in messages
    assert (tmp_path / "sample.txt").read_text(encoding="utf-8") == "new\n"
    assert messages[-1] == "done"


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
    _seed_plan(agent, "read sample")
    agent.model_client = FakeModelClient()

    response = agent.run("read sample")

    assert response["actions"][-1]["message_for_complete"] == "done"
    assert len(agent.model_client.user_prompts) == 3
    assert "Read sample.txt 0:1" in _blocks_text(agent.latest_tool_call_blocks)
    assert agent.recent_tool_call_blocks == []


def test_agent_run_prunes_tool_result_store_when_next_run_starts(tmp_path):
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
    agent.latest_tool_call_blocks = ["old tool call"]
    agent.model_client = FakeModelClient()

    agent.run("read samples")

    assert len(session.state.tool_result_store) == 51
    assert list(session.state.tool_result_store)[0] == "tr.1"

    agent.model_client.responses = [{"actions": [{"type": "chat", "text": "ok"}]}]
    agent.run("next task")

    assert len(session.state.tool_result_store) == 50
    assert list(session.state.tool_result_store)[:2] == ["tr.2", "tr.3"]
    assert list(session.state.tool_result_store)[-1] == "tr.51"
    assert session.state.tool_result_counter == 51
    assert agent.blackboard.goal == "read samples"
    assert agent.blackboard.plan == [nanocode.PlanItem(text="try answer")]
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
    _seed_plan(agent, "read sample")
    agent.model_client = FakeModelClient()
    messages = []

    response = agent.run("read sample", on_message=messages.append)

    assert response["actions"][-1]["message_for_complete"] == "done too early"
    assert "Retrying: Known was not reviewed after tool results." not in messages
    assert "done too early" in messages
    assert len(agent.model_client.user_prompts) == 2


def test_agent_run_requires_plan_before_first_tool(tmp_path):
    (tmp_path / "sample.txt").write_text("alpha\n", encoding="utf-8")

    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {
                    "actions": [
                        {"type": "goal", "text": "read sample", "complete": False},
                        {"type": "tool", "name": "Read", "intention": "read sample", "args": ["sample.txt", "0", "1"]},
                    ]
                },
                {
                    "actions": [
                        {"type": "tool", "name": "Read", "intention": "read sample", "args": ["sample.txt", "0", "1"]},
                    ]
                },
                {
                    "actions": [
                        {"type": "plan", "mode": "replace", "items": [{"id": "p1", "text": "Read sample", "status": "doing"}]},
                        {"type": "tool", "name": "Read", "intention": "read sample", "args": ["sample.txt", "0", "1"]},
                    ]
                },
                {"actions": _final_actions("read sample")},
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)
    agent.model_client = FakeModelClient()
    messages = []

    response = agent.run("read sample", on_message=messages.append)

    assert response["actions"][-1]["message_for_complete"] == "done"
    assert "Retrying: create a short plan before tools/workers." in messages
    assert len(session.state.tool_result_store) == 1
    assert [item.text for item in agent.blackboard.plan] == ["Read sample"]


def test_agent_run_requires_fresh_plan_when_goal_changes(tmp_path):
    (tmp_path / "sample.txt").write_text("alpha\n", encoding="utf-8")

    class FakeModelClient:
        def __init__(self):
            self.responses = [
                {
                    "actions": [
                        {"type": "goal", "text": "new goal", "complete": False},
                        {"type": "tool", "name": "Read", "intention": "read sample", "args": ["sample.txt", "0", "1"]},
                    ]
                },
                {
                    "actions": [
                        {"type": "goal", "text": "new goal", "complete": False},
                        {"type": "tool", "name": "Read", "intention": "read sample", "args": ["sample.txt", "0", "1"]},
                    ]
                },
                {
                    "actions": [
                        {
                            "type": "start",
                            "goal": "new goal",
                            "plan": [{"id": "p1", "text": "Read sample", "status": "doing"}],
                        },
                        {"type": "tool", "name": "Read", "intention": "read sample", "args": ["sample.txt", "0", "1"]},
                    ]
                },
                {"actions": _final_actions("new goal")},
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)
    agent.blackboard.goal = "old goal"
    agent.blackboard.plan = [nanocode.PlanItem(id="old", text="Old plan")]
    agent.model_client = FakeModelClient()
    messages = []

    response = agent.run("new goal", on_message=messages.append)

    assert response["actions"][-1]["message_for_complete"] == "done"
    assert "Retrying: new goal requires a fresh plan." in messages
    assert agent.blackboard.goal == "new goal"
    assert [item.text for item in agent.blackboard.plan] == ["Read sample"]
    assert len(session.state.tool_result_store) == 1


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
    _seed_plan(agent, "answer")
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
    _seed_plan(agent, "你好")
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
                {"actions": [{"type": "plan", "mode": "patch", "items": []}]},
                {"actions": _final_actions()},
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)
    _seed_plan(agent, "answer")
    agent.model_client = FakeModelClient()
    messages = []

    response = agent.run("answer", on_message=messages.append)

    assert response["actions"][-1]["message_for_complete"] == "done"
    assert "Continuing: goal is not complete yet." not in messages


def test_main_agent_rejects_standalone_sidecar_actions(tmp_path):
    class FakeModelClient:
        def __init__(self):
            self.responses = [
                {"actions": [{"type": "known", "items": ["fact"]}]},
                {"actions": _final_actions()},
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)
    _seed_plan(agent, "answer")
    agent.model_client = FakeModelClient()

    response = agent.run("answer")

    assert response["actions"][-1]["message_for_complete"] == "done"
    assert agent.blackboard.known == []
    assert any("standalone sidecar action is invalid" in error for error in agent.agent_feedback_errors)


def test_agent_run_reports_continuation_only_when_no_actions(tmp_path):
    class FakeModelClient:
        def __init__(self):
            self.responses = [
                {"actions": []},
                {"actions": []},
                {"actions": _final_actions()},
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)
    _seed_plan(agent, "answer")
    agent.model_client = FakeModelClient()
    messages = []

    response = agent.run("answer", on_message=messages.append)

    assert response["actions"][-1]["message_for_complete"] == "done"
    assert "Continuing: assistant must set current task's goal." in messages


def test_agent_run_enforces_verification_gate_before_completion(tmp_path):
    (tmp_path / "sample.txt").write_text("old\n", encoding="utf-8")
    verify_confirm_callbacks = []

    class FakeVerifyAgent:
        def run(self, *, confirm=None, on_auto_approve=None, on_message=None):
            verify_confirm_callbacks.append(confirm)
            if on_message is not None:
                on_message('[success] Git("diff", "--", "sample.txt")')
            return nanocode.VerifyReport(status="passed", method="git diff", summary="diff matches goal", evidence=["sample.txt changed"])

    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {
                    "actions": [
                        {"type": "goal", "text": "change file", "complete": False},
                        {
                            "type": "tool",
                            "name": "Edit",
                            "intention": "edit sample",
                            "args": ["sample.txt", "old", "new"],
                        },
                    ],
                },
                {
                    "actions": [
                        {"type": "goal", "text": "change file done", "complete": True, "message_for_complete": "done"},
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
    _seed_plan(agent, "change file")
    agent.model_client = FakeModelClient()
    agent._make_verify_agent = lambda *, goal, scope: FakeVerifyAgent()
    messages = []

    response = agent.run("change file", confirm=lambda call, tool: True, on_message=messages.append)

    assert response["actions"][-1]["message_for_complete"] == "done"
    assert len(agent.model_client.user_prompts) == 3
    assert len(verify_confirm_callbacks) == 1
    assert verify_confirm_callbacks[0] is not None
    assert agent.blackboard.verification.status == VerificationStatus.DONE
    assert agent.blackboard.verification.context == "diff matches goal"
    assert "Verifying: change_syntax_check change file done" in messages
    assert '[verify] [success] Git("diff", "--", "sample.txt")' in messages
    assert "Verify done: passed | git diff\n  diff matches goal" in messages
    assert (tmp_path / "sample.txt").read_text(encoding="utf-8") == "new\n"


def test_agent_run_feeds_failed_verify_report_into_next_prompt(tmp_path):
    (tmp_path / "sample.txt").write_text("old\n", encoding="utf-8")

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
                            "type": "tool",
                            "name": "Edit",
                            "intention": "edit sample badly",
                            "args": ["sample.txt", "old", "bad"],
                        },
                    ],
                },
                {"actions": [{"type": "goal", "text": "change file", "complete": True, "message_for_complete": "done"}]},
                {"actions": _observe_actions("unit verification failed: assertion failed.")},
                {
                    "actions": [
                        {
                            "type": "tool",
                            "name": "Edit",
                            "intention": "fix sample",
                            "args": ["sample.txt", "bad", "new"],
                        },
                    ],
                },
                {"actions": [{"type": "goal", "text": "change file", "complete": True, "message_for_complete": "done"}]},
                {"actions": [{"type": "goal", "text": "change file", "complete": True, "message_for_complete": "done"}]},
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)
    _seed_plan(agent, "change file")
    agent.model_client = FakeModelClient()
    verifier = FakeVerifyAgent()
    agent._make_verify_agent = lambda *, goal, scope: verifier

    response = agent.run("change file", confirm=lambda call, tool: True)

    assert response["actions"][-1]["message_for_complete"] == "done"
    assert len(agent.model_client.user_prompts) == 6
    assert "Worker Reports:" in agent.model_client.user_prompts[2]
    assert "<Agent_Reports>" not in agent.model_client.user_prompts[2]
    assert "assertion failed" in agent.model_client.user_prompts[2]
    assert (tmp_path / "sample.txt").read_text(encoding="utf-8") == "new\n"


def test_agent_run_does_not_repeat_failed_verification_before_fix(tmp_path):
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
                            "type": "tool",
                            "name": "Edit",
                            "intention": "edit sample badly",
                            "args": ["sample.txt", "old", "bad"],
                        },
                    ],
                },
                {"actions": [{"type": "goal", "text": "change file", "complete": True, "message_for_complete": "done"}]},
                {"actions": [{"type": "goal", "text": "change file", "complete": True, "message_for_complete": "done"}]},
                {"actions": [{"type": "goal", "text": "change file", "complete": True, "message_for_complete": "done"}]},
                {
                    "actions": [
                        {
                            "type": "tool",
                            "name": "Edit",
                            "intention": "fix sample",
                            "args": ["sample.txt", "bad", "new"],
                        },
                    ],
                },
                {"actions": [{"type": "goal", "text": "change file", "complete": True, "message_for_complete": "done"}]},
                {"actions": [{"type": "goal", "text": "change file", "complete": True, "message_for_complete": "done"}]},
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    (tmp_path / "sample.txt").write_text("old\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)
    _seed_plan(agent, "change file")
    agent.model_client = FakeModelClient()
    verifier = FakeVerifyAgent()
    agent._make_verify_agent = lambda *, goal, scope: verifier
    messages = []

    response = agent.run("change file", confirm=lambda call, tool: True, on_message=messages.append)

    assert response["actions"][-1]["message_for_complete"] == "done"
    assert verifier.reports == []
    assert "Retrying: verification failed; fix the reported issue first." in messages
    assert "assertion failed" in agent.model_client.user_prompts[2]
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
                        {
                            "type": "verify",
                            "kind": "change_check",
                            "method": "manual check",
                            "criteria": ["answer is correct"],
                            "status": "pending",
                            "context": "check answer",
                        },
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
    _seed_plan(agent, "answer")
    agent.blackboard.goal = "answer"
    agent.model_client = FakeModelClient()
    agent._make_verify_agent = lambda *, goal, scope: FakeVerifyAgent(goal=goal, scope=scope)
    messages = []

    response = agent.run("answer", on_message=messages.append)

    assert response["actions"][-1]["message_for_complete"] == "done"
    assert verifier_calls
    assert verifier_calls[0][0] == "manual check"
    assert "kind: change_check" in verifier_calls[0][1]
    assert "target: manual check" in verifier_calls[0][1]
    assert "expect: answer is correct" in verifier_calls[0][1]
    assert "context: check answer" in verifier_calls[0][1]
    assert "Verifying: change_check manual check" in messages
    assert len(agent.model_client.user_prompts) == 2


def test_agent_run_retries_repeated_pending_verify_after_passed(tmp_path):
    verifier_calls = []

    class FakeVerifyAgent:
        def __init__(self, *, goal, scope):
            self.goal = goal
            self.scope = scope

        def run(self, *, confirm=None, on_auto_approve=None, on_message=None):
            verifier_calls.append((self.goal, self.scope))
            return nanocode.VerifyReport(status="passed", method="cmake_build", summary="build passed")

    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {
                    "actions": [
                        {
                            "type": "verify",
                            "kind": "build",
                            "method": "cmake_build",
                            "criteria": ["build exits 0"],
                            "status": "pending",
                            "context": "verify build",
                        }
                    ],
                },
                {
                    "actions": [
                        {
                            "type": "verify",
                            "kind": "build",
                            "method": "cmake_build",
                            "criteria": ["build exits 0"],
                            "status": "pending",
                            "context": "verify build again",
                        }
                    ],
                },
                {
                    "actions": [
                        {
                            "type": "verify",
                            "kind": "build",
                            "method": "cmake_build",
                            "criteria": ["build exits 0"],
                            "status": "pending",
                            "context": "verify build once more",
                        }
                    ],
                },
                {"actions": [{"type": "goal", "text": "answer", "complete": True, "message_for_complete": "done"}]},
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    agent = MainAgent(Session(cwd=str(tmp_path)))
    _seed_plan(agent, "answer")
    agent.blackboard.goal = "answer"
    agent.model_client = FakeModelClient()
    agent._make_verify_agent = lambda *, goal, scope: FakeVerifyAgent(goal=goal, scope=scope)
    messages = []

    response = agent.run("answer", on_message=messages.append)

    assert response["actions"][-1]["message_for_complete"] == "done"
    assert len(verifier_calls) == 1
    assert "Retrying: observe latest results before new verification." in messages
    assert agent.blackboard.verification.status == VerificationStatus.DONE


def test_agent_run_treats_verify_scope_check_blocked_as_failed(tmp_path):
    class FakeVerifyAgent:
        def run(self, *, confirm=None, on_auto_approve=None, on_message=None):
            return nanocode.VerifyReport(status="blocked", method="scope_check", summary="missing target")

    agent = MainAgent(Session(cwd=str(tmp_path)))
    _seed_plan(agent, "answer")
    agent.blackboard.goal = "answer"
    agent._make_verify_agent = lambda *, goal, scope: FakeVerifyAgent()
    messages = []

    agent.handle_response(
        {
            "actions": [
                {
                    "type": "verify",
                    "kind": "change_check",
                    "method": "manual check",
                    "criteria": ["answer is correct"],
                    "status": "pending",
                }
            ]
        },
        on_message=messages.append,
    )
    agent.handle_response({"actions": [{"type": "goal", "text": "answer", "complete": True, "message_for_complete": "done"}]}, on_message=messages.append)
    agent.handle_response({"actions": [{"type": "goal", "text": "answer", "complete": True, "message_for_complete": "done"}]}, on_message=messages.append)

    assert "Verify blocked | scope_check\n  missing target" in messages
    assert "Retrying: verification failed; fix the reported issue first." in messages
    assert "done" not in messages
    assert agent.blackboard.verification.status == VerificationStatus.FAILED


def test_agent_run_prioritizes_pending_verify_over_same_response_tools(tmp_path):
    verifier_calls = []
    bash_confirmed = []

    class FakeVerifyAgent:
        def __init__(self, *, goal, scope):
            self.goal = goal
            self.scope = scope

        def run(self, *, confirm=None, on_auto_approve=None, on_message=None):
            verifier_calls.append((self.goal, self.scope))
            return nanocode.VerifyReport(status="passed", method="unit", summary="tests passed")

    class FakeModelClient:
        def __init__(self):
            self.responses = [
                {
                    "actions": [
                        {"type": "goal", "text": "answer", "complete": False},
                        {
                            "type": "verify",
                            "kind": "test",
                            "method": "run unit tests",
                            "criteria": ["tests pass"],
                            "status": "pending",
                            "context": "verify answer",
                        },
                        {"type": "tool", "name": "Bash", "intention": "run tests", "args": ["python -m pytest -q"]},
                    ],
                },
                {"actions": [{"type": "goal", "text": "answer", "complete": True, "message_for_complete": "done"}]},
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            return self.responses.pop(0)

    agent = MainAgent(Session(cwd=str(tmp_path)))
    _seed_plan(agent, "answer")
    agent.model_client = FakeModelClient()
    agent._make_verify_agent = lambda *, goal, scope: FakeVerifyAgent(goal=goal, scope=scope)

    response = agent.run("answer", confirm=lambda call, tool: bash_confirmed.append(call.executed) or True)

    assert response["actions"][-1]["message_for_complete"] == "done"
    assert len(verifier_calls) == 1
    assert verifier_calls[0][0] == "run unit tests"
    assert "kind: test" in verifier_calls[0][1]
    assert bash_confirmed == []
    assert agent.latest_tool_call_blocks == []


def test_agent_run_retries_pending_verify_without_kind_or_criteria(tmp_path):
    class FakeVerifyAgent:
        def run(self, *, confirm=None, on_auto_approve=None, on_message=None):
            raise AssertionError("invalid pending verify should not start VerifyAgent")

    class FakeModelClient:
        def __init__(self):
            self.responses = [
                {
                    "actions": [
                        {"type": "goal", "text": "answer", "complete": False},
                        {"type": "verify", "method": "manual check", "status": "pending", "context": "check answer"},
                    ],
                },
                {
                    "actions": [
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
            return self.responses.pop(0)

    agent = MainAgent(Session(cwd=str(tmp_path)))
    agent.model_client = FakeModelClient()
    agent._make_verify_agent = lambda *, goal, scope: FakeVerifyAgent()
    messages = []

    response = agent.run("answer", on_message=messages.append)

    assert response["actions"][-1]["message_for_complete"] == "done"
    assert "Retrying: pending verification needs kind and criteria." in messages
    assert agent.blackboard.verification.status == VerificationStatus.IDLE


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
                {"actions": [{"type": "goal", "text": "change file", "complete": False}]},
                {"actions": _final_actions("change file")},
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)
    _seed_plan(agent)
    agent.model_client = FakeModelClient()
    messages = []

    response = agent.run("change file", on_message=messages.append)

    assert response["actions"][-1]["message_for_complete"] == "done"
    assert len(agent.model_client.user_prompts) == 3
    assert "Retrying: verification is done but goal is not complete." in messages
    assert agent.blackboard.verification.status == VerificationStatus.DONE


def test_agent_run_retries_when_goal_complete_has_no_message(tmp_path):
    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {"actions": [{"type": "goal", "text": "answer", "complete": True}]},
                {"actions": [{"type": "goal", "text": "answer", "complete": True}]},
                {"actions": _final_actions()},
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)
    _seed_plan(agent)
    agent.model_client = FakeModelClient()
    messages = []

    response = agent.run("answer", on_message=messages.append)

    assert response["actions"][-1]["message_for_complete"] == "done"
    assert len(agent.model_client.user_prompts) == 3
    assert "Retrying: goal is complete but message_for_complete is missing." in messages
    assert agent.agent_feedback_errors
    assert agent.blackboard.goal_reached is False


def test_agent_run_retries_format_error_with_recent_tool_calls(tmp_path):
    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {"_format_error": "Invalid model output: plain answer", "actions": []},
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
    assert len(agent.model_client.user_prompts) == 3
    assert "Retrying: model returned invalid output: plain answer" in messages
    assert messages[-1] == "done"


def test_agent_feedback_survives_goal_complete_until_next_run(tmp_path):
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
    assert agent.agent_feedback_errors

    class ChatModelClient:
        def request(self, system_prompt, user_prompt, *, activity="main"):
            return {"actions": [{"type": "chat", "text": "ok"}]}

    agent.model_client = ChatModelClient()
    agent.run("next task")

    assert agent.agent_feedback_errors == []
    assert agent.blackboard.verification.status == VerificationStatus.IDLE


def test_agent_allows_progress_message_before_goal_complete(tmp_path):
    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {"actions": [{"type": "plan", "mode": "patch", "items": [], "progress": "progress"}]},
                {"actions": _final_actions()},
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)
    _seed_plan(agent)
    agent.model_client = FakeModelClient()

    messages = []
    response = agent.run("answer", on_message=messages.append)

    assert response["actions"][-1]["message_for_complete"] == "done"
    assert messages[0] == "progress"
    assert messages[-1] == "done"
    assert "progress" not in [item.content for item in session.state.conversation]
    assert agent.agent_feedback_errors == []


def test_agent_shows_progress_with_tool_action_without_storing_it(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\n", encoding="utf-8")

    class FakeModelClient:
        def __init__(self):
            self.responses = [
                {
                    "actions": [
                        {"type": "tool", "name": "Read", "intention": "read sample", "args": ["sample.txt"], "progress": "reading sample"},
                    ]
                },
                {"actions": _final_actions()},
            ]

        def request(self, system_prompt, user_prompt, *, activity="main"):
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = MainAgent(session)
    _seed_plan(agent)
    agent.model_client = FakeModelClient()

    messages = []
    response = agent.run("answer", on_message=messages.append)

    assert response["actions"][-1]["message_for_complete"] == "done"
    assert messages[0] == "reading sample"
    assert "reading sample" not in [item.content for item in session.state.conversation]


def test_agent_feedback_survives_keyboard_interrupt_until_next_run(tmp_path):
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
    agent.latest_tool_call_blocks = ["old tool call"]
    agent.model_client = FakeModelClient()

    try:
        agent.run("answer")
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("expected KeyboardInterrupt")

    assert agent.agent_feedback_errors
    assert agent.latest_tool_call_blocks == ["old tool call"]
    assert agent.recent_tool_call_blocks == []
    assert agent.blackboard.goal == "answer"
    assert agent.blackboard.plan == [nanocode.PlanItem(text="try answer")]
    assert agent.blackboard.known == ["keep this fact"]
    assert agent.blackboard.verification.status == VerificationStatus.IDLE
    assert agent.blackboard.goal_reached is False

    class ChatModelClient:
        def request(self, system_prompt, user_prompt, *, activity="main"):
            return {"actions": [{"type": "chat", "text": "ok"}]}

    agent.model_client = ChatModelClient()
    agent.run("next task")

    assert agent.agent_feedback_errors == []


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

    debug_session = _session(tmp_path, debug=True)
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

    session = _session(tmp_path, debug=True)
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
    assert len(agent.model_client.user_prompts) == 3
    assert "Retrying: goal is complete but message_for_complete is missing." in messages
    assert agent.agent_feedback_errors


def test_agent_run_uses_message_for_complete_even_when_progress_actions_exist(tmp_path):
    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {
                    "actions": [
                        {
                            "type": "goal",
                            "text": "answer",
                            "complete": True,
                            "message_for_complete": "fallback message",
                            "progress": "explicit progress",
                        },
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
    assert "explicit progress" not in [item.content for item in session.state.conversation]


def test_agent_run_ignores_message_for_complete_when_goal_not_complete(tmp_path):
    """message_for_complete should be ignored when complete=false."""
    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {"actions": [{"type": "goal", "text": "answer", "complete": False, "message_for_complete": "should be ignored"}]},
                {"actions": [{"type": "plan", "mode": "patch", "items": [], "progress": "done without goal"}]},
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
