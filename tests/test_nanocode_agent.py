import os
from dataclasses import replace

import nanocode
from nanocode import Agent, LLMError, ParsedToolCall, Session, VerificationStatus


def _verify_passed_action():
    return {"type": "verify", "method": "unit", "status": "passed", "context": "checked"}


def _final_actions(goal="answer", message="done"):
    return [
        _verify_passed_action(),
        {"type": "goal", "text": goal, "complete": True, "message_for_complete": message},
    ]


def _seed_plan(agent, goal="test goal"):
    agent.blackboard.goal = goal
    agent.blackboard.plan = [nanocode.PlanItem(text="test plan", status=nanocode.PlanStatus.DONE, context="seeded")]


def _blocks_text(blocks):
    return "\n".join(blocks)


def _observe_tool_result_context(agent):
    return "\n\n".join(agent.tool_context.unreduced_blocks(agent.blackboard.memory_checkpoint_tool_result_counter))


def _set_context_budget(monkeypatch, agent, **overrides):
    agent.session.settings.context_budget = "medium"
    monkeypatch.setitem(nanocode.CONTEXT_BUDGETS, "medium", replace(nanocode.CONTEXT_BUDGETS["medium"], **overrides))


def _session(
    tmp_path,
    *,
    api_url: str = "",
    api_key: str = "",
    model: str = "",
    stream: bool | None = None,
    timeout: int | None = None,
    first_token_timeout: int | None = None,
    temperature: float | None = None,
    reasoning: str = "",
    chat_reasoning: str = "",
    yolo: bool = False,
    plan_mode: bool = False,
    debug: bool = False,
    api: str = "",
) -> Session:
    provider: dict[str, object] = {"url": api_url, "key": api_key, "model": model}
    if api:
        provider["api"] = api
    if stream is not None:
        provider["stream"] = stream
    if timeout is not None:
        provider["timeout"] = timeout
    if first_token_timeout is not None:
        provider["first_token_timeout"] = first_token_timeout
    if temperature is not None:
        provider["temperature"] = temperature
    if reasoning:
        provider["reasoning"] = reasoning
    if chat_reasoning:
        provider["chat_reasoning"] = chat_reasoning
    data = {"provider": {"active": "default", "default": provider}, "paths": {"data_dir": str(tmp_path / ".nanocode")}}
    return Session(
        cwd=str(tmp_path),
        config=nanocode.Config.from_dict(data),
        settings=nanocode.RuntimeSettings.from_dict(data, yolo=yolo, plan_mode=plan_mode, debug=debug),
    )


def _chat_response(content: str = "ok", usage: dict | None = None) -> dict:
    return {"choices": [{"message": {"content": content}}], "usage": usage or {}}


def _stream_chunk(delta: dict | None = None, usage: dict | None = None, choices: bool = True) -> dict:
    return {"choices": [{"delta": delta or {}}] if choices else [], "usage": usage}


def _responses_response(content: str = "ok", usage: dict | None = None) -> dict:
    return {"output": [{"type": "message", "content": [{"type": "output_text", "text": content}]}], "usage": usage or {}}


def _responses_text_delta(text: str) -> dict:
    return {"type": "response.output_text.delta", "delta": text}


def _responses_reasoning_delta(text: str) -> dict:
    return {"type": "response.reasoning.delta", "delta": text}


def _responses_completed(usage: dict | None = None) -> dict:
    return {"type": "response.completed", "response": {"usage": usage or {}}}


def _sdk_payload(call: dict) -> dict:
    payload = dict(call)
    payload.update(payload.pop("extra_body", {}) or {})
    payload.pop("timeout", None)
    return payload


def _patch_openai(monkeypatch, responses):
    calls = []
    response_calls = []
    client_kwargs = []
    queue = list(responses) if isinstance(responses, tuple) else [responses]

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            response = responses() if callable(responses) else queue.pop(0)
            if isinstance(response, Exception):
                raise response
            return response

    class FakeResponses:
        def create(self, **kwargs):
            response_calls.append(kwargs)
            response = responses() if callable(responses) else queue.pop(0)
            if isinstance(response, Exception):
                raise response
            return response

    class FakeOpenAI:
        def __init__(self, **kwargs):
            client_kwargs.append(kwargs)
            self.chat = type("FakeChat", (), {"completions": FakeCompletions()})()
            self.responses = FakeResponses()

    monkeypatch.setattr(nanocode, "OpenAI", FakeOpenAI)
    return calls, response_calls, client_kwargs


def test_agent_tool_results_go_to_latest_tool_results_and_store(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)

    latest = agent.execute_tool_calls(
        [
            {
                "name": "Read",
                "intention": "read sample",
                "args": ["sample.txt", "0,1"],
            }
        ]
    )

    assert "alpha" in latest
    assert '- ok tool=Read args=["sample.txt","0,1"] key=tr.1' in latest
    assert "why: read sample" in latest
    assert "output:\n<ReadToolResult>" in latest
    assert session.state.tool_result_store["tr.1"].value.startswith("<ReadToolResult>")
    assert "alpha" in session.state.tool_result_store["tr.1"].value
    assert session.state.tool_result_store["tr.1"].log_path.startswith(os.path.join(".nanocode", "sessions"))
    assert session.state.tool_result_store["tr.1"].original_chars > 0
    assert session.state.tool_result_store["tr.1"].original_lines > 0
    assert session.state.tool_result_store["tr.1"].excerpted is False
    assert (tmp_path / session.state.tool_result_store["tr.1"].log_path).read_text(encoding="utf-8") == session.state.tool_result_store["tr.1"].value
    assert session.state.conversation == []
    assert os.path.isdir(session.tool_results_dir())


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
    assert list(session.state.tool_result_store) == ["tr.1"]
    assert "second read" in session.state.tool_result_store["tr.1"].description
    assert "first read" not in latest


def test_agent_does_not_dedupe_nonconsecutive_same_batch_readonly_tool_calls(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\nbeta\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)

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
    agent = Agent(session)

    agent.execute_tool_calls(
        [
            {"name": "Recall", "intention": "recall first", "args": ["tr.1"]},
            {"name": "Recall", "intention": "recall second", "args": ["tr.2"]},
            {"name": "Recall", "intention": "recall duplicated", "args": ["tr.1", "tr.2"]},
        ]
    )

    assert len(agent.tool_runner.latest_executions) == 1
    assert agent.tool_runner.latest_executions[0].call.args == ["tr.1", "tr.2"]


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
    assert list(session.state.tool_result_store) == ["tr.1", "tr.2"]
    assert path.read_text(encoding="utf-8") == "new\n"


def test_agent_tool_results_are_bounded_and_logged(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("H" * 5000 + "M" * 5000 + "T" * 5000 + "\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)

    latest = agent.execute_tool_calls([{"name": "Read", "intention": "read large sample", "args": ["sample.txt", "0,1"]}])

    item = session.state.tool_result_store["tr.1"]
    assert item.excerpted is True
    assert len(item.value) <= nanocode.MAX_TOOL_OUTPUT_CHARS
    assert "excerpted: true" in item.value
    assert "original_lines: " + str(item.original_lines) in item.value
    assert "original_chars: " + str(item.original_chars) in item.value
    assert "full_log:" not in item.value
    assert "H" * 50 in item.value
    assert "M" * 50 in item.value
    assert "T" * 50 in item.value
    assert "[tool result excerpt]" in latest
    assert (tmp_path / item.log_path).read_text(encoding="utf-8").startswith("<ReadToolResult>")


def test_agent_keeps_latest_batch_and_unreduced_tool_results(tmp_path, monkeypatch):
    for name in ["one.txt", "two.txt", "three.txt", "four.txt"]:
        (tmp_path / name).write_text(name + "\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    _set_context_budget(monkeypatch, agent, index_items=2, observe_after_results=4)

    for name in ["one.txt", "two.txt", "three.txt", "four.txt"]:
        agent.execute_tool_calls([{"name": "Read", "intention": "read " + name, "args": [name, "0,1"]}])

    latest = _blocks_text(agent.tool_context.latest)
    recent = _blocks_text(agent.tool_context.recent)
    assert "four.txt" in latest
    assert "four.txt" not in recent
    assert "one.txt" in recent
    assert "two.txt" in recent
    assert "three.txt" in recent
    assert "<ReadToolResult>" in latest
    assert "<ReadToolResult>" in recent
    assert len(agent.tool_context.recent) == 3
    assert agent.mode == nanocode.AgentMode.OBSERVE
    context = _observe_tool_result_context(agent)
    assert "one.txt" in context
    assert "two.txt" in context
    assert "three.txt" in context
    assert "four.txt" in context
    assert "<ReadToolResult>" in context
    assert len(agent.tool_context.unreduced_blocks(agent.blackboard.memory_checkpoint_tool_result_counter)) == 4


def test_agent_observes_full_latest_result_when_it_becomes_recent(tmp_path, monkeypatch):
    (tmp_path / "one.txt").write_text("one\n", encoding="utf-8")
    (tmp_path / "two.txt").write_text("two\n", encoding="utf-8")
    agent = Agent(Session(cwd=str(tmp_path)))
    _set_context_budget(monkeypatch, agent, raw_chars=10_000, observe_after_results=2)

    agent.execute_tool_calls([{"name": "Read", "intention": "read one", "args": ["one.txt", "0,1"]}])
    agent.execute_tool_calls([{"name": "Read", "intention": "read two", "args": ["two.txt", "0,1"]}])

    context = _observe_tool_result_context(agent)
    assert agent.mode == nanocode.AgentMode.OBSERVE
    assert "one.txt" in context
    assert "<ReadToolResult>" in context
    assert "one\n" in context
    assert "two.txt" in context
    recent = _blocks_text(agent.tool_context.recent)
    assert "key=tr.1" in recent
    assert "<ReadToolResult>" in recent
    assert agent.blackboard.memory_checkpoint_tool_result_counter == 0

    agent.handle_response(
        {
            "actions": [
                {"type": "keep", "source": ["tr.1"], "reason": "one.txt remains useful"},
                {"type": "forget", "source": ["tr.2"], "reason": "two.txt is not needed"},
            ]
        }
    )

    assert agent.blackboard.memory_checkpoint_tool_result_counter == 2
    assert agent.mode == nanocode.AgentMode.ACT
    assert agent.tool_context.unreduced_blocks(agent.blackboard.memory_checkpoint_tool_result_counter) == []
    assert "recall=tr.1" in _blocks_text(agent.tool_context.recent)
    assert "<ReadToolResult>" not in _blocks_text(agent.tool_context.recent)
    assert "recall=tr.2" in _blocks_text(agent.tool_context.latest)


def test_agent_act_context_keeps_pending_raw_after_latest_rotates(tmp_path, monkeypatch):
    (tmp_path / "one.txt").write_text("one\n", encoding="utf-8")
    (tmp_path / "two.txt").write_text("two\n", encoding="utf-8")
    agent = Agent(Session(cwd=str(tmp_path)))
    _set_context_budget(monkeypatch, agent, raw_chars=10_000)

    agent.execute_tool_calls([{"name": "Read", "intention": "read one", "args": ["one.txt", "0,1"]}])
    agent.execute_tool_calls([{"name": "Read", "intention": "read two", "args": ["two.txt", "0,1"]}])

    assert agent.mode == nanocode.AgentMode.ACT
    assert "key=tr.1" in _blocks_text(agent.tool_context.recent)
    index, unreduced, latest = agent._format_act_tool_result_context()
    assert "one.txt" in unreduced
    assert "one\n" in unreduced
    assert "two.txt" in latest
    assert "two\n" in latest
    assert "recall=tr.1" in index
    assert "recall=tr.2" in index
    assert "output:\n<ReadToolResult>" not in index


def test_empty_observe_compacts_unreduced_tool_results(tmp_path, monkeypatch):
    (tmp_path / "one.txt").write_text("one\n", encoding="utf-8")
    (tmp_path / "two.txt").write_text("two\n", encoding="utf-8")
    agent = Agent(Session(cwd=str(tmp_path)))
    _set_context_budget(monkeypatch, agent, raw_chars=300, observe_after_results=2)

    agent.execute_tool_calls([{"name": "Read", "intention": "read one", "args": ["one.txt", "0,1"]}])
    agent.execute_tool_calls([{"name": "Read", "intention": "read two", "args": ["two.txt", "0,1"]}])

    agent.handle_response({"actions": [], "_assistant_text": "checking result"})

    assert agent.blackboard.memory_checkpoint_tool_result_counter == 2
    assert agent.mode == nanocode.AgentMode.ACT
    assert agent.tool_context.unreduced_blocks(agent.blackboard.memory_checkpoint_tool_result_counter) == []


def test_assistant_text_does_not_mark_memory_checkpoint(tmp_path):
    (tmp_path / "sample.txt").write_text("alpha\n", encoding="utf-8")
    agent = Agent(Session(cwd=str(tmp_path)))
    agent.execute_tool_calls([{"name": "Read", "intention": "read sample", "args": ["sample.txt", "0,1"]}])

    agent.apply_response({"actions": [], "_assistant_text": "reading sample"})

    assert agent.blackboard.memory_checkpoint_tool_result_counter == 0


def test_known_action_accepts_source_references(tmp_path):
    agent = Agent(Session(cwd=str(tmp_path)))

    agent.handle_response({"actions": [{"type": "known", "items": [{"source": ["tr.1"], "text": "Router setup lives in app.py."}]}]})
    agent.handle_response({"actions": [{"type": "known", "items": [{"source": ["tr.2"], "text": "Router setup lives in app.py."}]}]})

    assert agent.blackboard.known == ["Router setup lives in app.py."]
    assert agent.blackboard.known[0].source == ("tr.1", "tr.2")
    assert "[tr.1, tr.2] Router setup lives in app.py." in agent.build_user_prompt()


def test_observe_prompt_uses_narrow_context(tmp_path):
    session = Session(cwd=str(tmp_path))
    session.state.conversation.append(nanocode.UserMessage(content="old conversation text"))
    session.state.tool_result_store["tr.9"] = nanocode.ToolResultItem(description="stored result", value="stored raw text")
    session.state.user_rules.add("always run tests")
    agent = Agent(session)
    agent.mode = nanocode.AgentMode.OBSERVE
    agent.blackboard.user_input = "fix bug"
    agent.blackboard.goal = "fix bug goal"
    agent.blackboard.plan = [nanocode.PlanItem(id="p1", text="inspect failing path", status=nanocode.PlanStatus.DOING)]
    agent.blackboard.hypotheses = [nanocode.Hypothesis(id="h1", text="cache branch", status=nanocode.HypothesisStatus.ACTIVE, source=("tr.1",))]
    agent.blackboard.known = ["known fact"]
    agent.blackboard.stable_knowledge = {"workflow": ["use pytest"]}
    agent.tool_context.kept_results = ['- ok tool=Read args=["old.py"] key=tr.1\n  output:\nselected result']
    agent.recent_edits = ["- sample.py: old edit"]
    agent.agent_feedback_errors = ["act error"]
    agent.observe_feedback_errors = ["observe error"]
    agent.tool_context.latest = ['- ok tool=Read args=["sample.py"] key=tr.2\n  output:\nraw alpha']

    prompt = agent.build_observe_prompt()

    assert "fix bug" in prompt
    assert "always run tests" not in prompt
    assert "fix bug goal" in prompt
    assert "inspect failing path" in prompt
    assert "cache branch" in prompt
    assert "known fact" in prompt
    assert "use pytest" in prompt
    assert "selected result" in prompt
    assert "raw alpha" in prompt
    assert "Observe Errors" in prompt
    assert "observe error" in prompt
    assert "act error" not in prompt
    assert "Conversation History" not in prompt
    assert "old conversation text" not in prompt
    assert "Tool Result Index" not in prompt
    assert "Archived Recall Index" not in prompt
    assert "stored raw text" not in prompt
    assert "Kept Tool Results" in prompt
    assert "Recent Edits" not in prompt
    assert "old edit" not in prompt


def test_act_prompt_includes_current_focus_from_doing_plan_item(tmp_path):
    agent = Agent(Session(cwd=str(tmp_path)))
    agent.blackboard.plan = [
        nanocode.PlanItem(id="p1", text="inspect config", status=nanocode.PlanStatus.DONE),
        nanocode.PlanItem(id="p2", text="edit command handler", status=nanocode.PlanStatus.DOING, context="next change"),
        nanocode.PlanItem(id="p3", text="run tests", status=nanocode.PlanStatus.TODO),
    ]

    prompt = agent.build_user_prompt()

    assert "Current Focus:\n- [◔ doing] edit command handler (id=p2)\n  context: next change" in prompt


def test_act_prompt_uses_first_todo_as_current_focus(tmp_path):
    agent = Agent(Session(cwd=str(tmp_path)))
    agent.blackboard.plan = [
        nanocode.PlanItem(id="p1", text="inspect config", status=nanocode.PlanStatus.DONE),
        nanocode.PlanItem(id="p2", text="edit command handler", status=nanocode.PlanStatus.TODO),
        nanocode.PlanItem(id="p3", text="run tests", status=nanocode.PlanStatus.TODO),
    ]

    prompt = agent.build_user_prompt()

    assert "Current Focus:\n- [○ todo] edit command handler (id=p2)" in prompt


def test_act_prompt_tells_model_to_reply_to_pending_feedback_first(tmp_path):
    agent = Agent(Session(cwd=str(tmp_path)))
    agent.session.state.pending_user_feedback = "focus on sed"

    prompt = agent.build_user_prompt()

    assert "Pending User Feedback:\nfocus on sed" in prompt
    assert "Pending feedback rules:" in prompt
    assert "first emit a brief assistant text response" in prompt
    assert "not a new task" in prompt
    assert "latest user language" in prompt
    assert "pending-feedback replies" in prompt


def test_act_prompt_includes_kept_tool_results(tmp_path):
    (tmp_path / "sample.txt").write_text("alpha unique\n", encoding="utf-8")
    (tmp_path / "other.txt").write_text("beta unique\n", encoding="utf-8")
    agent = Agent(Session(cwd=str(tmp_path)))

    agent.execute_tool_calls(
        [
            {"name": "Read", "intention": "read sample", "args": ["sample.txt", "0,1"]},
            {"name": "Read", "intention": "read other", "args": ["other.txt", "0,1"]},
        ]
    )
    agent.mode = nanocode.AgentMode.OBSERVE
    agent.handle_response(
        {
            "actions": [
                {"type": "keep", "source": ["tr.1"], "reason": "sample has alpha"},
                {"type": "forget", "source": ["tr.2"], "reason": "other.txt is not needed"},
            ]
        }
    )

    prompt = agent.build_user_prompt()
    assert "Kept Tool Results:" in prompt
    assert "alpha unique" in prompt
    assert "beta unique" not in prompt
    assert len(agent.tool_context.kept_results) == 1


def test_kept_tool_results_deduplicate_by_tool_key(tmp_path):
    (tmp_path / "sample.txt").write_text("alpha\n", encoding="utf-8")
    agent = Agent(Session(cwd=str(tmp_path)))

    agent.execute_tool_calls([{"name": "Read", "intention": "read sample", "args": ["sample.txt", "0,1"]}])
    agent.mode = nanocode.AgentMode.OBSERVE
    agent.handle_response(
        {
            "actions": [
                {"type": "keep", "source": ["tr.1", "tr.1"], "reason": "sample contains alpha"},
                {"type": "known", "items": [{"source": ["tr.1"], "text": "sample.txt was inspected."}]},
            ]
        }
    )

    assert len(agent.tool_context.kept_results) == 1
    assert agent.tool_context.kept_results[0].count("key=tr.1") == 1


def test_observe_reports_kept_tool_result_keys(tmp_path):
    (tmp_path / "sample.txt").write_text("alpha\n", encoding="utf-8")
    agent = Agent(Session(cwd=str(tmp_path)))
    agent.execute_tool_calls([{"name": "Read", "intention": "read sample", "args": ["sample.txt", "0,1"]}])
    agent.mode = nanocode.AgentMode.OBSERVE
    messages = []

    agent.handle_response(
        {"actions": [{"type": "keep", "source": ["tr.1"], "reason": "sample contains alpha"}]},
        on_message=messages.append,
    )

    assert "Tool Result Context: +tr.1" in messages


def test_forget_removes_kept_tool_result_but_keeps_known_source(tmp_path):
    agent = Agent(Session(cwd=str(tmp_path)))
    _seed_plan(agent, "debug branch")
    agent.tool_context.kept_results = [
        '- ok tool=Read args=["a"] key=tr.1\n  output:\na',
        '- ok tool=Read args=["b"] key=tr.2\n  output:\nb',
    ]
    agent.blackboard.known = [nanocode.KnownItem(text="a was ruled out.", source=("tr.1",))]
    messages = []

    result = agent.handle_response({"actions": [{"type": "forget", "source": ["tr.1"], "reason": "branch ruled out"}]}, on_message=messages.append)

    assert result.done is False
    assert "tr.1" not in _blocks_text(agent.tool_context.kept_results)
    assert "tr.2" in _blocks_text(agent.tool_context.kept_results)
    assert nanocode.KnownItem.source_of(agent.blackboard.known[0]) == ("tr.1",)
    assert messages == ["Tool Result Context: -tr.1"]


def test_hypothesis_action_updates_blackboard_and_report(tmp_path):
    agent = Agent(Session(cwd=str(tmp_path)))
    _seed_plan(agent, "debug branch")
    messages = []

    result = agent.handle_response(
        {
            "actions": [
                {
                    "type": "hypothesis",
                    "items": [
                        {
                            "id": "h1",
                            "text": "admin filtering drops history events",
                            "status": "active",
                            "source": ["tr.1"],
                            "context": "feed search",
                        }
                    ],
                }
            ]
        },
        on_message=messages.append,
    )

    assert result.done is False
    assert agent.blackboard.hypotheses == [
        nanocode.Hypothesis(
            text="admin filtering drops history events",
            status=nanocode.HypothesisStatus.ACTIVE,
            id="h1",
            source=("tr.1",),
            context="feed search",
        )
    ]
    assert messages == ["Hypotheses Updated\n  1. [active] h1: admin filtering drops history events [tr.1] context: feed search"]


def test_forget_rejects_active_hypothesis_source(tmp_path):
    agent = Agent(_session(tmp_path, debug=True))
    _seed_plan(agent, "debug branch")
    agent.tool_context.kept_results = ['- ok tool=Read args=["a"] key=tr.1\n  output:\na']
    agent.blackboard.hypotheses = [nanocode.Hypothesis(text="branch still possible", source=("tr.1",))]
    messages = []

    result = agent.handle_response({"actions": [{"type": "forget", "source": ["tr.1"], "reason": "branch ruled out"}]}, on_message=messages.append)

    assert result.done is False
    assert "tr.1" in _blocks_text(agent.tool_context.kept_results)
    assert any("active hypothesis source: tr.1" in error for error in agent.agent_feedback_errors)
    assert messages == ["ToolResult_Gate: active hypothesis source: tr.1."]


def test_forget_allows_source_when_hypothesis_is_closed_same_response(tmp_path):
    agent = Agent(Session(cwd=str(tmp_path)))
    _seed_plan(agent, "debug branch")
    agent.tool_context.kept_results = ['- ok tool=Read args=["a"] key=tr.1\n  output:\na']
    agent.blackboard.hypotheses = [nanocode.Hypothesis(id="h1", text="branch still possible", source=("tr.1",))]
    messages = []

    result = agent.handle_response(
        {
            "actions": [
                {
                    "type": "hypothesis",
                    "items": [{"id": "h1", "text": "branch ruled out", "status": "ruled_out", "source": ["tr.1"]}],
                },
                {"type": "forget", "source": ["tr.1"], "reason": "branch ruled out"},
            ]
        },
        on_message=messages.append,
    )

    assert result.done is False
    assert agent.blackboard.hypotheses[0].status == nanocode.HypothesisStatus.RULED_OUT
    assert "tr.1" not in _blocks_text(agent.tool_context.kept_results)
    assert messages == [
        "Hypotheses Updated\n  1. [ruled_out] h1: branch ruled out [tr.1]",
        "Tool Result Context: -tr.1",
    ]


def test_forget_allows_source_when_hypothesis_is_dropped_same_response(tmp_path):
    agent = Agent(Session(cwd=str(tmp_path)))
    _seed_plan(agent, "debug branch")
    agent.tool_context.kept_results = ['- ok tool=Read args=["a"] key=tr.1\n  output:\na']
    agent.blackboard.hypotheses = [nanocode.Hypothesis(id="h1", text="branch lost priority", source=("tr.1",))]
    messages = []

    result = agent.handle_response(
        {
            "actions": [
                {"type": "hypothesis", "items": [{"id": "h1", "text": "branch no longer matters", "status": "dropped", "source": ["tr.1"]}]},
                {"type": "forget", "source": ["tr.1"], "reason": "branch no longer matters"},
            ]
        },
        on_message=messages.append,
    )

    assert result.done is False
    assert agent.blackboard.hypotheses[0].status == nanocode.HypothesisStatus.DROPPED
    assert "tr.1" not in _blocks_text(agent.tool_context.kept_results)
    assert messages == [
        "Hypotheses Updated\n  1. [dropped] h1: branch no longer matters [tr.1]",
        "Tool Result Context: -tr.1",
    ]


def test_forget_rejects_missing_or_unknown_tool_result_key(tmp_path):
    agent = Agent(_session(tmp_path, debug=True))
    _seed_plan(agent, "debug branch")
    agent.tool_context.kept_results = ['- ok tool=Read args=["a"] key=tr.1\n  output:\na']
    messages = []

    result = agent.handle_response({"actions": [{"type": "forget", "source": ["tr.2"], "reason": "branch ruled out"}]}, on_message=messages.append)

    assert result.done is False
    assert "tr.1" in _blocks_text(agent.tool_context.kept_results)
    assert any("not in visible tool results: tr.2" in error for error in agent.agent_feedback_errors)
    assert messages == ["ToolResult_Gate: not in visible tool results: tr.2."]


def test_observe_forget_does_not_cover_latest_result_key(tmp_path):
    agent = Agent(_session(tmp_path, debug=True))
    agent.mode = nanocode.AgentMode.OBSERVE
    agent.tool_context.kept_results = ['- ok tool=Read args=["old"] key=tr.1\n  output:\nold']
    agent.tool_context.latest = ['- ok tool=Read args=["new"] key=tr.2\n  output:\nnew']
    messages = []

    result = agent.handle_response({"actions": [{"type": "forget", "source": ["tr.1"], "reason": "old branch ruled out"}]}, on_message=messages.append)

    assert result.done is False
    assert agent.mode == nanocode.AgentMode.ACT
    assert "tr.1" not in _blocks_text(agent.tool_context.kept_results)
    assert agent.tool_context.unreduced_blocks(agent.blackboard.memory_checkpoint_tool_result_counter) == []
    assert messages == ["Tool Result Context: -tr.1"]


def test_observe_can_forget_old_kept_result_while_forgetting_latest(tmp_path):
    agent = Agent(Session(cwd=str(tmp_path)))
    agent.mode = nanocode.AgentMode.OBSERVE
    agent.tool_context.kept_results = ['- ok tool=Read args=["old"] key=tr.1\n  output:\nold']
    agent.tool_context.latest = ['- ok tool=Read args=["new"] key=tr.2\n  output:\nnew']
    messages = []

    result = agent.handle_response(
        {
            "actions": [
                {"type": "forget", "source": ["tr.1"], "reason": "old branch ruled out"},
                {"type": "forget", "source": ["tr.2"], "reason": "new result is not useful"},
            ]
        },
        on_message=messages.append,
    )

    assert result.done is False
    assert agent.mode == nanocode.AgentMode.ACT
    assert agent.tool_context.kept_results == []
    assert agent.tool_context.unreduced_blocks(agent.blackboard.memory_checkpoint_tool_result_counter) == []
    assert messages == ["Tool Result Context: -tr.1 -tr.2"]


def test_pending_user_feedback_does_not_rewrite_goal_by_default(tmp_path):
    agent = Agent(Session(cwd=str(tmp_path)))
    _seed_plan(agent, "implement demo")
    agent.session.state.pending_user_feedback = "how many lines?"

    result = agent.handle_response({"actions": [{"type": "goal", "text": "answer line count"}]})

    assert result.done is False
    assert agent.blackboard.goal == "implement demo"
    assert agent.session.state.pending_user_feedback == ""
    assert any("Pending User Feedback is not a new task" in error for error in agent.agent_feedback_errors)


def test_keep_tool_results_ignore_non_tool_sources(tmp_path):
    (tmp_path / "sample.txt").write_text("alpha\n", encoding="utf-8")
    agent = Agent(Session(cwd=str(tmp_path)))

    agent.execute_tool_calls([{"name": "Read", "intention": "read sample", "args": ["sample.txt", "0,1"]}])
    agent.mode = nanocode.AgentMode.OBSERVE
    agent.handle_response(
        {
            "actions": [
                {"type": "keep", "source": ["note.1"], "reason": "invalid source is ignored"},
                {"type": "forget", "source": ["tr.1"], "reason": "invalid source is ignored"},
            ]
        }
    )

    assert agent.tool_context.kept_results == []
    assert "alpha\n" not in agent.build_user_prompt()


def test_keep_action_is_observe_only(tmp_path):
    agent = Agent(Session(cwd=str(tmp_path)))
    _seed_plan(agent, "answer")

    result = agent.handle_response({"actions": [{"type": "keep", "source": ["tr.1"], "reason": "fact"}]})

    assert result.done is False
    assert any("Invalid action(s): keep" in error for error in agent.agent_feedback_errors)


def test_observe_rejects_invalid_action_and_allows_empty_actions(tmp_path):
    (tmp_path / "sample.txt").write_text("alpha\n", encoding="utf-8")
    agent = Agent(Session(cwd=str(tmp_path)))
    agent.execute_tool_calls([{"name": "Read", "intention": "read sample", "args": ["sample.txt", "0,1"]}])
    agent.mode = nanocode.AgentMode.OBSERVE

    agent.handle_response({"actions": [{"type": "goal", "text": "answer", "complete": False}]})
    assert any("latest results must be observed" in error for error in agent.observe_feedback_errors)
    assert agent.mode == nanocode.AgentMode.OBSERVE

    agent.handle_response({"actions": []})

    assert agent.mode == nanocode.AgentMode.ACT
    assert agent.observe_feedback_errors == []
    assert agent.tool_context.unreduced_blocks(agent.blackboard.memory_checkpoint_tool_result_counter) == []


def test_observe_compacts_unmentioned_result_keys_by_default(tmp_path):
    agent = Agent(_session(tmp_path, debug=True))
    agent.mode = nanocode.AgentMode.OBSERVE
    agent.tool_context.latest = [
        '- ok tool=Read args=["a"] key=tr.1\n  output:\na',
        '- ok tool=Read args=["b"] key=tr.2\n  output:\nb',
    ]
    messages = []

    result = agent.handle_response(
        {"actions": [{"type": "keep", "source": ["tr.1"], "reason": "a matters"}]},
        on_message=messages.append,
    )

    assert result.done is False
    assert agent.mode == nanocode.AgentMode.ACT
    assert "tr.1" in _blocks_text(agent.tool_context.kept_results)
    assert agent.tool_context.unreduced_blocks(agent.blackboard.memory_checkpoint_tool_result_counter) == []
    assert messages == ["Tool Result Context: +tr.1"]


def test_observe_forget_source_covers_result_key(tmp_path):
    agent = Agent(Session(cwd=str(tmp_path)))
    agent.mode = nanocode.AgentMode.OBSERVE
    agent.tool_context.latest = ['- ok tool=Read args=["a"] key=tr.1\n  output:\na']

    result = agent.handle_response({"actions": [{"type": "forget", "source": ["tr.1"], "reason": "not useful"}]})

    assert result.done is False
    assert agent.mode == nanocode.AgentMode.ACT
    assert agent.tool_context.unreduced_blocks(agent.blackboard.memory_checkpoint_tool_result_counter) == []
    assert agent.tool_context.kept_results == []


def test_observe_known_source_compacts_result_key_by_default(tmp_path):
    agent = Agent(Session(cwd=str(tmp_path)))
    agent.mode = nanocode.AgentMode.OBSERVE
    agent.tool_context.latest = ['- ok tool=Read args=["a"] key=tr.1\n  output:\na']

    agent.handle_response({"actions": [{"type": "known", "items": [{"source": ["tr.1"], "text": "a exists"}]}]})

    assert agent.mode == nanocode.AgentMode.ACT
    assert [nanocode.KnownItem.format_item(item) for item in agent.blackboard.known] == ["[tr.1] a exists"]
    assert agent.tool_context.unreduced_blocks(agent.blackboard.memory_checkpoint_tool_result_counter) == []


def test_kept_tool_results_respect_char_budget(tmp_path, monkeypatch):
    agent = Agent(Session(cwd=str(tmp_path)))
    agent.mode = nanocode.AgentMode.OBSERVE
    _set_context_budget(monkeypatch, agent, kept_chars=100)
    agent.tool_context.latest = [
        '- ok tool=Read args=["a"] key=tr.1\n  output:\n' + ("a" * 30),
        '- ok tool=Read args=["b"] key=tr.2\n  output:\n' + ("b" * 30),
    ]

    agent.handle_response(
        {
            "actions": [
                {"type": "keep", "source": ["tr.1", "tr.2"], "reason": "both results matter"}
            ]
        }
    )

    context = _blocks_text(agent.tool_context.kept_results)
    assert "key=tr.1" not in context
    assert "key=tr.2" in context


def test_kept_tool_results_respect_per_block_char_budget(tmp_path, monkeypatch):
    agent = Agent(Session(cwd=str(tmp_path)))
    agent.mode = nanocode.AgentMode.OBSERVE
    _set_context_budget(monkeypatch, agent, kept_chars=10_000, kept_block_chars=300)
    agent.tool_context.latest = [
        '- ok tool=Read args=["large.py"] key=tr.1\n  output:\n' + ("head\n" + ("x" * 2000) + "\ntail")
    ]

    agent.handle_response({"actions": [{"type": "keep", "source": ["tr.1"], "reason": "large output matters"}]})

    assert len(agent.tool_context.kept_results[0]) <= agent.context_budget().kept_block_chars
    assert "key=tr.1" in agent.tool_context.kept_results[0]
    assert "[tool result excerpt]" in agent.tool_context.kept_results[0]


def test_observe_checkpoint_clears_observe_errors(tmp_path):
    (tmp_path / "sample.txt").write_text("alpha\n", encoding="utf-8")
    agent = Agent(Session(cwd=str(tmp_path)))
    agent.execute_tool_calls([{"name": "Read", "intention": "read sample", "args": ["sample.txt", "0,1"]}])
    agent.mode = nanocode.AgentMode.OBSERVE
    agent.observe_feedback_errors = ["old observe error"]

    agent.handle_response({"actions": [{"type": "keep", "source": ["tr.1"], "reason": "sample.txt contains alpha"}]})

    assert agent.mode == nanocode.AgentMode.ACT
    assert agent.observe_feedback_errors == []


def test_agent_tool_result_raw_budget_triggers_observe(tmp_path, monkeypatch):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    _set_context_budget(monkeypatch, agent, raw_chars=180, observe_after_results=99)
    path = tmp_path / "sample.txt"
    path.write_text("x" * 400 + "\n", encoding="utf-8")

    agent.execute_tool_calls([{"name": "Read", "intention": "read sample", "args": ["sample.txt", "0,1"]}])

    assert agent.mode == nanocode.AgentMode.OBSERVE
    assert agent.tool_context.raw_context_chars(agent.blackboard.memory_checkpoint_tool_result_counter) >= agent.context_budget().raw_chars
    observe_context = _observe_tool_result_context(agent)
    assert "sample.txt" in observe_context
    assert "x" * 50 in observe_context


def test_agent_tool_result_index_has_count_limit(tmp_path, monkeypatch):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    _set_context_budget(monkeypatch, agent, index_items=2)

    for index in range(4):
        agent.tool_context.append_recent(
            ['- ok tool=Read args=["' + str(index) + '"] key=tr.' + str(index + 1) + "\n  output:\n" + ("x" * 20)],
            max_index_items=agent.context_budget().index_items,
            checkpoint=999,
        )

    recent = _blocks_text(agent.tool_context.recent)
    assert "recall=tr.1" not in recent
    assert "recall=tr.2" not in recent
    assert "recall=tr.3" in recent
    assert "recall=tr.4" in recent
    assert len(agent.tool_context.recent) == 2


def test_tool_result_store_keeps_latest_256_items(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)

    for index in range(257):
        agent.tool_runner._store_tool_result(ParsedToolCall(name="Read", intention="", args=[str(index)]), "success", "output " + str(index))

    assert len(session.state.tool_result_store) == 256
    assert list(session.state.tool_result_store)[:2] == ["tr.2", "tr.3"]
    assert list(session.state.tool_result_store)[-1] == "tr.257"
    assert session.state.tool_result_counter == 257


def test_tool_result_store_trim_keeps_hypothesis_source_keys(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    agent.blackboard.hypotheses = [nanocode.Hypothesis(id="h1", text="kept branch", source=("tr.1",))]

    for index in range(257):
        agent.tool_runner._store_tool_result(ParsedToolCall(name="Read", intention="", args=[str(index)]), "success", "output " + str(index))

    assert len(session.state.tool_result_store) == 256
    assert "tr.1" in session.state.tool_result_store
    assert "tr.2" not in session.state.tool_result_store
    assert "tr.257" in session.state.tool_result_store


def test_agent_prunes_tool_result_store_but_keeps_referenced_result_keys(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)

    for index in range(52):
        key = "tr." + str(index + 1)
        session.state.tool_result_store[key] = nanocode.ToolResultItem(description=key, value="value")
    agent.tool_context.kept_results = ['- ok tool=Read args=["sample.txt"] key=tr.1\n  output:\nvalue']
    agent.blackboard.hypotheses = [nanocode.Hypothesis(id="h1", text="kept branch", source=("tr.2",))]

    agent._prune_tool_result_store()

    assert len(session.state.tool_result_store) == 50
    assert "tr.1" in session.state.tool_result_store
    assert "tr.2" in session.state.tool_result_store
    assert "tr.3" not in session.state.tool_result_store
    assert "tr.52" in session.state.tool_result_store


def test_agent_request_calls_chat_completions_and_returns_text(tmp_path, monkeypatch):
    calls, _response_calls, client_kwargs = _patch_openai(monkeypatch, _chat_response(usage={"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}))
    session = _session(tmp_path, api_url="https://example.test/v1", api_key="key", model="model", timeout=12, stream=False)

    response = Agent(session).request("system", "user")
    payload = _sdk_payload(calls[0])

    assert response == {"actions": [], "_assistant_text": "ok"}
    assert client_kwargs[0]["base_url"] == "https://example.test/v1"
    assert client_kwargs[0]["api_key"] == "key"
    assert client_kwargs[0]["timeout"] == 12
    assert calls[0]["timeout"] == 12
    assert payload["model"] == "model"
    assert payload["messages"] == [{"role": "system", "content": "system"}, {"role": "user", "content": "user"}]
    assert "temperature" not in payload
    assert "response_format" not in payload
    assert "reasoning_effort" not in payload
    assert "reasoning" not in payload
    assert session.state.last_prompt_tokens == 2
    assert session.state.last_completion_tokens == 3
    assert session.state.last_total_tokens == 5


def test_agent_request_manual_retry_resends_same_model_prompt(tmp_path):
    session = _session(tmp_path, api_url="https://example.test/v1", api_key="key", model="model", stream=False)
    agent = Agent(session)

    class FakeModelClient:
        def __init__(self):
            self.calls = 0

        def request(self, system_prompt, user_prompt, *, activity="agent", **_kwargs):
            self.calls += 1
            if self.calls == 1:
                raise nanocode.ModelRequestRetry()
            return {"actions": [{"type": "message", "text": system_prompt + "/" + user_prompt + "/" + activity}]}

    fake_client = FakeModelClient()
    agent.model_client = fake_client

    response = agent.request("system", "user", activity="observe")

    assert response == {"actions": [{"type": "message", "text": "system/user/observe"}]}
    assert fake_client.calls == 2
    assert session.state.status_notice == ""


def test_agent_request_sends_temperature_only_when_configured(tmp_path, monkeypatch):
    calls, _response_calls, _client_kwargs = _patch_openai(monkeypatch, _chat_response())
    session = _session(tmp_path, api_url="https://example.test/v1", api_key="key", model="model", stream=False, temperature=0.2)

    Agent(session).request("system", "user")

    assert _sdk_payload(calls[0])["temperature"] == 0.2


def test_agent_request_uses_responses_api_and_sdk_output_text(tmp_path, monkeypatch):
    class FakeResponse:
        output_text = "ok"

        def model_dump(self, mode="json"):
            return {"output": [], "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5}}

    calls, response_calls, _client_kwargs = _patch_openai(monkeypatch, FakeResponse())
    session = _session(
        tmp_path,
        api_url="https://api.openai.com/v1",
        api_key="key",
        model="model",
        api="responses",
        stream=False,
        reasoning="high",
    )

    response = Agent(session).request("system", "user")
    payload = _sdk_payload(response_calls[0])

    assert response == {"actions": [], "_assistant_text": "ok"}
    assert calls == []
    assert payload["model"] == "model"
    assert payload["instructions"] == "system"
    assert payload["input"] == "user"
    assert payload["store"] is False
    assert payload["reasoning"] == {"effort": "high"}
    assert session.state.last_prompt_tokens == 2
    assert session.state.last_completion_tokens == 3
    assert session.state.last_total_tokens == 5


def test_agent_request_responses_api_omits_reasoning_when_disabled(tmp_path, monkeypatch):
    calls, response_calls, _client_kwargs = _patch_openai(monkeypatch, _responses_response())
    session = _session(tmp_path, api_url="https://api.openai.com/v1", api_key="key", model="model", api="responses", stream=False)
    session.config.provider.reasoning = "off"

    Agent(session).request("system", "user")
    payload = _sdk_payload(response_calls[0])

    assert calls == []
    assert "reasoning" not in payload


def test_plan_mode_uses_runtime_plan_timeouts(tmp_path):
    session = _session(tmp_path, api_url="https://example.test/v1", api_key="key", model="model", timeout=12, first_token_timeout=5, plan_mode=True)
    session.settings.plan_timeout = 240
    session.settings.plan_first_token_timeout = 80
    client = nanocode.ModelClient(session)

    assert client._request_timeouts(session.config.provider, activity="agent") == (240, 80)
    assert client._request_timeouts(session.config.provider, activity="compact") == (12, 5)


def test_agent_request_retries_model_timeout(tmp_path, monkeypatch):
    class FakeModelClient:
        def __init__(self):
            self.calls = 0

        def request(self, system_prompt, user_prompt, *, activity="agent", **_kwargs):
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
    assert agent.session.state.turn_model_calls == 4
    assert sleeps == [3, 10, 20]


def test_agent_request_marks_first_token_timeout_notice(tmp_path, monkeypatch):
    class FakeModelClient:
        def __init__(self):
            self.calls = 0

        def request(self, system_prompt, user_prompt, *, activity="agent", **_kwargs):
            self.calls += 1
            if self.calls == 1:
                raise LLMError("request first token timeout")
            return {"actions": [{"type": "message", "text": "ok"}]}

    sleeps = []
    monkeypatch.setattr(nanocode.time, "sleep", sleeps.append)
    agent = Agent(Session(cwd=str(tmp_path)))
    agent.model_client = FakeModelClient()

    response = agent.request("system", "user")

    assert response["actions"][0]["text"] == "ok"
    assert agent.session.state.status_notice == "err:first_token"
    assert sleeps == [3]


def test_agent_request_hides_model_timeout_retries_without_debug(tmp_path, monkeypatch):
    class FakeModelClient:
        def __init__(self):
            self.calls = 0

        def request(self, system_prompt, user_prompt, *, activity="agent", **_kwargs):
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
    assert agent.session.state.turn_model_calls == 3
    assert sleeps == [3, 10]
    assert messages == []
    assert agent.session.state.status_notice == "err:timeout"

    debug_messages = []
    debug_agent = Agent(_session(tmp_path, debug=True))
    debug_agent.model_client = FakeModelClient()

    debug_agent.request("system", "user", on_message=debug_messages.append)

    assert debug_messages == [
        "Retrying: request model timeout; retry 1/6 in 3s.",
        "Retrying: request model timeout; retry 2/6 in 10s.",
    ]


def test_agent_gate_hides_retry_messages_without_debug(tmp_path):
    agent = Agent(Session(cwd=str(tmp_path)))
    messages = []

    agent._report_gate(messages.append, "Retrying: sample gate.", "Sample_Gate: first")
    agent._report_gate(messages.append, "Retrying: sample gate.", "Sample_Gate: second")
    agent._report_gate(messages.append, "Retrying: sample gate.", "Sample_Gate: third")

    assert messages == []
    assert agent.session.state.status_notice == "err:gate"


def test_agent_gate_does_not_overwrite_specific_recent_notice(tmp_path):
    agent = Agent(Session(cwd=str(tmp_path)))
    messages = []
    agent._set_status_notice("err:format")

    agent._report_gate(messages.append, "Retrying: sample gate.", "Sample_Gate: first")

    assert messages == []
    assert agent.session.state.status_notice == "err:format"


def test_agent_gate_reports_immediately_in_debug(tmp_path):
    agent = Agent(_session(tmp_path, debug=True))
    messages = []

    agent._report_gate(messages.append, "Retrying: sample gate.", "Sample_Gate: debug")

    assert messages == ["Sample_Gate: debug"]


def test_agent_request_stops_after_model_timeout_retries(tmp_path, monkeypatch):
    class FakeModelClient:
        def __init__(self):
            self.calls = 0

        def request(self, system_prompt, user_prompt, *, activity="agent", **_kwargs):
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
    assert agent.session.state.turn_model_calls == 7
    assert sleeps == [3, 10, 20, 30, 60, 120]


def test_agent_request_does_not_retry_other_llm_errors(tmp_path, monkeypatch):
    class FakeModelClient:
        def __init__(self):
            self.calls = 0

        def request(self, system_prompt, user_prompt, *, activity="agent", **_kwargs):
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
    assert agent.session.state.turn_model_calls == 1
    assert sleeps == []


def test_agent_request_sends_function_tool_schema_and_parses_tool_call(tmp_path, monkeypatch):
    calls, _response_calls, _client_kwargs = _patch_openai(
        monkeypatch,
        {
            "choices": [
                {
                    "message": {
                        "content": "Reading the file.",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "Read",
                                    "arguments": '{"intention":"read sample","args":["sample.txt","0","1"]}',
                                }
                            }
                        ],
                    }
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        },
    )
    session = _session(tmp_path, api_url="https://example.test/v1", api_key="key", model="model", stream=False)

    response = Agent(session).request("system", "user", tool_schemas=[nanocode.ReadTool.tool_schema()])
    payload = _sdk_payload(calls[0])

    assert payload["tools"][0]["function"]["name"] == "Read"
    assert payload["tool_choice"] == "auto"
    assert payload["parallel_tool_calls"] is True
    assert response == {
        "actions": [{"type": "tool", "name": "Read", "intention": "read sample", "args": ["sample.txt", "0", "1"]}],
        "_assistant_text": "Reading the file.",
    }
    assert session.state.last_total_tokens == 5


def test_agent_accepts_string_plan_items_from_function_call(tmp_path):
    agent = Agent(Session(cwd=str(tmp_path)))
    response = {"actions": [{"type": "plan", "mode": "replace", "items": ["Create demo", "Run smoke test"]}]}

    assert agent._build_response_context(response).has_fresh_plan_action is True
    agent.apply_response(response)

    assert agent.blackboard.plan == [
        nanocode.PlanItem(text="Create demo"),
        nanocode.PlanItem(text="Run smoke test"),
    ]


def test_agent_accepts_string_hypothesis_items_from_function_call(tmp_path):
    agent = Agent(Session(cwd=str(tmp_path)))

    agent.apply_response({"actions": [{"type": "hypothesis", "items": ["Admin filter excludes history"]}]})

    assert agent.blackboard.hypotheses == [
        nanocode.Hypothesis(text="Admin filter excludes history"),
    ]


def test_function_tool_schemas_define_items_for_every_array():
    def walk(value, path="schema"):
        if isinstance(value, dict):
            schema_type = value.get("type")
            if schema_type == "array" or (isinstance(schema_type, list) and "array" in schema_type):
                assert "items" in value, path
            for key, child in value.items():
                walk(child, path + "." + str(key))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, path + "[" + str(index) + "]")

    state_schemas = [nanocode._state_tool_schema(name) for name in nanocode.STATE_TOOL_PARAMS]
    repo_schemas = [tool.tool_schema() for tool in nanocode.TOOL_REGISTRY.values()]
    for schema in [*state_schemas, *repo_schemas, nanocode.COMPACT_TOOL_SCHEMA]:
        walk(schema)


def test_function_tool_schemas_do_not_emit_null_enum_values():
    def walk(value, path="schema"):
        if isinstance(value, dict):
            enum = value.get("enum")
            if isinstance(enum, list):
                assert None not in enum, path
            for key, child in value.items():
                walk(child, path + "." + str(key))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, path + "[" + str(index) + "]")

    state_schemas = [nanocode._state_tool_schema(name) for name in nanocode.STATE_TOOL_PARAMS]
    repo_schemas = [tool.tool_schema() for tool in nanocode.TOOL_REGISTRY.values()]
    for schema in [*state_schemas, *repo_schemas, nanocode.COMPACT_TOOL_SCHEMA]:
        walk(schema)


def test_agent_request_responses_api_parses_function_call(tmp_path, monkeypatch):
    _calls, response_calls, _client_kwargs = _patch_openai(
        monkeypatch,
        {
            "output": [
                {
                    "type": "function_call",
                    "name": "known",
                    "arguments": '{"items":["Project uses pytest."]}',
                }
            ],
            "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
        },
    )
    session = _session(tmp_path, api_url="https://api.openai.com/v1", api_key="key", model="model", api="responses", stream=False)

    response = Agent(session).request("system", "user", tool_schemas=[nanocode._state_tool_schema("known")])
    payload = _sdk_payload(response_calls[0])

    assert payload["tools"][0]["name"] == "known"
    assert payload["tool_choice"] == "auto"
    assert response == {"actions": [{"type": "known", "items": ["Project uses pytest."]}]}
    assert session.state.last_total_tokens == 5


def test_agent_request_chat_stream_parses_function_tool_event(tmp_path, monkeypatch):
    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return iter(
                [
                    _stream_chunk({"content": "Reading."}),
                    _stream_chunk({"tool_calls": [{"index": "0", "function": {"name": "Read", "arguments": '{"intention":"read sample",'}}]}),
                    _stream_chunk({"tool_calls": [{"index": "0", "function": {"arguments": '"args":["sample.txt","0","1"]}'}}]}),
                    _stream_chunk(usage={"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}, choices=False),
                ]
            )

    class FakeOpenAI:
        def __init__(self, **_kwargs):
            self.chat = type("FakeChat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr(nanocode, "OpenAI", FakeOpenAI)
    session = _session(tmp_path, api_url="https://example.test/v1", api_key="key", model="model")

    response = Agent(session).request("system", "user", tool_schemas=[nanocode.ReadTool.tool_schema()])

    assert calls[0]["tools"][0]["function"]["name"] == "Read"
    assert calls[0]["stream"] is True
    assert response == {
        "actions": [
            {
                "type": "tool",
                "name": "Read",
                "intention": "read sample",
                "args": ["sample.txt", "0", "1"],
            }
        ],
        "_assistant_text": "Reading.",
    }
    assert session.state.last_total_tokens == 5


def test_agent_request_responses_stream_parses_function_tool_event(tmp_path, monkeypatch):
    response_calls = []

    class FakeResponses:
        def create(self, **kwargs):
            response_calls.append(kwargs)
            return iter(
                [
                    {"type": "response.output_text.delta", "delta": "Recording."},
                    {
                        "type": "response.function_call_arguments.done",
                        "name": "known",
                        "arguments": '{"items":["Project uses pytest."]}',
                    },
                    {"type": "response.completed", "response": {"usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5}}},
                ]
            )

    class FakeOpenAI:
        def __init__(self, **_kwargs):
            self.responses = FakeResponses()

    monkeypatch.setattr(nanocode, "OpenAI", FakeOpenAI)
    session = _session(tmp_path, api_url="https://api.openai.com/v1", api_key="key", model="model", api="responses")

    response = Agent(session).request("system", "user", tool_schemas=[nanocode._state_tool_schema("known")])

    assert response_calls[0]["tools"][0]["name"] == "known"
    assert response_calls[0]["stream"] is True
    assert response == {"actions": [{"type": "known", "items": ["Project uses pytest."]}], "_assistant_text": "Recording."}
    assert session.state.last_total_tokens == 5


def test_agent_request_responses_stream_uses_output_item_function_name(tmp_path, monkeypatch):
    class FakeResponses:
        def create(self, **_kwargs):
            return iter(
                [
                    {
                        "type": "response.output_item.added",
                        "output_index": 0,
                        "item": {"id": "fc_1", "type": "function_call", "name": "goal", "arguments": ""},
                    },
                    {
                        "type": "response.function_call_arguments.done",
                        "item_id": "fc_1",
                        "arguments": '{"text":"Greet the user.","complete":true,"message_for_complete":"Hi!"}',
                    },
                    {"type": "response.completed", "response": {"usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5}}},
                ]
            )

    class FakeOpenAI:
        def __init__(self, **_kwargs):
            self.responses = FakeResponses()

    monkeypatch.setattr(nanocode, "OpenAI", FakeOpenAI)
    session = _session(tmp_path, api_url="https://api.openai.com/v1", api_key="key", model="model", api="responses")

    response = Agent(session).request("system", "user", tool_schemas=[nanocode._state_tool_schema("goal")])

    assert response == {"actions": [{"type": "goal", "text": "Greet the user.", "complete": True, "message_for_complete": "Hi!"}]}


def test_agent_request_responses_stream_error_event_raises_llm_error(tmp_path, monkeypatch):
    _patch_openai(monkeypatch, [{"code": "InvalidParameter", "message": "Unsupported model: 'deepseek-v4-flash'."}])
    session = _session(tmp_path, api_url="https://api.openai.com/v1", api_key="key", model="model", api="responses")

    try:
        Agent(session).request("system", "user")
    except LLMError as error:
        assert str(error) == "API request failed: InvalidParameter: Unsupported model: 'deepseek-v4-flash'."
    else:
        raise AssertionError("expected LLMError")


def test_agent_request_records_stream_rate_from_usage(tmp_path, monkeypatch):
    times = [100.0, 100.0, 100.0, 102.0]
    _patch_openai(
        monkeypatch,
        [
            _stream_chunk({"content": "ok"}),
            _stream_chunk(usage={"completion_tokens": 20, "total_tokens": 30}, choices=False),
        ],
    )
    monkeypatch.setattr(nanocode.time, "monotonic", lambda: times.pop(0) if times else 102.0)
    session = _session(tmp_path, api_url="https://example.test/v1", api_key="key", model="model")

    response = Agent(session).request("system", "user")

    assert response == {"actions": [], "_assistant_text": "ok"}
    assert session.state.last_model_call_rate == 10.0


def test_agent_request_stream_hard_timeout_becomes_model_timeout(tmp_path, monkeypatch):
    def stream():
        if False:
            yield {}
        while True:
            nanocode.signal.raise_signal(nanocode.signal.SIGALRM)
            yield {}

    sleeps = []
    _patch_openai(monkeypatch, stream)
    monkeypatch.setattr(nanocode.time, "sleep", sleeps.append)
    session = _session(tmp_path, api_url="https://example.test/v1", api_key="key", model="model", timeout=12)

    try:
        Agent(session).request("system", "user")
    except LLMError as error:
        assert str(error) == "request model timeout"
    else:
        raise AssertionError("expected LLMError")

    assert session.state.current_model_call_started_at == 0.0
    assert sleeps == [3, 10, 20, 30, 60, 120]


def test_agent_request_uses_configured_chat_reasoning(tmp_path, monkeypatch):
    calls, _response_calls, _client_kwargs = _patch_openai(monkeypatch, _chat_response())
    session = _session(
        tmp_path,
        api_url="https://example.test/v1",
        api_key="key",
        model="model",
        reasoning="high",
        chat_reasoning="reasoning",
        stream=False,
    )

    Agent(session).request("system", "user")
    payload = _sdk_payload(calls[0])

    assert payload["reasoning"] == {"effort": "high"}
    assert "reasoning_effort" not in payload


def test_agent_request_uses_configured_reasoning_effort_payload(tmp_path, monkeypatch):
    calls, _response_calls, _client_kwargs = _patch_openai(monkeypatch, _chat_response())
    session = _session(
        tmp_path,
        api_url="https://example.test/v1",
        api_key="key",
        model="model",
        reasoning="high",
        chat_reasoning="reasoning_effort",
        stream=False,
    )

    Agent(session).request("system", "user")
    payload = _sdk_payload(calls[0])

    assert payload["reasoning_effort"] == "high"
    assert "reasoning" not in payload


def test_agent_request_uses_configured_thinking_payload(tmp_path, monkeypatch):
    calls, _response_calls, _client_kwargs = _patch_openai(monkeypatch, _chat_response())
    session = _session(
        tmp_path,
        api_url="https://example.test/v1",
        api_key="key",
        model="model",
        reasoning="xhigh",
        chat_reasoning="thinking",
        stream=False,
    )

    Agent(session).request("system", "user")
    payload = _sdk_payload(calls[0])

    assert payload["thinking"] == {"type": "enabled"}
    assert payload["reasoning_effort"] == "max"
    assert "reasoning" not in payload


def test_agent_request_uses_configured_thinking_disabled_payload(tmp_path, monkeypatch):
    calls, _response_calls, _client_kwargs = _patch_openai(monkeypatch, _chat_response())
    session = _session(
        tmp_path,
        api_url="https://example.test/v1",
        api_key="key",
        model="model",
        chat_reasoning="thinking",
        stream=False,
    )
    session.config.provider.reasoning = "off"

    Agent(session).request("system", "user")
    payload = _sdk_payload(calls[0])

    assert payload["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in payload


def test_agent_request_auto_detects_chat_reasoning_from_provider_url(tmp_path, monkeypatch):
    calls, _response_calls, _client_kwargs = _patch_openai(monkeypatch, tuple(_chat_response() for _ in range(10)))

    Agent(
        _session(
            tmp_path,
            api_url="https://api.deepseek.com",
            api_key="key",
            model="model",
            reasoning="xhigh",
            stream=False,
        )
    ).request("system", "user")
    Agent(
        _session(
            tmp_path,
            api_url="https://openrouter.ai/api/v1",
            api_key="key",
            model="model",
            api="chat",
            reasoning="high",
            stream=False,
        )
    ).request("system", "user")
    Agent(
        _session(
            tmp_path,
            api_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key="key",
            model="qwen3.6-plus",
            api="chat",
            reasoning="high",
            stream=False,
        )
    ).request("system", "user")
    Agent(
        _session(
            tmp_path,
            api_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key="key",
            model="deepseek-v4-flash",
            api="chat",
            reasoning="xhigh",
            stream=False,
        )
    ).request("system", "user")
    Agent(
        _session(
            tmp_path,
            api_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key="key",
            model="glm-5.1",
            api="chat",
            reasoning="high",
            stream=False,
        )
    ).request("system", "user")
    Agent(
        _session(
            tmp_path,
            api_url="https://api.openai.com/v1",
            api_key="key",
            model="gpt-5",
            api="chat",
            reasoning="medium",
            stream=False,
        )
    ).request("system", "user")
    Agent(
        _session(
            tmp_path,
            api_url="https://opencode.ai/zen/go/v1",
            api_key="key",
            model="deepseek-v4-flash",
            reasoning="high",
            stream=False,
        )
    ).request("system", "user")
    Agent(
        _session(
            tmp_path,
            api_url="https://opencode.ai/zen/go/v1",
            api_key="key",
            model="kimi-k2.6",
            reasoning="high",
            stream=False,
        )
    ).request("system", "user")
    Agent(
        _session(
            tmp_path,
            api_url="https://not-openrouter.ai/api/v1",
            api_key="key",
            model="model",
            stream=False,
        )
    ).request("system", "user")
    Agent(
        _session(
            tmp_path,
            api_url="https://example.test/v1",
            api_key="key",
            model="model",
            stream=False,
        )
    ).request("system", "user")

    payloads = [_sdk_payload(call) for call in calls]
    assert payloads[0]["thinking"] == {"type": "enabled"}
    assert payloads[0]["reasoning_effort"] == "max"
    assert payloads[1]["reasoning"] == {"effort": "high"}
    assert payloads[2]["enable_thinking"] is True
    assert payloads[2]["thinking_budget"] == nanocode.ALIYUN_THINKING_BUDGET_BY_EFFORT["high"]
    assert payloads[3]["thinking"] == {"type": "enabled"}
    assert payloads[3]["reasoning_effort"] == "max"
    assert payloads[4] == {"model": "glm-5.1", "messages": [{"role": "system", "content": "system"}, {"role": "user", "content": "user"}], "stream": False}
    assert payloads[5]["reasoning_effort"] == "medium"
    assert payloads[6]["reasoning"] == {"effort": "high"}
    for payload in payloads[7:]:
        assert "reasoning" not in payload
        assert "reasoning_effort" not in payload
        assert "thinking" not in payload
        assert "enable_thinking" not in payload


def test_provider_config_auto_resolves_api_and_chat_reasoning_from_profiles():
    openai_provider = nanocode.ProviderConfig.from_dict({"url": "https://api.openai.com/v1", "api": "auto"})
    openai_reasoning_provider = nanocode.ProviderConfig.from_dict({"url": "https://api.openai.com/v1", "api": "chat", "model": "gpt-5"})
    openrouter_provider = nanocode.ProviderConfig.from_dict({"url": "https://openrouter.ai/api/v1", "api": "auto"})
    opencode_deepseek_provider = nanocode.ProviderConfig.from_dict({"url": "https://opencode.ai/zen/go/v1", "api": "auto", "model": "deepseek-v4-flash"})
    opencode_kimi_provider = nanocode.ProviderConfig.from_dict({"url": "https://opencode.ai/zen/go/v1", "api": "auto", "model": "kimi-k2.6"})
    dashscope_provider = nanocode.ProviderConfig.from_dict({"url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "api": "auto", "model": "qwen3.6-plus"})
    dashscope_deepseek_provider = nanocode.ProviderConfig.from_dict({"url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "api": "auto", "model": "deepseek-v4-flash"})
    unknown_provider = nanocode.ProviderConfig.from_dict({"url": "https://example.test/v1", "api": "auto"})

    assert openai_provider.resolved_api() == "responses"
    assert openai_provider.resolved_chat_reasoning() == "off"
    assert openai_reasoning_provider.resolved_api() == "chat"
    assert openai_reasoning_provider.resolved_chat_reasoning() == "reasoning_effort"
    assert openrouter_provider.resolved_api() == "responses"
    assert openrouter_provider.resolved_chat_reasoning() == "reasoning"
    assert opencode_deepseek_provider.resolved_api() == "chat"
    assert opencode_deepseek_provider.resolved_chat_reasoning() == "reasoning"
    assert opencode_kimi_provider.resolved_api() == "chat"
    assert opencode_kimi_provider.resolved_chat_reasoning() == "off"
    assert dashscope_provider.resolved_api() == "chat"
    assert dashscope_provider.resolved_chat_reasoning() == "enable_thinking"
    assert dashscope_deepseek_provider.resolved_api() == "chat"
    assert dashscope_deepseek_provider.resolved_chat_reasoning() == "thinking"
    assert unknown_provider.resolved_api() == "chat"
    assert unknown_provider.resolved_chat_reasoning() == "off"


def test_agent_request_off_chat_reasoning_disables_auto_detection(tmp_path, monkeypatch):
    calls, _response_calls, _client_kwargs = _patch_openai(monkeypatch, _chat_response())
    session = _session(
        tmp_path,
        api_url="https://api.deepseek.com",
        api_key="key",
        model="model",
        stream=False,
    )
    session.config.provider.chat_reasoning = "off"

    Agent(session).request("system", "user")
    payload = _sdk_payload(calls[0])

    assert "reasoning" not in payload
    assert "reasoning_effort" not in payload
    assert "thinking" not in payload


def test_agent_request_wraps_non_json_model_content_as_format_error(tmp_path, monkeypatch):
    _patch_openai(monkeypatch, _chat_response("plain answer"))
    session = _session(tmp_path, api_url="https://example.test/v1", api_key="key", model="model", stream=False)

    response = Agent(session).request("system", "user")

    assert response["actions"] == []
    assert response["_assistant_text"] == "plain answer"


def test_agent_request_wraps_missing_message_content_as_format_error(tmp_path, monkeypatch):
    _patch_openai(
        monkeypatch,
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": None},
                }
            ],
            "usage": {},
        },
    )
    session = _session(tmp_path, api_url="https://example.test/v1", api_key="key", model="model", stream=False)

    response = Agent(session).request("system", "user")

    assert response["actions"] == []
    assert "expected a function tool call" in response["_format_error"]
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

    assert agent.blackboard.known == ["Search only supports rg and Python fallback."]


def test_agent_dedupes_normalized_known_facts(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)

    agent.apply_response(
        {
            "actions": [
                {
                    "type": "known",
                    "items": [
                        "Preview logic exists in _preview_segments.",
                        "Preview logic exists in _preview_segments.",
                        "Preview logic exists in _preview_segments!",
                    ],
                }
            ]
        }
    )

    assert agent.blackboard.known == ["Preview logic exists in _preview_segments."]


def test_agent_replaces_redundant_known_fact_with_more_specific_fact(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)

    agent.apply_response(
        {
            "actions": [
                {
                    "type": "known",
                    "items": [
                        "_knowledge 方法位于 nanocode.py 第5138行，当前仅支持显示知识",
                        "_knowledge 方法位于 nanocode.py 第5138行，当前仅支持显示知识（无参数时）",
                    ],
                }
            ]
        }
    )

    assert agent.blackboard.known == ["_knowledge 方法位于 nanocode.py 第5138行，当前仅支持显示知识（无参数时）"]


def test_agent_keeps_latest_500_known_items(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)

    agent.apply_response({"actions": [{"type": "known", "items": ["fact " + str(index) for index in range(501)]}]})

    assert len(agent.blackboard.known) == 500
    assert agent.blackboard.known[0] == "fact 1"
    assert agent.blackboard.known[-1] == "fact 500"


def test_main_agent_applies_stable_knowledge_action(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)

    agent.apply_response(
        {
            "actions": [
                {"type": "known", "items": ["Read pyproject.toml."]},
                {
                    "type": "stable_knowledge",
                    "items": [
                        {"category": "workflow", "text": "Project test command is make test."},
                        {"category": "workflow", "text": "Project test command is make test."},
                    ],
                }
            ]
        }
    )

    assert agent.blackboard.known == ["Read pyproject.toml."]
    assert agent.blackboard.stable_knowledge == {"workflow": ["Project test command is make test."]}
    assert "  Stable_Knowledge\n" in agent.state_updater.latest_report
    assert "    workflow\n" in agent.state_updater.latest_report
    assert "      1. Project test command is make test." in agent.state_updater.latest_report


def test_main_agent_keeps_latest_30_stable_knowledge_items_per_category(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)

    agent.apply_response(
        {
            "actions": [
                {
                    "type": "stable_knowledge",
                    "items": [{"category": "workflow", "text": "stable fact " + str(index)} for index in range(31)],
                }
            ]
        }
    )

    assert len(agent.blackboard.stable_knowledge["workflow"]) == 30
    assert agent.blackboard.stable_knowledge["workflow"][0] == "stable fact 1"
    assert agent.blackboard.stable_knowledge["workflow"][-1] == "stable fact 30"


def test_main_agent_applies_user_rule_and_saves(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)

    agent.apply_response({"actions": [{"type": "user_rule", "text": "Prompt-only changes do not need tests."}]})

    with open(session.user_rules_path(), encoding="utf-8") as file:
        content = file.read()
    assert session.state.user_rules.format() == "# User Rules\n\n- Prompt-only changes do not need tests."
    assert content == "# User Rules\n\n- Prompt-only changes do not need tests.\n"
    assert "  User_Rules    updated" in agent.state_updater.latest_report


def test_user_rules_deduplicate(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)

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
    agent = Agent(session)

    class FakeModelClient:
        def __init__(self):
            self.calls = 0

        def request(self, system_prompt, user_prompt, *, activity="agent", **_kwargs):
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
    assert not any(message.startswith("State Updated") for message in messages)
    assert session.state.conversation[-1].content == "记住了。"


def test_main_agent_state_updates_show_in_debug(tmp_path):
    agent = Agent(_session(tmp_path, debug=True))

    class FakeModelClient:
        def request(self, system_prompt, user_prompt, *, activity="agent", **_kwargs):
            return {"actions": [{"type": "user_rule", "text": "Prompt-only changes do not need tests.", "message": "记住了。"}]}

    agent.model_client = FakeModelClient()
    messages = []

    agent.run("记住：prompt 改动不用测试", on_message=messages.append)

    assert "User Rules Updated\n  updated" in messages


def test_main_agent_state_updates_are_compact_without_debug(tmp_path):
    agent = Agent(Session(cwd=str(tmp_path)))

    agent.apply_response(
        {
            "actions": [
                {"type": "goal", "text": "inspect project", "complete": False},
                {
                    "type": "plan",
                    "items": [
                        {"id": "p1", "text": "List files", "status": "done"},
                        {"id": "p2", "text": "Read config", "status": "done"},
                        {"id": "p3", "text": "Update code", "status": "doing"},
                        {"id": "p4", "text": "Run tests", "status": "todo"},
                    ],
                },
                {"type": "known", "items": ["fact one", "fact two", "fact three", "fact four"]},
            ]
        }
    )

    report = agent.state_updater.compact_report()
    assert report.startswith("Goal + Plan + Known Updated")
    assert "\nGoal\n  inspect project\n" in report
    assert "\nPlan\n" in report
    assert "  ... 1 older\n  2. [✓ done] Read config\n  3. [◔ doing] Update code\n  4. [○ todo] Run tests" in report
    assert "\nKnown\n" in report
    assert "  ... 1 older\n  2. fact two\n  3. fact three\n  4. fact four" in report
    assert "State Updated" not in report


def test_main_agent_compact_report_labels_combined_hypotheses_and_known(tmp_path):
    agent = Agent(Session(cwd=str(tmp_path)))

    agent.apply_response(
        {
            "actions": [
                {
                    "type": "hypothesis",
                    "items": [{"id": "h1", "text": "admin selector starves history mode", "status": "active", "source": ["tr.2"]}],
                },
                {"type": "known", "items": [{"fact": "feed SSE request path is shared by admin and normal users", "source": ["tr.3"]}]},
            ]
        }
    )

    report = agent.state_updater.compact_report()
    assert report == "\n".join(
        [
            "Hypotheses + Known Updated",
            "Hypotheses",
            "  1. [active] h1: admin selector starves history mode [tr.2]",
            "Known",
            "  1. [tr.3] feed SSE request path is shared by admin and normal users",
        ]
    )


def test_main_agent_compact_plan_report_shows_changed_rows(tmp_path):
    agent = Agent(Session(cwd=str(tmp_path)))
    agent.blackboard.plan = [
        nanocode.PlanItem(id="p1", text="List files", status=nanocode.PlanStatus.DONE),
        nanocode.PlanItem(id="p2", text="Read config", status=nanocode.PlanStatus.TODO),
        nanocode.PlanItem(id="p3", text="Update code", status=nanocode.PlanStatus.TODO),
        nanocode.PlanItem(id="p4", text="Run tests", status=nanocode.PlanStatus.TODO),
    ]

    agent.apply_response({"actions": [{"type": "plan", "mode": "patch", "items": [{"id": "p2", "status": "doing"}]}]})

    report = agent.state_updater.compact_report()
    assert report == "Plan Updated\n  2. [◔ doing] Read config"


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

    assert agent.blackboard.known == [
        "Parser notes exist.",
        "Parser notes were captured.",
    ]


def test_agent_ignores_schema_placeholder_known_facts(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)

    agent.apply_response({"actions": [{"type": "known", "items": ["<fact from latest tool results>", "Real fact."]}]})

    assert agent.blackboard.known == ["Real fact."]


def test_agent_state_report_only_includes_real_plan_and_known_changes(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)

    response = {
        "actions": [
            {
                "type": "plan",
                "mode": "replace",
                "items": [{"id": "p1", "text": "Inspect file", "status": "todo"}],
            },
            {"type": "known", "items": ["Search uses rg."]},
        ]
    }

    agent.apply_response(response)

    assert "  Plan\n" in agent.state_updater.latest_report
    assert "    1. [○ todo] Inspect file" in agent.state_updater.latest_report
    assert "  Known\n" in agent.state_updater.latest_report
    assert "    1. Search uses rg." in agent.state_updater.latest_report

    agent.apply_response(response)

    assert agent.state_updater.latest_report == ""


def test_agent_ignores_empty_plan_replace(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    agent.blackboard.plan = [nanocode.PlanItem(id="p1", text="Inspect file", status=nanocode.PlanStatus.TODO)]

    agent.apply_response({"actions": [{"type": "plan", "mode": "replace", "items": []}]})

    assert [item.text for item in agent.blackboard.plan] == ["Inspect file"]
    assert agent.state_updater.latest_report == ""


def test_agent_treats_plan_without_mode_as_replace(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    agent.blackboard.plan = [
        nanocode.PlanItem(id="p1", text="Inspect old file", status=nanocode.PlanStatus.DONE),
        nanocode.PlanItem(id="p2", text="Edit old file", status=nanocode.PlanStatus.TODO),
    ]
    response = {"actions": [{"type": "plan", "items": [{"id": "p1", "text": "Inspect new file", "status": "doing"}]}]}

    assert agent._build_response_context(response).has_fresh_plan_action is True
    agent.apply_response(response)

    assert [item.text for item in agent.blackboard.plan] == ["Inspect new file"]
    assert agent.blackboard.plan[0].status == nanocode.PlanStatus.DOING


def test_agent_applies_partial_plan_patch(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    agent.blackboard.plan = [
        nanocode.PlanItem(id="p1", text="Inspect file", status=nanocode.PlanStatus.TODO, context="old"),
    ]

    agent.apply_response({"actions": [{"type": "plan", "mode": "patch", "items": [{"id": "p1", "status": "done", "context": "read file"}]}]})

    assert agent.blackboard.plan == [
        nanocode.PlanItem(id="p1", text="Inspect file", status=nanocode.PlanStatus.DONE, context="read file"),
    ]


def test_agent_applies_goal_and_plan_actions(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)

    agent.apply_response(
        {
            "actions": [
                {"type": "goal", "text": "change map", "complete": False},
                {
                    "type": "plan",
                    "items": [
                        {"id": "p1", "text": "Find map code", "status": "doing", "context": "need location"},
                        {"id": "p2", "text": "Edit map size", "status": "todo"},
                    ],
                }
            ]
        }
    )

    assert agent.blackboard.goal == "change map"
    assert agent.blackboard.task_code == nanocode.TaskCode.WORKING
    assert agent.blackboard.goal_reached is False
    assert [item.text for item in agent.blackboard.plan] == ["Find map code", "Edit map size"]
    assert agent.blackboard.plan[0].status == nanocode.PlanStatus.DOING
    assert "  Goal    change map" in agent.state_updater.latest_report
    assert "  Plan\n" in agent.state_updater.latest_report


def test_agent_accepts_goal_without_plan_for_new_task(tmp_path):
    agent = Agent(_session(tmp_path, debug=True))
    agent.blackboard.task_code = nanocode.TaskCode.NEW
    messages = []

    result = agent.handle_response({"actions": [{"type": "goal", "text": "change map", "work_mode": "normal", "complete": False}]}, on_message=messages.append)

    assert result.done is False
    assert agent.blackboard.goal == "change map"
    assert agent.blackboard.task_code == nanocode.TaskCode.WORKING
    assert agent.blackboard.plan == []
    assert messages == ["Goal Updated\n  change map"]


def test_new_goal_clears_task_local_kept_results_only(tmp_path):
    agent = Agent(Session(cwd=str(tmp_path)))
    agent.blackboard.goal = "old goal"
    agent.tool_context.kept_results = ['- ok tool=Read args=["old.py"] key=tr.1\n  output:\nselected result']
    agent.tool_context.latest = ['- ok tool=Read args=["latest.py"] key=tr.3\n  output:\nlatest raw']
    agent.tool_context.recent = ['- ok tool=Read args=["recent.py"] key=tr.4\n  out: 3 lines, 12 chars; recall=tr.4']

    agent.apply_response(
        {
            "actions": [
                {"type": "goal", "text": "new goal", "complete": False},
                {
                    "type": "plan",
                    "items": [{"id": "p1", "text": "Inspect new target", "status": "doing"}],
                },
            ]
        }
    )

    assert agent.tool_context.kept_results == []
    assert "latest.py" in _blocks_text(agent.tool_context.latest)
    assert "latest raw" not in _blocks_text(agent.tool_context.latest)
    assert "recent.py" in _blocks_text(agent.tool_context.recent)


def test_same_goal_keeps_task_local_tool_results(tmp_path):
    agent = Agent(Session(cwd=str(tmp_path)))
    agent.blackboard.goal = "same goal"
    agent.tool_context.kept_results = ['- ok tool=Read args=["old.py"] key=tr.1\n  output:\nselected result']
    agent.tool_context.latest = ['- ok tool=Read args=["new.py"] key=tr.2\n  output:\npending raw']

    agent.apply_response(
        {
            "actions": [
                {"type": "goal", "text": "same goal", "complete": False},
                {
                    "type": "plan",
                    "items": [{"id": "p1", "text": "Continue current target", "status": "doing"}],
                },
            ]
        }
    )

    assert "selected result" in _blocks_text(agent.tool_context.kept_results)
    assert "pending raw" in _blocks_text(agent.tool_context.latest)


def test_agent_state_report_does_not_repeat_goal_for_restarted_task_when_text_matches(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    agent.blackboard.goal = "change map"
    agent.blackboard.task_code = nanocode.TaskCode.WORKING

    agent.apply_response(
        {
            "actions": [
                {"type": "goal", "text": "change map", "complete": False},
                {
                    "type": "plan",
                    "items": [{"id": "p1", "text": "Find map code", "status": "doing"}],
                },
            ]
        }
    )

    assert "  Goal    change map" not in agent.state_updater.latest_report
    assert "  Plan\n" in agent.state_updater.latest_report


def test_agent_resets_verification_when_goal_changes(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
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
        {"actions": [{"type": "verify", "kind": "test", "method": "run tests", "criteria": ["tests pass"], "status": "passed", "context": "tests pass"}]}
    )

    assert agent.blackboard.verification.goal == "new goal"
    assert agent.blackboard.verification.status == VerificationStatus.DONE
    assert agent.blackboard.verification.kind == "test"
    assert agent.blackboard.verification.method == "run tests"
    assert agent.blackboard.verification.criteria == ["tests pass"]
    assert agent.blackboard.verification.context == "tests pass"

    agent.apply_response({"actions": [{"type": "goal", "text": "new goal", "complete": True}]})

    assert agent.blackboard.goal_reached is True


def test_agent_task_code_returns_to_working_after_verification_result(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    agent.blackboard.task_code = nanocode.TaskCode.VERIFYING

    agent.apply_response({"actions": [{"type": "verify", "status": "passed", "context": "checked"}]})

    assert agent.blackboard.task_code == nanocode.TaskCode.WORKING
    assert agent.blackboard.verification.status == VerificationStatus.DONE


def test_agent_accepts_combined_verification_kind_and_ignores_pending(tmp_path):
    agent = Agent(Session(cwd=str(tmp_path)))

    agent.apply_response(
        {
            "actions": [
                {
                    "type": "verify",
                    "kind": "syntax_check+test",
                    "method": "check edit",
                    "criteria": ["syntax passes", "tests pass"],
                    "status": "passed",
                }
            ]
        }
    )

    assert agent.blackboard.verification.kind == "syntax_check+test"
    assert agent.blackboard.verification.status == VerificationStatus.DONE

    agent.blackboard.verification.reset()
    result = agent.handle_response(
        {
            "actions": [
                {
                    "type": "verify",
                    "kind": "syntax_check+test",
                    "method": "check edit",
                    "criteria": ["syntax passes", "tests pass"],
                    "status": "pending",
                }
            ]
        }
    )

    assert result.done is False
    assert agent.blackboard.verification.status == VerificationStatus.IDLE
    assert agent.blackboard.verification.kind == ""
    assert any('ignored verify status="pending"' in error for error in agent.agent_feedback_errors)


def test_agent_execute_tool_calls_requests_confirmation_for_edit_tools(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("old\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
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
    agent = Agent(session)

    latest = agent.execute_tool_calls(
        [{"name": "Edit", "intention": "edit sample", "args": ["sample.txt", "old", "new"]}],
        confirm=lambda call, tool: "please inspect tests first",
    )

    assert "Cancelled: user refused: please inspect tests first" in latest
    assert path.read_text(encoding="utf-8") == "old\n"
    assert session.state.conversation == []
    assert os.path.isdir(session.tool_results_dir())
    assert any("please inspect tests first" in error for error in agent.agent_feedback_errors)
    assert "please inspect tests first" in agent.build_user_prompt()


def test_agent_execute_tool_calls_stops_batch_after_refusal(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("old\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)

    latest = agent.execute_tool_calls(
        [
            {"name": "Edit", "intention": "edit sample", "args": ["sample.txt", "old", "new"]},
            {"name": "Bash", "intention": "should not run", "args": ["touch should-not-exist"]},
        ],
        confirm=lambda call, tool: "use English question",
    )

    assert "Cancelled: user refused: use English question" in latest
    assert "Bash" not in latest
    assert [execution.call.name for execution in agent.tool_runner.latest_executions] == ["Edit"]
    assert path.read_text(encoding="utf-8") == "old\n"
    assert not (tmp_path / "should-not-exist").exists()


def test_agent_execute_tool_calls_skips_after_first_failure(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("old\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)

    latest = agent.execute_tool_calls(
        [
            {"name": "Bash", "intention": "fail first", "args": ["exit 7"]},
            {"name": "Bash", "intention": "should not run after failure", "args": ["touch should-not-exist"]},
        ],
        confirm=lambda call, tool: True,
    )

    assert [execution.outcome for execution in agent.tool_runner.latest_executions] == ["failure"]
    assert [execution.result_key for execution in agent.tool_runner.latest_executions] == ["tr.1"]
    assert list(session.state.tool_result_store) == ["tr.1"]
    assert agent.tool_runner.skipped_after_failure_count == 1
    assert agent.tool_runner.skipped_after_failure_key == "tr.1"
    assert "exit_code: 7" in latest
    assert "should not run after failure" not in latest
    assert "Skipped" not in latest
    assert not (tmp_path / "should-not-exist").exists()


def test_agent_execute_tool_calls_rejects_failed_preview_before_confirmation(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("old\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    confirmations = []

    latest = agent.execute_tool_calls(
        [{"name": "ReplaceRange", "intention": "edit stale range", "args": ["sample.txt", [["0", "1", "bad", "", "", "new"]]]}],
        confirm=lambda call, tool: confirmations.append((call.executed, tool.preview())) or True,
    )

    assert confirmations == []
    assert "ToolCallError: preview unavailable: fingerprint mismatch" in latest
    assert path.read_text(encoding="utf-8") == "old\n"


def test_agent_execute_tool_calls_returns_malformed_tool_call_error(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)

    latest = agent.execute_tool_calls([{"intention": "bad call", "args": []}])

    assert "ToolCallError: tool action missing required field: name" in latest
    assert '{"type":"tool","name":"Read","intention":"...","args":["path"]}' in latest
    assert "InvalidToolCall" in latest
    assert "bad call" not in latest
    assert session.state.conversation == []
    assert os.path.isdir(session.tool_results_dir())


def test_agent_execute_tool_calls_records_arg_errors_in_feedback(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)

    latest = agent.execute_tool_calls([{"name": "Read", "intention": "bad range", "args": ["sample.txt", "bad,1"]}])

    assert "ToolCallError: Read args error: invalid range token" in latest
    assert len(agent.agent_feedback_errors) == 1
    assert 'tool=Read args=["sample.txt","bad,1"]' in agent.agent_feedback_errors[0]
    assert "invalid range token" in agent.agent_feedback_errors[0]


def test_agent_execute_tool_calls_reports_arg_count_details(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)

    latest = agent.execute_tool_calls([{"name": "ReplaceRange", "intention": "bad edit", "args": ["sample.txt", "0", "1", "abc", "", ""]}])

    assert "ToolCallError: requires args: filepath, ranges" in latest
    assert "got 6 args, expected 2, extra: 4" in agent.agent_feedback_errors[0]
    assert "use ReplaceRange for read ranges" in agent.agent_feedback_errors[0]


def test_tool_arg_error_does_not_force_observe(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)

    agent.execute_tool_calls([{"name": "Read", "intention": "bad range", "args": ["sample.txt", "bad,1"]}])

    assert agent.mode == nanocode.AgentMode.ACT
    assert agent.agent_feedback_errors


def test_non_arg_tool_failure_stays_in_act_for_repair(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)

    agent.execute_tool_calls(
        [{"name": "Bash", "intention": "fail command", "args": ["exit 7"]}],
        confirm=lambda call, tool: True,
    )

    assert agent.mode == nanocode.AgentMode.ACT
    assert "exit 7" in _blocks_text(agent.tool_context.latest)


def test_agent_blocks_repeated_identical_failed_tool_call(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    _seed_plan(agent, "read sample")
    action = {"type": "tool", "name": "Read", "intention": "bad range", "args": ["sample.txt", "bad,1"]}

    agent.handle_response({"actions": [action]})
    agent.handle_response({"actions": [{"type": "forget", "source": ["tr.1"], "reason": "failed read has no useful result"}]})
    agent.handle_response({"actions": [action]})
    result = agent.handle_response({"actions": [action]})

    assert result.done is False
    assert session.state.tool_result_counter == 2
    assert any("repeated failed tool call" in error for error in agent.agent_feedback_errors)


def test_agent_execute_bash_does_not_require_verification(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)

    agent.execute_tool_calls([{"name": "Bash", "intention": "run command", "args": ["true"]}], confirm=lambda call, tool: True)

    assert agent.blackboard.verification_required is False


def test_agent_marks_nonzero_bash_exit_as_failed_tool_call(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)

    latest = agent.execute_tool_calls([{"name": "Bash", "intention": "run failing command", "args": ["exit 7"]}], confirm=lambda call, tool: True)

    assert agent.tool_runner.latest_executions[0].outcome == "failure"
    assert 'fail tool=Bash args=["exit 7"] key=tr.1' in latest
    assert "* exit_code: 7" in agent.tool_runner.latest_executions[0].output


def test_agent_execute_tool_calls_does_not_record_runtime_errors_in_feedback(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)

    latest = agent.execute_tool_calls([{"name": "Read", "intention": "missing file", "args": ["missing.txt", "0,1"]}])

    assert "ToolCallError: " in latest
    assert agent.agent_feedback_errors == []


def test_main_agent_accepts_search_tool(tmp_path):
    (tmp_path / "sample.py").write_text("class Foo:\n    pass\n", encoding="utf-8")
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)

    latest = agent.execute_tool_calls([{"name": "Search", "intention": "find symbol", "args": ["class Foo"]}])

    assert '- ok tool=Search args=["class Foo"] key=tr.1' in latest
    assert "sample.py" in latest


def test_agent_execute_tool_calls_shows_auto_approval_in_yolo_mode(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("old\n", encoding="utf-8")
    session = _session(tmp_path, yolo=True)
    agent = Agent(session)
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
    assert agent.blackboard.task_code == nanocode.TaskCode.VERIFYING
    assert agent.recent_edits == ["- sample.txt: edit sample"]


def test_agent_run_loops_tool_results_into_next_model_prompt(tmp_path):
    (tmp_path / "sample.txt").write_text("alpha\n", encoding="utf-8")

    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {
                    "actions": [{"type": "tool", "name": "Read", "intention": "read sample", "args": ["sample.txt", "0,1"]}]
                },
                {"actions": [{"type": "keep", "source": ["tr.1"], "reason": "keep useful result"}]},
                {
                    "actions": [
                        {
                            "type": "verify",
                            "method": "unit",
                            "status": "passed",
                            "context": "checked",
                        },
                        {"type": "known", "items": ["Read sample.txt and found alpha."]},
                        {"type": "goal", "text": "read sample", "complete": True, "message_for_complete": "done"},
                    ],
                },
            ]

        def request(self, system_prompt, user_prompt, *, activity="agent", **_kwargs):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    _seed_plan(agent, "read sample")
    fake_client = FakeModelClient()
    agent.model_client = fake_client

    messages = []
    response = agent.run("read sample", on_message=messages.append)

    assert response["actions"][-1]["message_for_complete"] == "done"
    assert messages[0].startswith("[success] Read sample.txt 0,1 -> tr.1")
    assert "why:" not in messages[0]
    assert "log: .nanocode/sessions/" not in messages[0]
    assert messages[-1] == "done"
    assert len(fake_client.user_prompts) == 3
    assert "<ReadToolResult>" in fake_client.user_prompts[1]
    assert "alpha" in fake_client.user_prompts[2]
    assert "Kept Tool Results:" in fake_client.user_prompts[2]
    assert "<ReadToolResult>" in fake_client.user_prompts[2]
    assert 'tool=Read args=["sample.txt","0,1"]' in _blocks_text(agent.tool_context.latest)
    assert agent.tool_context.recent == []
    assert agent.blackboard.known == ["Read sample.txt and found alpha."]
    assert agent.blackboard.user_input == "read sample"
    assert agent.blackboard.goal == "read sample"
    assert agent.blackboard.plan == [nanocode.PlanItem(text="test plan", status=nanocode.PlanStatus.DONE, context="seeded")]
    assert agent.blackboard.verification.status == VerificationStatus.DONE
    assert agent.blackboard.goal_reached is False
    assert agent.blackboard.verification_required is False


def test_agent_run_ingests_queued_user_input_before_next_model_call(tmp_path):
    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {"actions": [{"type": "goal", "text": "initial task"}]},
                {"actions": [{"type": "known", "items": ["queued feedback was visible"]}]},
                {"actions": [{"type": "goal", "complete": True, "message_for_complete": "done"}]},
            ]

        def request(self, system_prompt, user_prompt, *, activity="agent", **_kwargs):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    queued_inputs = [None, "use chinese", None]
    messages = []
    agent = Agent(Session(cwd=str(tmp_path)))
    agent.model_client = FakeModelClient()

    response = agent.run("initial task", on_message=messages.append, poll_user_input=lambda: queued_inputs.pop(0) if queued_inputs else None)

    assert response["actions"][0]["message_for_complete"] == "done"
    assert messages == ["Goal Updated\n  initial task", "sent: use chinese", "Known Updated\n  1. queued feedback was visible", "done"]
    assert [item.content for item in agent.session.state.conversation if isinstance(item, nanocode.UserMessage)] == ["initial task", "use chinese"]
    assert agent.blackboard.user_input == "use chinese"
    assert "use chinese" not in agent.model_client.user_prompts[0]
    assert "use chinese" in agent.model_client.user_prompts[1]
    assert "Pending User Feedback:\nuse chinese" in agent.model_client.user_prompts[1]
    assert "Pending User Feedback:\n(empty)" in agent.model_client.user_prompts[2]
    assert "Latest User Request:" in agent.model_client.user_prompts[1]


def test_agent_plan_mode_tool_gate_allows_only_readonly_tools(tmp_path):
    agent = Agent(_session(tmp_path, plan_mode=True))

    assert agent._plan_mode_tool_error([{"type": "tool", "name": "Read", "args": ["sample.txt"]}]) == ""
    assert agent._plan_mode_tool_error([{"type": "tool", "name": "Git", "args": ["status", "--short"]}]) == ""
    assert "blocked tool=Bash" in agent._plan_mode_tool_error([{"type": "tool", "name": "Bash", "args": ["echo hi"]}])
    assert "blocked tool=Edit" in agent._plan_mode_tool_error([{"type": "tool", "name": "Edit", "args": ["sample.txt", "old", "new"]}])
    assert "blocked tool=Git" in agent._plan_mode_tool_error([{"type": "tool", "name": "Git", "args": ["commit", "-m", "x"]}])
    assert "blocked tool=Lsp" in agent._plan_mode_tool_error([{"type": "tool", "name": "Lsp", "args": ["symbols"]}])


def test_agent_plan_mode_rejects_mutating_tool_before_execution(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("old\n", encoding="utf-8")
    agent = Agent(_session(tmp_path, plan_mode=True, debug=True))
    _seed_plan(agent, "plan change")
    messages = []

    result = agent.handle_response(
        {"actions": [{"type": "tool", "name": "Edit", "intention": "change sample", "args": ["sample.txt", "old", "new"]}]},
        confirm=lambda call, tool: True,
        on_message=messages.append,
    )

    assert result.done is False
    assert path.read_text(encoding="utf-8") == "old\n"
    assert agent.tool_runner.latest_executions == []
    assert messages == ['PlanMode_Gate: plan mode allows readonly discovery only; blocked tool=Edit args=["sample.txt","old","new"].']


def test_agent_plan_mode_rejects_invalid_action_instead_of_completing(tmp_path):
    agent = Agent(_session(tmp_path, plan_mode=True, debug=True))
    messages = []

    result = agent.handle_response({"actions": [{"type": "invalid", "text": "done"}]}, on_message=messages.append)

    assert result.done is False
    assert agent.session.state.conversation == []
    assert messages == ["Protocol_Gate: invalid action type(s): invalid."]


def test_agent_normalizes_direct_repo_tool_action_type(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("old\n", encoding="utf-8")
    agent = Agent(_session(tmp_path, debug=True))
    _seed_plan(agent, "change sample")
    messages = []

    result = agent.handle_response(
        {"actions": [{"type": "Edit", "intention": "change sample", "args": ["sample.txt", "old", "new"]}]},
        confirm=lambda call, tool: True,
        on_message=messages.append,
    )

    assert result.done is False
    assert path.read_text(encoding="utf-8") == "new\n"
    assert agent.tool_runner.latest_executions[0].call.name == "Edit"
    assert not any("Protocol_Gate" in message for message in messages)


def test_agent_plan_mode_stores_proposed_plan_completion(tmp_path):
    agent = Agent(_session(tmp_path, plan_mode=True))
    _seed_plan(agent, "plan change")
    message = "<proposed_plan>\n1. Inspect target.\n2. Patch code.\n3. Run tests.\n</proposed_plan>"

    result = agent.handle_response({"actions": [{"type": "goal", "text": "plan change", "complete": True, "message_for_complete": message}]})

    assert result.done is True
    assert isinstance(agent.session.state.conversation[-1], nanocode.AssistantMessage)
    assert agent.session.state.conversation[-1].content == message


def test_agent_plan_mode_requires_proposed_plan_completion_block(tmp_path):
    agent = Agent(_session(tmp_path, plan_mode=True, debug=True))
    _seed_plan(agent, "plan change")
    messages = []

    result = agent.handle_response(
        {"actions": [{"type": "goal", "text": "plan change", "complete": True, "message_for_complete": "plain plan"}]},
        on_message=messages.append,
    )

    assert result.done is False
    assert not agent.session.state.conversation
    assert messages == ["PlanMode_Gate: final plan must be wrapped in <proposed_plan>...</proposed_plan>."]


def test_agent_run_allows_readonly_answer_without_verification(tmp_path):
    (tmp_path / "sample.txt").write_text("alpha\n", encoding="utf-8")

    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {"actions": [{"type": "tool", "name": "Read", "intention": "read sample", "args": ["sample.txt", "0,1"]}]},
                {
                    "actions": [
                        {"type": "goal", "text": "answer sample", "complete": True, "message_for_complete": "sample contains alpha"},
                    ],
                },
            ]

        def request(self, system_prompt, user_prompt, *, activity="agent", **_kwargs):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    agent = Agent(Session(cwd=str(tmp_path)))
    agent.model_client = FakeModelClient()
    messages = []

    response = agent.run("answer sample", on_message=messages.append)

    assert response["actions"][-1]["message_for_complete"] == "sample contains alpha"
    assert "Retrying: verification must pass before completion." not in messages
    assert messages[-1] == "sample contains alpha"


def test_agent_run_executes_edit_tool_and_requires_verification(tmp_path):
    (tmp_path / "sample.txt").write_text("old\n", encoding="utf-8")

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
                {"actions": [{"type": "keep", "source": ["tr.1"], "reason": "keep useful result"}]},
                {"actions": [{"type": "goal", "text": "change sample", "complete": True, "message_for_complete": "done"}]},
                {"actions": [{"type": "tool", "name": "Read", "intention": "inspect changed sample", "args": ["sample.txt", "0,1"]}]},
                {"actions": [{"type": "keep", "source": ["tr.2"], "reason": "keep useful result"}]},
                {
                    "actions": [
                        {"type": "verify", "kind": "change_check", "method": "Read sample.txt", "criteria": ["sample text is new"], "status": "passed", "context": "sample.txt contains new"},
                        {"type": "goal", "text": "change sample", "complete": True, "message_for_complete": "done"},
                    ]
                },
            ]

        def request(self, system_prompt, user_prompt, *, activity="agent", **_kwargs):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    _seed_plan(agent, "change sample")
    agent.model_client = FakeModelClient()
    messages = []

    response = agent.run("change sample", confirm=lambda call, tool: True, on_message=messages.append)

    assert response["actions"][-1]["message_for_complete"] == "done"
    assert any(message.startswith("[success] Edit sample.txt") for message in messages)
    assert any(message.startswith("[success] Read sample.txt") for message in messages)
    assert not any(message.startswith("State Updated") for message in messages)
    assert agent.blackboard.verification.status == VerificationStatus.DONE
    assert agent.blackboard.verification.context == "sample.txt contains new"
    assert (tmp_path / "sample.txt").read_text(encoding="utf-8") == "new\n"
    assert messages[-1] == "done"


def test_agent_reports_edit_verification_gate_in_debug(tmp_path):
    agent = Agent(_session(tmp_path, debug=True))
    _seed_plan(agent, "change sample")
    agent.blackboard.goal_reached = True
    agent.blackboard.verification_required = True
    agent.blackboard.verification.status = VerificationStatus.REQUIRED
    ctx = agent._build_response_context({"actions": [{"type": "goal", "text": "change sample", "complete": True, "message_for_complete": "done"}]})
    messages = []

    result = agent._finish_or_continue(ctx, messages.append)

    assert result.done is False
    assert messages == ["Verification_Gate: edit completion requires verification."]


def test_agent_plain_text_cannot_finish_when_verification_required(tmp_path):
    agent = Agent(Session(cwd=str(tmp_path)))
    agent.blackboard.verification_required = True
    agent.blackboard.verification.status = VerificationStatus.REQUIRED
    agent.blackboard.task_code = nanocode.TaskCode.NEW
    ctx = agent._build_response_context({"actions": [], "_assistant_text": "Done."})
    messages = []

    result = agent._handle_text_response(ctx, messages.append)

    assert result is not None
    assert result.done is False
    assert agent.blackboard.task_code == nanocode.TaskCode.VERIFYING
    assert agent.agent_feedback_errors == [
        'Warning: assistant text cannot finish while verification is required. Rule: run verification tools, then report verify status="passed"|"failed"|"blocked".'
    ]
    assert messages == ["Done."]


def test_agent_run_keeps_tool_results_when_format_retry_happens(tmp_path, monkeypatch):
    (tmp_path / "sample.txt").write_text("alpha\n", encoding="utf-8")

    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {"actions": [{"type": "tool", "name": "Read", "intention": "read sample", "args": ["sample.txt", "0,1"]}]},
                {"_format_error": "Invalid function-tool response: plain answer", "actions": []},
                {"actions": [{"type": "keep", "source": ["tr.1"], "reason": "keep useful result"}]},
                {"actions": _final_actions("read sample")},
            ]

        def request(self, system_prompt, user_prompt, *, activity="agent", **_kwargs):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    _set_context_budget(monkeypatch, agent, observe_after_results=1)
    _seed_plan(agent, "read sample")
    agent.model_client = FakeModelClient()

    response = agent.run("read sample")

    assert response["actions"][-1]["message_for_complete"] == "done"
    assert len(agent.model_client.user_prompts) == 4
    assert "<ReadToolResult>" in agent.model_client.user_prompts[1]
    assert "<ReadToolResult>" in agent.model_client.user_prompts[2]
    assert "Kept Tool Results:" in agent.model_client.user_prompts[3]
    assert "<ReadToolResult>" in agent.model_client.user_prompts[3]
    assert 'tool=Read args=["sample.txt","0,1"]' in _blocks_text(agent.tool_context.latest)
    assert agent.tool_context.recent == []


def test_agent_run_prunes_tool_result_store_when_next_run_starts(tmp_path):
    for index in range(51):
        (tmp_path / f"sample-{index}.txt").write_text(f"line {index}\n", encoding="utf-8")

    class FakeModelClient:
        def __init__(self):
            self.responses = [
                {
                    "actions": [
                        {"type": "tool", "name": "Read", "intention": f"read {index}", "args": [f"sample-{index}.txt", "0,1"]}
                        for index in range(51)
                    ]
                },
                {"actions": [{"type": "forget", "source": ["tr." + str(index) for index in range(1, 52)], "reason": "bulk sample reads are not needed after execution"}]},
                {"actions": _final_actions("read samples")},
            ]

        def request(self, system_prompt, user_prompt, *, activity="agent", **_kwargs):
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    agent.blackboard.goal = "read samples"
    agent.blackboard.plan = [nanocode.PlanItem(text="try answer", status=nanocode.PlanStatus.DONE, context="seeded")]
    agent.blackboard.known = ["keep this fact"]
    agent.blackboard.stable_knowledge = {"workflow": ["Project test command is make test."]}
    agent.tool_context.latest = ["old tool call"]
    agent.model_client = FakeModelClient()

    agent.run("read samples")

    assert len(session.state.tool_result_store) == 51
    assert list(session.state.tool_result_store)[0] == "tr.1"

    agent.model_client.responses = [{"actions": [], "_assistant_text": "ok"}]
    agent.run("next task")

    assert len(session.state.tool_result_store) == 50
    assert list(session.state.tool_result_store)[:2] == ["tr.2", "tr.3"]
    assert list(session.state.tool_result_store)[-1] == "tr.51"
    assert session.state.tool_result_counter == 51
    assert agent.blackboard.goal == "read samples"
    assert agent.blackboard.plan == [nanocode.PlanItem(text="try answer", status=nanocode.PlanStatus.DONE, context="seeded")]
    assert agent.blackboard.known == ["keep this fact"]
    assert agent.blackboard.stable_knowledge == {"workflow": ["Project test command is make test."]}
    assert agent.blackboard.verification.status == VerificationStatus.IDLE
    assert agent.blackboard.goal_reached is False


def test_agent_run_observe_checkpoint_allows_completion_without_known(tmp_path):
    (tmp_path / "sample.txt").write_text("alpha\n", encoding="utf-8")

    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {"actions": [{"type": "tool", "name": "Read", "intention": "read sample", "args": ["sample.txt", "0,1"]}]},
                {"actions": [{"type": "forget", "source": ["tr.1"], "reason": "sample content is not needed"}]},
                {"actions": _final_actions("read sample", "done too early")},
            ]

        def request(self, system_prompt, user_prompt, *, activity="agent", **_kwargs):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    _seed_plan(agent, "read sample")
    agent.model_client = FakeModelClient()
    messages = []

    response = agent.run("read sample", on_message=messages.append)

    assert response["actions"][-1]["message_for_complete"] == "done too early"
    assert "done too early" in messages
    assert len(agent.model_client.user_prompts) == 3
    assert "<ReadToolResult>" in agent.model_client.user_prompts[1]
    assert "<ReadToolResult>" not in agent.model_client.user_prompts[2]


def test_agent_run_allows_readonly_tool_before_plan(tmp_path):
    (tmp_path / "sample.txt").write_text("alpha\n", encoding="utf-8")

    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {
                    "actions": [
                        {"type": "goal", "text": "read sample", "complete": False},
                        {"type": "tool", "name": "Read", "intention": "read sample", "args": ["sample.txt", "0,1"]},
                    ]
                },
                {
                    "actions": [
                        {"type": "plan", "items": [{"id": "p1", "text": "Read sample", "status": "done", "context": "read sample.txt"}]},
                        *_final_actions("read sample"),
                    ]
                },
            ]

        def request(self, system_prompt, user_prompt, *, activity="agent", **_kwargs):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    agent.model_client = FakeModelClient()
    messages = []

    response = agent.run("read sample", on_message=messages.append)

    assert response["actions"][-1]["message_for_complete"] == "done"
    assert "Retrying: create a short plan before tools." not in messages
    assert len(session.state.tool_result_store) == 1
    assert [item.text for item in agent.blackboard.plan] == ["Read sample"]


def test_agent_run_allows_readonly_discovery_when_goal_changes_before_plan(tmp_path):
    (tmp_path / "sample.txt").write_text("alpha\n", encoding="utf-8")

    class FakeModelClient:
        def __init__(self):
            self.responses = [
                {
                    "actions": [
                        {"type": "goal", "text": "new goal", "complete": False},
                        {"type": "tool", "name": "Read", "intention": "read sample", "args": ["sample.txt", "0,1"]},
                    ]
                },
                {
                    "actions": [
                        {"type": "goal", "text": "new goal", "complete": False},
                        {"type": "tool", "name": "Read", "intention": "read sample", "args": ["sample.txt", "0,1"]},
                    ]
                },
                {
                    "actions": [
                        {"type": "goal", "text": "new goal", "complete": False},
                        {
                            "type": "plan",
                            "items": [{"id": "p1", "text": "Read sample", "status": "doing"}],
                        },
                        {"type": "tool", "name": "Read", "intention": "read sample", "args": ["sample.txt", "0,1"]},
                    ]
                },
                {"actions": [{"type": "keep", "source": ["tr.1"], "reason": "keep useful result"}]},
                {
                    "actions": [
                        {"type": "plan", "items": [{"id": "p1", "text": "Read sample", "status": "done", "context": "read sample.txt"}]},
                        *_final_actions("new goal"),
                    ]
                },
            ]

        def request(self, system_prompt, user_prompt, *, activity="agent", **_kwargs):
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    agent.blackboard.goal = "old goal"
    agent.blackboard.plan = [nanocode.PlanItem(id="old", text="Old plan")]
    agent.model_client = FakeModelClient()
    messages = []

    response = agent.run("new goal", on_message=messages.append)

    assert response["actions"][-1]["message_for_complete"] == "done"
    assert "Retrying: new goal requires a fresh plan." not in messages
    assert agent.blackboard.goal == "new goal"
    assert [item.text for item in agent.blackboard.plan] == ["Read sample"]
    assert len(session.state.tool_result_store) == 3


def test_agent_run_requires_task_alignment_before_work_with_old_context(tmp_path):
    (tmp_path / "sample.txt").write_text("alpha\n", encoding="utf-8")

    class FakeModelClient:
        def __init__(self):
            self.responses = [
                {"actions": [{"type": "tool", "name": "Read", "intention": "read sample", "args": ["sample.txt", "0,1"]}]},
                {
                    "actions": [
                        {"type": "goal", "text": "run lint", "complete": False},
                        {
                            "type": "plan",
                            "items": [{"id": "p1", "text": "Read sample", "status": "doing"}],
                        },
                        {"type": "tool", "name": "Read", "intention": "read sample", "args": ["sample.txt", "0,1"]},
                    ]
                },
                {
                    "actions": [
                        {"type": "plan", "items": [{"id": "p1", "text": "Read sample", "status": "done", "context": "read sample.txt"}]},
                        *_final_actions("run lint"),
                    ]
                },
            ]

        def request(self, system_prompt, user_prompt, *, activity="agent", **_kwargs):
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    agent.blackboard.goal = "review dirty changes"
    agent.blackboard.plan = [nanocode.PlanItem(id="p1", text="Review dirty changes", status=nanocode.PlanStatus.DONE, context="reviewed")]
    agent.model_client = FakeModelClient()
    messages = []

    response = agent.run("ruff", on_message=messages.append)

    assert response["actions"][-1]["message_for_complete"] == "done"
    assert agent.blackboard.goal == "run lint"
    assert [item.text for item in agent.blackboard.plan] == ["Read sample"]
    assert "previous task context is still present" in " ".join(agent.agent_feedback_errors)


def test_agent_run_ignores_goal_rewrite_after_task_is_working(tmp_path):
    (tmp_path / "sample.txt").write_text("alpha\n", encoding="utf-8")

    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {
                    "actions": [
                        {"type": "goal", "text": "read sample", "complete": False},
                        {
                            "type": "plan",
                            "items": [{"id": "p1", "text": "Read sample", "status": "doing"}],
                        },
                    ]
                },
                {"actions": [{"type": "goal", "text": "read sample again", "complete": False}]},
                {"actions": [{"type": "tool", "name": "Read", "intention": "read sample", "args": ["sample.txt", "0,1"]}]},
                {"actions": [{"type": "keep", "source": ["tr.1"], "reason": "keep useful result"}]},
                {
                    "actions": [
                        {"type": "plan", "items": [{"id": "p1", "text": "Read sample", "status": "done", "context": "read sample.txt"}]},
                        *_final_actions("read sample"),
                    ]
                },
            ]

        def request(self, system_prompt, user_prompt, *, activity="agent", **_kwargs):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    agent = Agent(Session(cwd=str(tmp_path)))
    agent.model_client = FakeModelClient()

    response = agent.run("read sample")

    assert response["actions"][-1]["message_for_complete"] == "done"
    assert agent.blackboard.goal == "read sample"
    assert [item.text for item in agent.blackboard.plan] == ["Read sample"]
    assert len(agent.tool_runner.latest_executions) == 1
    assert "rewrote Goal after the task was active" in " ".join(agent.agent_feedback_errors)


def test_agent_allows_plan_with_multiple_doing_items(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    agent.blackboard.task_code = nanocode.TaskCode.NEW

    result = agent.handle_response(
        {
            "actions": [
                {"type": "goal", "text": "answer", "complete": False},
                {
                    "type": "plan",
                    "items": [
                        {"id": "p1", "text": "first", "status": "doing"},
                        {"id": "p2", "text": "second", "status": "doing"},
                    ],
                }
            ]
        }
    )

    assert result.done is False
    assert [item.id for item in agent.blackboard.plan] == ["p1", "p2"]
    assert [item.status for item in agent.blackboard.plan] == [nanocode.PlanStatus.DOING, nanocode.PlanStatus.TODO]
    assert agent.agent_feedback_errors == []


def test_agent_ignores_goal_rewrite_after_task_is_working(tmp_path):
    agent = Agent(Session(cwd=str(tmp_path)))
    agent.blackboard.task_code = nanocode.TaskCode.WORKING
    agent.blackboard.goal = "read sample"
    agent.blackboard.plan = [nanocode.PlanItem(id="p1", text="Read sample", status=nanocode.PlanStatus.DOING)]

    result = agent.handle_response({"actions": [{"type": "goal", "text": "read sample again", "complete": False}]})

    assert result.done is False
    assert agent.blackboard.goal == "read sample"
    assert [item.text for item in agent.blackboard.plan] == ["Read sample"]
    assert "rewrote Goal after the task was active" in " ".join(agent.agent_feedback_errors)


def test_agent_run_continues_when_no_tool_calls_and_goal_not_reached(tmp_path):
    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {"actions": [{"type": "goal", "text": "answer", "complete": False}]},
                {"actions": _final_actions()},
            ]

        def request(self, system_prompt, user_prompt, *, activity="agent", **_kwargs):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    _seed_plan(agent, "answer")
    agent.model_client = FakeModelClient()
    messages = []

    response = agent.run("answer", on_message=messages.append)

    assert response["actions"][-1]["message_for_complete"] == "done"
    assert len(agent.model_client.user_prompts) == 2
    assert "Continuing: goal is not complete yet." not in messages
    assert not any(message.startswith("State Updated") for message in messages)


def test_agent_run_stops_after_assistant_text(tmp_path):
    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []

        def request(self, system_prompt, user_prompt, *, activity="agent", **_kwargs):
            self.user_prompts.append(user_prompt)
            return {"actions": [], "_assistant_text": "你好"}

    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    _seed_plan(agent, "你好")
    agent.model_client = FakeModelClient()
    messages = []

    response = agent.run("你好", on_message=messages.append)

    assert response == {"actions": [], "_assistant_text": "你好"}
    assert messages == ["你好"]
    assert len(agent.model_client.user_prompts) == 1
    assert agent.blackboard.task_code == nanocode.TaskCode.DONE


def test_agent_run_does_not_report_continuation_for_action_only_turn(tmp_path):
    class FakeModelClient:
        def __init__(self):
            self.responses = [
                {"actions": [{"type": "plan", "mode": "patch", "items": []}]},
                {"actions": _final_actions()},
            ]

        def request(self, system_prompt, user_prompt, *, activity="agent", **_kwargs):
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    _seed_plan(agent, "answer")
    agent.model_client = FakeModelClient()
    messages = []

    response = agent.run("answer", on_message=messages.append)

    assert response["actions"][-1]["message_for_complete"] == "done"
    assert "Continuing: goal is not complete yet." not in messages


def test_main_agent_accepts_memory_actions_during_act_turn(tmp_path):
    class FakeModelClient:
        def __init__(self):
            self.responses = [
                {"actions": [{"type": "known", "items": ["fact"]}]},
                {"actions": _final_actions()},
            ]

        def request(self, system_prompt, user_prompt, *, activity="agent", **_kwargs):
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    _seed_plan(agent, "answer")
    agent.model_client = FakeModelClient()

    response = agent.run("answer")

    assert response["actions"][-1]["message_for_complete"] == "done"
    assert agent.blackboard.known == ["fact"]
    assert any("state update-only turn" in error for error in agent.agent_feedback_errors)


def test_agent_warns_when_discovery_runs_long_without_plan(tmp_path, monkeypatch):
    agent = Agent(Session(cwd=str(tmp_path)))
    agent.blackboard.goal = "investigate"
    _set_context_budget(monkeypatch, agent, planless_discovery_tool_calls=2)

    agent.handle_response({"actions": [{"type": "tool", "name": "ListDir", "intention": "inspect root", "args": ["."]}]})
    agent.handle_response({"actions": [{"type": "tool", "name": "ListDir", "intention": "inspect root again", "args": ["."]}]})

    assert any("Plan is empty after discovery" in error for error in agent.agent_feedback_errors)


def test_agent_run_reports_continuation_only_when_no_actions(tmp_path):
    class FakeModelClient:
        def __init__(self):
            self.responses = [
                {"actions": []},
                {"actions": _final_actions()},
            ]

        def request(self, system_prompt, user_prompt, *, activity="agent", **_kwargs):
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    _seed_plan(agent, "answer")
    agent.model_client = FakeModelClient()
    messages = []

    response = agent.run("answer", on_message=messages.append)

    assert response["actions"][-1]["message_for_complete"] == "done"
    assert "Continuing: assistant must set current task's goal." not in messages


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

        def request(self, system_prompt, user_prompt, *, activity="agent", **_kwargs):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    _seed_plan(agent, "answer")
    agent.model_client = FakeModelClient()
    messages = []

    response = agent.run("change file", on_message=messages.append)

    assert response["actions"][-1]["message_for_complete"] == "done"
    assert len(agent.model_client.user_prompts) == 3
    assert "Retrying: verification is done but goal is not complete." not in messages
    assert agent.blackboard.verification.status == VerificationStatus.DONE


def test_agent_run_retries_when_plan_complete_without_verification(tmp_path):
    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {
                    "actions": [
                        {
                            "type": "plan",
                            "items": [{"id": "p1", "text": "run tests", "status": "done", "context": "tests passed"}],
                        }
                    ]
                },
                {
                    "actions": [
                        {
                            "type": "verify",
                            "kind": "test",
                            "method": "pytest",
                            "criteria": ["tests pass"],
                            "status": "passed",
                            "context": "tests passed",
                        }
                    ]
                },
                {"actions": _final_actions("change file")},
            ]

        def request(self, system_prompt, user_prompt, *, activity="agent", **_kwargs):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    agent.blackboard.goal = "change file"
    agent.blackboard.plan = [nanocode.PlanItem(id="p1", text="run tests", status=nanocode.PlanStatus.DOING)]
    agent.model_client = FakeModelClient()

    response = agent.run("change file")

    assert response["actions"][-1]["message_for_complete"] == "done"
    assert len(agent.model_client.user_prompts) == 3
    assert any("Plan is complete but verification is not recorded" in error for error in agent.agent_feedback_errors)
    assert agent.blackboard.verification.status == VerificationStatus.DONE


def test_agent_allows_tool_after_completed_plan_and_verification(tmp_path):
    (tmp_path / "sample.txt").write_text("alpha\n", encoding="utf-8")
    agent = Agent(_session(tmp_path, debug=True))
    _seed_plan(agent, "inspect")
    agent.blackboard.verification.status = VerificationStatus.DONE
    agent.blackboard.verification.context = "syntax check passed"
    messages = []

    result = agent.handle_response(
        {
            "actions": [
                {"type": "tool", "name": "Read", "intention": "inspect again", "args": ["sample.txt", "0,1"]}
            ]
        },
        on_message=messages.append,
    )

    assert result.done is False
    assert len(agent.tool_runner.latest_executions) == 1
    assert agent.tool_runner.latest_executions[0].outcome == "success"
    assert not any("Completion_Gate: completed plan and verification" in message for message in messages)
    assert any("Plan and verification are complete" in error for error in agent.agent_feedback_errors)


def test_agent_allows_tool_after_reopening_completed_plan_with_context(tmp_path):
    (tmp_path / "sample.txt").write_text("alpha\n", encoding="utf-8")
    agent = Agent(Session(cwd=str(tmp_path)))
    _seed_plan(agent, "inspect")
    agent.blackboard.verification.status = VerificationStatus.DONE
    agent.blackboard.verification.context = "syntax check passed"

    result = agent.handle_response(
        {
            "actions": [
                {
                    "type": "plan",
                    "mode": "patch",
                    "items": [
                        {
                            "id": "p2",
                            "text": "Inspect the remaining issue",
                            "status": "doing",
                            "context": "user reported the visual state still looks wrong",
                        }
                    ],
                },
                {"type": "tool", "name": "Read", "intention": "inspect sample", "args": ["sample.txt", "0,1"]},
            ]
        }
    )

    assert result.done is False
    assert len(agent.tool_runner.latest_executions) == 1
    assert agent.tool_runner.latest_executions[0].outcome == "success"
    assert agent.blackboard.plan[-1] == nanocode.PlanItem(
        id="p2",
        text="Inspect the remaining issue",
        status=nanocode.PlanStatus.DOING,
        context="user reported the visual state still looks wrong",
    )


def test_agent_allows_tool_after_reopening_completed_plan_without_context(tmp_path):
    (tmp_path / "sample.txt").write_text("alpha\n", encoding="utf-8")
    agent = Agent(_session(tmp_path, debug=True))
    _seed_plan(agent, "inspect")
    agent.blackboard.verification.status = VerificationStatus.DONE
    agent.blackboard.verification.context = "syntax check passed"
    messages = []

    result = agent.handle_response(
        {
            "actions": [
                {
                    "type": "plan",
                    "mode": "patch",
                    "items": [{"id": "p2", "text": "Inspect the remaining issue", "status": "doing"}],
                },
                {"type": "tool", "name": "Read", "intention": "inspect sample", "args": ["sample.txt", "0,1"]},
            ]
        },
        on_message=messages.append,
    )

    assert result.done is False
    assert len(agent.tool_runner.latest_executions) == 1
    assert agent.tool_runner.latest_executions[0].outcome == "success"
    assert not any("Completion_Gate: reopened plan item missing context" in message for message in messages)
    assert any("Continuing tools after completed Plan" in error for error in agent.agent_feedback_errors)


def test_agent_blocks_verify_blocked_completion_without_manual_context(tmp_path):
    agent = Agent(_session(tmp_path, debug=True))
    _seed_plan(agent, "verify")
    messages = []

    result = agent.handle_response(
        {
            "actions": [
                {"type": "verify", "status": "blocked", "context": "pytest unavailable"},
                {"type": "goal", "text": "verify", "complete": True, "message_for_complete": "done"},
            ]
        },
        on_message=messages.append,
    )

    assert result.done is False
    assert messages[-1] == "Verification_Gate: verify blocked requires blocker=user before completion."
    assert not agent.session.state.conversation


def test_agent_allows_verify_blocked_completion_with_user_blocker(tmp_path):
    agent = Agent(Session(cwd=str(tmp_path)))
    _seed_plan(agent, "verify")
    messages = []

    result = agent.handle_response(
        {
            "actions": [
                {
                    "type": "verify",
                    "status": "blocked",
                    "blocker": "user",
                    "context": "needs user confirmation",
                },
                {"type": "goal", "text": "verify", "complete": True, "message_for_complete": "done"},
            ]
        },
        on_message=messages.append,
    )

    assert result.done is True
    assert agent.blackboard.verification.blocker == nanocode.VerificationBlocker.USER
    assert messages[-1] == "done"


def test_agent_run_uses_fallback_when_goal_complete_has_no_message(tmp_path):
    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {"actions": [{"type": "goal", "text": "answer", "complete": True}]},
            ]

        def request(self, system_prompt, user_prompt, *, activity="agent", **_kwargs):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    _seed_plan(agent, "answer")
    agent.model_client = FakeModelClient()
    messages = []

    response = agent.run("answer", on_message=messages.append)

    assert response["actions"][-1] == {"type": "goal", "text": "answer", "complete": True}
    assert messages[-1] == "Done."
    assert len(agent.model_client.user_prompts) == 1
    assert "Retrying: goal is complete but message_for_complete is missing." not in messages
    assert agent.agent_feedback_errors
    assert agent.blackboard.goal_reached is False


def test_agent_run_allows_assistant_text_without_task_context(tmp_path):
    class FakeModelClient:
        def request(self, system_prompt, user_prompt, *, activity="agent", **_kwargs):
            return {"actions": [], "_assistant_text": "hello"}

    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    agent.model_client = FakeModelClient()
    messages = []

    response = agent.run("hi", on_message=messages.append)

    assert response == {"actions": [], "_assistant_text": "hello"}
    assert messages == ["hello"]
    assert session.state.conversation[-1].content == "hello"


def test_agent_run_treats_assistant_text_as_progress_with_unfinished_task_context(tmp_path):
    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {"actions": [], "_assistant_text": "done too early"},
                {
                    "actions": [
                        {"type": "plan", "mode": "patch", "items": [{"id": "p1", "status": "done", "context": "answered"}]},
                        *_final_actions("answer"),
                    ]
                },
            ]

        def request(self, system_prompt, user_prompt, *, activity="agent", **_kwargs):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    agent.blackboard.goal = "answer"
    agent.blackboard.task_code = nanocode.TaskCode.WORKING
    agent.blackboard.plan = [nanocode.PlanItem(id="p1", text="answer", status=nanocode.PlanStatus.DOING)]
    agent.model_client = FakeModelClient()
    messages = []

    response = agent.run("answer", on_message=messages.append)

    assert response["actions"][-1]["message_for_complete"] == "done"
    assert messages[-1] == "done"
    assert "done too early" in messages
    assert len(agent.model_client.user_prompts) == 2
    assert "done too early" in [item.content for item in session.state.conversation]
    assert not any("assistant text cannot finish an active task" in error for error in agent.agent_feedback_errors)


def test_agent_run_retries_goal_complete_with_unfinished_plan(tmp_path):
    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {"actions": [{"type": "goal", "text": "answer", "complete": True, "message_for_complete": "done"}]},
                {
                    "actions": [
                        {
                            "type": "plan",
                            "items": [{"id": "p1", "text": "answer", "status": "done", "context": "answered"}],
                        },
                        {"type": "goal", "text": "answer", "complete": True, "message_for_complete": "done"},
                    ]
                },
            ]

        def request(self, system_prompt, user_prompt, *, activity="agent", **_kwargs):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    agent.blackboard.goal = "answer"
    agent.blackboard.plan = [nanocode.PlanItem(id="p1", text="answer", status=nanocode.PlanStatus.DOING)]
    agent.model_client = FakeModelClient()
    messages = []

    response = agent.run("answer", on_message=messages.append)

    assert response["actions"][-1]["message_for_complete"] == "done"
    assert len(agent.model_client.user_prompts) == 2
    assert any("before Plan was complete" in error for error in agent.agent_feedback_errors)
    assert agent.blackboard.plan == [nanocode.PlanItem(id="p1", text="answer", status=nanocode.PlanStatus.DONE, context="answered")]


def test_investigate_completion_requires_root_cause_hypothesis(tmp_path):
    agent = Agent(_session(tmp_path, debug=True))
    _seed_plan(agent, "find bug")
    agent.blackboard.work_mode = nanocode.WorkMode.INVESTIGATE
    messages = []

    result = agent.handle_response(
        {
            "actions": [
                _verify_passed_action(),
                {"type": "goal", "text": "find bug", "complete": True, "message_for_complete": "done"},
            ]
        },
        on_message=messages.append,
    )

    assert result.done is False
    assert agent.blackboard.goal_reached is False
    assert any("confirmed hypothesis" in error for error in agent.agent_feedback_errors)
    assert messages[-1] == "Completion_Gate: investigate completion requires a confirmed hypothesis."

    result = agent.handle_response(
        {
            "actions": [
                {
                    "type": "hypothesis",
                    "items": [{"id": "h1", "text": "bad admin filter", "status": "confirmed", "source": ["tr.1"]}],
                },
                _verify_passed_action(),
                {"type": "goal", "text": "find bug", "complete": True, "message_for_complete": "done"},
            ]
        },
        on_message=messages.append,
    )

    assert result.done is True
    assert agent.blackboard.hypotheses[0].status == nanocode.HypothesisStatus.CONFIRMED
    assert messages[-1] == "done"


def test_goal_declares_investigate_work_mode(tmp_path):
    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []

        def request(self, system_prompt, user_prompt, *, activity="agent", **_kwargs):
            self.user_prompts.append(user_prompt)
            return {
                "actions": [
                    {
                        "type": "goal",
                        "text": "find bug",
                        "work_mode": "investigate",
                        "complete": False,
                    },
                    {
                        "type": "plan",
                        "items": [{"id": "p1", "text": "identify root cause", "status": "done", "context": "reasoned"}],
                    },
                    {"type": "hypothesis", "items": [{"id": "h1", "text": "bad filter", "status": "confirmed", "source": ["tr.1"]}]},
                    _verify_passed_action(),
                    {"type": "goal", "text": "find bug", "complete": True, "message_for_complete": "done"},
                ]
            }

    agent = Agent(Session(cwd=str(tmp_path)))
    agent.model_client = FakeModelClient()

    result = agent.run("为什么 admin history 不出现")

    assert result["actions"][-1]["message_for_complete"] == "done"
    assert agent.blackboard.work_mode == nanocode.WorkMode.INVESTIGATE
    assert "Work Mode:\nnormal" in agent.model_client.user_prompts[0]


def test_agent_run_retries_goal_complete_when_plan_done_without_context(tmp_path):
    class FakeModelClient:
        def __init__(self):
            self.responses = [
                {
                    "actions": [
                        {"type": "plan", "items": [{"id": "p1", "text": "answer", "status": "done"}]},
                        {"type": "goal", "text": "answer", "complete": True, "message_for_complete": "done"},
                    ]
                },
                {
                    "actions": [
                        {
                            "type": "plan",
                            "items": [{"id": "p1", "text": "answer", "status": "done", "context": "answered"}],
                        },
                        {"type": "goal", "text": "answer", "complete": True, "message_for_complete": "done"},
                    ]
                },
            ]

        def request(self, system_prompt, user_prompt, *, activity="agent", **_kwargs):
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    agent.blackboard.goal = "answer"
    agent.blackboard.plan = [nanocode.PlanItem(id="p1", text="answer", status=nanocode.PlanStatus.DOING)]
    agent.model_client = FakeModelClient()
    messages = []

    response = agent.run("answer", on_message=messages.append)

    assert response["actions"][-1]["message_for_complete"] == "done"
    assert any("before Plan was complete" in error for error in agent.agent_feedback_errors)
    assert agent.agent_feedback_errors
    assert agent.blackboard.plan == [nanocode.PlanItem(id="p1", text="answer", status=nanocode.PlanStatus.DONE, context="answered")]


def test_agent_run_retries_format_error_with_tool_result_context(tmp_path):
    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {"_format_error": "Invalid function-tool response: plain answer", "actions": []},
                {"_format_error": "Invalid function-tool response: plain answer", "actions": []},
                {"actions": _final_actions()},
            ]

        def request(self, system_prompt, user_prompt, *, activity="agent", **_kwargs):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    agent.model_client = FakeModelClient()
    messages = []

    response = agent.run("answer", on_message=messages.append)

    assert response["actions"][-1]["message_for_complete"] == "done"
    assert len(agent.model_client.user_prompts) == 3
    assert "Retrying: invalid function/tool response: plain answer" not in messages
    assert messages[-1] == "done"


def test_agent_feedback_survives_goal_complete_until_next_run(tmp_path):
    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {"_format_error": "Invalid function-tool response: plain answer", "actions": []},
                {"actions": [{"type": "goal", "text": "answer", "complete": False}]},
                {"actions": _final_actions()},
            ]

        def request(self, system_prompt, user_prompt, *, activity="agent", **_kwargs):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    agent.model_client = FakeModelClient()

    response = agent.run("answer")

    assert response["actions"][-1]["message_for_complete"] == "done"
    assert len(agent.model_client.user_prompts) == 3
    assert agent.agent_feedback_errors

    class ChatModelClient:
        def request(self, system_prompt, user_prompt, *, activity="agent", **_kwargs):
            return {"actions": [], "_assistant_text": "ok"}

    agent.model_client = ChatModelClient()
    agent.run("next task")

    assert agent.agent_feedback_errors == []
    assert agent.blackboard.verification.status == VerificationStatus.IDLE


def test_agent_allows_progress_message_before_goal_complete(tmp_path):
    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {
                    "actions": [
                        {
                            "type": "verify",
                            "kind": "light",
                            "method": "check",
                            "criteria": ["progress can be emitted before completion"],
                            "status": "passed",
                            "blocker": None,
                            "context": "progress context",
                        }
                    ],
                    "_assistant_text": "progress",
                },
                {"actions": _final_actions()},
            ]

        def request(self, system_prompt, user_prompt, *, activity="agent", **_kwargs):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    _seed_plan(agent)
    agent.model_client = FakeModelClient()

    messages = []
    response = agent.run("answer", on_message=messages.append)

    assert response["actions"][-1]["message_for_complete"] == "done"
    assert "progress" in messages
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
                        {"type": "tool", "name": "Read", "intention": "read sample", "args": ["sample.txt"]},
                    ],
                    "_assistant_text": "reading sample",
                },
                {"actions": [{"type": "forget", "source": ["tr.1"], "reason": "progress-only read result is not needed"}]},
                {"actions": _final_actions()},
            ]

        def request(self, system_prompt, user_prompt, *, activity="agent", **_kwargs):
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    _seed_plan(agent, "answer")
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
                {"_format_error": "Invalid function-tool response: plain answer", "actions": []},
                KeyboardInterrupt(),
            ]

        def request(self, system_prompt, user_prompt, *, activity="agent", **_kwargs):
            response = self.responses.pop(0)
            if isinstance(response, KeyboardInterrupt):
                raise response
            return response

    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    agent.blackboard.goal = "answer"
    agent.blackboard.plan = [nanocode.PlanItem(text="try answer")]
    agent.blackboard.known = ["keep this fact"]
    agent.blackboard.verification.status = VerificationStatus.REQUIRED
    agent.tool_context.latest = ["old tool call"]
    agent.model_client = FakeModelClient()

    try:
        agent.run("answer")
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("expected KeyboardInterrupt")

    assert agent.agent_feedback_errors
    assert agent.tool_context.latest == ["old tool call"]
    assert agent.tool_context.recent == []
    assert agent.blackboard.goal == "answer"
    assert agent.blackboard.plan == [nanocode.PlanItem(text="try answer")]
    assert agent.blackboard.known == ["keep this fact"]
    assert agent.blackboard.verification.status == VerificationStatus.IDLE
    assert agent.blackboard.goal_reached is False

    class ChatModelClient:
        def request(self, system_prompt, user_prompt, *, activity="agent", **_kwargs):
            return {"actions": [], "_assistant_text": "ok"}

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

        def request(self, system_prompt, user_prompt, *, activity="agent", **_kwargs):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    agent.model_client = FakeModelClient()

    response = agent.run("answer")

    assert response["actions"][-1]["message_for_complete"] == "done"
    assert len(agent.model_client.user_prompts) == 2


def test_agent_run_shows_debug_gate_details_when_debug_enabled(tmp_path):
    class FakeModelClient:
        def __init__(self):
            self.responses = [
                {"_format_error": "Invalid function-tool response: plain answer", "_format_bad_output": "plain answer", "actions": []},
                {"actions": _final_actions()},
            ]

        def request(self, system_prompt, user_prompt, *, activity="agent", **_kwargs):
            return self.responses.pop(0)

    session = _session(tmp_path, debug=True)
    agent = Agent(session)
    agent.model_client = FakeModelClient()
    messages = []

    agent.run("answer", on_message=messages.append)

    assert messages[0] == "Format_Gate: retrying function/tool response. Invalid function-tool response: plain answer\nFull bad output:\nplain answer"


def test_agent_run_stops_after_repeated_format_errors(tmp_path):
    class FakeModelClient:
        def __init__(self):
            self.calls = 0

        def request(self, system_prompt, user_prompt, *, activity="agent", **_kwargs):
            self.calls += 1
            return {"_format_error": "Invalid function-tool response: missing content", "actions": []}

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
    assert "invalid function/tool response 3 times in a row" in message
    assert messages[-1] == "Stopped: invalid function/tool response 3 times in a row."


def test_agent_run_no_retry_when_goal_complete_has_message_for_complete(tmp_path):
    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {"actions": [{"type": "goal", "text": "answer", "complete": True, "message_for_complete": "Task completed successfully"}]},
                {"actions": _final_actions()},
            ]

        def request(self, system_prompt, user_prompt, *, activity="agent", **_kwargs):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    agent.model_client = FakeModelClient()
    messages = []

    response = agent.run("answer", on_message=messages.append)

    assert response["actions"][0]["message_for_complete"] == "Task completed successfully"
    assert len(agent.model_client.user_prompts) == 1
    assert "Task completed successfully" in messages
    assert "Retrying: goal is complete but message_for_complete is missing." not in " ".join(messages)

def test_agent_run_uses_fallback_when_goal_complete_has_empty_message_for_complete(tmp_path):
    class FakeModelClient:
        def __init__(self):
            self.user_prompts = []
            self.responses = [
                {"actions": [{"type": "goal", "text": "answer", "complete": True, "message_for_complete": ""}]},
            ]

        def request(self, system_prompt, user_prompt, *, activity="agent", **_kwargs):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    agent.model_client = FakeModelClient()
    messages = []

    response = agent.run("answer", on_message=messages.append)

    assert response["actions"][-1]["message_for_complete"] == ""
    assert messages[-1] == "Done."
    assert len(agent.model_client.user_prompts) == 1
    assert "Retrying: goal is complete but message_for_complete is missing." not in messages
    assert agent.agent_feedback_errors


def test_agent_run_uses_message_for_complete_even_when_assistant_text_exists(tmp_path):
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
                        },
                    ],
                    "_assistant_text": "explicit progress",
                },
                {"actions": _final_actions()},
            ]

        def request(self, system_prompt, user_prompt, *, activity="agent", **_kwargs):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    agent.model_client = FakeModelClient()
    messages = []

    response = agent.run("answer", on_message=messages.append)

    assert response["actions"][0]["message_for_complete"] == "fallback message"
    assert "explicit progress" not in messages
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
                {"actions": _final_actions()},
            ]

        def request(self, system_prompt, user_prompt, *, activity="agent", **_kwargs):
            self.user_prompts.append(user_prompt)
            return self.responses.pop(0)

    session = Session(cwd=str(tmp_path))
    agent = Agent(session)
    agent.model_client = FakeModelClient()
    messages = []

    response = agent.run("answer", on_message=messages.append)

    assert response["actions"][-1]["message_for_complete"] == "done"
    assert len(agent.model_client.user_prompts) == 2
    assert "should be ignored" not in messages
    assert not agent.agent_feedback_errors
