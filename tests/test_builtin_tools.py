"""Provider-side builtin tools: config parsing, pass-through on every protocol, results, and echoes."""

import json

import pytest
from agent_harness import session as agent_session
from model_harness import _AnthropicMockClientFactory, _AnthropicStreamClientFactory, _MockClientFactory, _session, _StreamClientFactory

from minacode.base import (
    PAUSED_TURN_KEY,
    SEARCH_SOURCES_KEY,
    ConfigError,
    ConfigFile,
    ModelError,
    ProviderConfig,
    ToolCall,
    builtin_tool_label,
)
from minacode.context import ContextManager
from minacode.engine import Agent
from minacode.model import ModelClient
from minacode.render import search_sources_footer
from minacode.runner import ToolRunner
from minacode.skill import SkillLibrary

WEB_SEARCH = {"type": "web_search"}
FUNCTION_TOOL = {
    "type": "function",
    "function": {"name": "Bash", "description": "Run a command", "parameters": {"type": "object", "properties": {}}},
}


def _responses_body(status="completed", output=None):
    return {
        "id": "resp_test",
        "object": "response",
        "created_at": 1,
        "status": status,
        "model": "gpt-5",
        "output": output or [],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }


def test_builtin_tools_parse_as_tables_with_a_type():
    provider = ProviderConfig.from_dict({"builtin_tools": [{"type": "web_search", "search_context_size": "high"}]})

    assert provider.builtin_tools == ({"type": "web_search", "search_context_size": "high"},)


def test_builtin_tools_default_to_empty():
    assert ProviderConfig.from_dict({}).builtin_tools == ()


@pytest.mark.parametrize(
    "value",
    [
        {"type": "web_search"},  # a bare table, not a list of them
        [{"name": "web_search"}],  # every documented builtin tool carries a type
        [{"type": ""}],
        ["web_search"],
    ],
)
def test_builtin_tools_reject_shapes_no_provider_accepts(value):
    with pytest.raises(ConfigError):
        ProviderConfig.from_dict({"builtin_tools": value})


def test_builtin_tools_are_not_shared_with_the_loaded_config(tmp_path):
    """A request must not be able to mutate config that outlives it."""
    s = _session(tmp_path, api="responses", model="gpt-5", builtin_tools=(dict(WEB_SEARCH),))

    ModelClient(s).builtin_tools()[0]["type"] = "mutated"

    assert s.config.provider.builtin_tools == ({"type": "web_search"},)


def test_responses_request_appends_builtin_tools_after_function_schemas(tmp_path, monkeypatch):
    s = _session(tmp_path, url="https://api.openai.com/v1", api="responses", model="gpt-5", stream=False, builtin_tools=(WEB_SEARCH,))
    model = ModelClient(s)
    factory = _MockClientFactory([(200, _responses_body())])
    monkeypatch.setattr(model, "client", factory)

    model.request([{"role": "user", "content": "hi"}], [FUNCTION_TOOL])

    body = json.loads(factory.calls[0].content)
    assert [tool.get("name") or tool["type"] for tool in body["tools"]] == ["Bash", "web_search"]
    assert body["tools"][1] == {"type": "web_search"}


def test_chat_request_appends_builtin_tools(tmp_path, monkeypatch):
    """Z.AI and Kimi express builtin tools in the Chat tools array, not the request body."""
    zai_search = {"type": "web_search", "web_search": {"enable": "True"}}
    s = _session(tmp_path, model="glm-5", stream=False, builtin_tools=(zai_search,))
    model = ModelClient(s)
    factory = _MockClientFactory(
        [
            (
                200,
                {
                    "id": "c",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "glm-5",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
                },
            )
        ]
    )
    monkeypatch.setattr(model, "client", factory)

    model.request([{"role": "user", "content": "hi"}], [FUNCTION_TOOL])

    body = json.loads(factory.calls[0].content)
    assert body["tools"] == [FUNCTION_TOOL, zai_search]


def test_anthropic_request_appends_builtin_tools(tmp_path, monkeypatch):
    search = {"type": "web_search_20250305", "name": "web_search", "max_uses": 5}
    s = _session(tmp_path, model="claude-3", api="anthropic", stream=False, builtin_tools=(search,))
    model = ModelClient(s)
    factory = _AnthropicMockClientFactory(
        [
            (
                200,
                {
                    "id": "m",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-3",
                    "content": [{"type": "text", "text": "hi"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )
        ]
    )
    monkeypatch.setattr(model, "anthropic_client", factory)

    model.request([{"role": "user", "content": "hi"}], [FUNCTION_TOOL])

    body = json.loads(factory.calls[0].content)
    assert [tool["name"] for tool in body["tools"]] == ["Bash", "web_search"]
    assert body["tools"][1] == search


def test_builtin_tools_are_sent_without_any_function_tools(tmp_path, monkeypatch):
    """Compaction and live follow-ups request with no function tools; search must still be offered."""
    s = _session(tmp_path, api="responses", model="gpt-5", stream=False, builtin_tools=(WEB_SEARCH,))
    model = ModelClient(s)
    factory = _MockClientFactory([(200, _responses_body())])
    monkeypatch.setattr(model, "client", factory)

    model.request([{"role": "user", "content": "hi"}], [])

    assert json.loads(factory.calls[0].content)["tools"] == [{"type": "web_search"}]


def test_builtin_tools_change_the_prompt_cache_key(tmp_path):
    """Enabling search changes the provider-rendered tool prefix, so the cached prefix differs."""
    plain = _session(tmp_path, api="responses", model="gpt-5")
    searching = _session(tmp_path, api="responses", model="gpt-5", builtin_tools=(WEB_SEARCH,))

    plain_key = ModelClient(plain).prompt_cache_key(plain.config.provider, [FUNCTION_TOOL])
    searching_key = ModelClient(searching).prompt_cache_key(searching.config.provider, [FUNCTION_TOOL])

    assert plain_key and searching_key and plain_key != searching_key


def test_responses_result_collects_openai_citations_and_qwen_sources(tmp_path, monkeypatch):
    """OpenAI cites inline; Qwen reports sources only on the search call. Both must be read."""
    s = _session(tmp_path, api="responses", model="gpt-5", stream=False, builtin_tools=(WEB_SEARCH,))
    model = ModelClient(s)
    output = [
        {
            "id": "ws_1",
            "type": "web_search_call",
            "status": "completed",
            "action": {"type": "search", "query": "httpx timeout", "sources": [{"url": "https://qwen.example/a", "title": "A"}]},
        },
        {
            "id": "msg_1",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": "sunny",
                    "annotations": [{"type": "url_citation", "url": "https://openai.example/b", "title": "B"}],
                }
            ],
        },
    ]
    factory = _MockClientFactory([(200, _responses_body(output=output))])
    monkeypatch.setattr(model, "client", factory)

    assistant, _, content = model.request([{"role": "user", "content": "hi"}], [])

    assert content == "sunny"
    assert assistant[SEARCH_SOURCES_KEY] == [
        {"url": "https://qwen.example/a", "title": "A"},
        {"url": "https://openai.example/b", "title": "B"},
    ]


def test_anthropic_result_collects_cited_and_raw_search_results(tmp_path, monkeypatch):
    s = _session(tmp_path, model="claude-3", api="anthropic", stream=False)
    model = ModelClient(s)
    content_blocks = [
        {"type": "server_tool_use", "id": "srv_1", "name": "web_search", "input": {"query": "shannon"}},
        {
            "type": "web_search_tool_result",
            "tool_use_id": "srv_1",
            "content": [{"type": "web_search_result", "url": "https://wiki.example/s", "title": "Shannon", "encrypted_content": "x"}],
        },
        {
            "type": "text",
            "text": "born 1916",
            "citations": [{"type": "web_search_result_location", "url": "https://wiki.example/s", "title": "Shannon", "cited_text": "…"}],
        },
    ]
    factory = _AnthropicMockClientFactory(
        [
            (
                200,
                {
                    "id": "m",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-3",
                    "content": content_blocks,
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )
        ]
    )
    monkeypatch.setattr(model, "anthropic_client", factory)

    assistant, _, _ = model.request([{"role": "user", "content": "hi"}], [])

    # The same URL is both a raw result and a citation; it is reported once.
    assert assistant[SEARCH_SOURCES_KEY] == [{"url": "https://wiki.example/s", "title": "Shannon"}]


def test_anthropic_search_error_reports_no_sources(tmp_path, monkeypatch):
    """A failed search returns an error object where results normally are, and cites nothing."""
    s = _session(tmp_path, model="claude-3", api="anthropic", stream=False)
    model = ModelClient(s)
    blocks = [
        {"type": "web_search_tool_result", "tool_use_id": "srv_1", "content": {"type": "web_search_tool_result_error", "error_code": "max_uses_exceeded"}},
        {"type": "text", "text": "could not search"},
    ]
    factory = _AnthropicMockClientFactory(
        [
            (
                200,
                {
                    "id": "m",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-3",
                    "content": blocks,
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )
        ]
    )
    monkeypatch.setattr(model, "anthropic_client", factory)

    assistant, _, _ = model.request([{"role": "user", "content": "hi"}], [])

    assert SEARCH_SOURCES_KEY not in assistant


def test_chat_result_collects_message_annotations(tmp_path, monkeypatch):
    """OpenRouter's web-search server tool cites through message annotations."""
    s = _session(
        tmp_path,
        url="https://openrouter.ai/api/v1",
        model="openai/gpt-5",
        stream=False,
        builtin_tools=({"type": "openrouter:web_search"},),
    )
    model = ModelClient(s)
    message = {
        "role": "assistant",
        "content": "hi",
        "annotations": [{"type": "url_citation", "url_citation": {"url": "https://router.example/c", "title": "C"}}],
    }
    factory = _MockClientFactory(
        [
            (
                200,
                {
                    "id": "c",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "openai/gpt-5",
                    "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                },
            )
        ]
    )
    monkeypatch.setattr(model, "client", factory)

    assistant, _, _ = model.request([{"role": "user", "content": "hi"}], [])

    assert assistant[SEARCH_SOURCES_KEY] == [{"url": "https://router.example/c", "title": "C"}]


def test_stored_sources_never_replay_to_the_provider(tmp_path, monkeypatch):
    """Sources are presentation state: they persist, but no protocol sends them back."""
    s = _session(tmp_path, model="gpt-4", stream=False)
    model = ModelClient(s)
    history = [{"role": "assistant", "content": "hi", SEARCH_SOURCES_KEY: [{"url": "https://example.com", "title": "T"}]}]
    factory = _MockClientFactory(
        [
            (
                200,
                {
                    "id": "c",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-4",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                },
            )
        ]
    )
    monkeypatch.setattr(model, "client", factory)

    model.request(history, [])

    sent = json.loads(factory.calls[0].content)["messages"]
    assert sent == [{"role": "assistant", "content": "hi"}]
    assert ModelClient(s).responses_input(history) == [{"role": "assistant", "content": "hi"}]


def test_responses_stream_reports_a_search_in_progress(tmp_path, monkeypatch):
    """A provider-side search has no tool line of its own; the status label is the only signal."""
    s = _session(tmp_path, api="responses", model="gpt-5")
    model = ModelClient(s)
    streamed = []
    model.on_stream = lambda kind, delta: streamed.append((kind, delta))
    events = [
        {"type": "response.output_item.added", "item": {"id": "ws_1", "type": "web_search_call", "status": "in_progress"}},
        {"type": "response.output_text.delta", "delta": "sunny"},
        {
            "type": "response.completed",
            "response": _responses_body(
                output=[{"id": "m", "type": "message", "status": "completed", "role": "assistant", "content": [{"type": "output_text", "text": "sunny"}]}]
            ),
        },
    ]
    monkeypatch.setattr(model, "client", _StreamClientFactory(events))

    model.request([{"role": "user", "content": "hi"}], [])

    assert ("Web Search", "") in streamed


def test_responses_stream_reports_a_search_the_terminal_output_drops(tmp_path, monkeypatch):
    """Qwen streams the call but leaves it out of response.completed.output.

    The transcript line must come from the live stream event, since the parsed result has
    nothing to scan; without the live report the search would be invisible in the transcript."""
    s = _session(tmp_path, api="responses", model="qwen3-max", builtin_tools=(WEB_SEARCH,))
    model = ModelClient(s)
    reported = []
    model.on_stream = lambda kind, delta: None
    model.on_builtin_call = lambda label, detail: reported.append((label, detail))
    events = [
        {"type": "response.output_item.added", "item": {"id": "ws_1", "type": "web_search_call", "status": "in_progress"}},
        {
            "type": "response.output_item.done",
            "item": {"id": "ws_1", "type": "web_search_call", "status": "completed", "action": {"type": "search", "query": "qwen release date"}},
        },
        {"type": "response.output_text.delta", "delta": "sunny"},
        {
            "type": "response.completed",
            "response": _responses_body(
                output=[{"id": "m", "type": "message", "status": "completed", "role": "assistant", "content": [{"type": "output_text", "text": "sunny"}]}]
            ),
        },
    ]
    monkeypatch.setattr(model, "client", _StreamClientFactory(events))

    model.request([{"role": "user", "content": "hi"}], [])

    assert reported == [("Web Search", "qwen release date")]


def test_responses_stream_reports_a_search_once_when_the_terminal_output_keeps_it(tmp_path, monkeypatch):
    """OpenAI retains the call in the terminal output; the live report and the scan must not double it."""
    s = _session(tmp_path, api="responses", model="gpt-5", builtin_tools=(WEB_SEARCH,))
    model = ModelClient(s)
    reported = []
    model.on_stream = lambda kind, delta: None
    model.on_builtin_call = lambda label, detail: reported.append((label, detail))
    call = {"id": "ws_1", "type": "web_search_call", "status": "completed", "action": {"type": "search", "query": "httpx timeout"}}
    events = [
        {"type": "response.output_item.added", "item": {"id": "ws_1", "type": "web_search_call", "status": "in_progress"}},
        {"type": "response.output_item.done", "item": call},
        {
            "type": "response.completed",
            "response": _responses_body(
                output=[call, {"id": "m", "type": "message", "status": "completed", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]}]
            ),
        },
    ]
    monkeypatch.setattr(model, "client", _StreamClientFactory(events))

    model.request([{"role": "user", "content": "hi"}], [])

    assert reported == [("Web Search", "httpx timeout")]


def test_responses_stream_does_not_double_an_id_less_call_the_terminal_output_keeps(tmp_path, monkeypatch):
    """An id-less call cannot be matched by id, so the scan must stay silent on a streamed request.

    Otherwise the live report and the parsed-result scan each emit the same id-less call."""
    s = _session(tmp_path, api="responses", model="gpt-5", builtin_tools=(WEB_SEARCH,))
    model = ModelClient(s)
    reported = []
    model.on_stream = lambda kind, delta: None
    model.on_builtin_call = lambda label, detail: reported.append((label, detail))
    call = {"type": "web_search_call", "status": "completed", "action": {"type": "search", "query": "missing id"}}
    events = [
        {"type": "response.output_item.added", "item": {**call, "status": "in_progress"}},
        {"type": "response.output_item.done", "item": call},
        {
            "type": "response.completed",
            "response": _responses_body(
                output=[call, {"id": "m", "type": "message", "status": "completed", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]}]
            ),
        },
    ]
    monkeypatch.setattr(model, "client", _StreamClientFactory(events))

    model.request([{"role": "user", "content": "hi"}], [])

    assert reported == [("Web Search", "missing id")]


def test_anthropic_stream_reports_a_search_in_progress(tmp_path, monkeypatch):
    s = _session(tmp_path, model="claude-3", api="anthropic")
    model = ModelClient(s)
    streamed = []
    model.on_stream = lambda kind, delta: streamed.append((kind, delta))
    message = {
        "id": "m",
        "type": "message",
        "role": "assistant",
        "model": "claude-3",
        "content": [],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    events = [
        ("message_start", {"type": "message_start", "message": message}),
        (
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "server_tool_use", "id": "srv_1", "name": "web_search", "input": {}}},
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("content_block_start", {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "sunny"}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 1}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}}),
        ("message_stop", {"type": "message_stop"}),
    ]
    monkeypatch.setattr(model, "anthropic_client", _AnthropicStreamClientFactory(events))

    model.request([{"role": "user", "content": "hi"}], [])

    assert ("Web Search", "") in streamed


def test_anthropic_stream_reports_a_search_live_before_the_stream_ends(tmp_path, monkeypatch):
    """The transcript line must fire while the stream is still running, not after it returns.

    Anthropic's assembled final message retains the server_tool_use block, so the parsed-result
    scan would also report it; the timeline proves the report came from the live content_block_stop
    (before the stream's closing on_stream sentinel) and that the scan did not double it."""
    s = _session(tmp_path, model="claude-3", api="anthropic", builtin_tools=({"type": "web_search_20250305", "name": "web_search"},))
    model = ModelClient(s)
    timeline: list[tuple] = []
    model.on_stream = lambda kind, delta: timeline.append(("stream", kind, delta))
    model.on_builtin_call = lambda label, detail: timeline.append(("builtin", label, detail))
    message = {
        "id": "m",
        "type": "message",
        "role": "assistant",
        "model": "claude-3",
        "content": [],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    events = [
        ("message_start", {"type": "message_start", "message": message}),
        (
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "server_tool_use", "id": "srv_1", "name": "web_search", "input": {}}},
        ),
        (
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": '{"query":"shannon birth date"}'}},
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("content_block_start", {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "1916"}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 1}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}}),
        ("message_stop", {"type": "message_stop"}),
    ]
    monkeypatch.setattr(model, "anthropic_client", _AnthropicStreamClientFactory(events))

    model.request([{"role": "user", "content": "hi"}], [])

    report = ("builtin", "Web Search", "shannon birth date")
    assert report in timeline
    # Live: reported during the stream, before the closing on_stream("", "") sentinel the scan follows.
    assert timeline.index(report) < timeline.index(("stream", "", ""))
    # De-duplicated: the parsed-result scan must not add a second line for the same call.
    assert sum(1 for entry in timeline if entry[0] == "builtin") == 1


def test_anthropic_stream_reads_the_query_carried_on_the_start_block(tmp_path, monkeypatch):
    """Some hosts put the whole input on content_block_start with no input_json_delta.

    The live report must use that query, not an empty string."""
    s = _session(tmp_path, model="claude-3", api="anthropic", builtin_tools=({"type": "web_search_20250305", "name": "web_search"},))
    model = ModelClient(s)
    reported = []
    model.on_stream = lambda kind, delta: None
    model.on_builtin_call = lambda label, detail: reported.append((label, detail))
    message = {
        "id": "m",
        "type": "message",
        "role": "assistant",
        "model": "claude-3",
        "content": [],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    events = [
        ("message_start", {"type": "message_start", "message": message}),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "server_tool_use", "id": "srv_1", "name": "web_search", "input": {"query": "already present"}},
            },
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("content_block_start", {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "ok"}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 1}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}}),
        ("message_stop", {"type": "message_stop"}),
    ]
    monkeypatch.setattr(model, "anthropic_client", _AnthropicStreamClientFactory(events))

    model.request([{"role": "user", "content": "hi"}], [])

    assert reported == [("Web Search", "already present")]


def test_responses_result_reports_each_search_for_the_transcript(tmp_path, monkeypatch):
    """The log line is the only lasting record: the status label vanishes when the turn ends."""
    s = _session(tmp_path, api="responses", model="gpt-5", stream=False, builtin_tools=(WEB_SEARCH,))
    model = ModelClient(s)
    reported = []
    model.on_builtin_call = lambda label, detail: reported.append((label, detail))
    output = [
        {"id": "ws_1", "type": "web_search_call", "status": "completed", "action": {"type": "search", "query": "httpx timeout configuration"}},
        {"id": "fc_1", "type": "function_call", "call_id": "c1", "name": "Bash", "arguments": "{}"},
        {"id": "m", "type": "message", "status": "completed", "role": "assistant", "content": [{"type": "output_text", "text": "sunny"}]},
    ]
    monkeypatch.setattr(model, "client", _MockClientFactory([(200, _responses_body(output=output))]))

    model.request([{"role": "user", "content": "hi"}], [FUNCTION_TOOL])

    # The local function call has its own tool line already; only the provider-side call is reported.
    assert reported == [("Web Search", "httpx timeout configuration")]


def test_anthropic_result_reports_each_search_for_the_transcript(tmp_path, monkeypatch):
    s = _session(tmp_path, model="claude-3", api="anthropic", stream=False)
    model = ModelClient(s)
    reported = []
    model.on_builtin_call = lambda label, detail: reported.append((label, detail))
    blocks = [
        {"type": "server_tool_use", "id": "srv_1", "name": "web_search", "input": {"query": "shannon birth date"}},
        {"type": "web_search_tool_result", "tool_use_id": "srv_1", "content": []},
        {"type": "text", "text": "1916"},
    ]
    factory = _AnthropicMockClientFactory(
        [
            (
                200,
                {
                    "id": "m",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-3",
                    "content": blocks,
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )
        ]
    )
    monkeypatch.setattr(model, "anthropic_client", factory)

    model.request([{"role": "user", "content": "hi"}], [])

    assert reported == [("Web Search", "shannon birth date")]


def test_searches_are_reported_with_streaming_disabled(tmp_path, monkeypatch):
    """Reporting comes from the parsed result, so it does not depend on stream events."""
    s = _session(tmp_path, api="responses", model="gpt-5", stream=False, builtin_tools=(WEB_SEARCH,))
    model = ModelClient(s)
    reported = []
    model.on_builtin_call = lambda label, detail: reported.append((label, detail))
    model.on_stream = None
    output = [{"id": "ws_1", "type": "web_search_call", "status": "completed", "action": {"type": "search", "query": "q"}}]
    monkeypatch.setattr(model, "client", _MockClientFactory([(200, _responses_body(output=output))]))

    model.request([{"role": "user", "content": "hi"}], [])

    assert reported == [("Web Search", "q")]


def test_a_search_without_a_query_still_reports(tmp_path, monkeypatch):
    """Qwen omits the action query; the call is still worth a line."""
    s = _session(tmp_path, api="responses", model="qwen3-max", stream=False, builtin_tools=(WEB_SEARCH,))
    model = ModelClient(s)
    reported = []
    model.on_builtin_call = lambda label, detail: reported.append((label, detail))
    output = [{"id": "ws_1", "type": "web_search_call", "status": "completed"}]
    monkeypatch.setattr(model, "client", _MockClientFactory([(200, _responses_body(output=output))]))

    model.request([{"role": "user", "content": "hi"}], [])

    assert reported == [("Web Search", "")]


def test_builtin_labels_read_as_one_phase_across_protocols():
    """The same tool is named differently by each protocol and must still read alike."""
    assert builtin_tool_label("web_search_call") == "Web Search"  # Responses output item
    assert builtin_tool_label("web_search") == "Web Search"  # Messages server tool
    assert builtin_tool_label("$web_search") == "Web Search"  # Kimi builtin function
    assert builtin_tool_label("code_interpreter_call") == "Code Interpreter"
    assert builtin_tool_label("") == "Provider Tool"


def test_sources_footer_dedupes_by_url_and_keeps_first_title():
    sources = [
        {"url": "https://a.example", "title": "First"},
        {"url": "https://a.example", "title": "Second"},
        {"url": "https://b.example", "title": ""},
    ]

    footer = search_sources_footer(sources)

    assert footer.splitlines() == ["", "**Sources**", "", "1. a.example", "2. b.example"]


def test_sources_footer_caps_a_long_list():
    footer = search_sources_footer([{"url": f"https://e.example/{index}", "title": f"T{index}"} for index in range(14)])

    assert footer.splitlines()[-1] == "…and 4 more"
    assert footer.count("e.example") == 10


def test_no_sources_render_nothing():
    assert search_sources_footer([]) == ""
    assert search_sources_footer([{"title": "no url"}]) == ""


def test_default_config_template_documents_builtin_tools():
    assert "builtin_tools" in ConfigFile.DEFAULT_TEXT


def test_paused_turn_is_reported_and_replays_unchanged(tmp_path, monkeypatch):
    """A paused search must be resumed by sending the assistant message back exactly as received."""
    s = _session(tmp_path, model="claude-3", api="anthropic", stream=False, builtin_tools=({"type": "web_search_20250305", "name": "web_search"},))
    model = ModelClient(s)
    blocks = [
        {"type": "server_tool_use", "id": "srv_1", "name": "web_search", "input": {"query": "httpx timeout"}},
        {
            "type": "web_search_tool_result",
            "tool_use_id": "srv_1",
            "content": [{"type": "web_search_result", "url": "https://e.example", "title": "E", "encrypted_content": "keep-me"}],
        },
    ]
    paused = {
        "id": "m1",
        "type": "message",
        "role": "assistant",
        "model": "claude-3",
        "content": blocks,
        "stop_reason": "pause_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    factory = _AnthropicMockClientFactory([(200, paused)])
    monkeypatch.setattr(model, "anthropic_client", factory)

    assistant, calls, _ = model.request([{"role": "user", "content": "hi"}], [])

    assert assistant[PAUSED_TURN_KEY] is True
    assert calls == []
    # Replaying the paused message must preserve encrypted_content; the API rejects it otherwise.
    replayed = model.anthropic_messages([{"role": "user", "content": "hi"}, assistant])
    assert replayed[-1]["content"] == blocks


def test_an_unpaused_response_carries_no_pause_marker(tmp_path, monkeypatch):
    s = _session(tmp_path, model="claude-3", api="anthropic", stream=False)
    model = ModelClient(s)
    factory = _AnthropicMockClientFactory(
        [
            (
                200,
                {
                    "id": "m",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-3",
                    "content": [{"type": "text", "text": "done"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )
        ]
    )
    monkeypatch.setattr(model, "anthropic_client", factory)

    assistant, _, _ = model.request([{"role": "user", "content": "hi"}], [])

    assert PAUSED_TURN_KEY not in assistant


def test_agent_continues_a_paused_turn_instead_of_answering(tmp_path):
    """A pause carries no tool call of ours, so without this the turn would end early."""
    s = agent_session(tmp_path)
    s.skills = SkillLibrary({})
    agent = Agent(s, output_fn=lambda text: None)

    class PausingModel:
        def __init__(self):
            self.requests = []

        def request(self, messages, tools=None):
            self.requests.append(messages)
            if len(self.requests) == 1:
                return {"role": "assistant", "content": None, PAUSED_TURN_KEY: True}, [], ""
            return {"role": "assistant", "content": "found it"}, [], "found it"

    agent.model = PausingModel()

    assert agent.run("look it up") == "found it"
    assert len(agent.model.requests) == 2
    # The paused message is part of the conversation the second request sends back.
    assert agent.model.requests[1][-1].get(PAUSED_TURN_KEY) is True
    assert s.messages[-1]["content"] == "found it"


def test_a_paused_turn_is_bounded_by_max_steps(tmp_path):
    """A provider that never stops pausing must still end the turn."""
    s = agent_session(tmp_path)
    s.skills = SkillLibrary({})
    s.settings.max_steps = 3
    agent = Agent(s, output_fn=lambda text: None)

    class AlwaysPausing:
        def __init__(self):
            self.count = 0

        def request(self, messages, tools=None):
            self.count += 1
            return {"role": "assistant", "content": None, PAUSED_TURN_KEY: True}, [], ""

    agent.model = AlwaysPausing()

    assert "Stopped after max_agent_steps=3" in agent.run("look it up")
    assert agent.model.count == 3


def test_builtin_function_names_are_collected_from_config():
    provider = ProviderConfig.from_dict({"builtin_tools": [{"type": "web_search"}, {"type": "builtin_function", "function": {"name": "$web_search"}}]})

    assert provider.builtin_function_names() == ("$web_search",)


def test_a_declared_builtin_function_call_is_answered_with_its_arguments(tmp_path):
    """Kimi runs the search itself; the documented client side is to echo the arguments back."""
    s = agent_session(tmp_path)
    s.config.providers["default"].builtin_tools = ({"type": "builtin_function", "function": {"name": "$web_search"}},)
    logged = []
    runner = ToolRunner(s, ContextManager(s), input_fn=lambda *a: "", output_fn=logged.append)

    messages = runner.run([ToolCall("c1", "$web_search", [{"search_query": "httpx timeout"}])])

    assert messages == [{"role": "tool", "tool_call_id": "c1", "name": "$web_search", "content": '{"search_query": "httpx timeout"}'}]
    # No confirmation was asked for, and nothing was stored as a recallable result.
    assert s.tool_records == []
    assert logged and "Web Search" in str(logged[0])


def test_an_undeclared_builtin_function_call_is_still_an_unknown_tool(tmp_path):
    """The echo path is opened by config alone; it must not swallow arbitrary unknown names."""
    s = agent_session(tmp_path)
    runner = ToolRunner(s, ContextManager(s), input_fn=lambda *a: "", output_fn=lambda text: None)

    messages = runner.run([ToolCall("c1", "$web_search", [{"search_query": "x"}])])

    assert "unknown tool $web_search" in messages[0]["content"]


def test_a_batch_mixing_an_echo_and_a_real_tool_runs_both(tmp_path):
    s = agent_session(tmp_path)
    s.config.providers["default"].builtin_tools = ({"type": "builtin_function", "function": {"name": "$web_search"}},)
    (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")
    runner = ToolRunner(s, ContextManager(s), input_fn=lambda *a: "", output_fn=lambda text: None)

    messages = runner.run([ToolCall("c1", "$web_search", [{"search_query": "q"}]), ToolCall("c2", "Read", [{"path": "a.txt", "ranges": [[0, 1]]}])])

    assert [message["tool_call_id"] for message in messages] == ["c1", "c2"]
    assert messages[0]["content"] == '{"search_query": "q"}'
    assert "<Read" in messages[1]["content"]


def _chat_body(model="qwen3.8-max-preview"):
    return {
        "id": "c",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
    }


def test_qwen_chat_rejects_responses_builtin_tools_before_the_transport(tmp_path, monkeypatch):
    """The reported failure: Responses-only entries must fail locally on the Chat wire."""
    s = _session(
        tmp_path,
        url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen3.8-max-preview",
        api="chat",
        stream=False,
        builtin_tools=({"type": "web_search"}, {"type": "web_extractor"}),
    )
    model = ModelClient(s)
    factory = _MockClientFactory([(200, _chat_body())])
    monkeypatch.setattr(model, "client", factory)

    with pytest.raises(ModelError) as excinfo:
        model.request([{"role": "user", "content": "hi"}], [FUNCTION_TOOL])

    # Deterministic configuration error: refused before any SDK/network I/O, no retry.
    assert factory.calls == []
    message = str(excinfo.value)
    assert "chat" in message
    assert "web_search" in message
    assert "web_extractor" in message
    assert "api=responses" in message
    assert "extra_body.enable_search" in message


def test_qwen_responses_keeps_builtin_tools_unchanged(tmp_path, monkeypatch):
    """The same configuration stays valid on the Responses wire, untouched by any Chat wrapper."""
    s = _session(
        tmp_path,
        url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen3.8-max-preview",
        api="responses",
        stream=False,
        builtin_tools=({"type": "web_search"}, {"type": "web_extractor"}),
    )
    model = ModelClient(s)
    factory = _MockClientFactory([(200, _responses_body())])
    monkeypatch.setattr(model, "client", factory)

    model.request([{"role": "user", "content": "hi"}], [FUNCTION_TOOL])

    body = json.loads(factory.calls[0].content)
    tools = body["tools"]
    # Local function tools use Responses' flat function schema; provider entries follow unchanged.
    assert [tool["type"] for tool in tools] == ["function", "web_search", "web_extractor"]
    assert tools[1:] == [{"type": "web_search"}, {"type": "web_extractor"}]


def test_unknown_provider_keeps_generic_builtin_tools_pass_through(tmp_path, monkeypatch):
    """Unmatched hosts keep the pass-through path for private and future providers."""
    entry = {"type": "web_search", "custom_field": "kept"}
    s = _session(tmp_path, model="made-up-model", api="chat", stream=False, builtin_tools=(entry,))
    model = ModelClient(s)
    factory = _MockClientFactory([(200, _chat_body("made-up-model"))])
    monkeypatch.setattr(model, "client", factory)

    model.request([{"role": "user", "content": "hi"}], [FUNCTION_TOOL])

    assert json.loads(factory.calls[0].content)["tools"] == [FUNCTION_TOOL, entry]


@pytest.mark.parametrize("url", ["https://api.z.ai/api/paas/v4", "https://open.bigmodel.cn/api/paas/v4"])
def test_zai_and_bigmodel_chat_accept_their_documented_builtin_tool(tmp_path, monkeypatch, url):
    """Both GLM hosts place provider-native web_search inside the Chat tools array."""
    zai_search = {"type": "web_search", "web_search": {"enable": "True"}}
    s = _session(tmp_path, url=url, model="glm-5", api="chat", stream=False, builtin_tools=(zai_search,))
    model = ModelClient(s)
    factory = _MockClientFactory([(200, _chat_body("glm-5"))])
    monkeypatch.setattr(model, "client", factory)

    model.request([{"role": "user", "content": "hi"}], [FUNCTION_TOOL])

    assert json.loads(factory.calls[0].content)["tools"] == [FUNCTION_TOOL, zai_search]


def test_kimi_chat_accepts_builtin_function_unchanged(tmp_path, monkeypatch):
    """Kimi declares $web_search as a Chat builtin_function that keeps its handshake."""
    kimi_search = {"type": "builtin_function", "function": {"name": "$web_search"}}
    s = _session(tmp_path, url="https://api.moonshot.ai/v1", model="kimi-k3", api="chat", stream=False, builtin_tools=(kimi_search,))
    model = ModelClient(s)
    factory = _MockClientFactory([(200, _chat_body("kimi-k3"))])
    monkeypatch.setattr(model, "client", factory)

    model.request([{"role": "user", "content": "hi"}], [FUNCTION_TOOL])

    assert json.loads(factory.calls[0].content)["tools"] == [FUNCTION_TOOL, kimi_search]


def test_anthropic_builtin_tools_are_protocol_scoped(tmp_path, monkeypatch):
    """Anthropic server tools travel over Messages; forcing them onto Chat fails locally."""
    search = {"type": "web_search_20250305", "name": "web_search", "max_uses": 5}
    s = _session(tmp_path, url="https://api.anthropic.com/v1", model="claude-3", api="anthropic", stream=False, builtin_tools=(search,))
    model = ModelClient(s)
    factory = _AnthropicMockClientFactory(
        [
            (
                200,
                {
                    "id": "m",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-3",
                    "content": [{"type": "text", "text": "hi"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )
        ]
    )
    monkeypatch.setattr(model, "anthropic_client", factory)

    model.request([{"role": "user", "content": "hi"}], [FUNCTION_TOOL])
    assert [tool["name"] for tool in json.loads(factory.calls[0].content)["tools"]] == ["Bash", "web_search"]

    # The same known-provider configuration forced onto Chat is rejected locally.
    s.config.provider.api = "chat"
    with pytest.raises(ModelError) as excinfo:
        model.request([{"role": "user", "content": "hi"}], [FUNCTION_TOOL])
    message = str(excinfo.value)
    assert "anthropic" in message and "chat" in message


@pytest.mark.parametrize("api", ["chat", "responses"])
def test_openrouter_sends_supported_server_tools_unchanged_on_both_wires(tmp_path, monkeypatch, api):
    """OpenRouter documents these server tools on both Chat and Responses."""
    server_tools = (
        {"type": "openrouter:web_search", "parameters": {"max_results": 5}},
        {"type": "openrouter:web_fetch"},
        {"type": "openrouter:datetime", "parameters": {"timezone": "Asia/Shanghai"}},
    )
    s = _session(tmp_path, url="https://openrouter.ai/api/v1", model="vendor/model", api=api, stream=False, builtin_tools=server_tools)
    model = ModelClient(s)
    response = _chat_body("vendor/model") if api == "chat" else _responses_body()
    factory = _MockClientFactory([(200, response)])
    monkeypatch.setattr(model, "client", factory)

    model.request([{"role": "user", "content": "hi"}], [FUNCTION_TOOL])

    body = json.loads(factory.calls[0].content)
    expected_local = FUNCTION_TOOL if api == "chat" else ModelClient.responses_tool_schemas([FUNCTION_TOOL])[0]
    assert body["tools"] == [expected_local, *server_tools]


@pytest.mark.parametrize(
    ("url", "model", "api", "entry", "supported"),
    [
        ("https://api.moonshot.ai/v1", "kimi-k3", "chat", {"type": "builtin_function"}, "builtin_function/$web_search"),
        (
            "https://api.moonshot.ai/v1",
            "kimi-k3",
            "chat",
            {"type": "builtin_function", "function": {"name": "$other"}},
            "builtin_function/$web_search",
        ),
        ("https://api.z.ai/api/paas/v4", "glm-5", "chat", {"type": "web_search"}, "web_search object"),
        ("https://api.anthropic.com/v1", "claude-3", "anthropic", {"type": "web_search_20250305"}, "name=web_search"),
    ],
)
def test_known_provider_rejects_incomplete_or_different_supported_type_shapes(tmp_path, url, model, api, entry, supported):
    """A matching type alone must not claim support for a different provider lifecycle."""
    s = _session(tmp_path, url=url, model=model, api=api, builtin_tools=(entry,))

    with pytest.raises(ModelError) as excinfo:
        ModelClient(s).builtin_tools()

    message = str(excinfo.value)
    assert "not supported" in message
    assert supported in message


def test_known_provider_rejects_unsupported_builtin_tool_types(tmp_path):
    """Unsupported provider-side tools fail locally instead of leaking lifecycle gaps."""
    cases = [
        ("https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen3.8-max-preview", "responses", {"type": "code_interpreter"}, "web_search, web_extractor"),
        ("https://api.openai.com/v1", "gpt-5", "responses", {"type": "file_search"}, "web_search"),
        ("https://api.anthropic.com/v1", "claude-3", "anthropic", {"type": "web_fetch_20250628"}, "web_search_20250305"),
        ("https://api.z.ai/api/paas/v4", "glm-5", "chat", {"type": "retrieval"}, "web_search"),
        ("https://api.moonshot.ai/v1", "kimi-k3", "chat", {"type": "web_search"}, "builtin_function"),
    ]
    for url, model, api, entry, supported in cases:
        s = _session(tmp_path, url=url, model=model, api=api, builtin_tools=(entry,))
        with pytest.raises(ModelError) as excinfo:
            ModelClient(s).builtin_tools()
        message = str(excinfo.value)
        assert entry["type"] in message
        assert "not supported" in message
        assert supported in message


def test_known_providers_without_server_tools_reject_builtin_tools(tmp_path):
    """DeepSeek, Kimi Code, and OpenCode have no provider-side tools contract."""
    cases = [
        ("https://api.deepseek.com/v1", "deepseek-chat", "chat"),
        ("https://api.kimi.com/coding/v1", "k3", "chat"),
        ("https://opencode.ai/zen/v1", "gpt-5.5", "responses"),
    ]
    for url, model, api in cases:
        s = _session(tmp_path, url=url, model=model, api=api, builtin_tools=(WEB_SEARCH,))
        with pytest.raises(ModelError) as excinfo:
            ModelClient(s).builtin_tools()
        assert "no documented provider-side tools" in str(excinfo.value)


def test_estimation_and_send_share_the_builtin_tools_policy(tmp_path, monkeypatch):
    """The estimator must not estimate a payload the send path would reject, and vice versa."""
    s = _session(
        tmp_path, url="https://dashscope.aliyuncs.com/compatible-mode/v1", model="qwen3.8-max-preview", api="chat", stream=False, builtin_tools=(WEB_SEARCH,)
    )
    model = ModelClient(s)

    # Both paths reject the same mismatch through the same policy.
    with pytest.raises(ModelError):
        model.estimated_request_tokens([{"role": "user", "content": "hi"}], [FUNCTION_TOOL])
    with pytest.raises(ModelError):
        model.chat_request([{"role": "user", "content": "hi"}], [FUNCTION_TOOL], allow_stream=False)

    # On the valid wire, the estimator consumes the same builtin entry the request sends.
    s.config.provider.api = "responses"
    factory = _MockClientFactory([(200, _responses_body())])
    monkeypatch.setattr(model, "client", factory)
    with_builtin = model.estimated_request_tokens([{"role": "user", "content": "hi"}], [FUNCTION_TOOL])
    s.config.provider.builtin_tools = ()
    without_builtin = model.estimated_request_tokens([{"role": "user", "content": "hi"}], [FUNCTION_TOOL])
    assert with_builtin > without_builtin
    s.config.provider.builtin_tools = (WEB_SEARCH,)
    model.request([{"role": "user", "content": "hi"}], [FUNCTION_TOOL])
    assert json.loads(factory.calls[0].content)["tools"][-1] == WEB_SEARCH
