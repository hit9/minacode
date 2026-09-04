"""builtin tools stream (split from tests/test_builtin_tools.py)."""
from model_harness import _AnthropicMockClientFactory, _AnthropicStreamClientFactory, _MockClientFactory, _session, _StreamClientFactory
from test_builtin_tools import FUNCTION_TOOL, WEB_SEARCH, _responses_body

from wizolt.base import (
    builtin_tool_label,
)
from wizolt.config import (
    ConfigFile,
)
from wizolt.model import ModelClient
from wizolt.render import search_sources_footer


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

    model.request_sync([{"role": "user", "content": "hi"}], [])

    assert ("Web Search", "") in streamed

def test_responses_stream_reports_a_search_the_terminal_output_drops(tmp_path, monkeypatch):
    """Qwen streams the call but leaves it out of response.completed.output.

    The transcript line must come from the live stream event, since the parsed result has
    nothing to scan; without the live report the search would be invisible in the transcript."""
    s = _session(tmp_path, api="responses", model="qwen3-max", builtin_tools=(WEB_SEARCH,))
    model = ModelClient(s)
    reported = []
    streamed = []
    model.on_stream = lambda kind, delta: streamed.append((kind, delta))
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

    model.request_sync([{"role": "user", "content": "hi"}], [])

    assert reported == [("Web Search", "qwen release date")]
    # Qwen omits response.output_text.done, so response.completed is the terminal fallback that
    # hands the completed answer off before the preview is cleared.
    promoted = ("output_done", "sunny")
    assert streamed.count(promoted) == 1
    assert streamed.index(promoted) < streamed.index(("", ""))

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

    model.request_sync([{"role": "user", "content": "hi"}], [])

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

    model.request_sync([{"role": "user", "content": "hi"}], [])

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

    model.request_sync([{"role": "user", "content": "hi"}], [])

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

    model.request_sync([{"role": "user", "content": "hi"}], [])

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

    model.request_sync([{"role": "user", "content": "hi"}], [])

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

    model.request_sync([{"role": "user", "content": "hi"}], [FUNCTION_TOOL])

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

    model.request_sync([{"role": "user", "content": "hi"}], [])

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

    model.request_sync([{"role": "user", "content": "hi"}], [])

    assert reported == [("Web Search", "q")]

def test_a_search_without_a_query_still_reports(tmp_path, monkeypatch):
    """Qwen omits the action query; the call is still worth a line."""
    s = _session(tmp_path, api="responses", model="qwen3-max", stream=False, builtin_tools=(WEB_SEARCH,))
    model = ModelClient(s)
    reported = []
    model.on_builtin_call = lambda label, detail: reported.append((label, detail))
    output = [{"id": "ws_1", "type": "web_search_call", "status": "completed"}]
    monkeypatch.setattr(model, "client", _MockClientFactory([(200, _responses_body(output=output))]))

    model.request_sync([{"role": "user", "content": "hi"}], [])

    assert reported == [("Web Search", "")]

def test_builtin_labels_read_as_one_phase_across_protocols():
    """The same tool is named differently by each protocol and must still read alike."""
    assert builtin_tool_label("web_search_call") == "Web Search"  # Responses output item
    assert builtin_tool_label("web_search") == "Web Search"  # Messages server tool
    assert builtin_tool_label("$web_search") == "Web Search"  # Kimi builtin function
    assert builtin_tool_label("code_interpreter_call") == "Code Interpreter"
    assert builtin_tool_label("") == "Provider Tool"

def test_sources_footer_dedupes_by_url_in_first_mention_order():
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
