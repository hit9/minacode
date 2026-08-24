"""compaction slices (split from tests/test_context.py)."""

import json
import threading

import pytest
from agent_harness import session, session_with_provider

from minacode.base import (
    SESSION_EVENT_KEY,
)
from minacode.context import ContextManager
from minacode.model import ModelClient, compaction
from minacode.prompts import (
    COMPACTION_SUMMARY_TITLE,
    LIVE_FOLLOWUP_PREFIX,
    PREVIOUS_CONTEXT_TRIMMED,
)
from minacode.session import Session


def test_history_segments_keep_only_the_newest_window(tmp_path):
    """Every compaction stores a span and nothing used to drop one, so a long session carried each
    one it ever evicted. Keys keep counting past the bound: reusing a number the model has already
    seen would answer a stale recall with a different span instead of saying it is gone."""
    s = session(tmp_path)
    context = ContextManager(s)
    limit = ContextManager.MAX_HISTORY_SEGMENTS

    for index in range(limit + 5):
        context.store_history_segment([{"role": "user", "content": f"span {index}"}], scope="history", trigger="auto", fallback=False)

    assert len(s.history) == limit
    assert [segment.key for segment in s.history] == [f"seg.{number}" for number in range(6, limit + 6)]
    # The sixth span is the oldest one still retained; the five before it are gone with their text.
    assert [segment.text for segment in s.history] == [f"user:\nspan {index}" for index in range(5, limit + 5)]


def test_pruned_history_survives_a_snapshot_round_trip(tmp_path):
    """The snapshot writes segments as an append-only delta while the list only grows; pruning
    shortens it, and the digest guard has to notice and rewrite the whole list instead."""
    s = session(tmp_path)
    context = ContextManager(s)
    for index in range(ContextManager.MAX_HISTORY_SEGMENTS + 3):
        context.store_history_segment([{"role": "user", "content": f"span {index}"}], scope="history", trigger="auto", fallback=False)
        s.save_snapshot()

    restored = Session.load_snapshot(s.uid, config=s.config)

    assert [segment.key for segment in restored.history] == [segment.key for segment in s.history]
    assert [segment.text for segment in restored.history] == [segment.text for segment in s.history]


def test_compaction_reuses_the_agent_prefix_and_keeps_real_messages(tmp_path):
    """The summary request is the agent's own request truncated, with an instruction appended, so
    the provider cache already covers it -- and the compactor sees tool calls, which the flattened
    payload drops (ImageInputs.label_text reads only content)."""
    live = session(tmp_path)
    live.messages = [{"role": "user", "content": "check the call sites"}]
    for index in range(6):
        live.messages.extend(
            [
                {
                    "role": "assistant",
                    "content": "Looking.",
                    "tool_calls": [{"id": f"c{index}", "type": "function", "function": {"name": "Bash", "arguments": "{}"}}],
                },
                {"role": "tool", "tool_call_id": f"c{index}", "content": f"tool tr.{index} Bash rg -n _run_workflow"},
            ]
        )
    live.messages.append({"role": "user", "content": "继续 Part B"})
    live.messages.append({"role": "assistant", "content": "ok"})
    context = ContextManager(live)
    compacted, _keep = compaction.Compactor(context).parts()
    assert compacted  # there is a head to summarize

    built = compaction.Compactor(context).request(compacted)
    assert built is not None
    messages, tools = built

    # Every message but the appended instruction is byte-identical to what the turn already sent,
    # which is what a prefix cache requires -- not merely "starts with the same system prompt".
    sent = context.model_messages(live.system_prompt)
    assert messages[:-1] == sent[: len(messages) - 1]
    assert tools  # carried so the prefix matches; tool_choice stays exactly as an ordinary request
    # The tool calls survive as structure, where the flattened payload would have dropped them.
    assert any(message.get("tool_calls") for message in messages)
    assert "Compact the minacode working context." in messages[-1]["content"]
    assert "END OF CONVERSATION TO COMPACT" in messages[-1]["content"]


def test_compaction_prefix_survives_an_earlier_summary_and_repeated_schemas(tmp_path):
    """The prefix has to be a slice of the list the turn actually sent. Rebuilding a lookalike from
    `compacted` diverges at the first earlier summary (dropped by without_compaction_summaries) and
    at every repeated schema (collapsed by the dedup only the live projection runs) -- which cost
    the whole conversation and left just the header cached."""
    live = session(tmp_path)
    describe = 'tool tr.%d MCP\n<MCPDescribe server="x" tool="y">SCHEMA</MCPDescribe>'
    live.messages = [
        {"role": "user", "content": COMPACTION_SUMMARY_TITLE + "\nSummary: earlier", SESSION_EVENT_KEY: "compaction_checkpoint"},
        {"role": "user", "content": "task A"},
        {"role": "assistant", "content": "looking"},
        {"role": "tool", "content": describe % 1},
        {"role": "tool", "content": describe % 2},
        # Past the recent window, so the earlier summary and the repeated schema are on the
        # compacted side where the divergence this pins used to happen.
        *({"role": "assistant", "content": f"filler {index}"} for index in range(10)),
        {"role": "user", "content": "task B"},
        {"role": "assistant", "content": "ok"},
    ]
    context = ContextManager(live)
    compacted, _keep = compaction.Compactor(context).parts()
    messages, _ = compaction.Compactor(context).request(compacted)

    sent = context.model_messages(live.system_prompt)
    assert messages[:-1] == sent[: len(messages) - 1]
    # Not just the header: the conversation is in the shared prefix too.
    assert len(messages) - 1 > len(context.model_header(live.system_prompt))


def test_turn_scope_compaction_slices_the_same_projection(tmp_path):
    """A turn-scope span sits after the stored conversation rather than at the head of it, but it is
    the same projection and the same kind of prefix -- only the offset differs."""
    live = session(tmp_path)
    live.messages = [{"role": "user", "content": "task"}, {"role": "assistant", "content": "starting"}]
    turn = [{"role": "assistant", "content": f"step {index}"} for index in range(24)]
    context = ContextManager(live)
    compacted, _keep = compaction.Compactor(context).turn_parts(turn)
    assert compacted

    messages, _ = compaction.Compactor(context).request(compacted, turn)

    sent = context.model_messages(live.system_prompt, turn)
    assert messages[:-1] == sent[: len(messages) - 1]
    # The slice reaches into the turn, not merely up to the end of stored history.
    assert len(messages) - 1 > len(context.model_header(live.system_prompt)) + len(live.messages)


def test_compaction_falls_back_to_the_flat_payload_on_a_separate_provider(tmp_path):
    """A [compaction] entry elsewhere is a different cache namespace, so rebuilding the prefix for
    it would pay the whole history at full rate to save nothing."""
    live = session(tmp_path)
    live.messages = [{"role": "user", "content": "hello"}]
    live.config.compaction_provider = "cheap"
    assert compaction.Compactor(ContextManager(live)).request(list(live.messages)) is None


def test_compaction_leaves_tool_choice_exactly_as_an_ordinary_request_sets_it(tmp_path):
    """Forcing tool_choice would look safer and cost the prize: it invalidates the messages cache,
    which is the whole conversation this request exists to reuse. Every wire is treated alike."""
    live = session(tmp_path)
    live.messages = [{"role": "user", "content": "hello"}, *({"role": "assistant", "content": f"step {index}"} for index in range(12))]
    for api in ("chat", "responses", "anthropic"):
        live.config.provider.api = api
        built = compaction.Compactor(ContextManager(live)).request(list(live.messages))
        assert built is not None, api  # no wire is excluded any more


def test_state_apply_takes_a_bare_string_where_a_list_was_asked_for(tmp_path):
    """Ignoring the wrong type is worse than accepting it: the previous compaction's value survives
    as though this one had confirmed it, and is fed back as current on the next pass."""
    live = session(tmp_path)
    live.state.apply({"known": ["stale fact"], "plan": [{"status": "done", "text": "old step"}]})

    live.state.apply({"known": "the API is rate limited", "plan": "finish Part B"})
    assert live.state.known == ["the API is rate limited"]
    assert [item.text for item in live.state.plan] == ["finish Part B"]

    # An empty string clears rather than silently keeping the old value.
    live.state.apply({"known": "   "})
    assert live.state.known == []
    # A type that is neither is still refused, as before.
    live.state.apply({"known": 17})
    assert live.state.known == []


def test_turn_scope_prefix_stops_where_the_turn_keeps(tmp_path):
    """The scopes split differently when the list holds no user message -- the ordinary shape of a
    long turn. Counting a turn the history way sent every message it was about to keep."""
    live = session(tmp_path)
    live.messages = [{"role": "user", "content": "task"}]
    turn = [{"role": "assistant", "content": f"step {index}"} for index in range(14)]
    context = ContextManager(live)
    compacted, keep = compaction.Compactor(context).turn_parts(turn)
    assert compacted and keep  # the split is real, not a degenerate all-or-nothing

    messages, _ = compaction.Compactor(context).request(compacted, turn)

    head = len(context.model_header(live.system_prompt)) + len(live.messages)
    assert len(messages) - 1 - head == len(compacted)


def test_echo_source_covers_the_message_the_slice_adds(tmp_path):
    """In the non-contiguous case -- the latest user message falling before the recent window, a
    worker given one order that then ran many steps -- the contiguous slice carries that message
    while `compacted` does not. It is a user message, the shape the observed echo copied, so
    checking `compacted` would leave the guard blind to it."""
    live = session(tmp_path)
    order = "继续 Part B 收尾：检查 _run_workflow 的所有调用点。"
    live.messages = [{"role": "user", "content": order}, *({"role": "assistant", "content": f"step {index}"} for index in range(20))]
    context = ContextManager(live)
    compacted, keep = compaction.Compactor(context).parts()
    assert order not in [message["content"] for message in compacted]
    assert order in [message["content"] for message in keep]

    messages, _ = compaction.Compactor(context).request(compacted)

    assert order not in compaction.Compactor(context).echo_source(compacted)  # what it used to be checked against
    assert order in compaction.Compactor(context).echo_source(messages[:-1])  # what the model is actually handed


def test_minimum_recent_fallback_carries_everything_it_evicts(tmp_path):
    """Both scopes re-split with COMPACT_MINIMUM_RECENT when the ordinary window leaves nothing to
    compact. Counting the slice with the default window then cut it short of `compacted`, evicting
    messages the summarizer was never shown -- a silent loss, since the summary is what replaces
    them."""
    live = session(tmp_path)
    live.messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": "Bash", "arguments": "{}"}} for i in range(5)],
        },
        *({"role": "tool", "tool_call_id": f"c{i}", "content": f"t{i}"} for i in range(5)),
        *({"role": "assistant", "content": f"a{i}"} for i in range(5)),
    ]
    context = ContextManager(live)
    assert not compaction.Compactor(context).parts()[0]  # the ordinary window yields nothing, forcing the fallback
    compacted, _keep = compaction.Compactor(context).parts(compaction.Compactor.COMPACT_MINIMUM_RECENT)

    messages, _ = compaction.Compactor(context).request(compacted, recent=compaction.Compactor.COMPACT_MINIMUM_RECENT)

    carried = messages[len(context.model_header(live.system_prompt)) : -1]
    assert len(carried) >= len(compacted)
    for message in compacted:  # nothing evicted may go unseen by the summarizer
        assert message in carried


def test_reasoning_boundary_matches_the_live_request_in_every_slice_shape(tmp_path):
    """Providers with chat_reasoning_history="current_turn" replay reasoning only after the last
    user message, so where the appended instruction sits relative to that boundary decides whether
    the summary request strips the same set the live request did.

    It differs by shape, and a fix for one shape broke the others. Marked when the boundary is
    inside the slice, so the instruction does not displace it; unmarked when the boundary is beyond
    the slice -- kept by the recent window, or living in the current turn -- so the instruction
    becomes the boundary and strips everything here, which is what the live request also does."""

    def diverges_at(live_session, request, turn=None):
        context, model = ContextManager(live_session), ModelClient(live_session)
        live = model.chat_messages(context.model_messages(live_session.system_prompt, turn))
        summary = model.chat_messages(request)
        pairs = zip(live, summary)
        return next((index for index, (a, b) in enumerate(pairs) if a != b), None), len(summary) - 1

    def reasoning_history(path):
        live_session = session_with_provider(path)
        live_session.config.provider.url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        assert live_session.config.provider.resolve().chat_reasoning_history == "current_turn"
        live_session.messages = [
            {"role": "user", "content": "earlier task"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "thought",
                "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "Bash", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "out"},
        ]
        return live_session

    # The boundary is kept by the recent window, so it sits beyond the slice.
    outside = reasoning_history(tmp_path / "outside")
    outside.messages.extend({"role": "assistant", "content": f"step {index}"} for index in range(10))
    outside.messages.extend([{"role": "user", "content": "latest"}, {"role": "assistant", "content": "answer"}])
    context = ContextManager(outside)
    compacted, _keep = compaction.Compactor(context).parts()
    at, body = diverges_at(outside, compaction.Compactor(context).request(compacted)[0])
    assert at == body, "the slice must match the live request; only the appended instruction may differ"

    # The non-contiguous shape: the boundary falls before the window, so it is inside the slice.
    inside = reasoning_history(tmp_path / "inside")
    inside.messages.extend({"role": "assistant", "content": f"step {index}"} for index in range(20))
    context = ContextManager(inside)
    compacted, _keep = compaction.Compactor(context).parts()
    assert compaction.Compactor(context).keep_start(inside.messages, None) > context.latest_user_index(inside.messages)
    at, body = diverges_at(inside, compaction.Compactor(context).request(compacted)[0])
    assert at == body

    # History-scope compaction during a turn: the boundary lives in the turn, outside the slice.
    midturn = reasoning_history(tmp_path / "midturn")
    midturn.messages.extend({"role": "assistant", "content": f"step {index}"} for index in range(12))
    turn = [{"role": "user", "content": "this turn"}, {"role": "assistant", "content": "working"}]
    context = ContextManager(midturn)
    compacted, _keep = compaction.Compactor(context).parts()
    # Through the real call chain, so the turn actually reaches the projection: history-scope
    # compaction receives it as `tool_messages`, and reading the boundary without it marked the
    # instruction in a shape where the live request had already stripped everything.
    captured = {}

    class CapturingModel:
        def __init__(self, session):
            self.session = session
            self.cancel_requested = threading.Event()
            self.last_compaction_model = ""

        def api_request(self, messages, _tools, *, allow_stream, response_timeout, provider, json_object):
            captured["messages"] = messages
            return None, [], '{"summary": "done"}'

        @staticmethod
        def parse_json_object(text):
            return json.loads(text)

    # Snapshot the live projection first: _compact_messages rewrites session.messages, and the
    # prefix being ridden is the one that existed before it did.
    before = ModelClient(midturn).chat_messages(context.model_messages(midturn.system_prompt, turn))
    assert compaction.Compactor(context, CapturingModel(midturn)).run(compacted, _keep, PREVIOUS_CONTEXT_TRIMMED, tool_messages=turn)
    request = captured["messages"]
    summary = ModelClient(midturn).chat_messages(request)
    at = next((index for index, (a, b) in enumerate(zip(before, summary)) if a != b), None)
    assert at == len(summary) - 1
    assert SESSION_EVENT_KEY not in ModelClient(midturn).chat_messages(request)[-1]  # never on the wire


def test_flat_payload_is_not_built_when_the_inline_form_is_used(tmp_path, monkeypatch):
    """compact() ignores `context` whenever inline messages are given, and flattening the span is
    proportional to a conversation large enough to need compacting."""
    live = session(tmp_path)
    live.messages = [{"role": "user", "content": "task"}]
    for index in range(12):
        live.messages.append({"role": "assistant", "content": f"step {index}"})
    context = ContextManager(live)
    monkeypatch.setattr(compaction.Compactor, "input", lambda _self, _messages: pytest.fail("flattened the span despite the inline form"))

    class FakeModel:
        def __init__(self, session):
            self.session = session
            self.cancel_requested = threading.Event()
            self.last_compaction_model = ""

        def api_request(self, _messages, _tools, *, allow_stream, response_timeout, provider, json_object):
            return None, [], '{"summary": "done"}'

        @staticmethod
        def parse_json_object(text):
            return json.loads(text)

    compacted, keep = compaction.Compactor(context).parts()
    assert compaction.Compactor(context, FakeModel(live)).run(compacted, keep, PREVIOUS_CONTEXT_TRIMMED)


def test_recent_window_is_a_floor_for_small_messages_and_a_ceiling_for_large_ones(tmp_path):
    """The window used to be measured only after the latest user message, which made it a cap: a
    /compact run just after a turn answered kept two messages out of a hundred and eighteen. It now
    spans the whole tail, bounded by size -- because a count is not a size, and keeping eight
    enormous messages leaves the request over budget with nothing left to compact."""
    small = session(tmp_path)
    for index in range(58):
        small.messages.extend([{"role": "user", "content": f"q{index}"}, {"role": "assistant", "content": f"a{index}"}])
    small.messages.extend([{"role": "user", "content": "latest"}, {"role": "assistant", "content": "answer"}])
    _, keep = compaction.Compactor(ContextManager(small)).parts()
    assert len(keep) == compaction.Compactor.COMPACT_RECENT_MESSAGES  # was 2

    large = session(tmp_path / "large")
    large.messages = [{"role": "user", "content": "go"}]
    for index in range(10):
        large.messages.append({"role": "assistant", "content": f"step {index} " + "x" * 400_000})
    compacted, keep = compaction.Compactor(ContextManager(large)).parts()
    assert compacted  # the tail collapses by size, so there is still something to evict
    assert len(keep) < compaction.Compactor.COMPACT_RECENT_MESSAGES


def test_the_slice_and_the_split_agree_on_where_the_cut_is(tmp_path):
    """Two expressions of the cut drifted apart twice: once when the MINIMUM_RECENT fallback
    re-split with a different window, once when the split grew a size bound the count did not have.
    Both times the request carried less than was evicted, and the summary lost the difference."""
    live = session(tmp_path)
    live.messages = [{"role": "user", "content": "go"}]
    for index in range(30):
        live.messages.append({"role": "assistant", "content": f"step {index} " + "y" * 20_000})
    context = ContextManager(live)

    for recent in (None, compaction.Compactor.COMPACT_MINIMUM_RECENT):
        compacted, _keep = compaction.Compactor(context).parts(recent)
        if not compacted:
            continue
        messages, _ = compaction.Compactor(context).request(compacted, recent=recent)
        carried = messages[len(context.model_header(live.system_prompt)) : -1]
        for message in compacted:
            assert message in carried, f"recent={recent} evicted a message the summary never saw"


def test_the_slice_follows_the_request_being_built_not_the_one_already_sent(tmp_path):
    """A follow-up queued mid-turn joins the request before compaction runs, so the projection the
    slice is cut from carries it and the previously sent request does not. The slice therefore
    aligns with the request about to go out rather than the one that wrote the cache.

    That is deliberate, and it is not the boundary bug it resembles. The queued message moves the
    reasoning boundary for the ordinary request too, so the divergence from what was sent exists
    with or without compaction -- at the same position. Aligning to the outgoing request keeps the
    hit rather than losing it: the summary writes a prefix the larger request that follows reads
    back. Aligning to the sent one instead would need the last projection carried as state, to
    move a hit from one request to the other."""
    live = session(tmp_path)
    live.config.provider.url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    live.messages = [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "earlier answer"}]
    sent_turn = [
        {"role": "user", "content": "do X"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "thought",
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "Bash", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "out"},
        *({"role": "assistant", "content": f"step {index}"} for index in range(16)),
    ]
    queued_turn = [*sent_turn, {"role": "user", "content": LIVE_FOLLOWUP_PREFIX + "also do Y"}]

    context, model = ContextManager(live), ModelClient(live)
    sent = model.chat_messages(context.model_messages(live.system_prompt, sent_turn))
    outgoing = model.chat_messages(context.model_messages(live.system_prompt, queued_turn))
    compacted, _keep = compaction.Compactor(context).turn_parts(queued_turn)
    summary = model.chat_messages(compaction.Compactor(context).request(compacted, queued_turn)[0])

    def diverges_at(left, right):
        return next((index for index, (a, b) in enumerate(zip(left, right)) if a != b), None)

    # The summary matches the outgoing request everywhere but its appended instruction.
    assert diverges_at(summary, outgoing) == len(summary) - 1
    # It parts from the sent request earlier -- at exactly where the queued message already parted.
    assert diverges_at(summary, sent) == diverges_at(outgoing, sent)
