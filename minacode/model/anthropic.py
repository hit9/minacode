"""Anthropic Messages wire conversion: params, history replay, stream reassembly, and result parsing.

Pure conversion with explicit inputs — no client state and no retry or orchestration logic.
ModelClient keeps the flow decisions and delegates the conversion here.
"""

from __future__ import annotations

import contextlib
import json
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from minacode.base import (
    ANTHROPIC_CONTENT_KEY,
    PAUSED_TURN_KEY,
    SEARCH_SOURCES_KEY,
    Json,
    ModelOutputTruncated,
    ToolCall,
    builtin_tool_label,
)
from minacode.config import ProviderConfig
from minacode.image import ImageInputs
from minacode.providers.compat import ResolvedProvider

if TYPE_CHECKING:
    from anthropic import Anthropic


def anthropic_params(
    messages: list[Json],
    tools: list[Json] | None,
    provider: ProviderConfig,
    resolved: ResolvedProvider,
    *,
    provider_origin: Callable[[ProviderConfig | None], str],
    replayable_echo: Callable[[Json, str], bool],
    images: ImageInputs,
    builtin_tools: Callable[[ResolvedProvider | None], list[Json]],
    apply_request: Callable[[Json, ProviderConfig, ResolvedProvider], Json],
    text_only: bool = False,
) -> Json:
    """Assemble the Messages request body from normalized messages and provider settings.

    The caller supplies the ModelClient hooks the body depends on: endpoint identity
    (`provider_origin`), the echo replay gate (`replayable_echo`), image content substitution
    (`images`), builtin tool schemas (`builtin_tools`), and the request-recipe application
    (`apply_request`, the Anthropic wire's thinking/effort fold). `text_only` is the route's
    image-delivery verdict: raw blocks are suppressed but readable labels and asset paths stay.
    """
    system_text = "\n\n".join(str(message.get("content") or "") for message in messages if message.get("role") == "system").strip()
    # Anthropic prompt caching is a prefix match that only takes effect at explicit
    # cache_control breakpoints; without one, every turn reprocesses the whole prompt from
    # scratch. Render order is tools -> system -> messages, so this breakpoint caches the stable
    # tools+system prefix, and `mark_prompt_cache_tail` adds the rolling one that covers the
    # conversation itself. Two breakpoints, well under the four a request may carry.
    system: str | list[Json] = [{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}] if system_text else system_text
    params: Json = {
        "model": provider.model,
        "system": system,
        "messages": mark_prompt_cache_tail(
            anthropic_messages(
                messages, provider_origin(provider), provider_origin=provider_origin, replayable_echo=replayable_echo, images=images, text_only=text_only
            )
        ),
        "max_tokens": resolved.output_max_tokens,
    }
    # Thinking pins temperature to its default; sending any other value is rejected.
    if request_tools := [*anthropic_tool_schemas(tools or []), *builtin_tools(resolved)]:
        params["tools"] = request_tools
        params["tool_choice"] = {"type": "auto"}
    # The generation recipe (messages.manual / messages.adaptive, selected from the catalog by
    # model version) fills in thinking and output_config; temperature is suppressed while thinking
    # is active because the API pins it then.
    params = apply_request(params, provider, resolved)
    thinking = params.get("thinking")
    thinking_active = resolved.reasoning_mandatory or (isinstance(thinking, dict) and thinking.get("type") in ("enabled", "adaptive"))
    # Anthropic SDK 1.0 removed the top-level `temperature` parameter; `extra_body` is merged
    # into the wire body by both 0.104.1 and 1.0.0, so the value still goes out the same way.
    if provider.temperature is not None and not thinking_active:
        params.setdefault("extra_body", {})["temperature"] = provider.temperature
    return params


def anthropic_messages(
    messages: list[Json],
    origin: str,
    *,
    provider_origin: Callable[[ProviderConfig | None], str],
    replayable_echo: Callable[[Json, str], bool],
    images: ImageInputs,
    text_only: bool = False,
) -> list[Json]:
    """Convert normalized messages to Messages history, merging consecutive same-role turns.

    System messages are dropped (they live in `system`), assistant messages replay their saved
    content blocks when the origin matches, and tool results become user tool_result blocks.
    `text_only` is the route's image-delivery verdict: raw blocks are suppressed but readable
    labels and asset paths stay.
    """
    origin = origin or provider_origin(None)
    converted: list[Json] = []
    for message in messages:
        role = message.get("role")
        if role == "system":
            continue
        if role == "user":
            append_anthropic_message(converted, "user", images.anthropic_content(message, text_only=text_only))
        elif role == "assistant":
            blocks = anthropic_assistant_blocks(message, origin, provider_origin=provider_origin, replayable_echo=replayable_echo)
            if blocks:
                append_anthropic_message(converted, "assistant", blocks)
        elif role == "tool":
            block = {"type": "tool_result", "tool_use_id": str(message.get("tool_call_id") or ""), "content": str(message.get("content") or "")}
            append_anthropic_message(converted, "user", [block])
    # One shape for every turn: text content is a block list, never a bare string. The wire
    # accepts both, but the rolling cache breakpoint has to land on a block, and a turn that
    # rendered as blocks while it was last (marked) and as a string once it is history would be
    # two different prefixes -- the cache read would miss on exactly the span it was written for.
    for message in converted:
        if isinstance(text := message.get("content"), str) and text:
            message["content"] = [{"type": "text", "text": text}]
    return converted or [{"role": "user", "content": ""}]


# Blocks the API documents as carrying cache_control. `thinking` is deliberately absent: the API
# verifies replayed thinking blocks against the signature it issued, so they are echoed untouched.
CACHE_BREAKPOINT_BLOCKS = ("text", "image", "tool_use", "tool_result", "document")


def mark_prompt_cache_tail(messages: list[Json]) -> list[Json]:
    """Put a rolling cache_control breakpoint on the last block of the conversation.

    Cache writes happen only at a breakpoint, so the system breakpoint alone caches tools+system
    and leaves the conversation body -- the part that grows to a hundred thousand tokens -- paid
    for in full on every single turn. This marker writes the history through this turn; the next
    turn's marker reads it back as its prefix. The block is copied rather than annotated in place
    because assistant blocks are replayed from session state and must not pick up wire-only fields.
    """
    content = messages[-1].get("content") if messages else None
    if not isinstance(content, list):
        return messages
    for index in range(len(content) - 1, -1, -1):
        if isinstance(block := content[index], dict) and block.get("type") in CACHE_BREAKPOINT_BLOCKS:
            content[index] = {**block, "cache_control": {"type": "ephemeral"}}
            break
    return messages


def append_anthropic_message(messages: list[Json], role: str, content: str | list[Json]) -> None:
    """Append a Messages content block or text, merging into the previous message of the same role.

    The Messages wire alternates roles, so adjacent same-role turns are merged in place: lists are
    extended, and str/list pairs are normalized to blocks before merging.
    """
    if messages and messages[-1].get("role") == role:
        previous = messages[-1].get("content")
        if isinstance(previous, list) and isinstance(content, list):
            previous.extend(content)
            return
        if isinstance(previous, list) and isinstance(content, str):
            if content:
                previous.append({"type": "text", "text": content})
            return
        if isinstance(previous, str) and isinstance(content, list):
            messages[-1]["content"] = ([{"type": "text", "text": previous}] if previous else []) + content
            return
        if isinstance(previous, str) and isinstance(content, str):
            messages[-1]["content"] = (previous + "\n\n" + content).strip()
            return
    messages.append({"role": role, "content": content})


def anthropic_assistant_blocks(
    message: Json,
    origin: str,
    *,
    provider_origin: Callable[[ProviderConfig | None], str],
    replayable_echo: Callable[[Json, str], bool],
) -> list[Json]:
    """Rebuild the assistant content blocks for one normalized message, replaying saved blocks when replayable.

    The API verifies that thinking blocks come back exactly as it produced them, signature
    included, so a turn it produced is echoed rather than rebuilt from text and tool calls.
    `provider_origin` supplies the endpoint identity for an empty `origin`, and `replayable_echo`
    is the replay gate — both ModelClient hooks.
    """
    # The API verifies that thinking blocks come back exactly as it produced them, signature
    # included, so a turn it produced is echoed rather than rebuilt from text and tool calls.
    saved = message.get(ANTHROPIC_CONTENT_KEY) if replayable_echo(message, origin or provider_origin(None)) else None
    if isinstance(saved, list) and saved:
        return [block for block in saved if isinstance(block, dict) and (message.get("content") is not None or block.get("type") != "text")]
    blocks: list[Json] = []
    content = message.get("content")
    if isinstance(content, str) and content:
        blocks.append({"type": "text", "text": content})
    for raw in message.get("tool_calls") or []:
        if not isinstance(raw, dict):
            continue
        raw_function = raw.get("function")
        function = raw_function if isinstance(raw_function, dict) else {}
        try:
            # strict=False: tool-call argument strings often contain literal newlines
            # (e.g. a multi-line git commit message), which are not valid JSON otherwise.
            payload = json.loads(str(function.get("arguments") or "{}"), strict=False)
        except json.JSONDecodeError:
            payload = {}
        blocks.append(
            {
                "type": "tool_use",
                "id": str(raw.get("id") or uuid.uuid4().hex),
                "name": str(function.get("name") or ""),
                "input": payload if isinstance(payload, dict) else {"args": [payload]},
            }
        )
    return blocks


def anthropic_tool_schemas(tools: list[Json]) -> list[Json]:
    """Convert normalized tool schemas to Messages `tools` entries."""

    def convert(schema: Json) -> Json:
        raw_function = schema.get("function")
        function = raw_function if isinstance(raw_function, dict) else {}
        return {
            "name": str(function.get("name") or ""),
            "description": str(function.get("description") or ""),
            "input_schema": function.get("parameters") if isinstance(function.get("parameters"), dict) else {},
        }

    return [convert(schema) for schema in tools]


def reassemble_stream(
    client: Anthropic,
    params: Json,
    *,
    message_field: Callable[[Any, str], Any],
    raise_if_inactive: Callable[[], None],
    emit: Callable[[str, str], None],
    report_builtin_call: Callable[[str, object], None],
) -> Any:
    """Consume Messages blocks and promote text once both text and tool blocks are known.

    Content blocks need not put text before `tool_use`, so block start/stop events feed the same
    order-independent transition as Responses. Input JSON may continue after promotion when the
    completed text block came first.

    `message_field` reads a field off an SDK object or dict, `raise_if_inactive` aborts a
    cancelled request, `emit` publishes stream deltas, and `report_builtin_call` records
    provider-side tool calls — all ModelClient hooks passed in by the caller.
    """
    output: list[str] = []
    text_blocks: set[int] = set()
    server_tools: dict[int, dict[str, str]] = {}
    text_done = handoff_seen = output_promoted = False

    def promote_output() -> None:
        nonlocal output_promoted
        if text_done and handoff_seen and output and not output_promoted:
            emit("output_done", "".join(output))
            output_promoted = True

    try:
        with client.messages.stream(**params) as stream:
            for event in stream:
                raise_if_inactive()
                event_type = message_field(event, "type")
                if event_type == "content_block_start":
                    block = message_field(event, "content_block")
                    block_type = message_field(block, "type")
                    if block_type == "text":
                        text_blocks.add(int(message_field(event, "index") or 0))
                    elif block_type == "tool_use":
                        handoff_seen = True
                        promote_output()
                    elif block_type == "server_tool_use":
                        # A provider-side tool is the same durable tool boundary as a local
                        # tool_use: completed text before it is final and must be handed off now,
                        # before the live status below covers the preview.
                        handoff_seen = True
                        promote_output()
                        emit(builtin_tool_label(str(message_field(block, "name") or "")), "")
                        # The query streams in via input_json_delta and is only whole at content_block_stop,
                        # so register the block now and report it there, showing the search in the transcript live.
                        # Some hosts put the whole input on content_block_start instead of streaming
                        # it via input_json_delta; keep that query as the fallback the stop handler
                        # uses when no partial_json ever arrived.
                        start_input = message_field(block, "input")
                        server_tools[int(message_field(event, "index") or 0)] = {
                            "id": str(message_field(block, "id") or ""),
                            "name": str(message_field(block, "name") or ""),
                            "json": "",
                            "query": str(start_input.get("query") or "") if isinstance(start_input, dict) else "",
                        }
                    continue
                if event_type == "content_block_stop":
                    index = int(message_field(event, "index") or 0)
                    if index in text_blocks:
                        text_done = True
                        promote_output()
                    elif index in server_tools:
                        info = server_tools.pop(index)
                        # Defensively establish the boundary here too; the durable report below
                        # must not be the first durable tool signal after a finished text block.
                        handoff_seen = True
                        promote_output()
                        query = info["query"]
                        if info["json"]:
                            with contextlib.suppress(json.JSONDecodeError):
                                parsed = json.loads(info["json"])
                                if isinstance(parsed, dict) and parsed.get("query"):
                                    query = str(parsed["query"])
                        report_builtin_call(info["name"], query)
                    continue
                if event_type != "content_block_delta":
                    continue
                delta = message_field(event, "delta")
                delta_type = message_field(delta, "type")
                if delta_type == "thinking_delta":
                    emit("reasoning", str(message_field(delta, "thinking") or ""))
                elif delta_type == "text_delta":
                    text = str(message_field(delta, "text") or "")
                    output.append(text)
                    emit("output", text)
                elif delta_type == "input_json_delta":
                    index = int(message_field(event, "index") or 0)
                    if index in server_tools:
                        server_tools[index]["json"] += str(message_field(delta, "partial_json") or "")
            raise_if_inactive()
            return stream.get_final_message()
    finally:
        emit("", "")


def anthropic_result(
    result: Any,
    streamed: bool,
    *,
    message_field: Callable[[Any, str], Any],
    dump_message_item: Callable[[Any], Json],
    tool_call: Callable[[str, str, object], ToolCall],
    report_builtin_call: Callable[[str, object], None],
    truncated_output_error: Callable[[Any], ModelOutputTruncated],
    collect_sources: Callable[..., list[Json]],
) -> tuple[Json, list[ToolCall], str]:
    """Parse a Messages result into the normalized (assistant message, tool calls, text) triple.

    Replays the parsed content blocks under ANTHROPIC_CONTENT_KEY, reports provider-side calls
    when the request was not streamed (the stream already reported them live), marks the
    turn-paused state, and raises the truncated-output error when the response stopped empty at
    the output cap.
    """
    text_parts: list[str] = []
    tool_calls: list[Json] = []
    calls: list[ToolCall] = []
    content_blocks = message_field(result, "content") or []
    saved_content = [dump_message_item(block) for block in content_blocks]
    for block in content_blocks:
        block_type = message_field(block, "type")
        # Streaming already reported each server tool live; the scan is the only source otherwise.
        if block_type == "server_tool_use" and not streamed:
            raw_input = message_field(block, "input")
            query = raw_input.get("query") if isinstance(raw_input, dict) else ""
            report_builtin_call(str(message_field(block, "name") or ""), query)
        if block_type == "text":
            text_parts.append(str(message_field(block, "text") or ""))
        elif block_type == "tool_use":
            raw_input = message_field(block, "input")
            payload = raw_input if isinstance(raw_input, dict) else {}
            name = str(message_field(block, "name") or "")
            call_id = str(message_field(block, "id") or uuid.uuid4().hex)
            arguments = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            tool_calls.append({"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}})
            calls.append(tool_call(call_id, name, payload))
    text = "".join(text_parts)
    if not calls and not text.strip() and message_field(result, "stop_reason") == "max_tokens":
        raise truncated_output_error(message_field(result, "usage"))
    assistant: Json = {"role": "assistant", "content": text or None, ANTHROPIC_CONTENT_KEY: [block for block in saved_content if block]}
    # A long server-side tool run can be paused and handed back mid-turn. The turn continues by
    # sending this message back unchanged, which the saved content blocks above already do.
    if message_field(result, "stop_reason") == "pause_turn":
        assistant[PAUSED_TURN_KEY] = True
    if sources := anthropic_sources(saved_content, collect_sources):
        assistant[SEARCH_SOURCES_KEY] = sources
    if tool_calls:
        assistant["tool_calls"] = tool_calls
    return assistant, calls, text


def anthropic_sources(saved_content: list[Json], collect_sources: Callable[..., list[Json]]) -> list[Json]:
    """Sources from a Messages response: cited text first, then the raw search results.

    A `web_search_tool_result` carries an error object rather than a result list when the
    search itself failed, which `collect_sources` skips as having no URL."""
    groups: list[Any] = []
    for block in saved_content:
        if not isinstance(block, dict):
            continue
        groups.append(block.get("citations"))
        if block.get("type") == "web_search_tool_result":
            content = block.get("content")
            groups.append(content if isinstance(content, list) else None)
    return collect_sources(*groups)
