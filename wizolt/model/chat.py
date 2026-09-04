"""Chat Completions wire conversion: history replay, request params, and stream reassembly.

Pure conversion with explicit inputs — no client state and no retry or orchestration logic.
ModelClient keeps the flow decisions (stream on/off, finish_reason=length, usage recording) and
delegates the conversion here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from wizolt.base import PROVIDER_ECHO_KEYS, SESSION_EVENT_KEY, Json, ModelError, Text
from wizolt.config import ProviderConfig
from wizolt.image import IMAGE_REFS_KEY, TOOL_IMAGE_OBSERVATION_KEY, ImageInputs
from wizolt.model.history import keeps_reasoning
from wizolt.providers.compat import ResolvedProvider

if TYPE_CHECKING:
    from openai import AsyncOpenAI


def chat_messages(
    messages: list[Json],
    provider: ProviderConfig,
    resolved: ResolvedProvider,
    images: ImageInputs,
    latest_user_position: Callable[[list[Json]], int],
    *,
    text_only: bool = False,
    image_payloads: dict[str, bytes] | None = None,
) -> list[Json]:
    """Build Chat Completions history from normalized messages under the provider's replay contract.

    Scrubs provider-echo keys and reasoning per the resolved reasoning_history, substituting
    image content for local image refs. `latest_user_position` is the boundary the caller uses for
    "current turn" reasoning — ModelClient.latest_user_position — so the same rule cannot drift.
    `text_only` is the route's image-delivery verdict: raw blocks are suppressed but readable
    labels and stable asset paths stay. `image_payloads` is the request-local ref→bytes mapping
    loaded once before building the wire; the image parts read from it instead of reopening each
    asset file on the loop.
    """
    converted: list[Json] = []
    latest_user = latest_user_position(messages)
    for index, message in enumerate(messages):
        clean = {
            key: value for key, value in message.items() if key not in (*PROVIDER_ECHO_KEYS, IMAGE_REFS_KEY, TOOL_IMAGE_OBSERVATION_KEY, SESSION_EVENT_KEY)
        }
        if message.get("role") == "assistant" and not keeps_reasoning(resolved.reasoning_history, message, index, latest_user):
            for key in ("reasoning_content", "reasoning", "reasoning_details"):
                clean.pop(key, None)
        if message.get("role") == "user" and images.refs(message):
            clean["content"] = images.chat_content(message, text_only=text_only, payloads=image_payloads)
        converted.append(clean)
    return Text.value(converted)


def chat_params(
    messages: list[Json],
    tools: list[Json] | None,
    provider: ProviderConfig,
    resolved: ResolvedProvider,
    *,
    stream: bool,
    json_object: bool,
    builtin_tools: Callable[[ResolvedProvider | None], list[Json]],
    derive_cache_key: Callable[[ProviderConfig, list[Json] | None], str],
    apply_provider_params: Callable[[Json, ProviderConfig, ResolvedProvider | None], None],
) -> Json:
    """Assemble the Chat Completions request body from normalized messages and provider settings.

    The caller supplies the stream decision (`stream`), the constrained-decoding flag
    (`json_object`), and the ModelClient hooks the body depends on: builtin tool schemas
    (`builtin_tools`), cache-key derivation (`prompt_cache_key`), and the provider-specific
    reasoning/temperature fold (`apply_provider_params`).
    """
    params: Json = {"model": provider.model, "messages": messages, "stream": stream}
    # Constrained decoding beats asking nicely: where the provider implements it, a reply that
    # is not a JSON object becomes unreachable rather than merely discouraged. Gated on the
    # catalog because an unsupporting gateway answers 400, and the prompt reminder plus the
    # retry are what carry the providers left out.
    if json_object and resolved.json_response_format:
        params["response_format"] = {"type": "json_object"}
    if provider.max_tokens > 0:
        params["max_tokens"] = provider.max_tokens
    if request_tools := [*(tools or []), *builtin_tools(resolved)]:
        params["tools"] = request_tools
        params["tool_choice"] = "auto"
        params["parallel_tool_calls"] = True
    prompt_cache_key = derive_cache_key(provider, tools)
    if prompt_cache_key:
        params["prompt_cache_key"] = prompt_cache_key
    apply_provider_params(params, provider, resolved)
    if stream:
        params["stream_options"] = {"include_usage": True}
    return params


async def reassemble_stream(
    client: AsyncOpenAI,
    params: Json,
    *,
    message_field: Callable[[Any, str], Any],
    dump_message_item: Callable[[Any], Json],
    raise_if_inactive: Callable[[], None],
    emit: Callable[[str, str], None],
) -> tuple[Json, Any, str]:
    """Reassemble a streamed chat completion into one assistant message and its finish reason.

    Tool calls are the hard part. The spec streams them as deltas keyed by `index`, but providers
    variously omit it, restart it, or send only `id`. `resolve_tool_call_index` recovers the
    association from whatever a chunk carries, in decreasing order of reliability, and raises
    instead of guessing when nothing identifies the call: a wrong association concatenates two
    calls' argument fragments into one call with corrupt JSON, which the model cannot correct
    because it looks like something it wrote.

    Unlike Responses, Chat has no separate text-done event. Do not promote on the first tool
    delta: compatible providers can vary their delta order. `finish_reason=tool_calls` is the
    first protocol boundary that proves this assistant message is complete.

    `message_field` reads a field off an SDK object or dict, `dump_message_item` normalizes a
    stream item to JSON, `raise_if_inactive` aborts a cancelled request, and `emit` publishes
    stream deltas — all ModelClient hooks passed in by the caller.
    """
    content: list[str] = []
    reasoning_content: list[str] = []
    reasoning: list[str] = []
    reasoning_details: list[Json] = []
    tool_calls: dict[int, Json] = {}
    tool_call_functions: dict[int, Json] = {}
    tool_call_ids: dict[str, int] = {}
    tool_call_positions: dict[int, int] = {}
    next_index = 0
    usage: Any = None
    output_promoted = False
    finish_reason = ""

    def allocate_tool_call() -> int:
        nonlocal next_index
        while next_index in tool_calls:
            next_index += 1
        index = next_index
        next_index += 1
        return index

    def resolve_tool_call_index(raw_index: object, call_id: str, position: int, chunk_size: int) -> int:
        nonlocal next_index
        if isinstance(raw_index, int):
            index = raw_index
        elif call_id and call_id in tool_call_ids:
            index = tool_call_ids[call_id]
        elif call_id:
            index = allocate_tool_call()
        elif chunk_size == 1 and len(tool_calls) == 1:
            index = next(iter(tool_calls))
        elif position in tool_call_positions and chunk_size == len(tool_call_positions):
            index = tool_call_positions[position]
        elif position not in tool_call_positions:
            index = allocate_tool_call()
        else:
            raise ModelError("Chat stream tool-call delta omitted both index and id; cannot associate it safely")
        next_index = max(next_index, index + 1)
        tool_call_positions[position] = index
        if call_id:
            tool_call_ids[call_id] = index
        return index

    try:
        async for chunk in await client.chat.completions.create(**params):
            raise_if_inactive()
            if chunk_usage := message_field(chunk, "usage"):
                usage = chunk_usage
            choices = message_field(chunk, "choices") or []
            if not choices:
                continue
            choice = choices[0]
            delta = message_field(choice, "delta")
            reasoning_content_delta = str(message_field(delta, "reasoning_content") or "")
            reasoning_delta = str(message_field(delta, "reasoning") or "")
            if reasoning_content_delta:
                reasoning_content.append(reasoning_content_delta)
                emit("reasoning", reasoning_content_delta)
            elif reasoning_delta:
                reasoning.append(reasoning_delta)
                emit("reasoning", reasoning_delta)
            raw_details = message_field(delta, "reasoning_details") or []
            details = [dump_message_item(item) for item in raw_details]
            reasoning_details.extend(item for item in details if item)
            if not reasoning_content_delta and not reasoning_delta:
                for detail in details:
                    text = detail.get("text") if detail.get("type") == "reasoning.text" else detail.get("summary")
                    if text:
                        emit("reasoning", str(text))
            if content_delta := str(message_field(delta, "content") or ""):
                content.append(content_delta)
                emit("output", content_delta)
            raw_tool_calls = message_field(delta, "tool_calls") or []
            for position, raw in enumerate(raw_tool_calls):
                raw_index = message_field(raw, "index")
                call_id = str(message_field(raw, "id") or "")
                index = resolve_tool_call_index(raw_index, call_id, position, len(raw_tool_calls))
                if index not in tool_calls:
                    function_target: Json = {"name": "", "arguments": ""}
                    tool_calls[index] = {"id": "", "type": "function", "function": function_target}
                    tool_call_functions[index] = function_target
                call = tool_calls[index]
                if call_id:
                    call["id"] = call_id
                function = message_field(raw, "function")
                target = tool_call_functions[index]
                if name := message_field(function, "name"):
                    target["name"] = str(name)
                if arguments := message_field(function, "arguments"):
                    target["arguments"] = str(target["arguments"]) + str(arguments)
            if chunk_finish_reason := str(message_field(choice, "finish_reason") or ""):
                finish_reason = chunk_finish_reason
            if finish_reason == "tool_calls" and content and tool_calls and not output_promoted:
                emit("output_done", "".join(content))
                output_promoted = True
    finally:
        emit("", "")
    message: Json = {"content": "".join(content) or None}
    if reasoning_content:
        message["reasoning_content"] = "".join(reasoning_content)
    if reasoning:
        message["reasoning"] = "".join(reasoning)
    if reasoning_details:
        # This optional compatible-wire field is an ordered delta sequence; replay it unchanged.
        message["reasoning_details"] = reasoning_details
    if tool_calls:
        message["tool_calls"] = [tool_calls[index] for index in sorted(tool_calls)]
    return message, usage, finish_reason
