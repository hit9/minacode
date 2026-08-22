"""Responses wire conversion: input replay, request extras, stream reassembly, and result parsing.

Pure conversion with explicit inputs — no client state and no retry or orchestration logic.
ModelClient keeps the flow decisions and delegates the conversion here.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from minacode.base import (
    RESPONSES_OUTPUT_KEY,
    SEARCH_SOURCES_KEY,
    Json,
    ModelError,
    ModelOutputTruncated,
    Text,
    ToolCall,
    builtin_tool_label,
)
from minacode.config import ProviderConfig
from minacode.image import ImageInputs

if TYPE_CHECKING:
    from openai import OpenAI


def responses_input(
    messages: list[Json],
    origin: str,
    *,
    provider_origin: Callable[[ProviderConfig | None], str],
    replayable_echo: Callable[[Json, str], bool],
    images: ImageInputs,
    text_only: bool = False,
) -> list[Json]:
    """Convert normalized messages to Responses input items.

    Replays saved output items when the origin still matches the endpoint, rebuilds function
    calls from normalized tool_calls, and substitutes image content for local image refs.
    `provider_origin` is the caller's endpoint identity (ModelClient.provider_origin) and
    `replayable_echo` its replay gate (ModelClient.replayable_echo). `text_only` is the route's
    image-delivery verdict: raw blocks are suppressed but readable labels and asset paths stay.
    """
    origin = origin or provider_origin(None)
    converted: list[Json] = []
    seen_output_ids: set[str] = set()
    for message in messages:
        role = str(message.get("role") or "")
        content = message.get("content")
        saved_output = message.get(RESPONSES_OUTPUT_KEY) if replayable_echo(message, origin) else None
        if role == "assistant" and isinstance(saved_output, list):
            for item in saved_output:
                if not isinstance(item, dict) or not replayable_output_item(item):
                    continue
                if content is None and item.get("type") == "message":
                    continue
                item_id = str(item.get("id") or "")
                if item_id and item_id in seen_output_ids:
                    continue
                if item_id:
                    seen_output_ids.add(item_id)
                converted.append(item)
            continue
        if role == "tool":
            converted.append(
                {
                    "type": "function_call_output",
                    "call_id": str(message.get("tool_call_id") or ""),
                    "output": str(message.get("content") or ""),
                }
            )
            continue
        if role not in ("system", "developer", "user", "assistant"):
            continue
        if content is not None:
            converted.append(
                {
                    "role": role,
                    "content": (
                        images.responses_content(message, text_only=text_only)
                        if role == "user" and images.refs(message)
                        # A pre-built content list (a vision-bridge request) is already in the
                        # Responses wire shape; str() would flatten it into one text blob.
                        else content
                        if isinstance(content, list)
                        else str(content)
                    ),
                }
            )
        if role == "assistant":
            for raw in message.get("tool_calls") or []:
                if not isinstance(raw, dict):
                    continue
                raw_function = raw.get("function")
                function = raw_function if isinstance(raw_function, dict) else {}
                converted.append(
                    {
                        "type": "function_call",
                        "call_id": str(raw.get("id") or uuid.uuid4().hex),
                        "name": str(function.get("name") or ""),
                        "arguments": str(function.get("arguments") or "{}"),
                    }
                )
    return converted


def replayable_output_item(item: Json) -> bool:
    """Whether a saved output item still carries something a later request can use.

    Stateless reasoning travels in the encrypted payload, which the id alone cannot stand in
    for once the response was never stored. A host that returns neither that payload nor any
    readable reasoning leaves an empty shell, so it is dropped instead of replayed."""
    return item.get("type") != "reasoning" or any(item.get(key) for key in ("encrypted_content", "content", "summary"))


def responses_tool_schemas(tools: list[Json]) -> list[Json]:
    """Convert normalized tool schemas to Responses `function` entries."""
    converted: list[Json] = []
    for schema in tools:
        raw_function = schema.get("function")
        function = raw_function if isinstance(raw_function, dict) else {}
        converted.append(
            {
                "type": "function",
                "name": str(function.get("name") or ""),
                "description": str(function.get("description") or ""),
                "parameters": function.get("parameters") if isinstance(function.get("parameters"), dict) else {},
                "strict": bool(function.get("strict", False)),
            }
        )
    return converted


def responses_extra_body(extra_body: Json, params: Json) -> Json:
    """Fold configured `reasoning` fields into the managed object instead of replacing it.

    `extra_body` is merged over the request body, so a whole object configured there would drop
    the fields minacode manages inside it — settling `reasoning.context` would silently take the
    resolved `effort` with it. Merging per field keeps a documented extra reachable while
    `/reason` stays authoritative, mirroring how the Chat path folds `thinking`.
    """
    merged = dict(extra_body)
    configured = merged.pop("reasoning", None)
    managed = params.get("reasoning")
    if isinstance(configured, dict):
        params["reasoning"] = {**configured, **managed} if isinstance(managed, dict) else configured
    elif configured is not None:
        # Nothing to merge field by field: a scalar keeps the plain override an unknown host may
        # want, rather than being second-guessed here.
        merged["reasoning"] = configured
    return merged


def reassemble_stream(
    client: OpenAI,
    params: Json,
    *,
    message_field: Callable[[Any, str], Any],
    raise_if_inactive: Callable[[], None],
    emit: Callable[[str, str], None],
    report_builtin_call: Callable[[str, object], None],
) -> Any:
    """Consume a Responses stream, promoting completed text before tool arguments finish.

    Text completion and function-call discovery are independent events and either can arrive
    first. Promotion is therefore a two-condition state transition, not an ordering assumption;
    the terminal response is still consumed normally for history, tool calls, and usage.

    `message_field` reads a field off an SDK object or dict, `raise_if_inactive` aborts a
    cancelled request, `emit` publishes stream deltas, and `report_builtin_call` records
    provider-side tool calls — all ModelClient hooks passed in by the caller.
    """

    terminal: Any = None
    output: list[str] = []
    text_done = handoff_seen = output_promoted = False

    def promote_output() -> None:
        nonlocal output_promoted
        if text_done and handoff_seen and output and not output_promoted:
            emit("output_done", "".join(output))
            output_promoted = True

    try:
        for event in client.responses.create(**params):
            raise_if_inactive()
            event_type = str(message_field(event, "type") or "")
            # Two spellings of the same event: hosts that summarize reasoning stream the summary,
            # hosts that expose the raw chain stream the text. DeepSeek only ever sends the
            # latter and documents that it generates no summary at all, so listening for one
            # spelling leaves a thinking model with no preview.
            # Evidence: https://api-docs.deepseek.com/guides/responses_api
            if event_type in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta"):
                emit("reasoning", str(message_field(event, "delta") or ""))
            elif event_type in ("response.output_text.delta", "response.refusal.delta"):
                delta = str(message_field(event, "delta") or "")
                output.append(delta)
                emit("output", delta)
            elif event_type in ("response.output_text.done", "response.refusal.done"):
                text_done = True
                promote_output()
            elif event_type == "response.output_item.added":
                item = message_field(event, "item")
                item_type = str(message_field(item, "type") or "")
                if item_type == "function_call":
                    handoff_seen = True
                    promote_output()
                elif item_type.endswith("_call"):
                    # A provider-side tool is the same durable tool boundary as a local function
                    # call: completed text before it must be handed off now, so the live status
                    # below never covers a finished answer. A provider-side tool also runs inside
                    # the request with no local tool line to show for it, so the status label is
                    # the only sign the turn is still moving.
                    handoff_seen = True
                    promote_output()
                    emit(builtin_tool_label(item_type), "")
            elif event_type == "response.output_item.done":
                item = message_field(event, "item")
                item_type = str(message_field(item, "type") or "")
                # A provider-side call has no local tool line of its own, so report it the moment
                # the stream completes it and the transcript shows it live. The stream and the
                # terminal output carry the same calls, so the parsed-result scan stays silent on
                # streaming requests; reporting here is the one and only record for them.
                if item_type.endswith("_call") and item_type != "function_call":
                    # Some compatible providers omit the matching output_item.added event, so the
                    # durable report below must also establish the promotion boundary itself.
                    handoff_seen = True
                    promote_output()
                    action = message_field(item, "action")
                    query = message_field(action, "query") if action is not None else ""
                    report_builtin_call(item_type, str(query or ""))
            elif event_type == "response.function_call_arguments.delta":
                handoff_seen = True
                promote_output()
            elif event_type in ("response.completed", "response.incomplete"):
                # Compatible providers may omit response.output_text.done; the accepted terminal
                # response proves the streamed text is final, so it is the terminal fallback for
                # text completion. The tool boundary guard keeps plain responses unpromoted.
                text_done = True
                promote_output()
                terminal = message_field(event, "response")
            elif event_type == "response.failed":
                terminal = message_field(event, "response")
    finally:
        emit("", "")
    if terminal is None:
        raise ModelError("Responses stream ended without a terminal response")
    return terminal


def responses_result(
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
    """Parse a Responses result into the normalized (assistant message, tool calls, text) triple.

    Replays the parsed output items under RESPONSES_OUTPUT_KEY, reports provider-side calls when
    the request was not streamed (the stream already reported them live), and raises the
    truncated-output error when the response stopped empty at the output cap.
    """
    if message_field(result, "status") == "failed":
        error = message_field(result, "error") or "unknown error"
        raise ModelError(f"Responses request failed: {error}")
    output = message_field(result, "output") or []
    saved_output = [dump_message_item(item) for item in output]
    text_parts: list[str] = []
    tool_calls: list[Json] = []
    calls: list[ToolCall] = []
    for item in output:
        item_type = message_field(item, "type")
        if item_type == "message":
            for part in message_field(item, "content") or []:
                part_type = message_field(part, "type")
                if part_type == "output_text":
                    text_parts.append(str(message_field(part, "text") or ""))
                elif part_type == "refusal":
                    text_parts.append(str(message_field(part, "refusal") or ""))
        elif item_type == "function_call":
            name = str(message_field(item, "name") or "")
            call_id = str(message_field(item, "call_id") or message_field(item, "id") or uuid.uuid4().hex)
            arguments = str(message_field(item, "arguments") or "{}")
            try:
                payload = json.loads(arguments, strict=False)
            except json.JSONDecodeError:
                payload = {}
            tool_calls.append({"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}})
            calls.append(tool_call(call_id, name, payload))
    text = "".join(text_parts) or str(message_field(result, "output_text") or "")
    # Streaming already reported every provider-side call live, and the stream and the terminal
    # output carry the same calls, so scanning again would double each one — and a call without an
    # id could not be de-duplicated at all. The scan is the only source for non-streaming requests.
    if not streamed:
        for item in saved_output:
            item_type = str(item.get("type") or "")
            if item_type.endswith("_call") and item_type != "function_call":
                action = item.get("action")
                query = action.get("query") if isinstance(action, dict) else ""
                report_builtin_call(item_type, query if isinstance(query, str) else "")
    if not calls and not text.strip() and message_field(result, "status") == "incomplete":
        details = message_field(result, "incomplete_details")
        if message_field(details, "reason") == "max_output_tokens":
            raise truncated_output_error(message_field(result, "usage"))
    assistant: Json = {"role": "assistant", "content": text or None, RESPONSES_OUTPUT_KEY: saved_output}
    if sources := responses_sources(saved_output, collect_sources):
        assistant[SEARCH_SOURCES_KEY] = sources
    if tool_calls:
        assistant["tool_calls"] = tool_calls
    return assistant, calls, text


def responses_sources(saved_output: list[Json], collect_sources: Callable[..., list[Json]]) -> list[Json]:
    """Sources a Responses host attached to one response.

    Two hosts, two places: OpenAI cites inline through `url_citation` annotations on the
    message, while Qwen returns no citations at all and reports sources only on the search
    call. Reading both keeps one renderer honest across them."""
    groups: list[Any] = []
    for item in saved_output:
        if item.get("type") == "message":
            for part in item.get("content") or []:
                if isinstance(part, dict):
                    groups.append(part.get("annotations"))
            continue
        action = item.get("action")
        groups.append(action.get("sources") if isinstance(action, dict) else None)
        groups.append(item.get("results"))
    return collect_sources(*groups)


def dump_message_item(item: Any) -> Json:
    """Normalize one SDK message item (dict or model) to JSON, or {} when it carries nothing usable."""
    if isinstance(item, dict):
        return Text.value(item)
    if hasattr(item, "model_dump"):
        dumped = item.model_dump(mode="json", exclude_none=True)
        if isinstance(dumped, dict):
            return Text.value(dumped)
    return {}
