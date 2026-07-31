"""Test-only OpenAI HTTP behavior model for prompt-cache black-box tests.

This is deliberately not a tokenizer or a complete API emulator. It models the observable contract
the tests need: Chat Completions and Responses request shapes, implicit breakpoints at user/tool
boundaries, longest exact-prefix reads, and cache-write accounting.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from openai import OpenAI


class OpenAIMockServer:
    def __init__(self, answers: list[str | dict[str, Any]]):
        self.answers = list(answers)
        self.requests: list[dict[str, Any]] = []
        self.cache_events: list[tuple[int, int, int]] = []
        self.cache: dict[str, set[str]] = {}

    @staticmethod
    def _serialized(value: object) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def _tokens(cls, value: object) -> int:
        return max(1, (len(cls._serialized(value)) + 3) // 4)

    @staticmethod
    def _is_breakpoint(item: dict[str, Any], *, responses: bool) -> bool:
        if responses:
            return (item.get("role") == "user") or item.get("type") == "function_call_output"
        return item.get("role") in {"user", "tool"}

    def _cache_usage(self, body: dict[str, Any], *, responses: bool) -> tuple[int, int, int]:
        items = body.get("input" if responses else "messages") or []
        tools = body.get("tools") or []
        candidates = [
            self._serialized({"tools": tools, "prefix": items[: index + 1]})
            for index, item in enumerate(items)
            if isinstance(item, dict) and self._is_breakpoint(item, responses=responses)
        ]
        scope = str(body.get("prompt_cache_key") or "implicit")
        cached = self.cache.setdefault(scope, set())
        hits = [candidate for candidate in candidates if candidate in cached]
        cached_tokens = max((self._tokens(candidate) for candidate in hits), default=0)
        latest = candidates[-1] if candidates else ""
        cache_write_tokens = self._tokens(latest) if latest and latest not in cached else 0
        if latest:
            cached.add(latest)
        prompt_tokens = self._tokens({"tools": tools, "items": items})
        return prompt_tokens, cached_tokens, cache_write_tokens

    def _handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        self.requests.append(body)
        responses = request.url.path.endswith("/responses")
        prompt_tokens, cached_tokens, cache_write_tokens = self._cache_usage(body, responses=responses)
        self.cache_events.append((prompt_tokens, cached_tokens, cache_write_tokens))
        answer = self.answers.pop(0)
        output_tokens = self._tokens(answer)
        if responses:
            if isinstance(answer, dict):
                output = [
                    {
                        "id": f"fc_{len(self.requests)}",
                        "type": "function_call",
                        "status": "completed",
                        "call_id": f"call_{len(self.requests)}",
                        "name": answer["tool"],
                        "arguments": json.dumps(answer["arguments"]),
                    }
                ]
            else:
                output = [
                    {
                        "id": f"msg_{len(self.requests)}",
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": answer, "annotations": [], "logprobs": []}],
                    }
                ]
            response = {
                "id": f"resp_{len(self.requests)}",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": body.get("model", "gpt-5.6"),
                "output": output,
                "usage": {
                    "input_tokens": prompt_tokens,
                    "input_tokens_details": {
                        "cached_tokens": cached_tokens,
                        "cache_write_tokens": cache_write_tokens,
                    },
                    "output_tokens": output_tokens,
                    "total_tokens": prompt_tokens + output_tokens,
                },
            }
        else:
            message: dict[str, Any]
            finish_reason = "stop"
            if isinstance(answer, dict):
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call_{len(self.requests)}",
                            "type": "function",
                            "function": {"name": answer["tool"], "arguments": json.dumps(answer["arguments"])},
                        }
                    ],
                }
                finish_reason = "tool_calls"
            else:
                message = {"role": "assistant", "content": answer}
            response = {
                "id": f"chatcmpl_{len(self.requests)}",
                "object": "chat.completion",
                "created": 1,
                "model": body.get("model", "gpt-5.6"),
                "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": prompt_tokens + output_tokens,
                    "prompt_tokens_details": {
                        "cached_tokens": cached_tokens,
                        "cache_write_tokens": cache_write_tokens,
                    },
                },
            }
        return httpx.Response(200, json=response)

    def client(self) -> OpenAI:
        transport = httpx.MockTransport(self._handle)
        return OpenAI(
            api_key="sk-test",
            base_url="http://test",
            http_client=httpx.Client(transport=transport),
            max_retries=0,
        )
