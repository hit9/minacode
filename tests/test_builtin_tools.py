"""Provider-side builtin tools: config parsing, pass-through on every protocol, results, and echoes."""




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
















































































def _chat_body(model="qwen3.8-max-preview"):
    return {
        "id": "c",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
    }
























