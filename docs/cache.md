# Prompt cache

Some model providers can reuse work when the beginning of a request is unchanged. nanocode
keeps its instructions, environment, and tool schemas stable so supported providers can cache
that prefix.

What gets cached depends on the provider. OpenAI-compatible providers may reuse matching
conversation prefixes automatically. For Anthropic, nanocode explicitly marks the stable tools
and system instructions for caching. Older messages and tool results may become part of a
reusable prefix; the newest content has no matching earlier request yet.

`/status` shows the cached prompt tokens reported by your provider for the whole session and
the latest request. The ratio varies with the provider, model, prompt length, and conversation;
when the request prefixes line up, it can reach 90–99%. This is an observation, not a
guaranteed rate.
