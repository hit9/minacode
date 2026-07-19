# Context and caching

Each request contains more than the latest message. nanocode arranges the model's context so
the stable parts come first, the active conversation follows, and frequently changing task
state comes last. This keeps the agent informed while giving supported providers a long,
reusable prompt prefix.

## What the model receives

<div class="context-stack" role="img" aria-label="The message context, ordered from system instructions first to task memory last.">
  <div class="context-stack-edge">Context begins</div>
  <div class="context-stack-layer"><strong>System instructions</strong><span>How the agent should operate</span></div>
  <div class="context-stack-layer"><strong>Project environment</strong><span>Directory · OS · shell · detected commands</span></div>
  <div class="context-stack-layer context-stack-optional"><strong>Installed skills</strong><span>Index only · omitted when none are installed</span></div>
  <div class="context-stack-layer context-stack-optional"><strong>Connected MCP servers</strong><span>Names and tools · omitted until connected</span></div>
  <div class="context-stack-layer"><strong>Conversation history</strong><span>User messages · agent replies · tool calls and results</span></div>
  <div class="context-stack-layer"><strong>Current turn</strong><span>The latest messages and tool results</span></div>
  <div class="context-stack-layer"><strong>Task memory</strong><span>Goal · plan · facts · checks · recent errors</span></div>
  <div class="context-stack-edge">Context ends</div>
</div>

Tool definitions are sent beside this message stack: built-in tools, `Skill` when skills are
installed, and MCP tools and resources from <span class="marker">currently connected servers</span>.
Configured but disconnected servers do not consume context.

## Keeping context manageable

Large tool results are shortened before they enter the conversation; the agent can retrieve the
complete result later with `Recall`. Repeated skill instructions and MCP descriptions are replaced
with references to their first full copy instead of being sent in full again.

When the estimated context reaches `runtime.max_context_tokens`, nanocode compacts older history
while retaining the recent conversation and current task state. Compaction may discard detail, so
important decisions and facts are kept in task memory.

## Prompt caching

Prompt caching lets a provider reuse work for an unchanged beginning of a request. The next request
usually begins with the same instructions, environment, tools, and earlier conversation, so only
the new tail needs to be processed. Connecting an MCP server, changing installed skills, switching
models, or otherwise changing an early section can reduce the next request's cache hit.

<div class="cache-comparison" role="img" aria-label="A previous and next request share an unchanged prefix. The provider reuses that prefix and processes the changed tail.">
  <div class="cache-comparison-label">Previous request</div>
  <div class="cache-comparison-bar"><span class="cache-comparison-hit">Unchanged prefix</span><span class="cache-comparison-tail">Previous tail</span></div>
  <div class="cache-comparison-label">Next request</div>
  <div class="cache-comparison-bar"><span class="cache-comparison-hit">Reused by provider</span><span class="cache-comparison-tail">New or changed tail</span></div>
  <div class="cache-match"><span>stable key<sup>*</sup></span><b>→</b><span>provider compares exact prefix</span><b>→</b><span>cached tokens in <code>/status</code></span></div>
  <small><sup>*</sup> Where supported; Anthropic uses an explicit cacheable prefix.</small>
</div>

The provider reuses the request only up to the first difference. Its reported cached-token count,
shown by `/status`, confirms how much matched. A change near the beginning, such as connecting an
MCP server, shortens the reusable prefix.

OpenAI-compatible providers may reuse matching prefixes automatically; nanocode supplies a stable
cache key where the provider supports one. For Anthropic, nanocode explicitly marks the tools and
system instructions as an ephemeral cacheable prefix. Provider support and accounting differ.

`/status` shows the cached prompt tokens reported by the provider for the whole session and the
latest request. The ratio varies with the provider, model, prompt length, and conversation. When
request prefixes line up, it <span class="marker">can reach 90–99%</span>. This is an observation,
not a guaranteed rate.
