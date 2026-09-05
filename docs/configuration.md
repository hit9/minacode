# Configuration

wizolt reads a single TOML file, `~/.wizolt/config.toml` by default. Generate a
commented starter with `wizolt --init-config`, or point at another file with
`--config <path>`.

<span class="marker">Only the `[provider]` block is required.</span> Every other key falls back to
a built-in default, so a minimal config is just a provider. Inspect the resolved configuration
at any time with `/config`.

## Providers

wizolt supports OpenAI-compatible Chat Completions and Responses APIs, plus the Anthropic
Messages API. Define one or more `[provider.<name>]` blocks and select one with
`[provider] active`:

```toml
[provider]
active = "default"

[provider.default]
url = "https://api.deepseek.com"
key = "sk-..."
model = "deepseek-v4-flash"
```

These three fields are enough for most endpoints. wizolt selects the usual protocol and applies
only necessary, documented compatibility adjustments. Explicit settings always take precedence.
Use `/config` to inspect the result. [Compatibility catalog](catalog.md) explains which provider
and model facts wizolt maintains, how they update, and what happens to unknown endpoints.

Define additional blocks to use more providers. Switch between them with `/provider [NAME]`, and
switch the active model with `/model [MODEL]`.

### API protocol

Leave `api = "auto"` unless your endpoint needs an explicit protocol:

| Value | Meaning |
|---|---|
| `auto` | Infer the protocol when possible; otherwise use Chat Completions |
| `chat` | OpenAI-compatible Chat Completions |
| `responses` | OpenAI-compatible Responses |
| `anthropic` | Anthropic-compatible Messages |

A URL ending in `/chat/completions`, `/responses`, or `/messages` also selects that protocol.

### Optional provider settings

Most users can leave these unset.

| Key | Default | Meaning |
|---|---|---|
| `api` | `auto` | API protocol shown above |
| `stream` | `true` | Stream model output; disable for endpoints that reject streaming or Chat `stream_options` |
| `reasoning` | `medium` | Reasoning effort: `off`, `minimal`, `low`, `medium`, `high`, `xhigh`, or `max`; change it during a session with `/reason` |
| `available_models` | — | Additional models shown by `/model` |
| `temperature` | — | Sampling temperature; omitted by default |
| `max_tokens` | `0` | Output-token cap per model request, reasoning included; `0` leaves it to the provider (Anthropic sends a conservative 8K). 16K is still reserved from the input budget for the answer, trading against `max_context_tokens` one for one |
| `max_context_tokens` | `0` | How much of *this* entry's model window to use; `0` inherits `runtime.max_context_tokens`. Set it per entry when entries point at models with different windows |
| `timeout` | `120` | Transport inactivity timeout in seconds |
| `response_timeout` | `600` | Total generation limit in seconds; `0` disables it |
| `prompt_cache_key` | `auto` | Stable prompt-cache key; set `off` to omit it |
| `strict_tools` | `false` | Request strict function schemas where supported; toggle with `/strict` |
| `headers` | `{}` | Extra HTTP headers sent with every request to this entry; see below |
| `omit_body` | `[]` | Request fields this endpoint rejects; see below |
| `extra_body` | `{}` | Extra fields for an OpenAI-compatible request body. Fields inside an object wizolt also manages are merged rather than replacing it, so `extra_body.reasoning.context` reaches a Responses host without dropping the resolved effort |
| `builtin_tools` | `[]` | Tools the provider runs itself, passed through verbatim; see below |
| `chat_reasoning` | `auto` | Provider-specific Chat reasoning format; normally leave on `auto` |
| `reasoning_history` | `auto` | Reasoning replay policy on Chat, Responses, and Anthropic: catalog-selected by default; `all`, `current_turn`, or `tool_calls` explicitly overrides it |

### Extra HTTP headers

Some features live in the header rather than the request body, where `extra_body` cannot reach
them. `headers` sends whatever the provider documents alongside every request from that entry:

```toml
[provider.cmd]
url = "https://api.commandcode.ai/provider/v1"
key = "..."
model = "deepseek/deepseek-v4-flash"
headers = { x-cmd-zdr = "1" }   # Command Code: route only to zero-retention upstreams
```

Values are ASCII strings or plain integers. `key` still supplies authentication, so a header is
only needed for what the provider documents separately — zero-retention routing, a gateway's
tenant or routing key. The same headers are used when `/model` asks the endpoint for its model
list. `/config` lists the headers in effect.

### Fields an endpoint rejects

Some endpoints answer `400` for a field wizolt sends. Name it in `omit_body` and it is left out:

```toml
[provider.gw]
omit_body = ["reasoning_effort", "stream_options"]
```

`extra_body` is the other half — it adds fields, `omit_body` removes them. A name is dropped
wherever it sits in the request. `model`, `messages`, `input`, and `stream` cannot be removed;
use `stream = false` when an endpoint does not stream. `/config` lists what is being omitted.

Streaming is enabled by default for all three protocols. If a compatible endpoint does not
support it, set `stream = false` in that provider block, or use `/set provider.stream off` for
the current session.

`timeout` detects a connection that stops delivering data. Streaming reasoning can keep that
timer active indefinitely, so `response_timeout` separately limits the complete model response to
ten minutes by default. Reaching the total limit cancels the request without automatic retries;
set it to `0` only when deliberately allowing unbounded generations.

`/reason` offers the levels the active model documents, and sends the one you pick. DeepSeek
models offer `off, low, high, max`; a model wizolt has no evidence about keeps the full scale
(`minimal, low, medium, high, xhigh, max`). Unknown OpenAI-compatible endpoints and model names
stay on the generic path rather than an allowlist; set `api` and `chat_reasoning` explicitly if
automatic selection is wrong. `/config` lists the levels in `provider.supported_reasoning`, while
`/status` shows the active model and cache usage reported by the provider.

When the list is shorter than the full scale, `/reason` shows why underneath it, with the page it
came from:

```
Reasoning effort
   1. off - disable reasoning
   2. low
   3. high
   4. max
  ──────────────────────────────────
  │ Why these levels
  │ DeepSeek documents low/high/max; medium and xhigh are served as high
  │ https://api-docs.deepseek.com/guides/thinking_mode/
```

Switching model or provider can leave an effort the new model has no level for. wizolt moves it
to the nearest one and says so once:

```
Reasoning medium is not offered by deepseek-v4-flash, using high
```

### Effort levels a model accepts

When wizolt's list is wrong for your model, say what the model accepts. Write the levels weakest
first, under a model name or a glob:

```toml
[provider.gw.models]
"gpt-5.6*" = { reasoning = ["low", "medium", "high", "ultra"] }
```

Those levels become what `/reason` offers for models the glob matches, replacing wizolt's own.
They can include names wizolt does not know — `ultra` above — since they are sent as written.
Each list must contain unique levels and must not include `off`; wizolt adds `off` unless the
catalog documents that the model always reasons. Worker and compaction reasoning overrides use
the scale declared for their effective model too.

## Provider-side tools

Some providers can run web search themselves; see
[Provider-side tools](tools.md#provider-side-tools) for what that looks like in a session. List
the ones you want in `builtin_tools`, written the way your provider documents them:

```toml
[provider]
url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
model = "qwen3-max"
api = "responses"
builtin_tools = [{ type = "web_search" }, { type = "web_extractor" }]
```

| Provider | Entry |
|---|---|
| OpenAI (Responses) | `{ type = "web_search" }`, optionally with `search_context_size` or `filters` |
| Qwen (Responses) | `{ type = "web_search" }`; also `web_extractor` |
| Anthropic | `{ type = "web_search_20250305", name = "web_search", max_uses = 5 }` |
| Z.AI / BigModel | `{ type = "web_search", web_search = { enable = "True" } }` |
| Kimi / Moonshot | `{ type = "builtin_function", function = { name = "$web_search" } }` |
| OpenRouter | `{ type = "openrouter:web_search" }`; also `openrouter:web_fetch`, `openrouter:datetime` |

One provider configures search elsewhere, through [`extra_body`](#optional-provider-settings):
Qwen's Chat Completions endpoint takes `enable_search`. DeepSeek has no web search.

Builtin tools only work with the APIs shown in the table. If you switch to another API, wizolt
keeps the setting but does not send those tools; switching back enables them again. Use `/config`
to check whether they are active. If wizolt reports an unsupported entry, compare it with the
example for your provider.

## Vision model

An optional `[vision]` entry gives image input a fallback when the active model is text-only:

```toml
[vision]
provider = "vision"

[provider.vision]
url = "https://api.deepseek.com"
key = "sk-..."
model = "deepseek-v4-flash-vision-exp"
```

An attached image is first sent to the active model unless wizolt knows it rejects images:
documented text-only families from a static catalog, or the same model returning HTTP 400 for an
image in this session. In those cases `[vision]` describes the image once and its text replaces
the raw image, and `ViewImage` falls back the same way. Each fallback description costs one
vision-model request. Without `[vision]`, a text-only route keeps sending to the active model and
you see the provider's own error.

## Runtime

Optional; the defaults shown are used when omitted.

| Key | Default | Meaning |
|---|---|---|
| `yolo` | `false` | Start without confirmation prompts |
| `max_context_tokens` | `262144` (256K) | Default for every provider entry that does not set its own `max_context_tokens`. How much of the model's context window to use, which sets the automatic-compaction budget — a budget, not the window's size: raise it for a 1M-window model, lower it for a smaller one |
| `max_agent_steps` | `400` | Maximum tool steps in one turn |
| `shell_timeout` | `60` | Maximum shell-command lifetime, in seconds |
| `bash_wait_timeout` | `10` | Foreground wait before a running command becomes a background job; `0` disables promotion |
| `max_parallel_tools` | `4` | Maximum read-only tool calls executed concurrently; `1` disables parallelism |
| `session_retention_days` | `7` | Delete saved sessions untouched for this many days, swept in the background at startup; `0` keeps them indefinitely |
| `theme` | `auto` | Terminal color scheme: `auto`, `light`, or `dark`; overridden by `--theme`. `auto` reads `COLORFGBG` and falls back to `dark` |
| `worker` | `false` | Let the model delegate to a second in-process session; see below |
| `language` | `auto` | Force the reply language (`auto` follows your messages and injects nothing); set a name like `Chinese` to append a fixed `LANGUAGE OVERRIDE` block to the system prompt. Change for the current session with `/language` |
| `agents_md` | `true` | Inject the project's `AGENTS.md` (falling back to `CLAUDE.md`) into every request as a bounded "Project instructions" section of the Environment block |

Selected tuning values can be changed for the current session with `/set` (Tab completion
lists the supported keys). `/yolo` toggles `yolo`.

## Worker delegation

The `Delegate` tool and `/worker` command hand bounded tasks to a second in-process session.
Concepts, quick start, and what you see in the terminal: [Worker delegation](worker.md).

Worker keys inherit the `[worker]` provider entry by default:

| Key | Default | Meaning |
|---|---|---|
| `[worker] provider` | — | Worker provider entry key; unset disables delegation |
| `[worker] model` | inherit | Override the entry's model; empty inherits |
| `[worker] reasoning` | inherit | Override the entry's reasoning effort; empty inherits |
| `[worker] api` | inherit | Override the entry's wire protocol; empty inherits |

`[runtime] worker` and `[runtime] language` are in the [Runtime](#runtime) table above.

## Compaction model

Context compaction (the summary request that makes room in the context window) runs on the active
provider by default. A `[compaction]` section overrides it per field, mirroring `[worker]`: an
empty `provider` means the active provider entry, and each empty override inherits that entry's
value. The context budget is unaffected — requests are still prepared against the active
provider's window; only the summary request itself uses this entry.

| Key | Default | Meaning |
|---|---|---|
| `[compaction] provider` | inherit | Base provider entry key; empty = the active provider |
| `[compaction] model` | inherit | Override the entry's model; empty inherits |
| `[compaction] reasoning` | inherit | Override the entry's reasoning effort; empty inherits |
| `[compaction] api` | inherit | Override the entry's wire protocol; empty inherits |

Each provider entry can also nest its own `compaction` table (`[provider.NAME.compaction]`) with
the same `model`/`reasoning`/`api` keys — there is no `provider` key there, the base entry comes
only from the global `[compaction] provider`. Per field the most specific value wins: the base
entry's nested table, then the global `[compaction]` section, then the entry's own value.

```toml
[provider.anthropic]
model = "claude-..."

  [provider.anthropic.compaction]
  model = "claude-haiku-..."
```

Write the nested table under a *named* entry, as above. In the short single-provider form, where
`[provider]` holds `url` and `key` directly, `[provider.compaction]` reads as a provider named
`compaction` and the config is rejected with `provider.active does not exist`.

Summaries can also run on a different vendor entirely — the entry supplies its own url, key, and
wire protocol, so the conversation stays on one host while summaries go to another:

```toml
[provider]
active = "anthropic"

[provider.anthropic]
url = "https://api.anthropic.com/v1"
key = "sk-ant-..."
model = "claude-..."

[provider.deepseek]
url = "https://api.deepseek.com/v1"
key = "sk-..."
model = "deepseek-..."

[compaction]
provider = "deepseek"
model = "deepseek-...-flash"
reasoning = "off"
```

Two things to check when picking a summarizer:

- **Its window must fit the span being compacted.** A small-window model asked to summarize a
  large conversation fails every time, and the context is then trimmed without a summary.
- **Its `response_timeout` applies to the summary.** A slow summarizer holds up the turn for that
  long before giving up. Lower it on that entry if you would rather trim early than wait.

`/config` shows the effective `compaction.*` values, and `/compact log` records which model
produced each stored segment. Summary
tokens are counted apart from the conversation, on their own `compaction usage` row in `/status`,
so each row can be read against one model's price.

## Data location

```toml
[paths]
data_dir = "~/.wizolt"   # sessions, input history, OAuth tokens, user skills, update cache
```

Sessions live under `<data_dir>/projects/<project>/`, one directory per working directory. Each
holds that project's session logs and a `latest` pointer, so a resume stays scoped to the project
it belongs to. A project directory is removed once its last session expires.

Beside each log sits a small `<uid>.meta.json` holding what the session picker shows — name,
opening line, round count. The log stays the source of truth; deleting a sidecar only costs that
session its label in the list.

`<data_dir>/history.txt` holds the input history that Up and Ctrl-P recall, across every project.
It is capped at 512 KB, keeping the most recent entries.
