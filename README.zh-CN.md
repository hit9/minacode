# nanocode

一个用 Python 写的小型终端编程代理。

[English](README.md)

[中文博客](https://hit9.dev/post/nanocode)

nanocode 仍是 1.0 前的软件。稳定版发布前，命令、配置和工具行为都可能变化。

![nanocode screenshot](snapshots/nanocode-snapshot.png)

## 概览

nanocode 是面向本地开发工作的终端编程代理。它把模型选择、历史搜索、确认、实时命令输出、排队输入、session 恢复和状态展示都放在一个 CLI 里。

核心能力：

- 通过 `+>` 提示符在代理运行时追加输入。
- 通过 `Read`、`Search`、`InspectCode` 和 `Edit` 构建文件感知上下文。
- 使用当前 `line:hash` 锚点保护编辑，避免修改漂移后的代码。
- 通过可选代码符号索引进行项目导航。
- 通过紧凑的 `tr.N` 引用和 `Recall` 恢复工具结果。
- 通过 `Note` 维护聚焦的工作记忆。
- 集成远程 HTTP 和本地 stdio MCP 服务器。
- 通过 `nanocode --resume` 恢复 append-only session。

## 安装

使用 uv 安装：

```sh
uv tool install nanocode-cli
```

升级：

```sh
uv tool upgrade nanocode-cli
```

本地开发：

```sh
uv sync --extra dev
uv run nanocode
```

## 快速开始

创建配置文件：

```sh
nanocode --init-config
```

编辑 `~/.nanocode/config.toml`，然后启动：

```sh
nanocode
```

常用参数：

- `--config <path>`：使用指定 TOML 配置文件。
- `--init-config`：创建默认配置文件。
- `--resume [UID]`：恢复已保存的 session；不传 `UID` 时恢复 `latest`。
- `--yolo`：跳过会修改环境的工具确认。
- `--mcp <selector>`：选择启用哪些已配置的 MCP 服务器。
- `--debug`：写入模型 I/O debug trace。
- `-v`, `--version`：显示版本。

代理运行中，可以在 `+>` 提示符里输入追加内容，发送到下一次模型请求。

## Sessions

nanocode 会把有可恢复内容的 session 保存到 `[paths] data_dir` 下，格式是 append-only JSONL snapshot。空 session 不会保存。

退出时，nanocode 会打印恢复命令：

```sh
Resume with: nanocode --resume <session-id>
```

恢复 session：

```sh
nanocode --resume <session-id>
nanocode --resume latest
nanocode --resume last
```

恢复后会在启动时重新渲染一次会话历史。工具执行摘要会再次显示，但不会打印原始 tool result 正文。`/status` 会显示当前 session id。

snapshot 只保存 nanocode 恢复所需的数据：会话消息、usage、工作笔记、tool records 和 tool errors。runtime settings、config、git branch 以及其他可重建状态会从当前环境和配置读取，不写入 snapshot。

启动时会删除早于 `runtime.session_retention_days` 的 session 文件。默认值是 `7`；设置为 `0` 可关闭保留期清理。

## CLI

命令：

- `/help`：显示命令和工具。
- `/status`：显示运行状态，包括当前 session id。
- `/config`：显示当前配置。
- `/api [auto|chat|anthropic]`：显示或设置 provider API 格式。
- `/debug [on|off]`：切换模型 I/O debug trace。
- `/compact`：立即压缩上下文。
- `/index [force]`：同步或重建代码符号索引。
- `/mcp [tools|login|logout|refresh] ...`：管理 MCP 服务器和工具。
- `/provider [NAME]`：显示或设置 provider。
- `/model [MODEL]`：显示或设置模型。
- `/reason`：选择 reasoning effort。
- `/set KEY VALUE`：设置当前 session 支持的 provider/runtime 值。
- `/yolo`：切换工具确认。
- `/exit`, `/quit`：退出。

交互选择器支持 `j`/`k`、方向键、`/` 搜索、Enter 和 Esc。输入框支持历史、补全和 `Ctrl-R` 历史搜索。

工具：

- 文件：`Read`, `LineCount`, `List`, `Find`, `Search`。
- 代码索引：`InspectCode`。
- 编辑：`Edit` 创建或修改文件内容。
- Shell：`Bash`, `Git`。
- 工具结果：`Recall`。
- 工作笔记：`Note`。
- 询问用户：`Question`。
- MCP：`MCP`。

`Read`、`Search` 和 `InspectCode` 会在合适时返回行锚点。`Edit` 使用当前 `line:hash` 锚点拒绝过期编辑。

## 配置

默认配置位置：

```text
~/.nanocode/config.toml
```

主要字段：

- `[provider] active = "name"`
- `[provider.<name>]`：`url`, `key`, `model`, `api`, `prompt_cache_key`, `available_models`, `reasoning`, `chat_reasoning`, `temperature`, `timeout`
- `[paths] data_dir`
- `[runtime] shell_timeout`, `max_agent_steps`, `max_context_tokens`, `check_updates`, `update_check_interval_hours`, `session_retention_days`, `yolo`, `debug`

`api = "auto"` 会根据 provider/model profile 在 Chat Completions 和 Anthropic Messages 之间选择。`prompt_cache_key = "auto"` 会根据 provider、model、workspace 和工具 schema 名称生成稳定 key。

`--yolo`、`--debug` 和 `--mcp` 等 runtime flags 对恢复的 session 同样生效。保存的 session 不会携带旧 runtime config。

## MCP

nanocode 可连接 [Model Context Protocol](https://modelcontextprotocol.io) 服务器，并通过 `MCP` 工具暴露其工具。每个服务器配置在 `[mcp.<name>]` 下，且只能二选一：`url`（远程）或 `command`（本地）。

通过 streamable HTTP 的远程服务器：

```toml
[mcp.example]
url = "https://example.com/mcp"
bearer_token_env_var = "EXAMPLE_MCP_TOKEN"  # 可选；发送 Authorization: Bearer
enabled = true

[mcp.oauth_example]
url = "https://example.com/mcp"
auth = "oauth"                              # 通过 /mcp login <server> 在浏览器登录
enabled = true
```

通过 stdio 的本地服务器：

```toml
[mcp.filesystem]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
env = { SOME_TOKEN = "value" }              # 可选；会合并到继承的环境变量之上
enabled = true
```

HTTP 鉴权选项（`auth`、`bearer_token_env_var`、`env_http_headers`）只对 `url` 服务器生效。`env_http_headers` 把 header 名映射到存放其值的环境变量。

运行时管理服务器：

- `/mcp`：列出已配置服务器及连接状态。
- `/mcp tools [server]`：列出已发现的工具。
- `/mcp refresh [server]`：重新发现服务器。
- `/mcp login <server>` / `/mcp logout <server>`：OAuth 登录和登出。

## Providers

以下 provider 已在 nanocode 中测试通过：

- **deepseek**：DeepSeek API
- **opencode**：OpenCode API
- **aliyun**：阿里云通义千问 API（Chat Completions）
- **llama.cpp**：通过 llama.cpp 服务端本地推理

## 上下文模型

每次模型请求都由 nanocode 手动构建成明确的 messages。稳定上下文在前，会话作为 messages 保留，工作记忆随后，最新文件状态放在末尾。

```text
model request
+--------------------------------------------------+
| system                                           |
|   concise agent contract and tool rules          |
+--------------------------------------------------+
| user                                             |
|   Environment                                    |
+--------------------------------------------------+
| user/assistant                                   |
|   conversation, compacted summaries, tools       |
+--------------------------------------------------+
| user                                             |
|   Memory: goal, plan, known, date                |
+--------------------------------------------------+
| user                                             |
|   FILE STATE: latest Read/Edit file view         |
+--------------------------------------------------+
```

核心规则：

- 回合中的 assistant 文本和用户追加输入都会作为 conversation 保留。
- 上下文过大时，较早 conversation 会压缩成明确的 summary。
- FILE STATE 由成功的 `Read` 和 `Edit` 更新，展示当前文件范围，最近文件优先。
- 更新的文件行会覆盖旧行；edit invalidation 会清理过期范围。
- 文件行展示前会通过当前文件 stat 或行 hash 校验。
- 成功的 `Read` 和 `Edit` 工具消息只指向 FILE STATE，不重复塞入文件正文。
- 其他工具输出在 conversation messages 中保持有界，并可通过 `tr.N` 召回。

## 安全

nanocode 会在启动它的环境中编辑文件和执行 shell 命令。它不提供 sandbox 保护。需要隔离时，请在你自己的 sandbox、容器、虚拟机或其他隔离环境中运行。
