<h1 align="center">nanocode-cli</h1>

<p align="center">
  单文件终端编程代理：控制流明确，session 设计对 prompt cache 友好。
</p>

<p align="center">
  <img src="snapshots/nanocode1.gif" alt="nanocode demo" width="600">
</p>
<p align="center"><sub>在一个交互式 session 中编辑代码并运行工具。</sub></p>

<p align="center"><a href="README.md">English</a></p>

## 安装

需要 Python 3.11+ 和 [uv](https://docs.astral.sh/uv/)。

```sh
uv tool install nanocode-cli
nanocode --init-config
```

编辑 `~/.nanocode/config.toml`，填入 OpenAI-compatible endpoint：

```toml
[provider]
active = "default"

[provider.default]
url = "https://api.openai.com/v1"
key = "YOUR_API_KEY"
model = "gpt-5"
```

然后启动：

```sh
nanocode
```

升级：`uv tool upgrade nanocode-cli`

常用参数：

- `--resume [UID]`：恢复已保存的 session，不传 UID 恢复最近一次
- `--yolo`：跳过工具确认，直接执行
- `--mcp <selector>`：选择启用哪些 MCP 服务器
- `--config <path>`：使用指定的 TOML 配置文件

## 范围与取舍

nanocode 是一个单文件终端编程代理，而不是一个极简代码示例。随着 session、MCP、skill、后台任务和交互式 CLI 的加入，实现规模已经增长，但这些功能仍集中在一个明确的 runtime 中，而不是分散到一套框架里。

这样更容易把 nanocode 作为一个整体引入、搜索和修改。相应的取舍是：`nanocode.py` 本身并不小，项目也不提供大型 agent 平台那样清晰的模块边界或广泛集成。

<p align="center">
  <img src="snapshots/nanocode2.gif" alt="nanocode session" width="600">
</p>
<p align="center"><sub>恢复保存的 session，包括对话和工具调用历史。</sub></p>

## 概览

| | |
|---|---|
| Provider | OpenAI, Anthropic, DeepSeek, OpenRouter, llama.cpp，以及任意 Chat-Completions 端点 |
| 编辑 | 结构化 patch 操作（`replace`, `insert_before`, `insert_after`, …）配 `line:hash` 锚点 |
| Session | 自动保存 JSONL 快照，支持 `--resume latest` / `--resume <id>` |
| MCP | 远程（HTTP streamable）和本地（stdio）服务器，支持 OAuth |
| Skills | 从项目和用户目录加载的可复用 Markdown 指令包 |
| 架构 | Runtime 以单个 `nanocode.py` 模块分发 |

## 常用命令

| 命令 | 用途 |
|---|---|
| `/help` | 显示完整命令和工具参考 |
| `/status` | 运行状态：token 用量、context 占比、缓存命中率 |
| `/context` | 模型上下文帧——环境、记忆（goal/plan/known/check） |
| `/diff` | 最新编辑 diff 和 session 整体净 diff（交互式，支持标签切换） |
| `/compact` | 立即压缩上下文 |
| `/mcp` | 管理 MCP 服务器和工具 |
| `/model [MODEL]` | 显示或切换模型 |
| `/yolo` | 切换工具确认 |

运行 `/help` 查看全部命令、工具和快捷键。交互选择器支持 `j`/`k`、方向键、`/` 搜索、Enter 和 Esc；输入支持历史补全和 `Ctrl-R` 历史搜索。

## 配置

配置文件：`~/.nanocode/config.toml`

生成的配置文件会说明常用 provider 和 runtime 选项。可以定义多个 `[provider.<name>]`，再通过 `[provider] active = "name"` 选择。使用 `/config` 查看当前配置，使用 `/help` 查看运行时命令。

## MCP

连接 [Model Context Protocol](https://modelcontextprotocol.io) 服务器，通过 `MCP` 工具暴露其能力。

远程服务器（HTTP）：

```toml
[mcp.example]
url = "https://example.com/mcp"
bearer_token_env_var = "EXAMPLE_MCP_TOKEN"  # 可选
enabled = true
```

本地服务器（stdio）：

```toml
[mcp.filesystem]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
enabled = true
```

运行时管理：`/mcp` 查看状态，`/mcp tools [server]` 列出工具，`/mcp login/logout <server>` 管理 OAuth。

## Skills

Skills 是 agent 按需加载的指令包。每个 skill 是一个包含 `SKILL.md` 的文件夹。

- **发现**：`.nanocode/skills/`（项目级）和 `~/.nanocode/skills/`（用户级），同名时项目优先
- **模型视角**：上下文只放 name + description 索引，正文在 `Skill(name)` 调用时加载
- **内联引用**：消息中 `$name` 直接引用（Tab 补全）
- **随附脚本**：`{skill_dir}` 展开为 skill 目录绝对路径，可通过 `Bash` 运行
- **内置**：默认自带 `nanocode-help` skill，包含使用手册和自动生成的命令/工具/配置清单

## 安全

**使用风险自负。** nanocode 会在启动它的环境中编辑文件和执行 shell 命令。它不提供 sandbox 保护。需要隔离时，请在你自己的 sandbox、容器、虚拟机或其他隔离环境中运行。
