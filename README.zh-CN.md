<h1 align="center">nanocode-cli</h1>

<p align="center">
  一个我自己在用、维护并定制的小型终端 coding agent — 单文件 Python 实现，可在交互式
  对话中读取和编辑代码、运行命令、恢复 session、连接 MCP server、按需加载 skill。
</p>

<p align="center">
  <img src="snapshots/nanocode1.gif" alt="nanocode 编辑代码并运行工具" width="600">
</p>

<p align="center"><a href="README.md">English</a></p>
## 安装

需要 macOS 或 Linux、Python 3.11+ 和 [uv](https://docs.astral.sh/uv/)。

```sh
uv tool install nanocode-cli
nanocode --init-config
```

在 `~/.nanocode/config.toml` 中填写 provider：

```toml
[provider]
active = "default"

[provider.default]
url = "https://api.deepseek.com"
key = "sk-..."
model = "deepseek-v4-flash"
```

然后运行：

```sh
nanocode
```

升级：`uv tool upgrade nanocode-cli`。

完整文档位于 [nanocode.readthedocs.io](https://nanocode.readthedocs.io)。

## 它是什么

nanocode 并不想发明一种新的 coding agent。它只是把熟悉的能力 — 读取和编辑文件、运行命令、实时追加指令、session 恢复、diff、MCP 和 skill — 整合到一个我真正在用、维护的 Python 模块里。

所有功能都在一个文件中，改行为只需要改一个文件。

<p align="center">
  <img src="snapshots/nanocode2.gif" alt="nanocode 恢复保存的 session" width="600">
</p>
<p align="center"><sub>恢复保存的 session，包括对话和工具调用历史。</sub></p>

## 亮点

- **实时追加指令：** agent 工作时仍可输入，排队内容会加入下一轮或打断当前请求。
- **锚点编辑：** 结构化编辑使用 `line:hash` 锚点，文件内容过期时会被拒绝。
- **可恢复 session：** 对话、工具调用、diff 和工作记忆可通过 `-c` 或 `--resume` 恢复。
- **内置 diff viewer：** `/diff` 展示最新一轮改动以及整个 session 的累计变更。
- **MCP 与 skills：** 按需连接 Model Context Protocol server，加载 Markdown 指令包。
- **Provider 兼容：** 支持 OpenAI-compatible API 与 Anthropic。

## 常用命令

| 命令 | 说明 |
|---|---|
| `/help` | 命令与工具参考 |
| `/status` | 运行状态、上下文、缓存和 MCP |
| `/diff` | 查看最新改动与 session 累计 diff |
| `/mcp` | 管理 MCP server 连接 |
| `/model [MODEL]` | 查看或切换模型 |
| `/yolo` | 开关确认提示 |

## 安全

**使用风险自负。** nanocode 会在启动环境中编辑文件并执行 shell 命令，不提供 sandbox 隔离。需要隔离时，请使用容器或虚拟机。
