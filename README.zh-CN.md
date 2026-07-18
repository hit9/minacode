<h1 align="center">nanocode-cli</h1>

<p align="center">
  一个我自己在用、维护并定制的小型终端 coding agent — 单文件 Python 实现，可在交互式
  对话中读取和编辑代码、运行命令、恢复 session、连接 MCP server、按需加载 skill。
</p>

<p align="center">
  <img src="snapshots/nanocode1.gif" alt="nanocode 编辑代码并运行工具" width="600">
</p>

<p align="center"><a href="README.md">English</a></p>


## 环境要求

- macOS 或 Linux
- Python 3.11 或更新版本
- [uv](https://docs.astral.sh/uv/)

暂不支持原生 Windows。请通过
[Windows Subsystem for Linux（WSL）](https://learn.microsoft.com/windows/wsl/)运行 nanocode。

## 安装

```sh
uv tool install nanocode-cli
nanocode --init-config
```

在 `~/.nanocode/config.toml` 中填写 provider URL、API key 和 model，然后启动：

```sh
nanocode
```

升级命令：`uv tool upgrade nanocode-cli`。

## 亮点

- Agent 工作时仍可实时追加指令
- 锚点编辑会拒绝修改内容已经过期的文件
- 可恢复的 session 和内置 diff viewer
- 支持 OpenAI-compatible 与 Anthropic provider
- 按需连接 MCP server、加载 Markdown skill
- 单个 Python 模块，便于直接定制

<p align="center">
  <img src="snapshots/nanocode2.gif" alt="nanocode 恢复保存的 session" width="600">
</p>
<p align="center"><sub>恢复保存的 session，包括对话和工具调用历史。</sub></p>

## 文档

完整文档位于 [`docs/`](docs/index.md)：

- [快速开始](docs/getting-started.md)
- [交互式使用](docs/usage.md)
- [配置](docs/configuration.md)
- [MCP](docs/mcp.md) 与 [Skills](docs/skills.md)
- [安全说明](docs/safety.md)

## 安全

**使用风险自负。** nanocode 会在启动环境中编辑文件并执行 shell 命令，且不提供
sandbox 隔离。需要隔离时，请使用容器或虚拟机。
