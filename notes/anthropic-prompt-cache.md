# Anthropic 路径：会话主体从未进入缓存

状态：待处理，未开工。发现于 compaction 前缀复用工作（分支 `toolscript-mcp-names-and-script-phase`）的收尾复查。

## 事实

Anthropic 没有隐式缓存。不写 `cache_control` 就完全不缓存 —— 这与 OpenAI 系不同，
后者对 ≥1024 token 的前缀自动缓存，无需任何参数（minacode 的 `prompt_cache_key`
只是给缓存分域/路由，不是开关）。

minacode 当前用**显式**断点，打在 system 块上：

    # minacode/model.py，anthropic_params
    system = [{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}]

官方文档：

> Cache prefixes are created in the following order: `tools`, `system`, then `messages`.
> ... up to and including the block designated with `cache_control`.
> **cache writes happen only at your breakpoint**, not at earlier positions.

断点在 system 结束处 ⇒ 写入只发生在那里 ⇒ **messages 从来没有被写进缓存**。

## 影响

不只是压缩请求，**每一个普通 turn 都是**：Anthropic 路径上对话主体每轮按全价重新
处理，缓存住的只有 tools + system（本仓库实测约 5.4k token 的工具 schema + 系统提示）。
一个 10 万 token 的会话，每轮为 9 万多 token 付全价。

连带后果：分支上那一整轮「压缩请求复用对话前缀缓存」的改动，**在 Anthropic 上收益为 0**，
因为它要复用的那段缓存根本不存在。收益只在 OpenAI 系兑现（deepseek / 阿里云 / kimi /
GLM / OpenRouter）。

`context.py` 的类 docstring 里那条设计原则 —— "Layer order exists for prompt-cache
stability … inserting a rebuilt block into the conversation prefix would invalidate later
cache reuse" —— 整个是按「会话主体在缓存里」写的，但在 Anthropic 上它从未成立。

## 可行的修法

SDK 0.104.1 已支持顶层字段（已验证 `anthropic.types.message_create_params`）：

    cache_control: Optional[CacheControlEphemeralParam]
    """Top-level cache control automatically applies a cache_control marker to the last
       cacheable block in the request."""

打在最后一个可缓存块上，也就是对话末尾 —— 正是 OpenAI 隐式缓存的行为。用它可以让
Anthropic 路径与其他 provider 行为一致，不再是特例。

## 开工前必须先确认的取舍

1. **写入溢价**：缓存写按 1.25x 计。每轮为新增段付写入，换旧段 0.1x 读取。对 10 万
   token 的 agent 循环几乎肯定大赚，但这是**所有 Anthropic 用户的计费形态变化**，
   不是纯优化。
2. **TTL 5 分钟**：turn 间隔超过 5 分钟就只付写不收读。当前"只缓存 tools+system"的
   做法在这点上是稳而少。
3. **断点能否并存（关键，未验证）**：理想形态可能是两个断点 —— system 长期命中 +
   末尾滚动，正是文档说的 "cache different sections that change at different
   frequencies"（上限 4 个）。若能并存，那才是正解，而不是把 system 断点换掉。

## 验证手段

`/status` 的 cache 行（`last` / `session` 命中率）是现成观测点。改前改后各跑一个
真实 session 对比即可。这件事影响每一个 turn，收益和风险都远大于分支上的压缩改动，
因此单独成一个分支做。

## 参考

- https://platform.claude.com/docs/en/build-with-claude/prompt-caching
