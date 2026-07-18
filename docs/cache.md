# Prompt cache

Most of nanocode's prompt — its instructions, environment, tool schemas — is the same across
every turn. The model provider sees identical request prefixes and reuses cached computations,
skipping the expensive inference work on those tokens.

When everything lines up, nanocode can reach 90–99% cache hit rates. This is the main reason
its API cost stays low even across long conversations.

| Cached | Not cached |
|---|---|
| System instructions | Your messages |
| Environment context (cwd, OS, commands) | Tool call results |
| Tool schemas (Read, Edit, Bash, etc.) | The model's replies |
| Recent conversation prefix | Each new turn's content |

