# Follow-ups: typing while it works

You don't have to wait for the agent to finish before typing your next request.
Press `Enter` to queue a message for the next turn, or `Ctrl-C` to interrupt and
take over immediately.

## How it works

```
  YOU                                    AGENT
  ───                                    ─────

  "fix the login bug" [Enter] ─────────► reads code, edits files...

  (agent is still working)                   │
                                             │
  "also add a test" [Enter] ──► queued ────►│ (saved for later)
                                             │
  /status [Enter] ─────────────► runs now ──►│ (read-only command)
                                             │
  "wait — refactor first" [Ctrl-C]           │
         │                                   │
         └──────── interrupts ──────────────►│ (stops current turn)
                                             │
         agent restarts with:                │
         "wait — refactor first"      ◄──────┘
```

1. **Type a message and press Enter** — it queues behind the current turn. The
   agent sees it on the next turn.
2. **Type a message and press Ctrl-C** — the agent stops what it's doing and
   your message becomes the next turn immediately.
3. **Run a read-only command** — `/status`, `/diff`, `/ps`, `/skills`, `/mcp`,
   `/help`, `/yolo`, and `/resend` all run right away without waiting.

## Edit a queued message

If you queued something and want to fix it before the agent reads it:

```
  Press Up (or Ctrl-P) ──► recalls the newest queued message
  Edit it
  Press Enter ──► replaces the original queued message
```

Press `Up` again to cycle through older queued messages.

## When to queue vs interrupt

| Situation | What to do |
|---|---|
| You have another task you want done after this one | Queue (Enter) |
| The agent is going down the wrong path | Interrupt (Ctrl-C) |
| You want to add context to the current task | Queue (Enter) |
| You forgot something important | Interrupt (Ctrl-C) |
| You want to check status or review diffs | Run `/status` or `/diff` |

## Keyboard reference

| Key | When | Effect |
|---|---|---|
| `Enter` | Typing a message | Queue for next turn |
| `Ctrl-C` | Typing a message | Interrupt agent now |
| `Ctrl-C` | Idle prompt | Clear input line |
| `Up` / `Ctrl-P` | Typing | Recall newest queued message |
