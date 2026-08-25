"""Worker handoff: the second in-process session a parent delegates to (see DESIGN.md).

Coverage follows WORKER_HANDOFF_PLAN.txt section 9; each numbered test maps to that list.
"""


from agent_harness import session

from minacode.engine import Agent


def _requested_system(tmp_path, custom=None):
    s = session(tmp_path)
    if custom is not None:
        s.system_prompt = custom
    agent = Agent(s, output_fn=lambda text: None)
    request = agent.prepare_request([{"role": "user", "content": "hi"}])
    return s, request.messages[0]["content"]


# 1. tool_names filtering: only the whitelisted schemas, in TOOL_REGISTRY order; empty tuple is
#    exactly the unfiltered behavior.


# 2. system_prompt comes from the session: the request payload's system content changes with it,
#    and the parent default is unchanged.




# 3. workers stay out of listings and never claim the latest pointer.




# 10. two registration gates: Delegate appears only when [worker] provider was set at session
#     start AND runtime.worker is on. The provider half is frozen per session, so a runtime
#     /worker provider change never flips the tool block; settings.worker stays the live half.
#     Closing is not reset (the snapshot stays).




# --- Delegation (steps 4-5): the worker is driven through DelegateTool with a scripted model. ---


class FakeModelClient:
    """Stands in for minacode.engine.ModelClient: records every request and replays a script of
    (assistant, tool_calls, content) triples, so the worker's loop is exercised without HTTP."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        self.received_tools = []
        self.last_compaction_model = ""

    def request(self, messages, request_tools=None):
        self.requests.append(messages)
        self.received_tools.append(request_tools)
        return self.script.pop(0)

    def estimated_request_tokens(self, messages, tools=None):
        return sum(len(str(message)) for message in messages) // 4

    def cancel(self):
        pass


def _delegate_session(tmp_path):
    parent = session(tmp_path)
    parent.config.worker_provider = "default"
    parent.settings.worker = True
    return parent


def _delegate_call(parent, runner, **args):
    from minacode.tools.delegate import DelegateTool

    tool = DelegateTool(parent, [args])
    tool.runner = runner
    return tool.call()


def _delegate_runner(parent):
    from minacode.context import ContextManager
    from minacode.runner import ToolRunner

    return ToolRunner(parent, ContextManager(parent), input_fn=lambda *a: "y", output_fn=lambda text: None)


# 4. context continuity: the second delegation's request carries the first order and its answer.


# 5b. the worker agent inherits the parent's lifecycle callbacks: retry backoff, provider-side
#     builtin calls, and automatic compaction surface in the parent TUI. Wiring is None-guarded
#     like model_stream: an uninjected runner leaves the worker's callbacks unset.


# 5. reset: after reset, the next send carries no prior history, and the snapshot file is gone.










# 6. diff reflux: an Edit inside the worker shows up in the parent's turn_diffs.


# 7. interrupt: cancellation lands on the worker, its turn settles with every tool call matched,
#    and the diff merge still runs.


# 7b. a failed delegation reports what the worker did before dying (steps, elapsed, files, alive,
#     rounds, context_percent) as a ToolError envelope, and the worker's history stays legal: the
#     unanswered tool call is settled with a Failed result and a turn-ended marker, so the next
#     send goes out instead of being rejected for a dangling call.


# 7b'. a batch that dies after its first call already ran keeps that call's side effects: the file is
#      written, its diff is merged and named in the failure envelope, while the settled history marks
#      every call of the dying batch as Failed — the executed call's result died with the crash, which
#      is the honest record — and the next send still goes out.


# 7c. Delegate status carries the last failure and the round it happened in until a send succeeds
#     and clears it, so the parent can confirm why the worker stopped without relying on memory.


# 7d. neither the status envelope nor the worker's permanent history takes a provider error
#     verbatim: an HTTP body carries quotes and newlines that break the attribute the model parses,
#     and it would ride every later request from inside the turn-ended marker.


# 8. cache prefix: system + tools + Environment are byte-identical across delegations.


# 12. settings isolation: a per-call max_steps override never touches the parent's budget, and the
#     worker sees the parent's current settings on every send.


# 11. user reset: /worker reset appends a SESSION_EVENT_KEY message to the parent's history tail, and
#     the message reaches the next request (render-hidden, never filtered from the model history).










# /worker's own status branch returns readable text for the human (the model-facing envelope stays
# in DelegateTool): no-live-worker, one line per fact, and the usage/state-context-percent values


# The engine publishes the model's own text as bare strings (content beside tool calls), so the
# worker output wrapper must wrap them into LogLine items in headless mode: LogBlock.walk crashes on a str item.






# 11b. a send opens with a visible start marker: one yellow [worker] line naming the worker's live
# provider/model and the one-line order summary, so the scrollback has a boundary before the
# finish block. This is the fallback when no worker_rule is wired; the wired path emits a
# full-width yellow rule label instead (see the test below).




# 11b2. the start divider uses the send's optional `title` when given: the human-readable label
# replaces the order-first-line summary on both the wired rule and the fallback [worker] ▶ line.








# 11c. language is a send parameter, not a setting: it lands in the order the worker receives as
#      an explicit language request covering the live stream and interim messages, not just the end.




# 11d. a forced runtime language is inherited: the worker rebuilds its settings from the parent on
#      every send, so the parent's /language value lands in the worker's system prompt too, while
#      the per-send `language` parameter stays an order-text directive (see test above).


# 12. the Agent lives on the worker Session, not in a module-level dict: a fresh worker object
#     (after /resume re-enters the same parent) always gets a fresh Agent bound to itself.


# 13. a snapshot-restored worker shares the parent's skills/mcp objects (review point 2): load
#     rebuilds its own copies, so the delegate caller must re-attach the shared ones.


# 14. stopped_at_max_steps in the envelope is a runtime fact from the Agent, never the answer's wording.


# 14b. The send envelope also carries the worker's token spend for this delegation: the program
#      subtracts worker.usage before/after (the fake model updates nothing, so 0/0), and the finish
#      summary renders it.


# 14c. delegate_result_summary formats the raw integer token counts like /status does, and keeps
#      parsing envelopes written before the tokens attribute existed.


# 14d. rounds and context_percent are the two attributes a reset decision is made from, so the
#      summary line shows both -- otherwise the model reads them and the user never sees them.


# 15. resolve_uid prefix search never resolves to a worker snapshot (review point 5): the parent's
#     uid prefix must resolve to the parent alone, without ambiguity.


# 16. the two blocks that must never drift between SYSTEM_PROMPT and WORKER_PROMPT are spliced from
#     the same module-level constants, so a wording change in one is a change in both (this is a
#     composition contract, not a prompt-literal test).


# 17. The two readable role prompts keep their role-specific behavior without exposing the
#     implementation as a collection of positional fragments.


# 18. The worker prompt may name only tools in its reduced tool set.


# 19. refactor-stability sentinel for the parent prompt: pure refactors of the prompt composition
#     must not change SYSTEM_PROMPT's text (its cache-prefix stability and this contract depend on
#     it). A deliberate, release-level edit to the parent prompt updates this hash in the same
#     commit and records the change in the changelog.


# 20. yolo covers editing files and running commands: those mistakes show up in the diff or the
#     command output at once. A delegation's mistake is the order text, and it only surfaces a whole
#     worker round later, so send is confirmed even under yolo. status and reset stay under it.


# 20b. Delegate send confirmation: Y/Enter approve, n refuses without a reason, any other input is
# a refusal reason passed back to the model. `a` is an ordinary reason now (the always key is
# retired), and only a whole-line "c"/"config" opens the worker configuration loop.












































# 21. [worker] model/reasoning/api parse like [worker] provider; reasoning and api validate their choices.








# 22. The Delegate registration gate is frozen per session: /worker provider stores the config for
#     the next spawn (and live-applies to a live worker) but never flips the tool block mid-
#     session, whether delegation was on or off at session start. A freshly constructed session
#     over the same config re-evaluates the gate (simulating a restart), and an unknown name is
#     rejected without touching the config.


# 23. "off" is the clearing word unless a provider entry is literally named "off": existence in
#     config.providers wins, so /worker provider off selects that entry.


# 24. /worker model and /worker reason store overrides, reject an invalid effort, and "default"
#     clears; "off" is a valid reasoning effort, never the clearing word.


# 25. spawn isolation: the worker's active ProviderConfig is a detached copy (never `is` the
#     parent's), [worker] model/reasoning overrides are applied to it, and mutating it does not
#     leak into the parent's providers entry. A snapshot-resumed worker picks up the overrides the
#     same way, because the load path receives the same freshly built config.


# 26. live switch: with a live worker, /worker model X replaces the worker's active entry
#     immediately while the parent's providers entry is untouched; "default" restores the
#     underlying entry's model on the live worker.


# 27. a live worker also takes /worker provider NAME immediately: its active entry is replaced with
#     a detached copy and the parent's entry is untouched.


# 28. a finished Delegate send renders as a proper log block: the confirmation root line is just
#     `Delegate send` (no argument blob), the finish block carries a steps/elapsed/files summary
#     and the worker's answer as an OUTPUT preview, and the raw envelope tags never reach the log.
#     This is the fallback when no worker_rule is wired; the wired path replaces the summary child
#     line with a yellow rule label (see the test below).




# 28b. a send with `title` carries the same human-readable label onto the done divider: the title
# is the first part of the `worker done` rule label, ahead of steps/elapsed/tokens/files.


# 28c. the worker's final report prints into the scrollback in full -- like its interim
# messages -- while the finish block's answer preview stays the folded three-line form: the
# scrollback block is the record, the preview only shows that it is there.


# 28e. with the loop wired in (worker_answer set), the worker's model text (interim and final)
# goes through the answer renderer (markdown) instead of the plain log lines.


# 29. a Delegate reset is a one-shot tool call, not a bracket: it keeps its ordinary tool root
#     and adds a plain done child stating what was cleared and what survives. No worker_rule rule
#     and no [worker] ◀ root.




# The worker's model stream forwards to the parent loop's live display, except
# `output_done`: the parent's promote would write the completed text a second
# time on top of what the worker's own output_fn already put in the scrollback,
# and the worker path never consumes the promoted-text marker. `output_done` is
# downgraded to a plain ("", "") preview clear; everything else forwards
# unchanged.


# 30. automatic compaction on the worker: the same ContextManager path the parent uses. The
#     budget estimate overruns, the compaction runs inline (bracketed by the on_compaction
#     lifecycle callback), the deterministic-trim fallback carries it when the model has no
#     `compact` (FakeModelClient does not), the compacted state persists into the worker snapshot,
#     and the next delegation runs on the compacted context without re-compacting.


def _worker_history_for_compaction(parent):
    """Return the spawned worker with a fat synthetic history appended after the first send."""
    worker = parent.worker
    assert worker is not None
    big = "x" * 100_000
    worker.messages.extend(
        [
            {"role": "user", "content": big + " u1"},
            {"role": "assistant", "content": big + " a1"},
            {"role": "user", "content": big + " u2"},
            {"role": "assistant", "content": big + " a2"},
            {"role": "user", "content": big + " u3"},
            {"role": "assistant", "content": big + " a3"},
            {"role": "user", "content": "final small request"},
        ]
    )
    return worker




