"""Compaction: turn one conversation span into a summary checkpoint.

The policy here decides what to compact (split/keep), how to ask for the summary (the inline
slice that reuses the turn's cached prefix, or the flattened payload), and how to run the
request with its format-retry and echo guard.

A `Compactor` binds the two collaborators the policy spans: the ContextManager (projection,
token budget, write-back) and the ModelClient (the request machinery). Planning methods
(`parts`, `split`, `request`, `input`) need only the context; `compact`/`compact_attempts`
need the model; `run` drives both.
"""

from __future__ import annotations

import asyncio
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, ClassVar

from wizolt.base import SESSION_EVENT_KEY, Billing, Json, ModelError, ModelResponseTimeout, Text
from wizolt.config import ProviderConfig, compaction_provider_config
from wizolt.model import ModelClient
from wizolt.prompts import (
    COMPACTION_ECHO_RETRY,
    COMPACTION_PROMPT,
    COMPACTION_REQUEST_EVENT,
    COMPACTION_RETRY,
    compaction_tail,
)
from wizolt.prompts import (
    compaction_input as format_compaction_input,
)
from wizolt.tools import Tool

if TYPE_CHECKING:
    from wizolt.context import ContextManager


class Compactor:
    """Turn one conversation span into a summary checkpoint.

    Construct with the ContextManager and the ModelClient the turn is using; both are required.
    The planning methods read projection and budget through `ctx`; the execution methods drive
    the summary request through `model`.
    """

    # A summary that is mostly one verbatim run copied out of the conversation is the echo failure
    # wearing valid JSON: constrained decoding forces the shape, never the task. Deliberately blunt
    # thresholds -- a real summary paraphrases, so 80% of it being a single unbroken copy is not a
    # close call, and quoting a path or an identifier is far below that.
    #
    # The floor is counted in characters, which are not equal: the same sentence is 138 characters
    # of English and 68 of Chinese. It is set for the denser script, because a floor tuned to
    # English would have excluded from this check every summary written in the language the failure
    # was first seen in.
    ECHO_MIN_CHARS: ClassVar[int] = 40
    ECHO_RATIO: ClassVar[float] = 0.8
    ECHO_COMPARE_CHARS: ClassVar[int] = 4000

    # Recent-window sizes for the split. The fallback is for when the ordinary window leaves
    # nothing to compact: the recent window is a message count, not a size, so a handful of very
    # large messages after the latest user message can blow the budget while all of them sit inside
    # the kept tail -- and then every request is over budget with an empty compactable head. Never
    # zero: the latest exchange has to survive.
    COMPACT_RECENT_MESSAGES: ClassVar[int] = 8
    COMPACT_MINIMUM_RECENT: ClassVar[int] = 2

    def __init__(self, ctx: ContextManager, model: ModelClient):
        self.ctx = ctx
        self.model = model

    async def run_async(
        self,
        compacted: list[Json],
        keep: list[Json],
        fallback_note: str,
        *,
        tool_messages: list[Json] | None = None,
        turn_messages: list[Json] | None = None,
        recent: int | None = None,
    ) -> bool:
        if not compacted:
            return False
        on_compaction = self.ctx.on_compaction
        if on_compaction is not None:
            on_compaction(True, "")
        error_detail = ""
        cancelled: BaseException | None = None
        try:
            try:
                req = self.request(compacted, turn_messages, recent, live_turn=tool_messages)
                # Checked against what the model is handed, which is not `compacted`: the inline
                # slice carries one message more, and the flattened payload carries `compacted`.
                sent = req[0][:-1] if req else compacted
                flat = self.input(compacted) if req is None else ""
                data = await self.compact_async(flat, *(req or ()), echo_source=self.echo_source(sent))
            except (asyncio.CancelledError, KeyboardInterrupt) as error:
                # The deterministic trim still runs: a cancelled turn must not be left with a
                # projection the provider would reject. Cancellation is re-raised after that.
                error_detail = "cancelled by user"
                cancelled = error
                data = None
            except Exception as error:  # noqa: BLE001 - compaction degrades to deterministic trimming on any model failure.
                error_detail = Text.clip_width(" ".join(str(error).split()) or type(error).__name__, 220)
                data = None
            self.ctx.apply_compaction(
                data,
                keep,
                tool_messages,
                turn_messages=turn_messages,
                fallback_note=fallback_note if data is None else "",
                compacted=compacted,
                model=self.model.last_compaction_model,
                title=self.title(data),
            )
        finally:
            if on_compaction is not None:
                on_compaction(False, error_detail)
        if cancelled is not None:
            raise cancelled
        return True

    async def compact_async(
        self,
        context: str,
        inline_messages: list[Json] | None = None,
        tools: list[Json] | None = None,
        echo_source: str = "",
    ) -> Json:
        model = self.model
        # The summary request runs on the [compaction]-resolved provider entry (empty [compaction]
        # = the active provider), resolved per call so a runtime /provider switch applies next
        # time. The context budget is untouched: compaction still measures against the main
        # provider's window, only the summary request itself uses this entry.
        provider = compaction_provider_config(model.session.config)
        entry_name = model.session.config.compaction_provider or model.session.config.active_provider
        # The client's own gate checks the active provider, which is the wrong entry when a
        # summary runs elsewhere: an incomplete [compaction] entry would otherwise reach the SDK
        # and come back as "Missing credentials", naming nothing the user can act on. Compaction
        # still degrades to deterministic trimming -- this only makes the fallback say why.
        if missing := provider.missing_fields():
            raise ModelError(f"compaction provider `{entry_name}` is missing {', '.join(missing)}; check [compaction] and [provider.{entry_name}]")
        model.last_compaction_model = ""
        # Two shapes. The inline form is the agent's own request with a compaction instruction
        # appended: same tools, same system, same conversation, so the provider's prefix cache --
        # already warm from the turn that just ran -- covers everything but the tail, and the
        # compactor sees real messages instead of a flattened re-rendering that drops tool calls.
        # The flattened form stays for every case the inline one cannot serve.
        inline = inline_messages is not None
        messages = inline_messages if inline else [{"role": "system", "content": COMPACTION_PROMPT}, {"role": "user", "content": Text.clean(context)}]
        # Compaction honors the configured total-generation limit instead of a hidden cap: a
        # summary is worth the user's configured wait, and the deterministic trim fallback still
        # catches whatever the provider rejects.
        response_timeout = provider.response_timeout
        # The status bar reads this to name the entry actually serving the summary; cleared in the
        # finally so a timeout, a cancel, or a provider error leaves no stale row behind.
        entry_label = f"{entry_name}/{provider.model}"
        model.session.state.compaction_entry = entry_label
        try:
            data = await self.compact_attempts_async(
                messages, provider, response_timeout, entry_label, tools=tools if inline else None, echo_source=echo_source
            )
        finally:
            model.session.state.compaction_entry = ""
        model.last_compaction_model = provider.model
        return data

    async def compact_attempts_async(
        self,
        messages: list[Json],
        provider: ProviderConfig,
        response_timeout: float,
        entry_label: str,
        tools: list[Json] | None = None,
        echo_source: str = "",
    ) -> Json:
        """Ask for the summary, and ask once more if what came back was not a JSON object.

        The failure this retries is a model ignoring the format and replying in prose -- usually by
        continuing the conversation it was handed instead of summarizing it. A bare resend would
        likely reproduce it, so the second attempt carries the previous reply and a correction,
        which is the same shape a person would use. Every error names the entry that served the
        request: compaction can run on its own `[compaction]` provider, and a cheaper model there is
        exactly the one that fails this way, so the message has to say which model to look at.
        """
        model = self.model
        attempt_messages = list(messages)
        for attempt in (1, 2):
            try:
                # Tools ride along, and tool_choice is deliberately left exactly as an ordinary
                # request sets it. Forcing "none" here looks safer and is not: changing tool_choice
                # invalidates the messages cache, which is the whole 100k-token conversation this
                # request exists to reuse -- it would spend the prize to buy the guarantee. The
                # instruction not to call tools lives in the appended message instead, and a model
                # that calls one anyway returns no text, which the retry below already handles.
                _, _, content = await model.api_request(
                    attempt_messages,
                    tools,
                    allow_stream=False,
                    response_timeout=response_timeout,
                    provider=provider,
                    json_object=True,
                    billing=Billing.COMPACTION,
                )
            except ModelResponseTimeout:
                raise ModelResponseTimeout(
                    f"compaction summary on `{entry_label}` exceeded provider.response_timeout={response_timeout:g}s; "
                    "set it to 0 to disable the total-generation limit"
                ) from None
            try:
                data = model.parse_json_object(content)
            except ModelError as error:
                if attempt == 2:
                    raise ModelError(f"{error} (compaction provider `{entry_label}`)") from None
                attempt_messages = [
                    *attempt_messages,
                    {"role": "assistant", "content": Tool.compact(content, 400)},
                    {"role": "user", "content": COMPACTION_RETRY},
                ]
                continue
            if not isinstance(data, dict):
                raise ModelError(f"compactor returned non-object JSON (compaction provider `{entry_label}`)")
            # Checked after parsing, not instead of it: this is the same failure the retry above
            # exists for, only it arrived shaped like a valid answer. Left unchecked it applies
            # cleanly into session.state and is fed back into every later compaction.
            if self.echoes_source(data.get("summary") or "", echo_source):
                if attempt == 2:
                    raise ModelError(f"compactor echoed the conversation instead of summarizing it (compaction provider `{entry_label}`)")
                attempt_messages = [
                    *attempt_messages,
                    {"role": "assistant", "content": Tool.compact(content, 400)},
                    {"role": "user", "content": COMPACTION_ECHO_RETRY},
                ]
                continue
            return data
        raise ModelError(f"compactor returned no usable JSON (compaction provider `{entry_label}`)")

    def request(
        self,
        compacted: list[Json],
        turn_messages: list[Json] | None = None,
        recent: int | None = None,
        live_turn: list[Json] | None = None,
    ) -> tuple[list[Json], list[Json]] | None:
        """The compaction request as the agent's own request truncated, plus one instruction, or
        None to use the flattened payload instead.

        This is the half of compaction that reads the cache; the rebuild that follows it does not
        (see ContextManager._summary_block and DESIGN.md, "Compaction reads the cache; the rebuild
        does not").

        A cache hit needs a byte-identical prefix, so this slices `model_messages` -- the very list
        the turn just sent -- rather than assembling a lookalike from `compacted`. They are not the
        same thing: `compacted` has had `without_compaction_summaries` applied and has not had the
        describe/skill dedup applied, so a request built from it diverges at the first earlier
        summary and at every repeated schema, which cost the whole conversation and left only the
        header cached.

        The slice therefore ends one message later than `compacted` does, taking in the latest user
        message that `compacted` deliberately excludes. That message is kept either way -- what is
        evicted is decided by `keep`, not by this request -- so the only effect is that the
        compactor sees what is being worked on right now, which helps it.

        Eligible only when the summary runs on the entry that served the turn: a `[compaction]`
        entry elsewhere is a different cache namespace, so rebuilding the prefix would cost the
        whole history at full rate to save nothing.

        One case builds the slice and gets no cache for it: a turn-scope pass that follows a
        history-scope pass in the same projection, where session.messages has just been rewritten
        and no request with that prefix has ever been sent. It is not a regression -- the flattened
        payload never hit either, and the real messages are still worth more to the summarizer than
        a rendering that drops tool calls -- but the reuse this method is named for does not apply
        there."""
        ctx = self.ctx
        if ctx.session.config.compaction_provider or ctx.session.system_info is None:
            return None
        base_system = ctx.session.system_prompt
        if not base_system or not compacted:
            return None
        # Projected with the turn attached whichever scope this is: the request being ridden
        # carries it, and its reasoning boundary is read off the whole projection below even when
        # only the stored half is being sliced.
        live = ctx.model_messages(base_system, turn_messages if turn_messages is not None else live_turn)
        header = len(ctx.model_header(base_system))
        # A turn-scope span sits after the stored conversation rather than at the head of it, so
        # its slice starts there. Both scopes are ordinary prefixes of the same projection; only
        # the offset differs.
        cut = header + (len(ctx.session.messages) + self.prefix_count(turn_messages, recent) if turn_messages is not None else self.prefix_count(recent=recent))
        if cut <= header:
            return None
        tail = compaction_tail(
            state=ctx.session.state.format(),
            previous_summary=ctx.session.state.summary,
            recent_count=min(self.COMPACT_RECENT_MESSAGES, len(compacted)),
        )
        # Where the appended instruction sits relative to the reasoning boundary decides whether
        # this request replays the same reasoning the live one did, and the answer differs by shape.
        # When the boundary is inside the slice, the instruction must not become it -- marked. When
        # it is beyond the slice (the recent window kept it, or it lives in the current turn), the
        # live request strips reasoning from everything here, so the instruction is left unmarked
        # and becomes the boundary itself, which strips exactly the same set. Marking in that shape
        # kept reasoning the live request had dropped and diverged four messages in.
        instruction: Json = {"role": "user", "content": tail}
        if -1 < ModelClient.latest_user_position(live) < cut:
            instruction[SESSION_EVENT_KEY] = COMPACTION_REQUEST_EVENT
        return [*live[:cut], instruction], Tool.resolved_schemas(ctx.session)

    def input(self, messages: list[Json]) -> str:
        older, recent = self.parts_for(messages)
        return format_compaction_input(
            state=self.ctx.session.state.format(),
            previous_summary=self.ctx.session.state.summary,
            older_messages=self.ctx.messages_text(older),
            recent_messages=self.ctx.messages_text(recent),
        )

    def prefix_count(self, turn_messages: list[Json] | None = None, recent: int | None = None) -> int:
        """How many messages of the scope's own list the summary request carries.

        One expression of the cut, shared with the split that produces `compacted`. Deriving it
        separately is what put the request out of step with what was being evicted twice already:
        once when the MINIMUM_RECENT fallback re-split with a different window, and again when the
        split grew a size bound this did not have."""
        ctx = self.ctx
        messages = ctx.session.messages if turn_messages is None else turn_messages
        if turn_messages is None and ctx.latest_user_index(messages) is None:
            # parts hands the whole list over when there is no request to keep.
            return len(messages)
        return self.keep_start(messages, recent)

    def echo_source(self, sent: list[Json]) -> str:
        """What a copied summary would have been copied from.

        Takes what the request actually carries, never `compacted`. The inline slice deliberately
        reaches one message further than `compacted` does, and that extra message is the latest user
        message -- precisely the text the failure this guard exists for reproduced. Checking against
        `compacted` left the guard blind to exactly the case it was written for.

        The tail, not the whole span: recency is what the failure follows, and this feeds a
        substring search over a span with no size limit."""
        ctx = self.ctx
        index = ctx.latest_user_index(sent)
        tail = sent if index is None else sent[index:]
        return ctx.messages_text(tail)[-4000:]

    def parts(self, recent: int | None = None) -> tuple[list[Json], list[Json]]:
        """Split history for manual compaction and the first automatic pass."""
        ctx = self.ctx
        messages = ctx.session.messages
        index = ctx.latest_user_index(messages)
        if index is None:
            return self.without_summaries(messages), []
        compacted, keep = self.split(messages, index, recent)
        return self.without_summaries(compacted), self.without_summaries(keep)

    def split(self, messages: list[Json], index: int, recent: int | None) -> tuple[list[Json], list[Json]]:
        """Split around a kept tail counted over the whole list, with the latest user message always
        in it.

        The window used to be measured only over what follows that message, which made it a cap
        rather than a floor: `/compact` run just after a turn answered has one message there, so a
        118-message session kept two and a checkpoint. Continuity wants the concrete recent work --
        the last tool results and file contents -- and the checkpoint is only prose.

        The kept tail stays non-contiguous in the one case that needs it. When the latest user
        message falls before the window (a worker given one order that then ran twenty steps),
        keeping everything from it onward would compact nothing at all, so it is carried on its own
        and the span between it and the window is compacted."""
        start = self.keep_start(messages, recent)
        if start <= index:
            return messages[:start], messages[start:]
        return messages[:index] + messages[index + 1 : start], [messages[index], *messages[start:]]

    def keep_start(self, messages: list[Json], recent: int | None) -> int:
        """Where the kept tail begins: at most `recent` messages, and at most a quarter of the
        request budget.

        The size bound is the reason the window could not simply be widened. A count is not a size,
        and a handful of very large messages inside the kept tail leaves the request over budget
        with nothing left to compact -- which is the failure COMPACT_MINIMUM_RECENT was added for.
        Bounding by both means small messages give the full window and large ones collapse it to
        the last exchange, which is what the per-line content verification achieved by accident."""
        limit = self.COMPACT_RECENT_MESSAGES if recent is None else recent
        share = max(1, self.ctx.request_token_budget() // 4)
        start = len(messages)
        while start > 0 and len(messages) - start < limit:
            # One message is always kept whatever its size; the bound only stops the tail growing.
            if start < len(messages) and self.ctx.request_tokens(messages[start - 1 :]) > share:
                break
            start -= 1
        return self.safe_cut(messages, start)

    @staticmethod
    def safe_cut(messages: list[Json], cut: int) -> int:
        """Move a cut back off the middle of a tool exchange. See parts_for."""
        if cut < len(messages) and messages[cut].get("role") == "tool":
            while cut > 0 and messages[cut - 1].get("role") == "tool":
                cut -= 1
            if cut > 0 and messages[cut - 1].get("role") == "assistant" and messages[cut - 1].get("tool_calls"):
                cut -= 1
        return cut

    def turn_parts(self, messages: list[Json], recent: int | None = None) -> tuple[list[Json], list[Json]]:
        ctx = self.ctx
        index = ctx.latest_user_index(messages)
        if index is None:
            start = self.keep_start(messages, recent)
            return self.without_summaries(messages[:start]), self.without_summaries(messages[start:])
        compacted, keep = self.split(messages, index, recent)
        return self.without_summaries(compacted), self.without_summaries(keep)

    def without_summaries(self, messages: list[Json]) -> list[Json]:
        return [message for message in messages if not self.ctx.is_compaction_summary(message)]

    def parts_for(self, messages: list[Json], recent: int | None = None) -> tuple[list[Json], list[Json]]:
        """Split messages into a compactable head and a recent tail, never inside a tool exchange.

        The cut walks back past a run of tool results and the assistant message that called them, since
        a history with tool calls whose results were summarized away -- or results whose call is gone --
        is rejected by every provider. Giving a few extra messages to the summary is the cheaper loss.
        That walk can reach zero, which is why a smaller `recent` does not always produce a head: a
        latest user message followed by one enormous tool result cannot be split here at all, and
        has to be bounded on the way in instead."""
        cut = self.safe_cut(messages, max(0, len(messages) - (self.COMPACT_RECENT_MESSAGES if recent is None else recent)))
        return messages[:cut], messages[cut:]

    @classmethod
    def echoes_source(cls, summary: str, source: str) -> bool:
        """True when `summary` reproduces `source` rather than describing it."""
        summary = " ".join(str(summary).split())[: cls.ECHO_COMPARE_CHARS]
        source = " ".join(str(source).split())[-cls.ECHO_COMPARE_CHARS :]
        if len(summary) < cls.ECHO_MIN_CHARS or not source:
            return False
        match = SequenceMatcher(None, summary, source, autojunk=False).find_longest_match(0, len(summary), 0, len(source))
        return match.size >= len(summary) * cls.ECHO_RATIO

    @staticmethod
    def title(data: Json | None) -> str:
        """The name the compactor gave this span, or "" when it gave none.

        The compaction request already returns a JSON object describing the span, so naming it
        costs a key rather than a call -- and the model is the only party that read the whole span.
        Bounded and flattened here: the key is free-form text from a model, and it lands in a
        segment listing, a checkpoint line, and a viewer column."""
        if not isinstance(data, dict) or not isinstance(data.get("title"), str):
            return ""
        return Tool.compact(" ".join(str(data["title"]).split()).strip("\"'"), 80)
