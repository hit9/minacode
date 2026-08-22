"""minacode context: model message projection, deduplication, and compaction."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Hashable
from typing import ClassVar, TypeVar

from minacode.base import (
    ANTHROPIC_CONTENT_KEY,
    MAX_AGENTS_MD_TOKENS,
    MAX_TOOL_OUTPUT_TOKENS,
    PROVIDER_ECHO_KEYS,
    RESPONSES_OUTPUT_KEY,
    SESSION_EVENT_KEY,
    TOOL_OUTPUT_ASSET_SUFFIX,
    Json,
    Text,
)
from minacode.image import IMAGE_REFS_KEY, IMAGE_TEXT_ONLY_KEY, TOOL_IMAGE_OBSERVATION_KEY, ImageInputs
from minacode.model import ModelClient
from minacode.prompts import (
    COMPACTION_REQUEST_EVENT,
    COMPACTION_SUMMARY_TITLE,
    CURRENT_TURN_CONTEXT_TRIMMED,
    PREVIOUS_CONTEXT_TRIMMED,
    compaction_tail,
    language_directive,
)
from minacode.prompts import (
    compaction_input as format_compaction_input,
)
from minacode.session import HistorySegment, Session, local_timestamp
from minacode.tools import (
    Tool,
)

_IdentityT = TypeVar("_IdentityT", bound=Hashable)


class ContextManager:
    """Project session state into one request's messages and compact it to fit the budget.

    Request projection is derived at the send boundary and never stored: replay transforms must not
    write back into history. Compaction is the deliberate persisted exception. When the projected
    request exceeds its budget, older messages are captured in retained history and replaced by one
    summary checkpoint. Layer order exists for prompt-cache stability — version-stable system and
    tools, then session-stable environment/capability context, then the append-only conversation and
    active turn. Mutable working state is written as tool history or compaction checkpoints; inserting
    a rebuilt block into the conversation prefix would invalidate later cache reuse.

    Request-local transforms belong here rather than in stored messages: repeated MCP schemas and
    skill loads collapse to a pointer at the first copy, re-promoted when compaction removes it.

    The budget is the context limit less the provider's output reserve and a safety margin, measured
    against the payload that actually crosses the wire. Over budget compacts prior history first, and
    the current turn only if still over.
    """

    COMPACT_RECENT_MESSAGES: ClassVar[int] = 8
    # Fallback window when the ordinary one leaves nothing to compact. The recent window is a message
    # count, not a size, so a handful of very large messages after the latest user message can blow
    # the budget while all of them sit inside the kept tail -- and then every request is over budget
    # with an empty compactable head. Never zero: the latest exchange has to survive.
    COMPACT_MINIMUM_RECENT: ClassVar[int] = 2
    # How many evicted spans stay recallable. Every compaction adds one and nothing removed them, so
    # a long session carried every span it ever evicted -- each holding a bounded excerpt -- in
    # memory and in the segment list rewritten into later snapshot deltas. The newest spans are what
    # recall reaches for; the oldest describe work that is long done. Same bound as tool records,
    # smaller because a segment is a whole conversation span rather than one tool result.
    MAX_HISTORY_SEGMENTS: ClassVar[int] = 50
    MCP_DESCRIBE_BLOCK: ClassVar[re.Pattern] = re.compile(r"<MCPDescribe server=(\".*?\") tool=(\".*?\")>.*?</MCPDescribe>", re.DOTALL)
    SKILL_BLOCK: ClassVar[re.Pattern] = re.compile(r"<Skill name=(\".*?\")>.*?</Skill>", re.DOTALL)
    TOOL_RECORD_KEY: ClassVar[re.Pattern] = re.compile(r"\btr\.\d+\b")
    SEGMENT_KEY: ClassVar[re.Pattern] = re.compile(r"seg\.(\d+)")

    def __init__(self, session: Session, model: ModelClient | None = None):
        self.session = session
        self.model = model
        # Automatic compaction runs inside request projection, below the UI layer. This lifecycle
        # hook lets orchestration expose that real phase without making context depend on a renderer.
        # False is emitted in a finally block, including model failures that fall back to trimming.
        self.on_compaction: Callable[[bool, str], None] | None = None
        # Message count per scope ("history"/"turn") at the last automatic compaction decision. See
        # _auto_compaction_allowed: this is what stops a compaction loop.
        self._auto_compacted_at: dict[str, int] = {}

    def model_header(self, base_system: str) -> list[Json]:
        """Everything ahead of the conversation: system, environment, skills, MCP tools.

        Factored out because it is exactly the span a provider caches, and the compaction request
        reuses it verbatim so its summary rides the same prefix the turn just paid for."""
        content = base_system.strip()
        # A forced reply language appends one fixed block to the system tail: stable text that
        # depends only on the value, so the cacheable system prefix is unchanged.
        directive = language_directive(self.session.settings.language)
        if directive:
            content += "\n\n" + directive
        messages: list[Json] = [
            {"role": "system", "content": content},
            {"role": "user", "content": "--- Environment ---\n" + (self.environment() or "(empty)")},
        ]
        for context in (self.skills_context(), self.mcp_tools_context()):
            if context:
                messages.append({"role": "user", "content": context})
        return messages

    def model_messages(self, base_system: str, turn_messages: list[Json] | None = None) -> list[Json]:
        messages = self.model_header(base_system)
        conversation = [*self.session.messages, *(turn_messages or [])]
        messages.extend(self.dedup_skill_loads(self.dedup_mcp_describes(conversation)))
        return Text.value(messages)

    def dedup_mcp_describes(self, messages: list[Json]) -> list[Json]:
        """Point repeats at the first full description, promoting the next after compaction."""
        return self._dedup_tool_blocks(
            messages,
            self.MCP_DESCRIBE_BLOCK,
            lambda match: (str(json.loads(match.group(1))), str(json.loads(match.group(2)))),
            lambda identity, key: f"(repeat describe of {identity[0]}.{identity[1]}; schema shown earlier at {key}, unchanged)",
        )

    def dedup_skill_loads(self, messages: list[Json]) -> list[Json]:
        return self._dedup_tool_blocks(
            messages,
            self.SKILL_BLOCK,
            lambda match: str(json.loads(match.group(1))),
            lambda name, key: f"(repeat load of skill {name}; instructions shown earlier at {key}, unchanged)",
        )

    @staticmethod
    def _dedup_tool_blocks(
        messages: list[Json],
        block: re.Pattern,
        identity_from: Callable[[re.Match[str]], _IdentityT],
        marker_for: Callable[[_IdentityT, str], str],
    ) -> list[Json]:
        seen: dict[_IdentityT, str] = {}
        result: list[Json] = []
        for message in messages:
            content = message.get("content")
            if message.get("role") != "tool" or not isinstance(content, str):
                result.append(message)
                continue
            match = block.search(content)
            if match is None:
                result.append(message)
                continue
            try:
                identity = identity_from(match)
            except (json.JSONDecodeError, ValueError):
                result.append(message)
                continue
            first_key = seen.get(identity)
            if first_key is None:
                key = ContextManager.TOOL_RECORD_KEY.search(content)
                seen[identity] = key.group(0) if key else "above"
                result.append(message)
                continue
            marker = marker_for(identity, first_key)
            result.append({**message, "content": block.sub(lambda _, marker=marker: marker, content)})
        return result

    def mcp_tools_context(self) -> str:
        return self.session.mcp.render_tools_index() if self.session.mcp else ""

    def skills_context(self) -> str:
        return self.session.skills.index() if self.session.skills else ""

    def request_token_budget(self) -> int:
        return self.session.request_token_budget()

    def request_tokens(self, messages: list[Json], tools: list[Json] | None = None) -> int:
        if self.model is not None:
            return self.model.estimated_request_tokens(messages, tools)
        return self.estimated_tokens(messages) + (self.estimated_tokens(tools) if tools else 0)

    def update_percent(self, messages: list[Json], tools: list[Json] | None = None) -> int:
        self.session.state.context_percent = min(100, self.request_tokens(messages, tools) * 100 // self.request_token_budget())
        return self.session.state.context_percent

    def update_current_tokens(self, base_system: str) -> int:
        messages = self.model_messages(base_system, self.session._active_turn_messages)
        tools = Tool.resolved_schemas(self.session)
        tokens = self.request_tokens(messages, tools)
        self.session.state.context_percent = min(100, tokens * 100 // self.request_token_budget())
        return tokens

    def prepare_messages(self, model: ModelClient, base_system: str, turn_messages: list[Json] | None = None, tools: list[Json] | None = None) -> list[Json]:
        messages = self.model_messages(base_system, turn_messages)
        budget = self.request_token_budget()
        raw = self.request_tokens(messages, tools)
        if raw < budget and not self._overdue_by_usage():
            return messages
        attempted = compacted_any = False
        if self._auto_compaction_allowed("history", self.session.messages):
            attempted = True
            recent = None
            compacted, keep = self.compaction_parts()
            if not compacted:
                recent = self.COMPACT_MINIMUM_RECENT
                compacted, keep = self.compaction_parts(recent)
            if self._compact_messages(model, compacted, keep, PREVIOUS_CONTEXT_TRIMMED, tool_messages=turn_messages, recent=recent):
                compacted_any = True
                messages = self.model_messages(base_system, turn_messages)
            self._auto_compacted_at["history"] = len(self.session.messages)
        if turn_messages is not None and self.request_tokens(messages, tools) >= budget and self._auto_compaction_allowed("turn", turn_messages):
            attempted = True
            recent = None
            compacted, keep = self.turn_compaction_parts(turn_messages)
            if not compacted:
                recent = self.COMPACT_MINIMUM_RECENT
                compacted, keep = self.turn_compaction_parts(turn_messages, recent)
            if self._compact_messages(model, compacted, keep, CURRENT_TURN_CONTEXT_TRIMMED, turn_messages=turn_messages, recent=recent):
                compacted_any = True
                messages = self.model_messages(base_system, turn_messages)
            self._auto_compacted_at["turn"] = len(turn_messages)
        # Only a real dead end is worth a word: a pass ran, freed nothing, and the request is still
        # over budget. Reporting per pass would fire on every ordinary turn, where the history pass
        # does the work and the short current turn has nothing to give. `attempted` keeps it to one
        # report -- once both scopes are marked, later requests skip the passes and stay quiet.
        if attempted and not compacted_any and self.request_tokens(messages, tools) >= budget:
            self._report_incompressible()
        return messages

    def _auto_compaction_allowed(self, scope: str, messages: list[Json]) -> bool:
        """Whether an automatic pass over `scope` may run, given it already ran at some message count.

        Compaction shrinks history but cannot always get under budget -- a single huge tool result
        is not splittable at all -- so "still over budget" is not by itself a reason to compact
        again. Deciding twice on the same messages is the runaway compaction this has regressed into
        before. Any change in the count (growth, or the shrink compaction itself caused) buys exactly
        one more attempt, which also keeps a fresh, shorter turn from inheriting the previous turn's
        mark. The count is what an attempt is spent on, so a failed attempt is spent too and the
        report below cannot repeat on every request.
        """
        return self._auto_compacted_at.get(scope) != len(messages)

    def _report_incompressible(self) -> None:
        """Say that the request is going out over budget. Nothing here can fix it: what is left is
        the latest exchange, which compaction may never drop -- so say so instead of silently
        sending a request the provider is likely to reject."""
        if self.on_compaction is not None:
            self.on_compaction(False, "context is over budget and nothing is left to compact: the latest exchange alone exceeds it")

    def _overdue_by_usage(self) -> bool:
        """The last completed request filled ~99% of its budget, so the next one compacts even if the
        estimate still fits. The estimate is the primary trigger; this is the last line of defense
        when it is off, at the cost of possibly compacting a smaller follow-up."""
        usage = self.session.usage
        return usage.last_prompt_budget > 0 and usage.last_prompt_tokens * 100 >= usage.last_prompt_budget * 99

    def compaction_request(
        self, compacted: list[Json], turn_messages: list[Json] | None = None, recent: int | None = None, live_turn: list[Json] | None = None
    ) -> tuple[list[Json], list[Json]] | None:
        """The compaction request as the agent's own request truncated, plus one instruction, or
        None to use the flattened payload instead.

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
        if self.session.config.compaction_provider or self.session.system_info is None:
            return None
        base_system = self.session.system_prompt
        if not base_system or not compacted:
            return None
        # Projected with the turn attached whichever scope this is: the request being ridden
        # carries it, and its reasoning boundary is read off the whole projection below even when
        # only the stored half is being sliced.
        live = self.model_messages(base_system, turn_messages if turn_messages is not None else live_turn)
        header = len(self.model_header(base_system))
        # A turn-scope span sits after the stored conversation rather than at the head of it, so
        # its slice starts there. Both scopes are ordinary prefixes of the same projection; only
        # the offset differs.
        cut = header + (
            len(self.session.messages) + self.compaction_prefix_count(turn_messages, recent)
            if turn_messages is not None
            else self.compaction_prefix_count(recent=recent)
        )
        if cut <= header:
            return None
        tail = compaction_tail(
            state=self.session.state.format(),
            previous_summary=self.session.state.summary,
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
        return [*live[:cut], instruction], Tool.resolved_schemas(self.session)

    def compaction_prefix_count(self, turn_messages: list[Json] | None = None, recent: int | None = None) -> int:
        """How many messages of the scope's own list the summary request carries.

        One expression of the cut, shared with the split that produces `compacted`. Deriving it
        separately is what put the request out of step with what was being evicted twice already:
        once when the MINIMUM_RECENT fallback re-split with a different window, and again when the
        split grew a size bound this did not have."""
        messages = self.session.messages if turn_messages is None else turn_messages
        if turn_messages is None and self.latest_user_index(messages) is None:
            # compaction_parts hands the whole list over when there is no request to keep.
            return len(messages)
        return self.compaction_keep_start(messages, recent)

    def compaction_echo_source(self, sent: list[Json]) -> str:
        """What a copied summary would have been copied from.

        Takes what the request actually carries, never `compacted`. The inline slice deliberately
        reaches one message further than `compacted` does, and that extra message is the latest user
        message -- precisely the text the failure this guard exists for reproduced. Checking against
        `compacted` left the guard blind to exactly the case it was written for.

        The tail, not the whole span: recency is what the failure follows, and this feeds a
        substring search over a span with no size limit."""
        index = self.latest_user_index(sent)
        tail = sent if index is None else sent[index:]
        return self.messages_text(tail)[-4000:]

    def _compact_messages(
        self,
        model: ModelClient,
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
        on_compaction = self.on_compaction
        if on_compaction is not None:
            on_compaction(True, "")
        error_detail = ""
        interrupted = False
        try:
            try:
                request = self.compaction_request(compacted, turn_messages, recent, live_turn=tool_messages)
                # Checked against what the model is handed, which is not `compacted`: the inline
                # slice carries one message more, and the flattened payload carries `compacted`.
                sent = request[0][:-1] if request else compacted
                flat = self.compaction_input(compacted) if request is None else ""
                data = model.compact(flat, *(request or ()), echo_source=self.compaction_echo_source(sent))
            except KeyboardInterrupt:
                error_detail = "cancelled by user"
                interrupted = True
                data = None
            except Exception as error:  # noqa: BLE001 - compaction degrades to deterministic trimming on any model failure.
                error_detail = Text.clip_width(" ".join(str(error).split()) or type(error).__name__, 220)
                data = None
            self.apply_compaction(
                data,
                keep,
                tool_messages,
                turn_messages=turn_messages,
                fallback_note=fallback_note if data is None else "",
                compacted=compacted,
                model=getattr(model, "last_compaction_model", ""),
            )
        finally:
            if on_compaction is not None:
                on_compaction(False, error_detail)
        if interrupted:
            raise KeyboardInterrupt
        return True

    def environment(self) -> str:
        info = self.session.system_info
        assert info is not None
        rows = [
            f"- cwd: {info.cwd}",
            f"- session_started_at: {self.session.created_at}",
            # Tell the model which executables it may drive through Bash.
            "- detected_commands (available via Bash): " + (", ".join(info.commands) or "(none)"),
            f"- os: {info.os}",
            f"- arch: {info.arch}",
            f"- shell_timeout: {self.session.settings.shell_timeout}s",
        ]
        if (entry := self.session.config.vision_provider) and (not self.session.tool_names or "ViewImage" in self.session.tool_names):
            provider = self.session.config.providers[entry]
            rows.append(f"- vision: {entry}/{provider.model or '(empty)'} (available through ViewImage)")
        if self.session.settings.agents_md and info.agents_md:
            content = info.agents_md
            total = self.estimated_text_tokens(content)
            if total > MAX_AGENTS_MD_TOKENS:
                # Bound the fixed prefix (DESIGN.md): keep the head and tail, mark the middle. The
                # marker counts against the cap too, so reserve it before splitting the rest between
                # the excerpts. Reserving against `total` overstates it -- the omitted count printed
                # is never larger -- which is what makes one pass enough to stay under the cap.
                def marker_of(omitted: int) -> str:
                    return f"... ({info.agents_md_source} truncated to fit the prefix; approximately {omitted} tokens omitted) ..."

                limit = max(2, MAX_AGENTS_MD_TOKENS * 4 - len(marker_of(total)) - 2)  # 2 = the newlines joining the three parts
                head_limit = max(1, limit * 2 // 5)
                head = self.head_excerpt(content, head_limit)
                tail = self.tail_excerpt(content, max(1, limit - head_limit))
                omitted = max(0, total - self.estimated_text_tokens(head) - self.estimated_text_tokens(tail))
                content = "\n".join(part for part in (head.rstrip(), marker_of(omitted), tail.lstrip()) if part)
            rows.append("")
            rows.append(f"--- Project instructions ({info.agents_md_source}) ---")
            rows.append(content)
        return "\n".join(rows)

    def compaction_input(self, messages: list[Json]) -> str:
        older, recent = self.compaction_parts_for(messages)
        return format_compaction_input(
            state=self.session.state.format(),
            previous_summary=self.session.state.summary,
            older_messages=self.messages_text(older),
            recent_messages=self.messages_text(recent),
        )

    def compaction_parts(self, recent: int | None = None) -> tuple[list[Json], list[Json]]:
        """Split history for manual compaction and the first automatic pass."""
        messages = self.session.messages
        index = self.latest_user_index(messages)
        if index is None:
            return self.without_compaction_summaries(messages), []
        compacted, keep = self.compaction_split(messages, index, recent)
        return self.without_compaction_summaries(compacted), self.without_compaction_summaries(keep)

    def compaction_split(self, messages: list[Json], index: int, recent: int | None) -> tuple[list[Json], list[Json]]:
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
        start = self.compaction_keep_start(messages, recent)
        if start <= index:
            return messages[:start], messages[start:]
        return messages[:index] + messages[index + 1 : start], [messages[index], *messages[start:]]

    def compaction_keep_start(self, messages: list[Json], recent: int | None) -> int:
        """Where the kept tail begins: at most `recent` messages, and at most a quarter of the
        request budget.

        The size bound is the reason the window could not simply be widened. A count is not a size,
        and a handful of very large messages inside the kept tail leaves the request over budget
        with nothing left to compact -- which is the failure COMPACT_MINIMUM_RECENT was added for.
        Bounding by both means small messages give the full window and large ones collapse it to
        the last exchange, which is what the old anchor achieved by accident."""
        limit = self.COMPACT_RECENT_MESSAGES if recent is None else recent
        share = max(1, self.request_token_budget() // 4)
        start = len(messages)
        while start > 0 and len(messages) - start < limit:
            # One message is always kept whatever its size; the bound only stops the tail growing.
            if start < len(messages) and self.request_tokens(messages[start - 1 :]) > share:
                break
            start -= 1
        return self.safe_cut(messages, start)

    def turn_compaction_parts(self, messages: list[Json], recent: int | None = None) -> tuple[list[Json], list[Json]]:
        index = self.latest_user_index(messages)
        if index is None:
            start = self.compaction_keep_start(messages, recent)
            return self.without_compaction_summaries(messages[:start]), self.without_compaction_summaries(messages[start:])
        compacted, keep = self.compaction_split(messages, index, recent)
        return self.without_compaction_summaries(compacted), self.without_compaction_summaries(keep)

    def without_compaction_summaries(self, messages: list[Json]) -> list[Json]:
        return [message for message in messages if not self.is_compaction_summary(message)]

    def compaction_parts_for(self, messages: list[Json], recent: int | None = None) -> tuple[list[Json], list[Json]]:
        """Split messages into a compactable head and a recent tail, never inside a tool exchange.

        The cut walks back past a run of tool results and the assistant message that called them, since
        a history with tool calls whose results were summarized away — or results whose call is gone —
        is rejected by every provider. Giving a few extra messages to the summary is the cheaper loss.
        That walk can reach zero, which is why a smaller `recent` does not always produce a head: a
        latest user message followed by one enormous tool result cannot be split here at all, and
        has to be bounded on the way in instead.
        """
        cut = self.safe_cut(messages, max(0, len(messages) - (self.COMPACT_RECENT_MESSAGES if recent is None else recent)))
        return messages[:cut], messages[cut:]

    @staticmethod
    def safe_cut(messages: list[Json], cut: int) -> int:
        """Move a cut back off the middle of a tool exchange. See compaction_parts_for."""
        if cut < len(messages) and messages[cut].get("role") == "tool":
            while cut > 0 and messages[cut - 1].get("role") == "tool":
                cut -= 1
            if cut > 0 and messages[cut - 1].get("role") == "assistant" and messages[cut - 1].get("tool_calls"):
                cut -= 1
        return cut

    def messages_text(self, messages: list[Json]) -> str:
        return "\n\n".join(f"{message.get('role', 'message')}:\n{ImageInputs.label_text(message)}" for message in messages) or "(empty)"

    def compaction_title(self, data: Json | None) -> str:
        """The name the compactor gave this span, or "" when it gave none.

        The compaction request already returns a JSON object describing the span, so naming it
        costs a key rather than a call — and the model is the only party that read the whole span.
        Bounded and flattened here: the key is free-form text from a model, and it lands in a
        segment listing, a checkpoint line, and a viewer column."""
        if not isinstance(data, dict) or not isinstance(data.get("title"), str):
            return ""
        return Tool.compact(" ".join(str(data["title"]).split()).strip("\"'"), 80)

    def history_title(self, messages: list[Json]) -> str:
        """Deterministic fallback name: the first plain user message of the span. Used when the
        compactor named nothing — a summarizer failure trimming the span without a model reply, or
        a reply with no usable title."""
        for message in messages:
            if (
                message.get("role") == "user"
                and not message.get(SESSION_EVENT_KEY)
                and not str(message.get("content") or "").startswith(COMPACTION_SUMMARY_TITLE)
                and not ImageInputs.is_tool_observation(message)
            ):
                return Tool.compact(str(message.get("content") or ""), 80)
        return Tool.compact(self.messages_text(messages[:1]), 80) or "compacted context"

    def next_segment_key(self) -> str:
        """The next `seg.N`, counting from the highest key ever issued rather than the list length.

        Dropping the oldest segments shortens the list, so a length-derived key would hand a number
        the model has already seen to different content — and a stale `RecallContext seg.7` would
        silently answer with someone else's span instead of saying it is gone."""
        numbers = [int(match.group(1)) for segment in self.session.history if (match := self.SEGMENT_KEY.fullmatch(segment.key))]
        return f"seg.{max(numbers, default=len(self.session.history)) + 1}"

    def store_history_segment(self, compacted: list[Json], *, scope: str, trigger: str, fallback: bool, model: str = "") -> HistorySegment:
        key = self.next_segment_key()
        text = self.bound_output(self.messages_text(compacted))
        segment = HistorySegment(
            key=key,
            title=self.history_title(compacted),
            text=text,
            created_at=local_timestamp(),
            scope=scope,
            trigger=trigger,
            fallback=fallback,
            messages=len(compacted),
            model=model,
        )
        self.session.history.append(segment)
        del self.session.history[: -self.MAX_HISTORY_SEGMENTS]  # newest kept; a shorter list is left alone
        return segment

    def _summary_block(self, segment: HistorySegment | None) -> list[Json]:
        """One durable checkpoint containing everything needed after the compacted prefix."""
        rows = [
            COMPACTION_SUMMARY_TITLE,
            "Summary:",
            self.session.state.summary or "(empty)",
            "",
            "Working state:",
            self.session.state.format(),
        ]
        if segment is not None:
            rows.extend(("", f"Stored history segment: {segment.key}: {segment.title}"))
        return [{"role": "user", "content": "\n".join(rows), SESSION_EVENT_KEY: "compaction_checkpoint"}]

    def apply_compaction(
        self,
        data: Json | None,
        keep: list[Json],
        tool_messages: list[Json] | None = None,
        *,
        turn_messages: list[Json] | None = None,
        fallback_note: str = "",
        compacted: list[Json] | None = None,
        trigger: str = "auto",
        model: str = "",
    ) -> None:
        self.session.state.compaction_count += 1
        # What this compaction was: the turn scope is the only caller that rewrites `turn_messages`,
        # and no summary data means the model call failed and `keep` is all that survives. Recorded
        # on the segment so `/compact log` can say which evictions were lossier than the rest.
        segment = (
            self.store_history_segment(
                compacted,
                scope="turn" if turn_messages is not None else "history",
                trigger=trigger,
                fallback=data is None,
                model=model,
            )
            if compacted
            else None
        )
        if data is not None:
            self.session.state.apply(data)
        if fallback_note:
            self.session.state.summary = (self.session.state.summary + "\n" + fallback_note).strip()
        if segment is not None:
            # After apply(): the summary worth keeping is the one this compaction just produced,
            # not the one it replaced. The compactor's own name for the span replaces the
            # deterministic one, which was only ever the first user message of the window and says
            # little once a span starts mid-work.
            segment.summary = self.session.state.summary
            segment.title = self.compaction_title(data) or segment.title
        summary_block = self._summary_block(segment)
        if turn_messages is None:
            self.session.messages = summary_block + keep
            prune_context = (self.session.messages if data is not None else [*keep]) + (tool_messages or [])
        else:
            index = self.latest_user_index(keep)
            insert = len(keep) if index is None else index + 1
            turn_messages[:] = keep[:insert] + summary_block + keep[insert:]
            prune_context = [*self.session.messages, *turn_messages]
        self.prune_tool_records(prune_context)
        # The recorded usage described the pre-compaction payload and no longer reflects what the
        # next request will carry (and a manual /compact ran a compaction request whose own usage
        # just overwrote the last-* fields). Clear them so the overdue guard and the status bar fall
        # back to the local estimate until the next ordinary request reports real usage again.
        # Cumulative totals stay: the compaction request was still billed.
        usage = self.session.usage
        usage.last_prompt_tokens = 0
        usage.last_prompt_budget = 0
        usage.last_cached_prompt_tokens = 0
        usage.last_cache_write_prompt_tokens = 0

    def prune_tool_records(self, keep_messages: list[Json]) -> None:
        records = self.session.tool_records
        keep = set(self.TOOL_RECORD_KEY.findall(self.messages_text(keep_messages)))
        self.session.tool_records = [record for record in records if record.key in keep][-400:]
        self.session.tool_results = {record.key: record.output for record in self.session.tool_records}

    def latest_user_index(self, messages: list[Json]) -> int | None:
        for index in range(len(messages) - 1, -1, -1):
            if (
                messages[index].get("role") == "user"
                and not messages[index].get(SESSION_EVENT_KEY)
                and not self.is_compaction_summary(messages[index])
                and not ImageInputs.is_tool_observation(messages[index])
            ):
                return index
        return None

    def is_compaction_summary(self, message: Json) -> bool:
        return message.get("role") == "user" and str(message.get("content") or "").startswith(COMPACTION_SUMMARY_TITLE)

    # What to do about the omitted middle, in the marker that reports it. Kept to one line: this is
    # paid on every truncated output, and the model needs the next move, not an explanation. Ordered
    # by what is certain to be there -- Search is built in, grep is near-universal, jq is neither.
    OMITTED_OUTPUT_HINT = "file holds this output in full; Search it for the part you need, or Bash grep/jq -- cheaper than paging the text back with Recall"
    OMITTED_OUTPUT_RECALL_HINT = "Recall this key with ranges to page the omitted lines back"

    def bound_output(self, text: str, key: str = "") -> str:
        estimated = self.estimated_text_tokens(text)
        if estimated <= MAX_TOOL_OUTPUT_TOKENS:
            return text
        limit = MAX_TOOL_OUTPUT_TOKENS * 4
        head_limit = max(1, limit * 2 // 5)
        tail_limit = max(1, limit - head_limit)
        head = self.head_excerpt(text, head_limit)
        tail = self.tail_excerpt(text, tail_limit)
        omitted_tokens = max(0, estimated - self.estimated_text_tokens(head) - self.estimated_text_tokens(tail))
        note = f'<bounded_output omitted="middle" max_tokens="{MAX_TOOL_OUTPUT_TOKENS}"'
        note += f' estimated_tokens="{estimated}" omitted_tokens="{omitted_tokens}"'
        note += f' recall="{key}"' if key else ""
        if key:
            path = self.materialize_output(key, text)
            if path:
                note += f' file="{path}"'
            # An attribute name is not an instruction: `recall` and `file` say where the rest of the
            # output is, not what to do about it. Say which one to reach for, because the cheap move
            # is the non-obvious one -- searching the file costs the matched lines, while recalling
            # pays context for every line it pages back, including all the ones that were skipped.
            note += f' hint="{self.OMITTED_OUTPUT_HINT if path else self.OMITTED_OUTPUT_RECALL_HINT}"'
        note += "/>"
        return "\n".join(part for part in (head.rstrip(), note, tail.lstrip()) if part)

    def materialize_output(self, key: str, text: str) -> str:
        """Write the full tool output next to the truncated marker as a navigable artifact.

        Derived cache only: session.tool_results and the jsonl stay the source of truth. Failures
        are swallowed so a read-only or full disk cannot break truncation itself.
        """
        try:
            directory = self.session.images.assets_dir()
            path = os.path.join(directory, key + TOOL_OUTPUT_ASSET_SUFFIX)
            os.makedirs(directory, exist_ok=True)
            with open(path, "w", encoding="utf-8") as file:
                file.write(text)
            return path
        except OSError:
            return ""

    # Snapping an excerpt to a line boundary may only cost this fraction of the budget. A payload
    # whose lines are longer than the budget -- one-line JSON is the common case, and exactly what
    # an MCP server returns -- would otherwise snap away nearly everything it was asked to keep.
    EXCERPT_SNAP_FLOOR = 0.5

    @classmethod
    def head_excerpt(cls, text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        window = text[:limit]
        snapped = window.rsplit("\n", 1)[0]
        return snapped if len(snapped) >= limit * cls.EXCERPT_SNAP_FLOOR else window

    @classmethod
    def tail_excerpt(cls, text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        window = text[-limit:]
        snapped = window.split("\n", 1)[-1]
        return snapped if len(snapped) >= limit * cls.EXCERPT_SNAP_FLOOR else window

    def estimated_tokens(self, messages: list[Json]) -> int:
        # Normalized assistant fields already contain visible text and tool calls, so provider
        # echoes would double-count them. Preserve only additional readable reasoning; ciphertext
        # and signatures are transport state whose byte length is not a prompt-token estimate.
        def readable_provider_context(message: Json) -> list[str]:
            readable: list[str] = []
            responses = message.get(RESPONSES_OUTPUT_KEY)
            if isinstance(responses, list):
                for item in responses:
                    if not isinstance(item, dict) or item.get("type") != "reasoning":
                        continue
                    readable.extend(str(item[key]) for key in ("content", "summary") if item.get(key))
            anthropic = message.get(ANTHROPIC_CONTENT_KEY)
            if isinstance(anthropic, list):
                for block in anthropic:
                    if isinstance(block, dict) and block.get("type") in ("thinking", "redacted_thinking") and block.get("thinking"):
                        readable.append(str(block["thinking"]))
            return readable

        payload: list[Json] = []
        for message in messages:
            estimated = {
                key: value
                for key, value in message.items()
                if key not in (*PROVIDER_ECHO_KEYS, IMAGE_REFS_KEY, IMAGE_TEXT_ONLY_KEY, TOOL_IMAGE_OBSERVATION_KEY, SESSION_EVENT_KEY)
            }
            if readable := readable_provider_context(message):
                estimated["_provider_context"] = readable
            payload.append(estimated)
        chars = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        images = self.session.images.estimated_tokens(messages)
        return (chars + 3) // 4 + images

    @staticmethod
    def estimated_text_tokens(text: str) -> int:
        return (len(text) + 3) // 4
