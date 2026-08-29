"""Reasoning-history selection shared by the three wire projections."""

from minacode.base import Json


def keeps_reasoning(policy: str, message: Json, index: int, latest_user: int) -> bool:
    """Whether one assistant turn keeps its provider reasoning on the next request.

    Reasoning attached to a tool call is continuation state. ``current_turn`` keeps only that
    state after the latest real user message; ``tool_calls`` keeps it across turns; ``all`` also
    keeps final-answer reasoning. Each wire decides which of its opaque blocks count as reasoning.
    """

    return policy == "all" or (bool(message.get("tool_calls")) and (policy == "tool_calls" or (policy == "current_turn" and index > latest_user)))
