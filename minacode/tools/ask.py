"""Ask tool: structured multiple-choice questions for the user."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from minacode.base import Json, ToolError
from minacode.tools.base import Tool


@dataclass(frozen=True)
class AskSpec:
    """One validated question the model wants to ask the user."""

    question: str
    choices: list[str] | None = None
    previews: list[str] | None = None
    recommended: int | None = None


class AskTool(Tool):
    NAME = "Ask"
    DESCRIPTION = "Ask the user one or more questions (asked in sequence) and wait for their answers. Use when intent is genuinely ambiguous, a choice affects the codebase's external shape (module layout, public API, naming), or you need prioritization; prefer offering choices with previews, and optionally a recommended index when one option is clearly best. Do NOT ask about trivial internal details or anything determinable from context (Read/InspectCode/Bash) or already specified; if a reasonable default exists, proceed."
    EXAMPLE = (
        'One question, recommending a choice. Example: {"questions":[{"question":"Which approach?","choices":["Refactor","Rewrite"],"previews":["auth/\\n  session.py  (new, +87)\\n  views.py    (-12)","auth.py -> deleted\\nauth/*      all new (+430)"],"recommended":0}]}',
        'Batch related questions. Example: {"questions":[{"question":"Target runtime?","choices":["Node","Deno"]},{"question":"Name the module?"}]}',
    )
    MUTATES = False
    STORES_RESULT = True
    question_fn: Callable[[AskSpec, str], str] | None = None

    @classmethod
    def params_schema(cls) -> Json:
        # fmt: off
        question = cls.object_schema({
            "question": {"type": "string", "description": "The question to ask the user"},
            "choices": {"type": "array", "items": {"type": "string"}, "description": "Optional predefined choices the user can pick from"},
            "previews": {"type": "array", "items": {"type": "string"}, "description": "Optional preview per choice, shown as the user navigates. Make it graphic and concrete, not a restatement of the label: a short code/diff snippet, an ASCII layout or tree, or a file/API shape. Multi-line is fine (use \\n); keep under ~10 lines"},
            "recommended": {"type": "integer", "minimum": 0, "description": "Optional 0-based index of the recommended choice; pre-selected and marked"},
        }, ["question"])
        return cls.object_schema({
            "questions": {"type": "array", "minItems": 1, "description": "Questions to ask, one after another", "items": question},
        }, ["questions"])
        # fmt: on

    def call(self) -> str:
        questions = self.single_dict_arg(f"{self.NAME} requires named fields").get("questions")
        if not isinstance(questions, list) or not questions:
            raise ToolError(f"{self.NAME} requires a non-empty 'questions' list")
        # Validate the whole batch up front, so a malformed later question never strands the
        # user after they have already answered earlier ones.
        prepared: list[AskSpec] = []
        for item in questions:
            if not isinstance(item, dict):
                raise ToolError("each question must be an object with a 'question' field")
            question = str(item.get("question", "")).strip()
            if not question:
                raise ToolError("each question requires a 'question' field")
            choices = item.get("choices")
            previews = item.get("previews")
            recommended = item.get("recommended")
            if choices is not None:
                if not isinstance(choices, list) or not all(isinstance(c, str) for c in choices):
                    raise ToolError(f"{self.NAME} choices must be a list of strings")
                if previews is not None:
                    if not isinstance(previews, list) or not all(isinstance(p, str) for p in previews):
                        raise ToolError(f"{self.NAME} previews must be a list of strings")
                    if len(previews) != len(choices):
                        raise ToolError(f"{self.NAME} previews must match choices length")
            if recommended is not None and (
                isinstance(recommended, bool) or not isinstance(recommended, int) or not choices or not 0 <= recommended < len(choices)
            ):
                raise ToolError(f"{self.NAME} recommended must be a valid 0-based choice index")
            prepared.append(AskSpec(question, choices, previews, recommended))
        total = len(prepared)
        answers: list[tuple[str, str]] = []
        for index, spec in enumerate(prepared):
            position = f"{index + 1}/{total}" if total > 1 else ""
            answers.append((spec.question, self.question_fn(spec, position) if self.question_fn else spec.question))
        if len(answers) == 1:
            return answers[0][1]
        return "\n\n".join(f"Q: {q}\nA: {a}" for q, a in answers)

    def short_args(self) -> list[str]:
        questions = self.args[0].get("questions") if self.args and isinstance(self.args[0], dict) else None
        if not isinstance(questions, list) or not questions:
            return [""]
        first = str((questions[0] or {}).get("question", "") or "").strip() if isinstance(questions[0], dict) else ""
        label = Tool.compact(first, 80)
        return [label + (f" (+{len(questions) - 1} more)" if len(questions) > 1 else "")]
