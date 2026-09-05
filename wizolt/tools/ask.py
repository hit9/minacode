"""Ask tool: structured multiple-choice questions for the user."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from wizolt.base import Json, ToolError
from wizolt.tools.base import Tool


@dataclass(frozen=True)
class AskSpec:
    """One validated question the model wants to ask the user."""

    question: str
    choices: list[str] | None = None
    previews: list[str] | None = None
    recommended: int | None = None


class AskTool(Tool):
    NAME = "Ask"
    DESCRIPTION = (
        "Ask and wait when missing intent or a choice changes the result; do not ask what tools or context can answer. Batch related questions "
        "and keep labels short. Previews can mix explanation with visual Markdown examples; use sections and blank lines, not repeated question "
        "text or controls."
    )
    EXAMPLE = ('Example: {"questions":[{"question":"Which approach?","choices":["Refactor","Rewrite"],"recommended":0}]}',)
    MUTATES = False
    STORES_RESULT = True
    # Injected by ToolRunner: asks the whole batch and awaits the answers. Awaitable because the
    # questions are put to the user on the runtime loop, which must stay responsive while they sit
    # unanswered -- an Ask can wait as long as the user takes.
    question_fn: Callable[[list[AskSpec]], Awaitable[list[str]]] | None = None

    @classmethod
    def params_schema(cls) -> Json:
        # fmt: off
        question = cls.object_schema({
            "question": {"type": "string", "minLength": 1, "description": "Required; never omit. Question shown to the user"},
            "choices": {"type": "array", "items": {"type": "string"}, "description": "Optional concise, mutually exclusive labels the user can pick from"},
            "previews": {"type": "array", "items": {"type": "string"}, "description": "Optional rich Markdown per choice: combine explanation with examples, diagrams, code, diffs, tables, or trees; use short sections and blank lines without redundant question text or controls"},
            "recommended": {"type": "integer", "minimum": 0, "description": "Optional 0-based index of the recommended choice; pre-selected and marked"},
        }, ["question"])
        return cls.object_schema({
            "questions": {"type": "array", "minItems": 1, "description": "One or more questions; every item requires question", "items": question},
        }, ["questions"])
        # fmt: on

    async def call(self) -> str:
        prepared = self._prepared()
        # Ask for the whole batch at once (the modal pages through it); fall back to the question
        # texts when no interactive question function is wired.
        answers = await self.question_fn(prepared) if self.question_fn else [spec.question for spec in prepared]
        if len(answers) == 1:
            return answers[0]
        return "\n\n".join(f"Q: {spec.question}\nA: {answer}" for spec, answer in zip(prepared, answers))

    def _prepared(self) -> list[AskSpec]:
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
        return prepared

    def short_args(self) -> list[str]:
        questions = self.args[0].get("questions") if self.args and isinstance(self.args[0], dict) else None
        if not isinstance(questions, list) or not questions:
            return [""]
        first = str((questions[0] or {}).get("question", "") or "").strip() if isinstance(questions[0], dict) else ""
        label = Tool.compact(first, 80)
        return [label + (f" (+{len(questions) - 1} more)" if len(questions) > 1 else "")]
