"""tool ask (split from tests/test_tools.py)."""
import pytest
from test_tools import _q, session

import wizolt
from wizolt.base import (
    ToolCall,
    ToolError,
)
from wizolt.context import ContextManager
from wizolt.runner import ToolRunner
from wizolt.tools import (
    TOOL_REGISTRY,
    TOOLS,
    AskTool,
)


async def test_ask_tool_call_basic(tmp_path):
    """call() returns question text when question_fn is None."""
    s = session(tmp_path)
    assert await AskTool(s, _q({"question": "Which approach?"})).call() == "Which approach?"

async def test_ask_tool_call_callback_passthrough_choices_none(tmp_path):
    """call() passes choices/previews/recommended as None when not provided."""
    s = session(tmp_path)
    calls = []

    async def fake_fn(specs):
        calls.append(specs)
        return ["free text answer"]

    tool = AskTool(s, _q({"question": "Name?"}))
    tool.question_fn = fake_fn
    assert await tool.call() == "free text answer"
    spec = calls[0][0]
    assert spec.choices is None
    assert spec.previews is None
    assert spec.recommended is None

async def test_ask_tool_call_empty_list_raises(tmp_path):
    """call() raises ToolError when questions list is missing or empty."""
    s = session(tmp_path)
    with pytest.raises(ToolError, match="non-empty 'questions' list"):
        await AskTool(s, [{"questions": []}]).call()
    with pytest.raises(ToolError, match="non-empty 'questions' list"):
        await AskTool(s, [{}]).call()

async def test_ask_tool_call_empty_question_raises(tmp_path):
    """call() raises ToolError for empty/missing question text."""
    s = session(tmp_path)
    with pytest.raises(ToolError, match="each question requires a 'question' field"):
        await AskTool(s, _q({"question": ""})).call()
    with pytest.raises(ToolError, match="each question requires a 'question' field"):
        await AskTool(s, _q({})).call()

async def test_ask_tool_call_invalid_args_raises(tmp_path):
    """call() raises ToolError for malformed top-level args."""
    s = session(tmp_path)
    with pytest.raises(ToolError, match="Ask requires named fields"):
        await AskTool(s, ["just a string"]).call()
    with pytest.raises(ToolError, match="Ask requires named fields"):
        await AskTool(s, []).call()

async def test_ask_tool_call_invalid_choices_raises(tmp_path):
    """call() validates choices type."""
    s = session(tmp_path)
    with pytest.raises(ToolError, match="Ask choices must be a list of strings"):
        await AskTool(s, _q({"question": "Q", "choices": "not-a-list"})).call()
    with pytest.raises(ToolError, match="Ask choices must be a list of strings"):
        await AskTool(s, _q({"question": "Q", "choices": [1, 2, 3]})).call()

async def test_ask_tool_call_invalid_previews_raises(tmp_path):
    """call() validates previews type and length."""
    s = session(tmp_path)
    with pytest.raises(ToolError, match="Ask previews must be a list of strings"):
        await AskTool(s, _q({"question": "Q", "choices": ["A"], "previews": [1]})).call()
    with pytest.raises(ToolError, match="Ask previews must match choices length"):
        await AskTool(s, _q({"question": "Q", "choices": ["A", "B"], "previews": ["only one"]})).call()

async def test_ask_tool_call_invalid_recommended_raises(tmp_path):
    """call() validates recommended is an in-range choice index."""
    s = session(tmp_path)
    with pytest.raises(ToolError, match="valid 0-based choice index"):
        await AskTool(s, _q({"question": "Q", "choices": ["A", "B"], "recommended": 2})).call()
    with pytest.raises(ToolError, match="valid 0-based choice index"):
        await AskTool(s, _q({"question": "Q", "recommended": 0})).call()  # no choices
    with pytest.raises(ToolError, match="valid 0-based choice index"):
        await AskTool(s, _q({"question": "Q", "choices": ["A"], "recommended": True})).call()  # bool not int

async def test_ask_tool_call_invokes_callback(tmp_path):
    """The whole batch goes to question_fn in one call, and its single answer comes back verbatim."""
    s = session(tmp_path)
    calls = []

    async def fake_fn(specs):
        calls.append(specs)
        return ["user chose B"]

    tool = AskTool(s, _q({"question": "A or B?", "choices": ["A", "B"], "previews": ["PA", "PB"], "recommended": 1}))
    tool.question_fn = fake_fn
    result = await tool.call()
    assert result == "user chose B"
    spec = calls[0][0]
    assert (spec.question, spec.choices, spec.previews, spec.recommended) == ("A or B?", ["A", "B"], ["PA", "PB"], 1)
    assert len(calls[0]) == 1  # the whole batch arrives in one call

async def test_ask_tool_call_multiple_questions(tmp_path):
    """The whole batch is asked at once, and the combined answers are labelled."""
    s = session(tmp_path)
    asked = []

    async def fake_fn(specs):
        asked.extend(spec.question for spec in specs)
        return [{"Runtime?": "Node", "Name?": "core"}[spec.question] for spec in specs]

    tool = AskTool(
        s,
        _q(
            {"question": "Runtime?", "choices": ["Node", "Deno"]},
            {"question": "Name?"},
        ),
    )
    tool.question_fn = fake_fn
    result = await tool.call()
    assert asked == ["Runtime?", "Name?"]  # batch order preserved
    assert result == "Q: Runtime?\nA: Node\n\nQ: Name?\nA: core"

async def test_ask_tool_call_no_previews_with_choices(tmp_path):
    """call() allows choices without previews."""
    s = session(tmp_path)
    assert await AskTool(s, _q({"question": "Q", "choices": ["A", "B"]})).call() == "Q"

async def test_ask_tool_call_with_choices(tmp_path):
    """call() accepts choices and returns fallback question text."""
    s = session(tmp_path)
    assert await AskTool(s, _q({"question": "Which?", "choices": ["A", "B"]})).call() == "Which?"

async def test_ask_tool_call_with_choices_and_previews(tmp_path):
    """call() accepts choices + previews."""
    s = session(tmp_path)
    tool = AskTool(
        s,
        _q(
            {
                "question": "Which?",
                "choices": ["A", "B"],
                "previews": ["Preview A", "Preview B"],
            }
        ),
    )
    assert await tool.call() == "Which?"

def test_ask_tool_registered():
    """AskTool is in TOOLS and TOOL_REGISTRY."""
    assert AskTool.NAME == "Ask"
    assert AskTool in TOOLS
    assert TOOL_REGISTRY["Ask"] is AskTool
    assert "Question" not in TOOL_REGISTRY
    assert not hasattr(wizolt, "QuestionTool")

def test_ask_tool_schema():
    """params_schema requires a questions array of question objects, strict."""
    schema = AskTool.params_schema()
    assert schema["type"] == "object"
    assert schema["required"] == ["questions"]
    assert schema["additionalProperties"] is False
    questions = schema["properties"]["questions"]
    assert questions["type"] == "array"
    assert questions["minItems"] == 1
    item = questions["items"]
    assert item["required"] == ["question"]
    assert item["additionalProperties"] is False
    props = item["properties"]
    assert props["question"]["type"] == "string"
    assert props["choices"]["items"]["type"] == "string"
    assert props["previews"]["items"]["type"] == "string"
    assert props["recommended"]["type"] == "integer"

def test_ask_tool_schema_strict(tmp_path):
    """schema() enforces additionalProperties=False at both levels."""
    schema = AskTool.schema()
    params = schema["function"]["parameters"]
    assert params["additionalProperties"] is False
    assert "questions" in params["properties"]
    item = params["properties"]["questions"]["items"]
    assert item["additionalProperties"] is False
    assert "question" in item["properties"]
    assert "choices" in item["properties"]
    assert "previews" in item["properties"]

def test_ask_tool_short_args(tmp_path):
    """short_args() shows the first question and a count of the rest."""
    s = session(tmp_path)
    tool = AskTool(s, _q({"question": "Which approach should I use?"}))
    args = tool.short_args()
    assert len(args) == 1
    assert "Which approach" in args[0]
    assert "more" not in args[0]
    multi = AskTool(s, _q({"question": "First?"}, {"question": "Second?"}))
    assert "(+1 more)" in multi.short_args()[0]
    assert len(AskTool(s, []).short_args()) == 1

async def test_ask_tool_validates_batch_before_asking(tmp_path):
    """A malformed later question raises before any question is asked."""
    s = session(tmp_path)
    asked = []

    async def fake_fn(specs):
        asked.extend(spec.question for spec in specs)
        return ["x"] * len(specs)

    tool = AskTool(
        s,
        _q(
            {"question": "First?", "choices": ["A"]},
            {"question": "Second?", "choices": ["A", "B"], "recommended": 5},  # out of range
        ),
    )
    tool.question_fn = fake_fn
    with pytest.raises(ToolError, match="valid 0-based choice index"):
        await tool.call()
    assert asked == []  # validation happens up front, so nothing was asked

def test_ask_tool_wired_in_tool_runner(tmp_path):
    """ToolRunner injects question_fn into AskTool instances."""
    s = session(tmp_path)
    ctx = ContextManager(s)
    captured = []

    async def fake_question_fn(specs):
        captured.append(specs)
        return ["test answer"]

    runner = ToolRunner(s, ctx, output_fn=lambda text: None)
    runner.question_fn = fake_question_fn
    results = runner.run_sync([ToolCall("q", "Ask", [{"questions": [{"question": "A or B?", "choices": ["A", "B"], "recommended": 0}]}])])
    assert len(results) == 1
    assert results[0]["tool_call_id"] == "q"
    assert results[0]["role"] == "tool"
    assert "test answer" in results[0]["content"]
    spec = captured[0][0]
    assert (spec.question, spec.choices, spec.recommended) == ("A or B?", ["A", "B"], 0)

def test_auto_approved_tool_prints_single_line_with_tag(tmp_path):
    # In yolo mode a confirmation-requiring tool without a preview (Bash) should print only the
    # result line tagged [auto], not a redundant "auto …" pre-line that duplicates the header.
    s = session(tmp_path)
    s.settings.yolo = True
    out = []
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: out.append(str(text)))
    runner.run_sync([ToolCall("b0", "Bash", [":"])])
    assert len(out) == 1
    assert out[0].startswith("  Bash  ")
    assert out[0].rstrip().endswith("[auto]")
    assert sum(line.startswith("  Bash  ") for line in out) == 1
