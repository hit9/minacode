"""Worker handoff: the second in-process session a parent delegates to (see DESIGN.md).

Coverage follows WORKER_HANDOFF_PLAN.txt section 9; each numbered test maps to that list.
"""

from agent_harness import session

from minacode.context import ContextManager
from minacode.engine import Agent
from minacode.prompts import SYSTEM_PROMPT
from minacode.tools import TOOL_REGISTRY, Tool


def _requested_system(tmp_path, custom=None):
    s = session(tmp_path)
    if custom is not None:
        s.system_prompt = custom
    agent = Agent(s, output_fn=lambda text: None)
    request = agent.prepare_request([{"role": "user", "content": "hi"}])
    return s, request.messages[0]["content"]


# 1. tool_names filtering: only the whitelisted schemas, in TOOL_REGISTRY order; empty tuple is
#    exactly the unfiltered behavior.
def test_tool_names_filter_resolved_schemas_and_keep_registry_order(tmp_path):
    s = session(tmp_path)
    names = ("Read", "Edit", "Bash")
    s.tool_names = names
    resolved = [schema["function"]["name"] for schema in Tool.resolved_schemas(s)]
    assert resolved == [name for name in TOOL_REGISTRY if name in names]

    # Empty tuple = no filtering; identical to a session that never set tool_names.
    s.tool_names = ()
    unfiltered = [schema["function"]["name"] for schema in Tool.resolved_schemas(s)]
    plain = [schema["function"]["name"] for schema in Tool.resolved_schemas(session(tmp_path))]
    assert unfiltered == plain
    assert "Read" in unfiltered and "Edit" in unfiltered and "Bash" in unfiltered


# 2. system_prompt comes from the session: the request payload's system content changes with it,
#    and the parent default is unchanged.
def test_system_prompt_comes_from_session(tmp_path):
    s, system = _requested_system(tmp_path, custom="CUSTOM WORKER ROLE")
    assert system == "CUSTOM WORKER ROLE"

    parent, parent_system = _requested_system(tmp_path)
    assert parent_system == SYSTEM_PROMPT.strip()


def test_system_prompt_default_matches_prompts_module(tmp_path):
    _, system = _requested_system(tmp_path)
    assert system == SYSTEM_PROMPT.strip()
    assert ContextManager(session(tmp_path)).model_messages(SYSTEM_PROMPT)[0]["content"] == SYSTEM_PROMPT.strip()
