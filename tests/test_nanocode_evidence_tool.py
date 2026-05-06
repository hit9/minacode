import pytest

from nanocode import Agent, EvidenceItem, EvidenceTool, KnownItem, Session, ToolCallError


def test_evidence_tool_gets_multiple_keys(tmp_path):
    session = Session(cwd=str(tmp_path))
    session.evidence = {"parser.notes": "line 1\nline 2"}

    result = EvidenceTool.make(session, ["parser.notes", "missing"]).call()
    prefixed_result = EvidenceTool.make(session, ["get", "parser.notes"]).call()

    assert '<Evidence key="parser.notes">\nline 1\nline 2\n  </Evidence>' in result
    assert '<Missing key="missing"/>' in result
    assert '<Evidence key="parser.notes">\nline 1\nline 2\n  </Evidence>' in prefixed_result


def test_prompt_shows_evidence_descriptions_and_active_values(tmp_path):
    session = Session(cwd=str(tmp_path))
    session.current.known = [KnownItem(fact="Parser notes were captured.", evidence_keys=["parser.notes"])]
    session.evidence = {"parser.notes": EvidenceItem(description="Parser notes from sample output.", value="line 1\nline 2")}

    prompt = Agent(session).build_user_prompt()

    assert "<Details_Keys>" not in prompt
    assert "<Evidence>" in prompt
    assert "<Active_Evidence>" in prompt
    assert "Parser notes were captured." in prompt
    assert "parser.notes" in prompt
    assert "Parser notes from sample output." in prompt
    assert "line 1" in prompt


def test_known_action_stores_evidence_descriptions_and_values(tmp_path):
    session = Session(cwd=str(tmp_path))
    agent = Agent(session)

    agent.apply_response(
        {
            "actions": [
                {
                    "type": "known",
                    "items": [
                        {
                            "fact": "Parser notes were captured.",
                            "evidence": [
                                {
                                    "key": "parser.notes",
                                    "description": "Parser notes from sample output.",
                                    "value": "line 1\nline 2",
                                }
                            ],
                        }
                    ],
                }
            ]
        }
    )

    assert session.current.known == [KnownItem(fact="Parser notes were captured.", evidence_keys=["parser.notes"])]
    assert session.evidence_store["parser.notes"] == EvidenceItem(description="Parser notes from sample output.", value="line 1\nline 2")
    assert "parser.notes" in agent.state_updater.latest_report
    assert "Parser notes from sample output." in agent.state_updater.latest_report
    assert "line 1" not in agent.state_updater.latest_report


def test_evidence_invalid_args(tmp_path):
    session = Session(cwd=str(tmp_path))

    with pytest.raises(ToolCallError, match="Evidence requires"):
        EvidenceTool.make(session, []).call()
