import pytest

from app.agent.models import VerificationResult
from app.agent.citations import CitationGrounder
from app.agent.nodes.verify_answer_node import verification_trace_event, verify_answer_node


class FakeVerifier:
    def __init__(self) -> None:
        self.calls = []

    def verify(self, answer: str, citations: list) -> VerificationResult:
        self.calls.append({"answer": answer, "citations": citations})
        return VerificationResult(
            passed=True,
            answer=f"{answer} [verified]",
            citations=citations[:1],
            issues=[],
            unsupported_claims=[],
            suggested_action="finalize",
        )


def test_verification_trace_event_records_verifier_decision():
    verification = VerificationResult(
        passed=False,
        answer="I don't know",
        citations=[],
        issues=["answer_references_no_valid_citations"],
        unsupported_claims=["unsupported claim"],
        suggested_action="retrieve_more",
    )

    trace = verification_trace_event([{"stage": "draft_answer"}], verification)

    assert trace == [
        {"stage": "draft_answer"},
        {
            "stage": "verify_answer",
            "status": "revised",
            "success": False,
            "issue_count": 1,
            "unsupported_claim_count": 1,
            "suggested_action": "retrieve_more",
        },
    ]


def test_verification_trace_event_marks_passed_answers():
    verification = VerificationResult(
        passed=True,
        answer="Supported answer [1].",
        citations=[],
        issues=[],
        unsupported_claims=[],
        suggested_action="finalize",
    )

    trace = verification_trace_event([], verification)

    assert trace == [
        {
            "stage": "verify_answer",
            "status": "passed",
            "success": True,
            "issue_count": 0,
            "unsupported_claim_count": 0,
            "suggested_action": "finalize",
        },
    ]


@pytest.mark.asyncio
async def test_verify_answer_node_updates_answer_citations_and_trace():
    verifier = FakeVerifier()
    citations = [object(), object()]

    state = await verify_answer_node(
        {
            "answer_verifier": verifier,
            "citation_grounder": CitationGrounder(),
            "answer": "Draft answer",
            "citations": citations,
            "trace": [{"stage": "draft_answer"}],
        }
    )

    assert verifier.calls == [{"answer": "Draft answer", "citations": citations}]
    assert state["answer"] == "Draft answer"
    assert state["citations"] == citations[:1]
    assert state["verification"].suggested_action == "finalize"
    assert state["trace"][-1] == {
        "stage": "verify_answer",
        "status": "passed",
        "success": True,
        "issue_count": 0,
        "unsupported_claim_count": 0,
        "suggested_action": "finalize",
    }
