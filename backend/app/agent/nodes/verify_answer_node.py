from app.agent.models import AgentTraceEvent, VerificationResult, append_trace
from app.agent.state import AgenticRAGState


def verification_trace_event(
    trace: list[AgentTraceEvent],
    verification: VerificationResult,
) -> list[AgentTraceEvent]:
    return append_trace(
        trace,
        "verify_answer",
        status="passed" if verification.passed else "revised",
        success=verification.passed,
        issue_count=len(verification.issues),
        unsupported_claim_count=len(verification.unsupported_claims),
        suggested_action=verification.suggested_action,
    )


async def verify_answer_node(state: AgenticRAGState) -> AgenticRAGState:
    verifier = state["answer_verifier"]
    citation_grounder = state["citation_grounder"]
    verification = verifier.verify(state.get("answer", ""), state.get("citations", []))
    display_answer = citation_grounder.display_answer_with_numbered_citations(
        verification.answer,
        verification.citations,
    )
    return {
        **state,
        "answer": display_answer,
        "citations": verification.citations,
        "verification": verification,
        "trace": verification_trace_event(state.get("trace", []), verification),
    }
