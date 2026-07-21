from app.agent.evaluators.context_quality import LATEST_QUERY_TERMS
from app.agent.models import append_trace
from app.agent.state import AgenticRAGState


async def classify_intent_node(state: AgenticRAGState) -> AgenticRAGState:
    request = state["request"]
    normalized_question = " ".join(request.question.lower().split())
    if any(term in normalized_question for term in LATEST_QUERY_TERMS):
        intent = "fresh_research"
        reason = "question_requests_current_or_recent_information"
    elif request.chat_history:
        intent = "follow_up_research"
        reason = "question_has_recent_chat_history"
    else:
        intent = "research_qa"
        reason = "default_research_question"

    return {
        **state,
        "intent": intent,
        "trace": append_trace(
            state.get("trace", []),
            "classify_intent",
            intent=intent,
            reason=reason,
        ),
    }
