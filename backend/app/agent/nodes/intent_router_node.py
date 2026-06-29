from app.agent.state import ResearchState


async def intent_router_node(state: ResearchState) -> ResearchState:
    query = state["query"].strip()
    intent = _classify_intent(query)
    return {
        **state,
        "query": query,
        "user_intent": intent,
    }


def _classify_intent(query: str) -> str:
    normalized = query.lower()

    summarize_terms = {
        "summarize",
        "summary",
        "tóm tắt",
        "tom tat",
        "summarise",
    }
    compare_terms = {
        "compare",
        "comparison",
        "versus",
        " vs ",
        "so sánh",
        "so sanh",
        "đối chiếu",
        "doi chieu",
    }
    report_terms = {
        "report",
        "survey",
        "literature review",
        "write a report",
        "viết báo cáo",
        "viet bao cao",
        "báo cáo",
        "bao cao",
    }
    question_terms = {
        "what",
        "why",
        "how",
        "when",
        "where",
        "which",
        "cái gì",
        "la gi",
        "là gì",
        "tại sao",
        "tai sao",
        "như thế nào",
        "nhu the nao",
    }

    if any(term in normalized for term in report_terms):
        return "full_research"
    if any(term in normalized for term in compare_terms):
        return "compare"
    if any(term in normalized for term in summarize_terms):
        return "summarize"
    if normalized.endswith("?") or any(term in normalized for term in question_terms):
        return "question_answering"
    return "full_research"
