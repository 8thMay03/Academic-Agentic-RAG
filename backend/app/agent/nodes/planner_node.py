from app.agent.state import ResearchState


async def planner_node(state: ResearchState) -> ResearchState:
    query = state["query"].strip()
    max_results = state.get("max_results", 5)
    user_intent = state.get("user_intent", "full_research")
    min_papers = min(max_results, 3)
    search_queries = _search_queries(query)

    plan = {
        "goal": query,
        "required_outputs": _required_outputs(user_intent),
        "selection_criteria": {
            "min_papers": min_papers,
            "max_papers": max_results,
            "prefer_pdf": True,
            "prefer_recent": True,
        },
        "quality_checks": [
            "answer_matches_query",
            "has_research_sources",
            "summaries_are_grounded",
            "comparison_is_synthetic",
            "report_is_present",
        ],
    }

    return {
        **state,
        "user_intent": user_intent,
        "plan": plan,
        "search_queries": search_queries,
        "attempted_queries": state.get("attempted_queries", []),
        "search_iterations": state.get("search_iterations", 0),
        "max_search_iterations": min(len(search_queries), 3),
        "reflection_iterations": state.get("reflection_iterations", 0),
        "max_reflection_iterations": 2,
        "papers": state.get("papers", []),
        "errors": state.get("errors", []),
    }


def _required_outputs(user_intent: str) -> list[str]:
    if user_intent == "summarize":
        return ["summary"]
    if user_intent == "compare":
        return ["summary", "comparison"]
    if user_intent == "question_answering":
        return ["summary", "report"]
    return ["summary", "comparison", "report"]


def _search_queries(query: str) -> list[str]:
    variants = [
        query,
        f"{query} survey",
        f"{query} benchmark evaluation",
    ]
    deduped: list[str] = []
    for variant in variants:
        normalized = " ".join(variant.split())
        if normalized and normalized.lower() not in {item.lower() for item in deduped}:
            deduped.append(normalized)
    return deduped
