from app.agent.state import ResearchState
from app.services.search_service import SearchService


async def search_node(state: ResearchState) -> ResearchState:
    search_queries = state.get("search_queries") or [state["query"]]
    attempted_queries = state.get("attempted_queries", [])
    search_iterations = state.get("search_iterations", 0)
    query = _next_query(search_queries, attempted_queries) or state["query"]

    existing_papers = state.get("papers", [])
    errors = state.get("errors", [])

    try:
        max_results = max(state.get("max_results", 5) - len(existing_papers), 1)
        new_papers = await SearchService().search(query, max_results)
    except Exception as exc:
        new_papers = []
        errors = [
            *errors,
            {
                "stage": "search",
                "query": query,
                "error": str(exc),
            },
        ]

    papers = _dedupe_papers([*existing_papers, *new_papers])
    return {
        **state,
        "papers": papers[: state.get("max_results", len(papers))],
        "attempted_queries": [*attempted_queries, query],
        "search_iterations": search_iterations + 1,
        "errors": errors,
    }


def _next_query(search_queries: list[str], attempted_queries: list[str]) -> str | None:
    attempted = {query.lower() for query in attempted_queries}
    for query in search_queries:
        if query.lower() not in attempted:
            return query
    return None


def _dedupe_papers(papers):
    deduped = {}
    for paper in papers:
        deduped.setdefault(paper.paper_id, paper)
    return list(deduped.values())
