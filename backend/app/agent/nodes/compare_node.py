from app.agent.state import ResearchState
from app.services.compare_service import CompareService
from app.services.llm_service import LLMService


async def compare_node(state: ResearchState) -> ResearchState:
    papers = state.get("selected_papers") or state.get("papers", [])
    errors = state.get("errors", [])

    if not papers:
        return {**state, "comparison": "", "errors": errors}

    try:
        comparison = await CompareService(LLMService()).compare(papers)
    except Exception as exc:
        comparison = _fallback_comparison(state)
        errors.append(
            {
                "stage": "compare",
                "error": str(exc),
            }
        )

    return {**state, "comparison": comparison, "errors": errors}


def _fallback_comparison(state: ResearchState) -> str:
    summaries_by_id = {
        summary["paper_id"]: summary["content"] for summary in state.get("summaries", [])
    }
    rows = [
        "| Paper | Year | Available evidence |",
        "| --- | --- | --- |",
    ]
    for paper in state.get("selected_papers") or state.get("papers", []):
        evidence = summaries_by_id.get(paper.paper_id) or paper.abstract or "Not summarized."
        evidence = " ".join(evidence.split())[:240]
        rows.append(f"| {paper.title} | {paper.published or 'n/a'} | {evidence} |")
    return "\n".join(rows)
