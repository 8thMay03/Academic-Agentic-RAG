from app.agent.state import ResearchState


async def select_papers_node(state: ResearchState) -> ResearchState:
    papers = state.get("papers", [])
    max_papers = state.get("max_results", len(papers))

    selected_papers = sorted(
        papers,
        key=lambda paper: (
            paper.pdf_url is not None,
            paper.published or "",
        ),
        reverse=True,
    )[:max_papers]
    selected_ids = {paper.paper_id for paper in selected_papers}
    rejected_papers = [
        {
            "paper_id": paper.paper_id,
            "title": paper.title,
            "reason": "Outside selected paper limit.",
        }
        for paper in papers
        if paper.paper_id not in selected_ids
    ]

    return {
        **state,
        "selected_papers": selected_papers,
        "rejected_papers": rejected_papers,
    }
