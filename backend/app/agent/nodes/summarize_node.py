from app.agent.state import ResearchState
from app.services.llm_service import LLMService
from app.services.summary_service import SummaryService


async def summarize_node(state: ResearchState) -> ResearchState:
    parsed_documents = state.get("parsed_documents", [])
    existing_summaries = state.get("summaries", [])
    summarized_ids = {summary["paper_id"] for summary in existing_summaries}
    summaries = [*existing_summaries]
    errors = state.get("errors", [])
    summary_service = SummaryService(LLMService())

    for document in parsed_documents:
        paper_id = document["paper_id"]
        if paper_id in summarized_ids:
            continue

        try:
            content = await summary_service.summarize(document["title"], document["text"])
            source = "llm"
        except Exception as exc:
            content = _fallback_summary(document["title"], document["text"])
            source = "fallback"
            errors.append(
                {
                    "stage": "summarize",
                    "paper_id": paper_id,
                    "error": str(exc),
                }
            )

        summaries.append(
            {
                "paper_id": paper_id,
                "title": document["title"],
                "content": content,
                "source": source,
            }
        )
        summarized_ids.add(paper_id)

    return {
        **state,
        "summaries": summaries,
        "summary": _combine_summaries(summaries),
        "errors": errors,
    }


def _fallback_summary(title: str, text: str, max_chars: int = 1200) -> str:
    excerpt = " ".join(text.split())[:max_chars]
    if len(text) > max_chars:
        excerpt = f"{excerpt.rstrip()}..."
    return (
        f"## {title}\n\n"
        "- LLM summary was unavailable; using extracted paper text instead.\n"
        f"- Evidence excerpt: {excerpt or 'Not specified.'}"
    )


def _combine_summaries(summaries: list[dict]) -> str:
    return "\n\n".join(
        f"### {summary['title']}\n\n{summary['content']}" for summary in summaries
    )
