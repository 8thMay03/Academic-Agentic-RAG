from app.services.retriever_service import RetrieverService


async def retriever_tool(
    query: str,
    top_k: int = 5,
    score_threshold: float | None = None,
    paper_ids: list[str] | None = None,
) -> list[dict]:
    return await RetrieverService().retrieve(query, top_k, score_threshold, paper_ids)
