from app.services.retriever_service import RetrieverService


async def retriever_tool(query: str, top_k: int = 5) -> list[dict]:
    return await RetrieverService().retrieve(query, top_k)

