from app.services.embedding_service import EmbeddingService


async def embedding_tool(texts: list[str]) -> list[list[float]]:
    return await EmbeddingService().embed_texts(texts)

