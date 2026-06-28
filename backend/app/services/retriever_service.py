from app.vectorstore.chroma import ChromaVectorStore


class RetrieverService:
    def __init__(self, vector_store: ChromaVectorStore | None = None) -> None:
        self._vector_store = vector_store or ChromaVectorStore()

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float | None = None,
    ) -> list[dict]:
        return await self._vector_store.similarity_search(query, top_k, score_threshold)
