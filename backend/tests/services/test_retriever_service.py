import pytest

from app.services.retriever_service import RetrieverService


class FakeVectorStore:
    def __init__(self) -> None:
        self.query = None
        self.top_k = None
        self.score_threshold = None
        self.paper_ids = None

    async def similarity_search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float | None = None,
        paper_ids: list[str] | None = None,
    ) -> list[dict]:
        self.query = query
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.paper_ids = paper_ids
        return [{"id": "chunk-1", "citation": {"page_number": 3}}]


@pytest.mark.asyncio
async def test_retriever_service_passes_top_k_and_score_threshold() -> None:
    vector_store = FakeVectorStore()
    service = RetrieverService(vector_store=vector_store)

    results = await service.retrieve(
        "agentic rag",
        top_k=3,
        score_threshold=0.75,
        paper_ids=["paper-1"],
    )

    assert vector_store.query == "agentic rag"
    assert vector_store.top_k == 3
    assert vector_store.score_threshold == 0.75
    assert vector_store.paper_ids == ["paper-1"]
    assert results == [{"id": "chunk-1", "citation": {"page_number": 3}}]
