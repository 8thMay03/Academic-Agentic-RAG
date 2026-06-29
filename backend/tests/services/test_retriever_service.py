import pytest

from app.services.retriever_service import RetrieverService


class FakeVectorStore:
    def __init__(self) -> None:
        self.vector_call = None
        self.keyword_call = None

    async def similarity_search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float | None = None,
        paper_ids: list[str] | None = None,
    ) -> list[dict]:
        self.vector_call = {
            "query": query,
            "top_k": top_k,
            "score_threshold": score_threshold,
            "paper_ids": paper_ids,
        }
        return [
            {
                "id": "chunk-vector",
                "text": "Planning agents decide when to retrieve evidence.",
                "metadata": {"chunk_id": "chunk-vector", "paper_id": "paper-1"},
                "score": 0.72,
                "citation": {"page_number": 3},
            },
            {
                "id": "chunk-shared",
                "text": "Agentic RAG retrieval planning.",
                "metadata": {"chunk_id": "chunk-shared", "paper_id": "paper-1"},
                "score": 0.8,
                "citation": {"page_number": 4},
            },
        ]

    async def keyword_search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float | None = None,
        paper_ids: list[str] | None = None,
    ) -> list[dict]:
        self.keyword_call = {
            "query": query,
            "top_k": top_k,
            "score_threshold": score_threshold,
            "paper_ids": paper_ids,
        }
        return [
            {
                "id": "chunk-keyword",
                "text": "Exact agentic rag planning keyword match.",
                "metadata": {"chunk_id": "chunk-keyword", "paper_id": "paper-1"},
                "score": 0.95,
                "citation": {"page_number": 5},
            },
            {
                "id": "chunk-shared",
                "text": "Agentic RAG retrieval planning.",
                "metadata": {"chunk_id": "chunk-shared", "paper_id": "paper-1"},
                "score": 1.0,
                "citation": {"page_number": 4},
            },
        ]


class FakeRerankerService:
    def __init__(self) -> None:
        self.query = None
        self.chunks = None

    def rerank(self, query: str, chunks: list[dict]) -> list[dict]:
        self.query = query
        self.chunks = chunks
        return sorted(chunks, key=lambda chunk: chunk["id"])


@pytest.mark.asyncio
async def test_retriever_service_merges_vector_and_keyword_results_before_reranking() -> None:
    vector_store = FakeVectorStore()
    reranker = FakeRerankerService()
    service = RetrieverService(
        vector_store=vector_store,
        reranker_service=reranker,
        vector_weight=0.6,
        keyword_weight=0.4,
        candidate_multiplier=3,
    )

    results = await service.retrieve(
        "agentic rag",
        top_k=3,
        score_threshold=0.75,
        paper_ids=["paper-1"],
    )

    assert vector_store.vector_call == {
        "query": "agentic rag",
        "top_k": 9,
        "score_threshold": None,
        "paper_ids": ["paper-1"],
    }
    assert vector_store.keyword_call == {
        "query": "agentic rag",
        "top_k": 9,
        "score_threshold": None,
        "paper_ids": ["paper-1"],
    }
    assert reranker.query == "agentic rag"
    assert {chunk["id"] for chunk in reranker.chunks} == {"chunk-keyword", "chunk-shared"}
    assert results == sorted(results, key=lambda chunk: chunk["id"])
    assert results[0]["id"] == "chunk-keyword"
    assert results[0]["retrieval_sources"] == ["keyword"]
    assert results[1]["id"] == "chunk-shared"
    assert results[1]["retrieval_sources"] == ["keyword", "vector"]
    assert results[1]["score"] == pytest.approx(0.88)
