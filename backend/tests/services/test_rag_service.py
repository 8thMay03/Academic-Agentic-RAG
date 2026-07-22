import pytest

from app.models.chat import ChatHistoryMessage
from app.services.embedding_service import EmbeddingUsage
from app.services.rag_service import RAGService


class FakeRetrieverService:
    def __init__(self, chunks: list[dict]) -> None:
        self.chunks = chunks
        self.query = None
        self.top_k = None
        self.score_threshold = None
        self.paper_ids = None
        self.calls = []
        self.last_embedding_usage = None

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float | None = None,
        paper_ids: list[str] | None = None,
        chat_id: str | None = None,
    ) -> list[dict]:
        self.query = query
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.paper_ids = paper_ids
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "score_threshold": score_threshold,
                "paper_ids": paper_ids,
                "chat_id": chat_id,
            }
        )
        self.last_embedding_usage = EmbeddingUsage(
            model="text-embedding-test",
            input_count=1,
            total_tokens=5,
            estimated_cost_usd=0.0001,
        )
        return self.chunks


class QueryAwareRetrieverService:
    def __init__(self, chunks_by_query: dict[str, list[dict]]) -> None:
        self.chunks_by_query = chunks_by_query
        self.calls = []

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float | None = None,
        paper_ids: list[str] | None = None,
        chat_id: str | None = None,
    ) -> list[dict]:
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "score_threshold": score_threshold,
                "paper_ids": paper_ids,
                "chat_id": chat_id,
            }
        )
        return self.chunks_by_query.get(query, [])


class ThresholdSensitiveRetrieverService:
    def __init__(self) -> None:
        self.calls = []

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float | None = None,
        paper_ids: list[str] | None = None,
        chat_id: str | None = None,
    ) -> list[dict]:
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "score_threshold": score_threshold,
                "paper_ids": paper_ids,
                "chat_id": chat_id,
            }
        )
        return [] if score_threshold else [RETRIEVED_CHUNK]


class FakeLLMService:
    def __init__(self, responses: str | list[str]) -> None:
        self.responses = [responses] if isinstance(responses, str) else list(responses)
        self.prompts: list[str] = []

    async def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if len(self.responses) > 1:
            return self.responses.pop(0)
        return self.responses[0]


RETRIEVED_CHUNK = {
    "id": "paper-1:p3:c0",
    "text": "Agentic RAG uses planning to decide when to retrieve evidence.",
    "metadata": {
        "paper_id": "paper-1",
        "title": "Agentic RAG",
        "page_number": "3",
        "chunk_id": "paper-1:p3:c0",
    },
    "score": 0.91,
    "rerank_score": 0.93,
    "cross_encoder_score": 2.6,
    "vector_score": 0.89,
    "keyword_score": 1.0,
    "reranker": "fake-cross-encoder",
    "retrieval_sources": ["keyword", "vector"],
    "citation": {
        "paper_id": "paper-1",
        "title": "Agentic RAG",
        "page_number": 3,
        "chunk_id": "paper-1:p3:c0",
        "text": "Agentic RAG uses planning to decide when to retrieve evidence.",
    },
}


@pytest.mark.asyncio
async def test_rag_service_retries_without_threshold_when_context_is_missing() -> None:
    retriever = FakeRetrieverService([])
    llm = FakeLLMService("not json")
    service = RAGService(retriever, llm)

    chunks = await service.retrieve_context("What is the method?", top_k=3, score_threshold=0.7)

    assert chunks == []
    assert len(llm.prompts) == 1
    assert "Generate up to 2 alternate retrieval queries" in llm.prompts[0]
    assert retriever.calls == [
        {
            "query": "What is the method?",
            "top_k": 3,
            "score_threshold": 0.7,
            "paper_ids": None,
            "chat_id": None,
        },
        {
            "query": "What is the method?",
            "top_k": 3,
            "score_threshold": None,
            "paper_ids": None,
            "chat_id": None,
        },
    ]
    assert service.last_embedding_usage is not None
    assert service.last_embedding_usage.input_count == 2
    assert service.last_embedding_usage.total_tokens == 10
    assert service.last_embedding_usage.estimated_cost_usd == pytest.approx(0.0002)


@pytest.mark.asyncio
async def test_rag_service_rewrites_follow_up_and_retrieves_multi_query() -> None:
    retriever = FakeRetrieverService([RETRIEVED_CHUNK])
    llm = FakeLLMService(
        [
            "How does Agentic RAG make retrieval decisions?",
            '["Agentic RAG planning retrieval evidence","retrieval decision mechanism"]',
        ]
    )
    service = RAGService(retriever, llm)
    history = [
        ChatHistoryMessage(
            role="user",
            content="How does Agentic RAG retrieve evidence?",
            created_at="2026-01-01T00:00:00+00:00",
        ),
        ChatHistoryMessage(
            role="assistant",
            content="It uses planning.",
            created_at="2026-01-01T00:00:01+00:00",
        ),
    ]

    await service.retrieve_context("What are its retrieval decisions?", chat_history=history)

    assert [call["query"] for call in retriever.calls] == [
        "How does Agentic RAG make retrieval decisions?",
        "Agentic RAG planning retrieval evidence",
        "retrieval decision mechanism",
    ]
    assert "Rewrite the current question as a standalone retrieval query" in llm.prompts[0]
    assert "Previous user questions:" in llm.prompts[0]
    assert "Generate up to 2 alternate retrieval queries" in llm.prompts[1]


@pytest.mark.asyncio
async def test_rag_service_filters_context_by_paper_ids() -> None:
    retriever = FakeRetrieverService([RETRIEVED_CHUNK])
    llm = FakeLLMService("[]")
    service = RAGService(retriever, llm)

    chunks = await service.retrieve_context("What is the method?", paper_ids=["paper-2"])

    assert chunks == []
    assert [call["paper_ids"] for call in retriever.calls] == [["paper-2"], ["paper-2"]]


@pytest.mark.asyncio
async def test_rag_service_retrieves_with_multi_query_and_deduplicates_chunks() -> None:
    lower_scored_chunk = {
        **RETRIEVED_CHUNK,
        "score": 0.4,
        "rerank_score": 0.5,
    }
    comparison_chunk = {
        **RETRIEVED_CHUNK,
        "id": "paper-1:p4:c0",
        "text": "The experiments compare retrieval planning against a non-planning baseline.",
        "metadata": {
            "paper_id": "paper-1",
            "title": "Agentic RAG",
            "page_number": "4",
            "chunk_id": "paper-1:p4:c0",
        },
        "score": 0.88,
        "rerank_score": 0.91,
        "citation": {
            "paper_id": "paper-1",
            "title": "Agentic RAG",
            "page_number": 4,
            "chunk_id": "paper-1:p4:c0",
            "text": "The experiments compare retrieval planning against a non-planning baseline.",
        },
    }
    retriever = QueryAwareRetrieverService(
        {
            "How does the method compare with baselines?": [lower_scored_chunk],
            "baseline comparison retrieval planning": [comparison_chunk],
            "non-planning baseline experiments": [RETRIEVED_CHUNK, comparison_chunk],
        }
    )
    llm = FakeLLMService('["baseline comparison retrieval planning","non-planning baseline experiments"]')
    service = RAGService(retriever, llm)

    chunks = await service.retrieve_context("How does the method compare with baselines?", top_k=2)

    assert [call["query"] for call in retriever.calls] == [
        "How does the method compare with baselines?",
        "baseline comparison retrieval planning",
        "non-planning baseline experiments",
    ]
    assert [chunk["id"] for chunk in chunks] == ["paper-1:p3:c0", "paper-1:p4:c0"]


@pytest.mark.asyncio
async def test_rag_service_filters_multi_query_results_by_original_question_anchor() -> None:
    cnn_chunk = {
        **RETRIEVED_CHUNK,
        "id": "cnn:p1:c0",
        "text": "CNN uses convolutional filters for local feature extraction.",
        "metadata": {
            "paper_id": "cnn-paper",
            "title": "CNN overview",
            "chunk_id": "cnn:p1:c0",
        },
        "rerank_score": 0.7,
        "citation": {
            "paper_id": "cnn-paper",
            "title": "CNN overview",
            "chunk_id": "cnn:p1:c0",
            "text": "CNN uses convolutional filters for local feature extraction.",
        },
    }
    lstm_chunk = {
        **RETRIEVED_CHUNK,
        "id": "web-ingest:lstm",
        "text": "LSTM is a recurrent neural network for sequence data.",
        "metadata": {
            "paper_id": "https://example.com/lstm",
            "title": "LSTM là gì?",
            "chunk_id": "web-ingest:lstm",
            "source": "web",
        },
        "rerank_score": 0.99,
        "citation": {
            "paper_id": "https://example.com/lstm",
            "title": "LSTM là gì?",
            "chunk_id": "web-ingest:lstm",
            "text": "LSTM is a recurrent neural network for sequence data.",
        },
    }
    retriever = QueryAwareRetrieverService(
        {
            "CNN là gì": [cnn_chunk],
            "neural network sequence model": [lstm_chunk],
        }
    )
    llm = FakeLLMService('["neural network sequence model"]')
    service = RAGService(retriever, llm)

    chunks = await service.retrieve_context("CNN là gì", top_k=3)

    assert [chunk["id"] for chunk in chunks] == ["cnn:p1:c0"]
    assert chunks[0]["matched_anchor_terms"] == ["cnn"]
