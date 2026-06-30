import pytest

from app.models.chat import ChatHistoryMessage
from app.services.chat_service import UNKNOWN_ANSWER, ChatService


class FakeRetrieverService:
    def __init__(self, chunks: list[dict]) -> None:
        self.chunks = chunks
        self.query = None
        self.top_k = None
        self.score_threshold = None
        self.paper_ids = None
        self.calls = []
        self.paper_ids = None
        self.calls = []

    async def retrieve(
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
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "score_threshold": score_threshold,
                "paper_ids": paper_ids,
            }
        )
        return self.chunks


class ThresholdSensitiveRetrieverService:
    def __init__(self) -> None:
        self.calls = []

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float | None = None,
        paper_ids: list[str] | None = None,
    ) -> list[dict]:
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "score_threshold": score_threshold,
                "paper_ids": paper_ids,
            }
        )
        return [] if score_threshold else [RETRIEVED_CHUNK]


class FakeLLMService:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.prompts: list[str] = []

    async def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answer

    async def stream_complete(self, prompt: str):
        self.prompts.append(prompt)
        for token in self.answer.split(" "):
            yield f"{token} "


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
    "vector_score": 0.89,
    "keyword_score": 1.0,
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
async def test_chat_service_returns_i_do_not_know_when_context_is_missing() -> None:
    retriever = FakeRetrieverService([])
    llm = FakeLLMService("This should not be called.")
    service = ChatService(retriever, llm)

    answer, citations = await service.answer("What is the method?", top_k=3, score_threshold=0.7)

    assert answer == UNKNOWN_ANSWER
    assert citations == []
    assert llm.prompts == []
    assert retriever.top_k == 3
    assert retriever.calls == [
        {
            "query": "What is the method?",
            "top_k": 3,
            "score_threshold": 0.7,
            "paper_ids": None,
        },
        {
            "query": "What is the method?",
            "top_k": 3,
            "score_threshold": None,
            "paper_ids": None,
        },
    ]
    assert retriever.paper_ids is None


@pytest.mark.asyncio
async def test_chat_service_answers_with_citations_from_context() -> None:
    retriever = FakeRetrieverService([RETRIEVED_CHUNK])
    llm = FakeLLMService("It uses planning for retrieval decisions (p. 3).")
    service = ChatService(retriever, llm)

    answer, citations = await service.answer("How does planning retrieve evidence?")

    assert answer == "It uses planning for retrieval decisions (p. 3)."
    assert citations[0].paper_id == "paper-1"
    assert citations[0].page_number == 3
    assert citations[0].chunk_id == "paper-1:p3:c0"
    assert citations[0].evidence_quality == "high"
    assert citations[0].retrieval_sources == ["keyword", "vector"]
    assert citations[0].matched_terms == ["planning", "retrieve", "evidence"]
    assert "If the context does not contain enough information" in llm.prompts[0]
    assert "I don't know" in llm.prompts[0]
    assert "Every factual claim supported by paper context" in llm.prompts[0]
    assert "[paper-1:p3:c0]" in llm.prompts[0]


@pytest.mark.asyncio
async def test_chat_service_uses_recent_history_for_follow_up_retrieval_and_prompt() -> None:
    retriever = FakeRetrieverService([RETRIEVED_CHUNK])
    llm = FakeLLMService("It uses planning for retrieval decisions [paper-1:p3:c0].")
    service = ChatService(retriever, llm)
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

    await service.answer("What are its retrieval decisions?", chat_history=history)

    assert "Recent conversation for resolving follow-up references" in retriever.query
    assert "How does Agentic RAG retrieve evidence?" in retriever.query
    assert "Current question: What are its retrieval decisions?" in retriever.query
    assert "Recent conversation:" in llm.prompts[0]
    assert "Use the recent conversation only to resolve" in llm.prompts[0]


@pytest.mark.asyncio
async def test_chat_service_streams_answer_tokens_with_citations() -> None:
    retriever = FakeRetrieverService([RETRIEVED_CHUNK])
    llm = FakeLLMService("It uses planning")
    service = ChatService(retriever, llm)

    token_stream, citations = await service.stream_answer("What is the method?")
    tokens = [token async for token in token_stream]

    assert tokens == ["It ", "uses ", "planning "]
    assert citations[0].paper_id == "paper-1"
    assert "Retrieved context" in llm.prompts[0]


@pytest.mark.asyncio
async def test_chat_service_retries_without_threshold_when_selected_paper_has_context() -> None:
    retriever = ThresholdSensitiveRetrieverService()
    llm = FakeLLMService("It uses planning for retrieval decisions (p. 3).")
    service = ChatService(retriever, llm)

    answer, citations = await service.answer(
        "What is the method?",
        paper_ids=["paper-1"],
        score_threshold=0.8,
    )

    assert answer == "It uses planning for retrieval decisions (p. 3)."
    assert citations[0].page_number == 3
    assert retriever.calls == [
        {
            "query": "What is the method?",
            "top_k": 5,
            "score_threshold": 0.8,
            "paper_ids": ["paper-1"],
        },
        {
            "query": "What is the method?",
            "top_k": 5,
            "score_threshold": None,
            "paper_ids": ["paper-1"],
        },
    ]


@pytest.mark.asyncio
async def test_chat_service_filters_context_by_paper_ids() -> None:
    retriever = FakeRetrieverService([RETRIEVED_CHUNK])
    llm = FakeLLMService("This should not be called.")
    service = ChatService(retriever, llm)

    answer, citations = await service.answer("What is the method?", paper_ids=["paper-2"])

    assert answer == UNKNOWN_ANSWER
    assert citations == []
    assert llm.prompts == []
