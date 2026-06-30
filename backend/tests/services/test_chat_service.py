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
    ) -> list[dict]:
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "score_threshold": score_threshold,
                "paper_ids": paper_ids,
            }
        )
        return self.chunks_by_query.get(query, [])


class FakeLLMService:
    def __init__(self, responses: str | list[str]) -> None:
        self.responses = [responses] if isinstance(responses, str) else list(responses)
        self.prompts: list[str] = []

    async def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if len(self.responses) > 1:
            return self.responses.pop(0)
        return self.responses[0]

    async def stream_complete(self, prompt: str):
        self.prompts.append(prompt)
        if len(self.responses) > 1:
            response = self.responses.pop(0)
        else:
            response = self.responses[0]
        for token in response.split(" "):
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
async def test_chat_service_returns_i_do_not_know_when_context_is_missing() -> None:
    retriever = FakeRetrieverService([])
    llm = FakeLLMService("This should not be called.")
    service = ChatService(retriever, llm)

    answer, citations = await service.answer("What is the method?", top_k=3, score_threshold=0.7)

    assert answer == UNKNOWN_ANSWER
    assert citations == []
    assert len(llm.prompts) == 1
    assert "Create a retrieval plan" in llm.prompts[0]
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

    assert answer == "It uses planning for retrieval decisions (p. 3). [paper-1:p3:c0]"
    assert citations[0].paper_id == "paper-1"
    assert citations[0].page_number == 3
    assert citations[0].chunk_id == "paper-1:p3:c0"
    assert citations[0].evidence_quality == "high"
    assert citations[0].retrieval_sources == ["keyword", "vector"]
    assert citations[0].cross_encoder_score == 2.6
    assert citations[0].reranker == "fake-cross-encoder"
    assert citations[0].matched_terms == ["planning", "retrieve", "evidence"]
    assert "Create a retrieval plan" in llm.prompts[0]
    assert "If the context does not contain enough information" in llm.prompts[1]
    assert "I don't know" in llm.prompts[1]
    assert "Every factual claim supported by paper context" in llm.prompts[1]
    assert "[paper-1:p3:c0]" in llm.prompts[1]


@pytest.mark.asyncio
async def test_chat_service_uses_recent_history_for_follow_up_retrieval_and_prompt() -> None:
    retriever = FakeRetrieverService([RETRIEVED_CHUNK])
    llm = FakeLLMService(
        [
            (
                '{"standalone_question":"How does Agentic RAG make retrieval decisions?",'
                '"search_queries":["Agentic RAG planning retrieval evidence",'
                '"retrieval decision mechanism"]}'
            ),
            "It uses planning for retrieval decisions [paper-1:p3:c0].",
        ]
    )
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

    assert [call["query"] for call in retriever.calls] == [
        "How does Agentic RAG make retrieval decisions?",
        "Agentic RAG planning retrieval evidence",
        "retrieval decision mechanism",
    ]
    assert "Create a retrieval plan" in llm.prompts[0]
    assert "Previous user questions:" in llm.prompts[0]
    assert "Recent conversation:" in llm.prompts[1]
    assert "Use the recent conversation only to resolve" in llm.prompts[1]


@pytest.mark.asyncio
async def test_chat_service_streams_answer_tokens_with_citations() -> None:
    retriever = FakeRetrieverService([RETRIEVED_CHUNK])
    llm = FakeLLMService("It uses planning")
    service = ChatService(retriever, llm)

    token_stream, citations = await service.stream_answer("What is the method?")
    tokens = [token async for token in token_stream]

    assert tokens == ["It ", "uses ", "planning "]
    assert citations[0].paper_id == "paper-1"
    assert "Retrieved context" in llm.prompts[1]


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

    assert answer == "It uses planning for retrieval decisions (p. 3). [paper-1:p3:c0]"
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
    assert len(llm.prompts) == 1
    assert "Create a retrieval plan" in llm.prompts[0]


@pytest.mark.asyncio
async def test_chat_service_removes_invalid_citations_from_answer() -> None:
    retriever = FakeRetrieverService([RETRIEVED_CHUNK])
    llm = FakeLLMService("It uses planning [made-up:p1:c0].")
    service = ChatService(retriever, llm)

    answer, citations = await service.answer("How does planning retrieve evidence?")

    assert answer == "It uses planning. [paper-1:p3:c0]"
    assert [citation.chunk_id for citation in citations] == ["paper-1:p3:c0"]


@pytest.mark.asyncio
async def test_chat_service_retrieves_with_multi_query_and_deduplicates_chunks() -> None:
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
    llm = FakeLLMService(
        [
            (
                '{"standalone_question":"How does the method compare with baselines?",'
                '"search_queries":["baseline comparison retrieval planning",'
                '"non-planning baseline experiments"]}'
            ),
            "It compares against a non-planning baseline [paper-1:p4:c0].",
        ]
    )
    service = ChatService(retriever, llm)

    answer, citations = await service.answer("How does the method compare with baselines?", top_k=2)

    assert [call["query"] for call in retriever.calls] == [
        "How does the method compare with baselines?",
        "baseline comparison retrieval planning",
        "non-planning baseline experiments",
    ]
    assert answer == "It compares against a non-planning baseline [paper-1:p4:c0]."
    assert [citation.chunk_id for citation in citations] == ["paper-1:p4:c0"]
