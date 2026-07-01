import pytest

from app.models.chat import ChatHistoryMessage
from app.services.agentic_chat_workflow import (
    UNKNOWN_ANSWER,
    AgenticChatWorkflow,
    ChatWorkflowRequest,
)
from app.services.chat_service import ChatService
from app.services.web_search_service import WebSearchResult


class FakeRAGService:
    def __init__(self, chunks: list[dict]) -> None:
        self.chunks = chunks
        self.calls = []

    async def retrieve_context(
        self,
        question: str,
        paper_ids: list[str] | None = None,
        top_k: int = 5,
        score_threshold: float | None = 0.65,
        chat_history: list[ChatHistoryMessage] | None = None,
    ) -> list[dict]:
        self.calls.append(
            {
                "question": question,
                "paper_ids": paper_ids,
                "top_k": top_k,
                "score_threshold": score_threshold,
                "chat_history": chat_history,
            }
        )
        return self.chunks


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


class FakeWebSearchService:
    def __init__(self, sources: list[dict] | None = None) -> None:
        self.sources = sources or []
        self.calls = []

    async def search_papers(self, query: str, max_results: int = 5) -> WebSearchResult:
        self.calls.append({"query": query, "max_results": max_results})
        return WebSearchResult(sources=self.sources)


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
    rag = FakeRAGService([])
    llm = FakeLLMService("This should not be called.")
    web = FakeWebSearchService()
    workflow = AgenticChatWorkflow(rag, llm, web)

    result = await workflow.run(
        ChatWorkflowRequest("What is the method?", top_k=3, score_threshold=0.7)
    )

    assert result.answer == UNKNOWN_ANSWER
    assert result.citations == []
    assert llm.prompts == []
    assert web.calls == [{"query": "What is the method?", "max_results": 3}]
    assert [event["stage"] for event in result.trace] == [
        "local_retrieve",
        "quality_gate",
        "web_search",
        "answer",
    ]
    assert result.trace[1]["sufficient"] is False
    assert rag.calls == [
        {
            "question": "What is the method?",
            "paper_ids": None,
            "top_k": 3,
            "score_threshold": 0.7,
            "chat_history": None,
        }
    ]


@pytest.mark.asyncio
async def test_chat_service_falls_back_to_web_when_local_context_is_missing() -> None:
    rag = FakeRAGService([])
    web = FakeWebSearchService(
        [
            {
                "title": "Agentic RAG vs CRAG",
                "url": "https://example.com/agentic-rag-crag",
                "content": "Agentic RAG uses agent planning; CRAG corrects retrieved context.",
                "score": 0.82,
            }
        ]
    )
    llm = FakeLLMService("Agentic RAG plans retrieval; CRAG corrects retrieval [web:1].")
    workflow = AgenticChatWorkflow(rag, llm, web)

    result = await workflow.run(
        ChatWorkflowRequest("How does Agentic RAG differ from CRAG?", top_k=4)
    )

    assert web.calls == [{"query": "How does Agentic RAG differ from CRAG?", "max_results": 4}]
    assert result.answer == "Agentic RAG plans retrieval; CRAG corrects retrieval [web:1]."
    assert result.citations[0].chunk_id == "web:1"
    assert result.citations[0].retrieval_sources == ["web"]
    assert result.citations[0].evidence_quality == "web"
    assert result.trace[2] == {
        "stage": "web_search",
        "chunk_count": 1,
        "trigger": "insufficient_local_context",
    }
    assert "Agentic RAG uses agent planning" in llm.prompts[0]


@pytest.mark.asyncio
async def test_chat_service_answers_with_citations_from_context() -> None:
    rag = FakeRAGService(
        [
            RETRIEVED_CHUNK,
            {
                **RETRIEVED_CHUNK,
                "id": "paper-1:p4:c0",
                "text": "Agentic RAG reflects on whether retrieved evidence is sufficient.",
                "metadata": {
                    **RETRIEVED_CHUNK["metadata"],
                    "page_number": "4",
                    "chunk_id": "paper-1:p4:c0",
                },
                "citation": {
                    **RETRIEVED_CHUNK["citation"],
                    "page_number": 4,
                    "chunk_id": "paper-1:p4:c0",
                    "text": "Agentic RAG reflects on whether retrieved evidence is sufficient.",
                },
            },
        ]
    )
    llm = FakeLLMService("It uses planning for retrieval decisions (p. 3).")
    web = FakeWebSearchService()
    workflow = AgenticChatWorkflow(rag, llm, web)

    result = await workflow.run(ChatWorkflowRequest("How does planning retrieve evidence?"))

    assert result.answer == "It uses planning for retrieval decisions (p. 3). [paper-1:p3:c0]"
    assert web.calls == []
    assert result.trace[1]["sufficient"] is True
    citations = result.citations
    assert citations[0].paper_id == "paper-1"
    assert citations[0].page_number == 3
    assert citations[0].chunk_id == "paper-1:p3:c0"
    assert citations[0].evidence_quality == "high"
    assert citations[0].retrieval_sources == ["keyword", "vector"]
    assert citations[0].cross_encoder_score == 2.6
    assert citations[0].reranker == "fake-cross-encoder"
    assert citations[0].matched_terms == ["planning", "retrieve", "evidence"]
    assert "If the context does not contain enough information" in llm.prompts[0]
    assert "I don't know" in llm.prompts[0]
    assert "Every factual claim supported by retrieved context" in llm.prompts[0]
    assert "[paper-1:p3:c0]" in llm.prompts[0]


@pytest.mark.asyncio
async def test_chat_service_passes_recent_history_to_rag_and_prompt() -> None:
    rag = FakeRAGService([RETRIEVED_CHUNK])
    llm = FakeLLMService("It uses planning for retrieval decisions [paper-1:p3:c0].")
    workflow = AgenticChatWorkflow(rag, llm, FakeWebSearchService())
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

    await workflow.run(
        ChatWorkflowRequest("What are its retrieval decisions?", chat_history=history)
    )

    assert rag.calls[0]["chat_history"] == history
    assert "Recent conversation:" in llm.prompts[0]
    assert "Use the recent conversation only to resolve" in llm.prompts[0]


@pytest.mark.asyncio
async def test_chat_service_streams_answer_tokens_with_citations() -> None:
    rag = FakeRAGService([RETRIEVED_CHUNK])
    llm = FakeLLMService("It uses planning")
    workflow = AgenticChatWorkflow(rag, llm, FakeWebSearchService())

    service = ChatService(workflow)
    token_stream, citations = await service.stream_answer("What is the method?")
    tokens = [token async for token in token_stream]

    assert tokens == ["It ", "uses ", "planning "]
    assert citations[0].paper_id == "paper-1"
    assert "Retrieved context" in llm.prompts[0]


@pytest.mark.asyncio
async def test_chat_service_removes_invalid_citations_from_answer() -> None:
    rag = FakeRAGService([RETRIEVED_CHUNK])
    llm = FakeLLMService("It uses planning [made-up:p1:c0].")
    workflow = AgenticChatWorkflow(rag, llm, FakeWebSearchService())

    result = await workflow.run(ChatWorkflowRequest("How does planning retrieve evidence?"))

    assert result.answer == "It uses planning. [paper-1:p3:c0]"
    assert [citation.chunk_id for citation in result.citations] == ["paper-1:p3:c0"]
