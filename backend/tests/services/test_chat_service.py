import pytest

from app.agent.models import ToolResult
from app.agent.tools.registry import ToolRegistry
from app.models.chat import ChatHistoryMessage
from app.agent.workflow import (
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
    def __init__(self, answer: str | list[str]) -> None:
        self.answers = [answer] if isinstance(answer, str) else list(answer)
        self.prompts: list[str] = []

    async def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if len(self.answers) > 1:
            return self.answers.pop(0)
        return self.answers[0]

    async def stream_complete(self, prompt: str):
        self.prompts.append(prompt)
        for token in self.answers[0].split(" "):
            yield f"{token} "


class FakeWebSearchService:
    def __init__(self, sources: list[dict] | None = None) -> None:
        self.sources = sources or []
        self.calls = []

    async def search(self, query: str, max_results: int = 5) -> WebSearchResult:
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

SECOND_RETRIEVED_CHUNK = {
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
}

SUFFICIENT_LOCAL_CHUNKS = [RETRIEVED_CHUNK, SECOND_RETRIEVED_CHUNK]

UNUSABLE_CITATION_CHUNKS = [
    {
        "text": "Agentic RAG planning retrieves evidence before answering.",
        "score": 0.91,
        "rerank_score": 0.91,
    },
    {
        "text": "Agentic RAG planning retrieves evidence and verifies the answer.",
        "score": 0.9,
        "rerank_score": 0.9,
    },
]


class SequenceLocalRetrieveTool:
    name = "local_retrieve"

    def __init__(self, chunks_by_call: list[list[dict]]) -> None:
        self.chunks_by_call = chunks_by_call
        self.calls = []

    async def run(self, input: dict) -> ToolResult:
        self.calls.append(input)
        call_index = min(len(self.calls) - 1, len(self.chunks_by_call) - 1)
        return ToolResult(
            tool_name=self.name,
            success=True,
            chunks=self.chunks_by_call[call_index],
        )


class StaticTool:
    def __init__(self, name: str, result: ToolResult) -> None:
        self.name = name
        self._result = result
        self.calls = []

    async def run(self, input: dict) -> ToolResult:
        self.calls.append(input)
        return self._result


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
        "classify_intent",
        "local_retrieve",
        "quality_gate",
        "plan",
        "execute_tool",
        "observe",
        "execute_tool",
        "observe",
        "draft_answer",
    ]
    assert result.trace[0]["intent"] == "research_qa"
    assert result.trace[2]["sufficient"] is False
    assert result.trace[3]["step_count"] == 2
    assert result.trace[4]["tool_name"] == "web_search"
    assert result.trace[6]["tool_name"] == "web_snippet_ingest"
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
    assert result.citations[0].url == "https://example.com/agentic-rag-crag"
    assert result.citations[0].retrieval_sources == ["web"]
    assert result.citations[0].evidence_quality == "web"
    assert result.trace[2]["reason"] == "no_local_context"
    assert result.trace[4]["tool_name"] == "web_search"
    assert result.trace[5]["chunk_count"] == 1
    assert "Agentic RAG uses agent planning" in llm.prompts[0]


@pytest.mark.asyncio
async def test_chat_workflow_ingests_fresh_research_for_latest_questions() -> None:
    fresh_chunk = {
        **RETRIEVED_CHUNK,
        "id": "fresh-paper:p2:c0",
        "text": "A 2026 survey discusses the latest Agentic RAG systems.",
        "metadata": {
            **RETRIEVED_CHUNK["metadata"],
            "paper_id": "fresh-paper",
            "title": "Latest Agentic RAG survey",
            "chunk_id": "fresh-paper:p2:c0",
            "source_type": "arxiv",
            "source_url": "https://arxiv.org/abs/2601.12345",
            "pdf_url": "https://arxiv.org/pdf/2601.12345",
            "trust_level": "high",
            "ingestion_status": "downloaded",
        },
        "citation": {
            **RETRIEVED_CHUNK["citation"],
            "paper_id": "fresh-paper",
            "title": "Latest Agentic RAG survey",
            "chunk_id": "fresh-paper:p2:c0",
            "text": "A 2026 survey discusses the latest Agentic RAG systems.",
        },
    }
    local_tool = SequenceLocalRetrieveTool(
        [
            [RETRIEVED_CHUNK, {**RETRIEVED_CHUNK, "id": "paper-1:p4:c0"}],
            [fresh_chunk],
        ]
    )
    registry = ToolRegistry(
        [
            local_tool,
            StaticTool(
                "arxiv_search",
                ToolResult(
                    tool_name="arxiv_search",
                    success=True,
                    artifacts=[
                        {
                            "title": "Latest Agentic RAG survey",
                            "pdf_url": "https://arxiv.org/pdf/2601.12345",
                        }
                    ],
                    metadata={
                        "paper_count": 1,
                        "pdf_urls": ["https://arxiv.org/pdf/2601.12345"],
                    },
                ),
            ),
            StaticTool(
                "pdf_download",
                ToolResult(
                    tool_name="pdf_download",
                    success=True,
                    artifacts=[
                        {
                            "path": "data/pdfs/2601.12345.pdf",
                            "filename": "2601.12345.pdf",
                        }
                    ],
                    metadata={"path": "data/pdfs/2601.12345.pdf"},
                ),
            ),
            StaticTool(
                "pdf_index",
                ToolResult(
                    tool_name="pdf_index",
                    success=True,
                    metadata={"chunks_indexed": 5},
                ),
            ),
        ]
    )
    rag = FakeRAGService([])
    web = FakeWebSearchService()
    llm = FakeLLMService("Latest systems add reflection loops [fresh-paper:p2:c0].")
    workflow = AgenticChatWorkflow(rag, llm, web, tool_registry=registry)

    result = await workflow.run(ChatWorkflowRequest("What is the latest Agentic RAG approach?"))

    assert web.calls == []
    assert [event.get("tool_name") for event in result.trace if event["stage"] == "execute_tool"] == [
        "arxiv_search",
        "pdf_download",
        "pdf_index",
        "local_retrieve",
    ]
    assert result.trace[0]["intent"] == "fresh_research"
    assert result.trace[2]["reason"] == "latest_query_requires_web"
    assert result.trace[2]["sufficient"] is False
    assert result.citations[0].chunk_id == "fresh-paper:p2:c0"
    assert result.citations[0].url == "https://arxiv.org/abs/2601.12345"
    assert result.citations[0].source_type == "arxiv"
    assert result.citations[0].source_url == "https://arxiv.org/abs/2601.12345"
    assert result.citations[0].pdf_url == "https://arxiv.org/pdf/2601.12345"
    assert result.citations[0].trust_level == "high"
    assert result.citations[0].ingestion_status == "downloaded"


@pytest.mark.asyncio
async def test_chat_workflow_rejects_low_score_context_before_self_check() -> None:
    low_score_chunk = {
        **RETRIEVED_CHUNK,
        "score": 0.2,
        "rerank_score": 0.2,
    }
    rag = FakeRAGService([low_score_chunk, {**low_score_chunk, "id": "paper-1:p4:c0"}])
    web = FakeWebSearchService()
    llm = FakeLLMService("This should not be used.")
    workflow = AgenticChatWorkflow(rag, llm, web)

    result = await workflow.run(ChatWorkflowRequest("How does planning retrieve evidence?"))

    assert result.answer == UNKNOWN_ANSWER
    assert result.trace[2]["reason"] == "low_top_score"
    assert result.trace[2]["self_check_used"] is False
    assert web.calls == [{"query": "How does planning retrieve evidence?", "max_results": 5}]
    assert llm.prompts == []


@pytest.mark.asyncio
async def test_chat_workflow_uses_llm_self_check_for_borderline_context() -> None:
    borderline_chunk = {
        **RETRIEVED_CHUNK,
        "score": 0.62,
        "rerank_score": 0.62,
    }
    rag = FakeRAGService([borderline_chunk, {**borderline_chunk, "id": "paper-1:p4:c0"}])
    web = FakeWebSearchService()
    llm = FakeLLMService(["YES", "It uses planning [paper-1:p3:c0]."])
    workflow = AgenticChatWorkflow(rag, llm, web)

    result = await workflow.run(ChatWorkflowRequest("How does planning retrieve evidence?"))

    assert result.answer == "It uses planning [paper-1:p3:c0]."
    assert result.trace[2]["reason"] == "llm_self_check_passed"
    assert result.trace[2]["self_check_used"] is True
    assert result.trace[2]["self_check_passed"] is True
    assert web.calls == []
    assert "Return only YES or NO" in llm.prompts[0]


@pytest.mark.asyncio
async def test_chat_workflow_searches_web_when_llm_self_check_rejects_context() -> None:
    borderline_chunk = {
        **RETRIEVED_CHUNK,
        "score": 0.62,
        "rerank_score": 0.62,
    }
    rag = FakeRAGService([borderline_chunk, {**borderline_chunk, "id": "paper-1:p4:c0"}])
    web = FakeWebSearchService(
        [
            {
                "title": "Planning retrieval",
                "url": "https://example.com/planning-retrieval",
                "content": "Planning retrieval uses explicit evidence selection.",
            }
        ]
    )
    llm = FakeLLMService(["NO", "Planning uses explicit evidence selection [web:1]."])
    workflow = AgenticChatWorkflow(rag, llm, web)

    result = await workflow.run(ChatWorkflowRequest("How does planning retrieve evidence?"))

    assert result.answer == "Planning uses explicit evidence selection [web:1]."
    assert result.trace[2]["reason"] == "llm_self_check_failed"
    assert result.trace[2]["self_check_passed"] is False
    assert result.trace[3]["reason"] == "llm_self_check_failed"
    assert result.trace[4]["tool_name"] == "web_search"
    assert web.calls == [{"query": "How does planning retrieve evidence?", "max_results": 5}]


@pytest.mark.asyncio
async def test_chat_service_answers_with_citations_from_context() -> None:
    rag = FakeRAGService(SUFFICIENT_LOCAL_CHUNKS)
    llm = FakeLLMService("It uses planning for retrieval decisions (p. 3).")
    web = FakeWebSearchService()
    workflow = AgenticChatWorkflow(rag, llm, web)

    result = await workflow.run(ChatWorkflowRequest("How does planning retrieve evidence?"))

    assert result.answer == "It uses planning for retrieval decisions (p. 3). [paper-1:p3:c0]"
    assert web.calls == []
    assert result.trace[2]["sufficient"] is True
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
    rag = FakeRAGService(SUFFICIENT_LOCAL_CHUNKS)
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
        ChatWorkflowRequest("How does planning retrieve evidence?", chat_history=history)
    )

    assert rag.calls[0]["chat_history"] == history
    assert "Recent conversation:" in llm.prompts[0]
    assert "Use the recent conversation only to resolve" in llm.prompts[0]


@pytest.mark.asyncio
async def test_chat_service_streams_answer_tokens_with_citations() -> None:
    rag = FakeRAGService(SUFFICIENT_LOCAL_CHUNKS)
    llm = FakeLLMService("It uses planning")
    workflow = AgenticChatWorkflow(rag, llm, FakeWebSearchService())

    service = ChatService(workflow)
    token_stream, citations, trace = await service.stream_answer("How does planning retrieve evidence?")
    tokens = [token async for token in token_stream]

    assert tokens == ["It ", "uses ", "planning ", "[paper-1:p3:c0]"]
    assert citations[0].paper_id == "paper-1"
    assert trace[0]["stage"] == "classify_intent"
    assert trace[1]["stage"] == "local_retrieve"
    assert trace[-1]["stage"] == "verify_answer"
    assert trace[-1]["suggested_action"] == "revise_answer"
    assert trace[-1]["unsupported_claim_count"] == 0
    assert "Retrieved context" in llm.prompts[0]


@pytest.mark.asyncio
async def test_chat_service_removes_invalid_citations_from_answer() -> None:
    rag = FakeRAGService(SUFFICIENT_LOCAL_CHUNKS)
    llm = FakeLLMService("It uses planning [made-up:p1:c0].")
    workflow = AgenticChatWorkflow(rag, llm, FakeWebSearchService())

    result = await workflow.run(ChatWorkflowRequest("How does planning retrieve evidence?"))

    assert result.answer == "It uses planning. [paper-1:p3:c0]"
    assert [citation.chunk_id for citation in result.citations] == ["paper-1:p3:c0"]
    assert result.trace[-1]["suggested_action"] == "revise_answer"


@pytest.mark.asyncio
async def test_chat_workflow_retrieves_more_when_verifier_requests_more_evidence() -> None:
    rag = FakeRAGService(UNUSABLE_CITATION_CHUNKS)
    web = FakeWebSearchService(
        [
            {
                "title": "Recovered evidence",
                "url": "https://example.com/recovered",
                "content": "Recovered web evidence explains Agentic RAG verification.",
            }
        ]
    )
    llm = FakeLLMService(
        [
            "Unsupported claim [fake:c0].",
            "Recovered web evidence explains verification [web:1].",
        ]
    )
    workflow = AgenticChatWorkflow(rag, llm, web)

    result = await workflow.run(ChatWorkflowRequest("How does Agentic RAG verify evidence?"))

    assert result.answer == "Recovered web evidence explains verification [web:1]."
    assert [citation.chunk_id for citation in result.citations] == ["web:1"]
    assert web.calls == [{"query": "How does Agentic RAG verify evidence?", "max_results": 5}]
    assert [event["stage"] for event in result.trace[-5:]] == [
        "execute_tool",
        "observe",
        "draft_answer",
        "generate_answer",
        "verify_answer",
    ]
    assert result.trace[-7]["stage"] == "verify_answer"
    assert result.trace[-7]["suggested_action"] == "retrieve_more"
    assert result.trace[-6]["stage"] == "plan"
    assert result.trace[-6]["status"] == "recovery"
    assert result.trace[-5]["tool_name"] == "web_search"
    assert result.trace[-1]["suggested_action"] == "finalize"


@pytest.mark.asyncio
async def test_chat_workflow_does_not_retrieve_more_when_web_search_is_disabled() -> None:
    rag = FakeRAGService(UNUSABLE_CITATION_CHUNKS)
    web = FakeWebSearchService(
        [
            {
                "title": "Recovered evidence",
                "url": "https://example.com/recovered",
                "content": "Recovered web evidence explains Agentic RAG verification.",
            }
        ]
    )
    llm = FakeLLMService("Unsupported claim [fake:c0].")
    workflow = AgenticChatWorkflow(rag, llm, web)

    result = await workflow.run(
        ChatWorkflowRequest(
            "How does Agentic RAG verify evidence?",
            enable_web_search=False,
        )
    )

    assert result.answer == UNKNOWN_ANSWER
    assert result.citations == []
    assert web.calls == []
    assert result.trace[-1]["stage"] == "verify_answer"
    assert result.trace[-1]["suggested_action"] == "retrieve_more"
