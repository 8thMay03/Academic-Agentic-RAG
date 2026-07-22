from pathlib import Path

import pytest

from app.agent.models import ContextQuality
from app.agent.workflow import AgenticChatWorkflow, ChatWorkflowRequest
from app.models.chat import ChatHistoryMessage
from app.services.chat_service import ChatService
from app.services.pdf_index_service import PDFIndexService
from tests.services.test_chat_service import FakeLLMService, FakeRAGService, FakeWebSearchService


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures"
FIXTURE_PDF = FIXTURE_DIR / "pdfs" / "agentic_rag_fixture.pdf"


class FixtureQualityEvaluator:
    async def evaluate(self, question: str, chunks: list[dict]) -> ContextQuality:
        return ContextQuality(
            sufficient=bool(chunks),
            reason="fixture_context" if chunks else "no_fixture_context",
            chunk_count=len(chunks),
            context_chars=sum(len(chunk.get("text", "")) for chunk in chunks),
            top_score=chunks[0].get("score", 0.0) if chunks else 0.0,
            average_score=sum(chunk.get("score", 0.0) for chunk in chunks) / len(chunks) if chunks else 0.0,
            source_count=len({chunk.get("metadata", {}).get("paper_id") for chunk in chunks}) if chunks else 0,
            query_coverage=1.0 if chunks else 0.0,
        )


class EmptyFirstQualityEvaluator:
    def __init__(self) -> None:
        self.calls = 0

    async def evaluate(self, question: str, chunks: list[dict]) -> ContextQuality:
        self.calls += 1
        return ContextQuality(
            sufficient=self.calls > 1 and bool(chunks),
            reason="fixture_context" if self.calls > 1 and chunks else "no_fixture_context",
            chunk_count=len(chunks),
            context_chars=sum(len(chunk.get("text", "")) for chunk in chunks),
            top_score=chunks[0].get("score", 0.0) if chunks else 0.0,
            average_score=sum(chunk.get("score", 0.0) for chunk in chunks) / len(chunks) if chunks else 0.0,
            source_count=len({chunk.get("metadata", {}).get("paper_id") for chunk in chunks}) if chunks else 0,
            query_coverage=1.0 if chunks else 0.0,
        )


async def _index_fixture_pdf(monkeypatch, tmp_path):
    indexed_chunks = []

    def fake_extract_text_from_pdf(path):
        assert path == FIXTURE_PDF
        return (
            "Abstract\n"
            "Agentic RAG plans retrieval actions before generation. "
            "It verifies whether evidence is sufficient before answering.\f"
            "Limitations\n"
            "The fixture does not report private benchmark accuracy."
        )

    async def fake_index_chunks(chunks):
        indexed_chunks.extend(chunks)

    monkeypatch.setattr("app.services.pdf_index_service.extract_text_from_pdf", fake_extract_text_from_pdf)
    monkeypatch.setattr("app.services.pdf_index_service.index_chunks", fake_index_chunks)

    result = await PDFIndexService(data_dir=tmp_path).index_pdf(
        FIXTURE_PDF,
        source_metadata={"source_type": "local_pdf", "trust_level": "high"},
    )
    return result, indexed_chunks


def _retrieved_chunks(indexed_chunks):
    chunks = []
    for chunk in indexed_chunks:
        chunk_id = chunk.metadata["chunk_id"]
        page_number = chunk.page_number or chunk.page or 1
        chunks.append(
            {
                "id": chunk_id,
                "text": chunk.text,
                "metadata": {
                    **chunk.metadata,
                    "page_number": str(page_number),
                    "chunk_id": chunk_id,
                },
                "score": 0.92,
                "retrieval_sources": ["vector", "keyword"],
                "citation": {
                    "paper_id": chunk.paper_id,
                    "title": chunk.metadata.get("title", "agentic_rag_fixture.pdf"),
                    "page_number": page_number,
                    "chunk_id": chunk_id,
                    "text": chunk.text,
                },
            }
        )
    return chunks


@pytest.mark.asyncio
async def test_fixture_pdf_indexes_and_answers_with_grounded_citation(monkeypatch, tmp_path) -> None:
    index_result, indexed_chunks = await _index_fixture_pdf(monkeypatch, tmp_path)
    retrieved_chunks = _retrieved_chunks(indexed_chunks)
    rag = FakeRAGService(retrieved_chunks)
    llm = FakeLLMService("Agentic RAG plans retrieval actions before generation [agentic_rag_fixture:p1:c0].")
    workflow = AgenticChatWorkflow(
        rag,
        llm,
        FakeWebSearchService(),
        quality_evaluator=FixtureQualityEvaluator(),
    )

    result = await workflow.run(
        ChatWorkflowRequest(
            question="What makes Agentic RAG different from a fixed RAG pipeline?",
            paper_ids=["agentic_rag_fixture"],
            enable_web_search=False,
        )
    )

    indexed_chunk_ids = {chunk.metadata["chunk_id"] for chunk in indexed_chunks}
    assert index_result.chunks_indexed == 2
    assert indexed_chunks[0].metadata["section_type"] == "abstract"
    assert result.answer == "Agentic RAG plans retrieval actions before generation [1]."
    assert result.citations
    assert {citation.chunk_id for citation in result.citations}.issubset(indexed_chunk_ids)
    assert result.stop_reason == "answered_with_sufficient_context"
    assert [event["stage"] for event in result.trace][-3:] == ["draft_answer", "generate_answer", "verify_answer"]
    verification_step = result.trace[-1]
    assert verification_step["claim_citation_map"] == [
        {
            "claim": "Agentic RAG plans retrieval actions before generation [agentic_rag_fixture:p1:c0].",
            "status": "supported",
            "supporting_chunk_ids": ["agentic_rag_fixture:p1:c0"],
            "reason": "claim_terms_overlap_cited_evidence",
        }
    ]


@pytest.mark.asyncio
async def test_unanswerable_fixture_question_stops_without_web(monkeypatch, tmp_path) -> None:
    await _index_fixture_pdf(monkeypatch, tmp_path)
    rag = FakeRAGService([])
    workflow = AgenticChatWorkflow(
        rag,
        FakeLLMService("This should not be generated."),
        FakeWebSearchService(),
        quality_evaluator=FixtureQualityEvaluator(),
    )

    result = await workflow.run(
        ChatWorkflowRequest(
            question="What accuracy did this paper report on a private benchmark?",
            paper_ids=["agentic_rag_fixture"],
            enable_web_search=False,
        )
    )

    assert result.answer == "I don't know"
    assert result.citations == []
    assert result.stop_reason == "web_search_disabled"
    assert result.trace[-1]["stage"] == "draft_answer"


@pytest.mark.asyncio
async def test_chat_service_passes_followup_history_into_retrieval(monkeypatch, tmp_path) -> None:
    _, indexed_chunks = await _index_fixture_pdf(monkeypatch, tmp_path)
    rag = FakeRAGService(_retrieved_chunks(indexed_chunks))
    workflow = AgenticChatWorkflow(
        rag,
        FakeLLMService("It verifies evidence sufficiency before answering [agentic_rag_fixture:p1:c0]."),
        FakeWebSearchService(),
        quality_evaluator=FixtureQualityEvaluator(),
    )
    service = ChatService(workflow)
    history = [
        ChatHistoryMessage(
            role="user",
            content="What is Agentic RAG?",
            created_at="2026-01-01T00:00:00+00:00",
        )
    ]

    result = await service.answer(
        question="And what does it verify?",
        paper_ids=["agentic_rag_fixture"],
        chat_history=history,
        enable_web_search=False,
    )

    assert rag.calls[0]["chat_history"] == history
    assert result.stop_reason == "answered_with_sufficient_context"
    assert result.citations[0].chunk_id in {chunk.metadata["chunk_id"] for chunk in indexed_chunks}


@pytest.mark.asyncio
async def test_comparison_question_uses_planned_multi_query_retrieval(monkeypatch, tmp_path) -> None:
    _, indexed_chunks = await _index_fixture_pdf(monkeypatch, tmp_path)
    rag = FakeRAGService(_retrieved_chunks(indexed_chunks))
    workflow = AgenticChatWorkflow(
        rag,
        FakeLLMService("Agentic RAG plans retrieval before generation [agentic_rag_fixture:p1:c0]."),
        FakeWebSearchService(),
        quality_evaluator=FixtureQualityEvaluator(),
    )

    result = await workflow.run(
        ChatWorkflowRequest(
            question="Compare RAG vs AGENTIC_RAG and explain the difference.",
            paper_ids=["agentic_rag_fixture"],
            top_k=6,
            score_threshold=0.3,
            enable_web_search=False,
        )
    )

    retrieve_step = next(event for event in result.trace if event["stage"] == "local_retrieve")
    assert retrieve_step["retrieval_mode"] == "comparative"
    assert retrieve_step["query_count"] == 4
    assert retrieve_step["per_query_top_k"] == 4
    assert retrieve_step["max_total_chunks"] == 12
    assert len(rag.calls) == 4


@pytest.mark.asyncio
async def test_paper_review_question_lowers_threshold_and_caps_chunk_budget(monkeypatch, tmp_path) -> None:
    _, indexed_chunks = await _index_fixture_pdf(monkeypatch, tmp_path)
    rag = FakeRAGService(_retrieved_chunks(indexed_chunks))
    workflow = AgenticChatWorkflow(
        rag,
        FakeLLMService("The paper discusses planning and verification [agentic_rag_fixture:p1:c0]."),
        FakeWebSearchService(),
        quality_evaluator=FixtureQualityEvaluator(),
    )

    result = await workflow.run(
        ChatWorkflowRequest(
            question="Give a summary of the paper, methodology, experiments, and limitations.",
            paper_ids=["agentic_rag_fixture"],
            top_k=8,
            score_threshold=0.4,
            enable_web_search=False,
        )
    )

    retrieve_step = next(event for event in result.trace if event["stage"] == "local_retrieve")
    assert retrieve_step["retrieval_mode"] == "paper_review"
    assert retrieve_step["query_count"] == 6
    assert retrieve_step["per_query_top_k"] == 3
    assert retrieve_step["score_threshold"] == pytest.approx(0.35)
    assert retrieve_step["max_total_chunks"] == 15


@pytest.mark.asyncio
async def test_multi_aspect_question_expands_retrieval_strategy(monkeypatch, tmp_path) -> None:
    _, indexed_chunks = await _index_fixture_pdf(monkeypatch, tmp_path)
    rag = FakeRAGService(_retrieved_chunks(indexed_chunks))
    workflow = AgenticChatWorkflow(
        rag,
        FakeLLMService("Agentic RAG covers mechanism, advantages, and limitations [agentic_rag_fixture:p1:c0]."),
        FakeWebSearchService(),
        quality_evaluator=FixtureQualityEvaluator(),
    )

    result = await workflow.run(
        ChatWorkflowRequest(
            question="Explain Agentic RAG advantages, limitations, applications, and how it works.",
            paper_ids=["agentic_rag_fixture"],
            top_k=5,
            score_threshold=0.3,
            enable_web_search=False,
        )
    )

    retrieve_step = next(event for event in result.trace if event["stage"] == "local_retrieve")
    assert retrieve_step["retrieval_mode"] == "expanded"
    assert retrieve_step["query_count"] == 6
    assert retrieve_step["per_query_top_k"] == 3
    assert retrieve_step["score_threshold"] == pytest.approx(0.25)
    assert retrieve_step["max_total_chunks"] == 15


@pytest.mark.asyncio
async def test_insufficient_local_context_recovers_with_web_search(monkeypatch, tmp_path) -> None:
    await _index_fixture_pdf(monkeypatch, tmp_path)
    web = FakeWebSearchService(
        [
            {
                "title": "Fresh Agentic RAG note",
                "url": "https://example.com/agentic-rag-note",
                "content": "Fresh Agentic RAG systems can recover by searching the web for missing evidence.",
                "score": 0.88,
            }
        ]
    )
    workflow = AgenticChatWorkflow(
        FakeRAGService([]),
        FakeLLMService("Agentic RAG can recover by searching the web for missing evidence [web:1]."),
        web,
        quality_evaluator=EmptyFirstQualityEvaluator(),
    )

    result = await workflow.run(
        ChatWorkflowRequest(
            question="How can Agentic RAG recover when local evidence is missing?",
            paper_ids=["agentic_rag_fixture"],
            enable_web_search=True,
            enable_research_ingest=False,
        )
    )

    assert web.calls == [{"query": "How can Agentic RAG recover when local evidence is missing?", "max_results": 5}]
    assert any(event["stage"] == "execute_tool" and event.get("tool_name") == "web_search" for event in result.trace)
    assert result.citations[0].chunk_id == "web:1"
    assert result.stop_reason == "answered_after_recovery"


@pytest.mark.asyncio
async def test_web_search_step_limit_stops_before_unbounded_recovery(monkeypatch, tmp_path) -> None:
    await _index_fixture_pdf(monkeypatch, tmp_path)
    workflow = AgenticChatWorkflow(
        FakeRAGService([]),
        FakeLLMService("This should not be generated."),
        FakeWebSearchService(),
        quality_evaluator=FixtureQualityEvaluator(),
    )

    result = await workflow.run(
        ChatWorkflowRequest(
            question="Find missing evidence with no agent steps available.",
            paper_ids=["agentic_rag_fixture"],
            enable_web_search=True,
            max_agent_steps=0,
        )
    )

    assert result.answer == "I don't know"
    assert result.stop_reason == "planner_no_valid_steps"
    plan_step = next(event for event in result.trace if event["stage"] == "plan")
    assert plan_step["step_count"] == 0


@pytest.mark.asyncio
async def test_verifier_rejects_fabricated_citation_without_calling_web_when_disabled(monkeypatch, tmp_path) -> None:
    _, indexed_chunks = await _index_fixture_pdf(monkeypatch, tmp_path)
    web = FakeWebSearchService()
    workflow = AgenticChatWorkflow(
        FakeRAGService(_retrieved_chunks(indexed_chunks)),
        FakeLLMService("Agentic RAG reports 99 percent benchmark accuracy [fake:c9]."),
        web,
        quality_evaluator=FixtureQualityEvaluator(),
    )

    result = await workflow.run(
        ChatWorkflowRequest(
            question="What benchmark accuracy does the fixture report?",
            paper_ids=["agentic_rag_fixture"],
            enable_web_search=False,
        )
    )

    verification_step = result.trace[-1]
    assert verification_step["stage"] == "verify_answer"
    assert verification_step["success"] is False
    assert verification_step["suggested_action"] == "retrieve_more"
    assert verification_step["unsupported_claim_count"] == 1
    assert result.answer == "I don't know"
    assert result.citations == []
    assert result.stop_reason == "verification_failed_answer_unknown"
    assert web.calls == []


@pytest.mark.asyncio
async def test_stream_events_returns_trace_tokens_citations_and_terminal_result(monkeypatch, tmp_path) -> None:
    _, indexed_chunks = await _index_fixture_pdf(monkeypatch, tmp_path)
    workflow = AgenticChatWorkflow(
        FakeRAGService(_retrieved_chunks(indexed_chunks)),
        FakeLLMService("Agentic RAG verifies evidence before answering [agentic_rag_fixture:p1:c0]."),
        FakeWebSearchService(),
        quality_evaluator=FixtureQualityEvaluator(),
    )

    events = [
        event
        async for event in workflow.stream_events(
            ChatWorkflowRequest(
                question="What does Agentic RAG verify?",
                paper_ids=["agentic_rag_fixture"],
                enable_web_search=False,
            )
        )
    ]

    assert any(event["type"] == "agent_step" and event["step"]["stage"] == "verify_answer" for event in events)
    assert "".join(event["content"] for event in events if event["type"] == "token").startswith("Agentic RAG verifies")
    citation_event = next(event for event in events if event["type"] == "citations")
    assert citation_event["citations"][0].chunk_id == "agentic_rag_fixture:p1:c0"
    result_event = events[-1]
    assert result_event["type"] == "result"
    assert result_event["result"].stop_reason == "answered_with_sufficient_context"
