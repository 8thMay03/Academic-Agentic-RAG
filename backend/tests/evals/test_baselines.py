import pytest

from evals.baselines import HybridRAGBaseline, VectorOnlyRAGBaseline
from evals.local_fixture import build_local_fixture_baselines
from evals.models import EvalCase
from evals.offline_fixture import OfflineFixtureBaseline


class FakeVectorStore:
    async def similarity_search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float | None = None,
        paper_ids: list[str] | None = None,
    ) -> list[dict]:
        return [retrieved_chunk()]


class FakeRetrieverService:
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float | None = None,
        paper_ids: list[str] | None = None,
    ) -> list[dict]:
        return [retrieved_chunk()]


class FakeLLMService:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "Agentic RAG uses planning to retrieve evidence [paper-1:p3:c0]."


@pytest.mark.asyncio
async def test_vector_only_baseline_returns_answer_citations_and_retrieved_ids() -> None:
    llm = FakeLLMService()
    baseline = VectorOnlyRAGBaseline(FakeVectorStore(), llm)

    result = await baseline.run_case(EvalCase(id="q1", question="How does Agentic RAG work?"))

    assert result.mode == "vector_only_rag"
    assert result.answer == "Agentic RAG uses planning to retrieve evidence [1]."
    assert result.citation_chunk_ids == ["paper-1:p3:c0"]
    assert result.retrieved_chunk_ids == ["paper-1:p3:c0"]
    assert result.error is None
    assert "Retrieved context:" in llm.prompts[0]


@pytest.mark.asyncio
async def test_hybrid_baseline_returns_unknown_when_no_chunks_are_retrieved() -> None:
    class EmptyRetriever:
        async def retrieve(self, *args, **kwargs) -> list[dict]:
            return []

    baseline = HybridRAGBaseline(EmptyRetriever(), FakeLLMService())

    result = await baseline.run_case(EvalCase(id="q1", question="Unknown?"))

    assert result.mode == "hybrid_rag"
    assert result.answer == "I don't know"
    assert result.citation_chunk_ids == []
    assert result.retrieved_chunk_ids == []


def retrieved_chunk() -> dict:
    return {
        "id": "paper-1:p3:c0",
        "text": "Agentic RAG uses planning to decide when to retrieve evidence.",
        "metadata": {
            "paper_id": "paper-1",
            "title": "Agentic RAG",
            "page_number": "3",
            "chunk_id": "paper-1:p3:c0",
        },
        "score": 0.91,
        "citation": {
            "paper_id": "paper-1",
            "title": "Agentic RAG",
            "page_number": 3,
            "chunk_id": "paper-1:p3:c0",
            "text": "Agentic RAG uses planning to decide when to retrieve evidence.",
        },
    }


@pytest.mark.asyncio
async def test_local_fixture_profile_indexes_temp_corpus_and_runs_agentic_baseline() -> None:
    case = EvalCase(
        id="fixture_factual_001",
        question="What makes Agentic RAG different from a fixed RAG pipeline?",
        paper_ids=["agentic_rag_fixture"],
        expected_answer_points=[
            "Agentic RAG plans retrieval actions",
            "Agentic RAG verifies whether evidence is sufficient",
        ],
        expected_citation_chunk_ids=["agentic_rag_fixture:p1:c0"],
    )
    baseline = build_local_fixture_baselines(["full_agentic_rag"])[0]

    result = await baseline.run_case(case)

    assert result.error is None
    assert "plans retrieval actions" in result.answer
    assert result.citation_chunk_ids == ["agentic_rag_fixture:p1:c0"]
    assert "agentic_rag_fixture:p1:c0" in result.retrieved_chunk_ids
    assert any(event["stage"] == "local_retrieve" for event in result.trace)


@pytest.mark.asyncio
async def test_offline_fixture_full_agentic_handles_fresh_case() -> None:
    case = EvalCase(
        id="fresh_001",
        question="What is latest?",
        answer_type="fresh_research",
        expected_answer_points=["Fresh research requires external search or arXiv retrieval"],
        expected_citation_chunk_ids=["fresh:p1:c0"],
        requires_fresh_context=True,
    )
    baseline = OfflineFixtureBaseline("full_agentic_rag")

    result = await baseline.run_case(case)

    assert result.answer != "I don't know"
    assert result.citation_chunk_ids == ["fresh:p1:c0"]
    assert result.retrieved_chunk_ids == ["fresh:p1:c0"]
    assert result.input_tokens > 0


@pytest.mark.asyncio
async def test_offline_fixture_vector_only_abstains_on_fresh_case() -> None:
    case = EvalCase(
        id="fresh_001",
        question="What is latest?",
        answer_type="fresh_research",
        expected_answer_points=["Fresh research requires external search or arXiv retrieval"],
        expected_citation_chunk_ids=["fresh:p1:c0"],
        requires_fresh_context=True,
    )
    baseline = OfflineFixtureBaseline("vector_only_rag")

    result = await baseline.run_case(case)

    assert result.answer == "I don't know"
    assert result.citation_chunk_ids == []
