from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from app.agent.citations import CitationGrounder
from app.agent.prompts.answer_prompt import AnswerPromptBuilder
from app.agent.workflow import AgenticChatWorkflow, ChatWorkflowRequest
from app.models.chat import ChatHistoryMessage
from app.models.citation import Citation
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService
from app.services.retriever_service import RetrieverService
from app.vectorstore.chroma import ChromaVectorStore
from evals.models import BaselineResult, EvalCase


class EvalBaseline(Protocol):
    mode: str

    async def run_case(self, case: EvalCase) -> BaselineResult:
        ...


class NoopReranker:
    def rerank(self, query: str, chunks: list[dict]) -> list[dict]:
        return chunks


class EvalLLM(Protocol):
    async def complete(self, prompt: str) -> str:
        ...


class VectorOnlyRAGBaseline:
    mode = "vector_only_rag"

    def __init__(
        self,
        vector_store: ChromaVectorStore | Any,
        llm_service: EvalLLM,
        prompt_builder: AnswerPromptBuilder | None = None,
        citation_grounder: CitationGrounder | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._llm_service = llm_service
        self._prompt_builder = prompt_builder or AnswerPromptBuilder()
        self._citation_grounder = citation_grounder or CitationGrounder()

    async def run_case(self, case: EvalCase) -> BaselineResult:
        return await _run_with_latency(self.mode, case, self._answer)

    async def _answer(self, case: EvalCase) -> tuple[str, list[Citation], list[dict], list[dict]]:
        chunks = await self._vector_store.similarity_search(
            case.question,
            top_k=5,
            score_threshold=None,
            paper_ids=case.paper_ids,
        )
        return await _generate_answer(
            case,
            chunks,
            self._llm_service,
            self._prompt_builder,
            self._citation_grounder,
            trace=[{"stage": "vector_only_retrieve", "chunk_count": len(chunks)}],
        )


class HybridRAGBaseline:
    mode = "hybrid_rag"

    def __init__(
        self,
        retriever_service: RetrieverService | Any,
        llm_service: EvalLLM,
        prompt_builder: AnswerPromptBuilder | None = None,
        citation_grounder: CitationGrounder | None = None,
    ) -> None:
        self._retriever_service = retriever_service
        self._llm_service = llm_service
        self._prompt_builder = prompt_builder or AnswerPromptBuilder()
        self._citation_grounder = citation_grounder or CitationGrounder()

    async def run_case(self, case: EvalCase) -> BaselineResult:
        return await _run_with_latency(self.mode, case, self._answer)

    async def _answer(self, case: EvalCase) -> tuple[str, list[Citation], list[dict], list[dict]]:
        chunks = await self._retriever_service.retrieve(
            case.question,
            top_k=5,
            score_threshold=None,
            paper_ids=case.paper_ids,
        )
        return await _generate_answer(
            case,
            chunks,
            self._llm_service,
            self._prompt_builder,
            self._citation_grounder,
            trace=[{"stage": "hybrid_retrieve", "chunk_count": len(chunks)}],
        )


class HybridRerankRAGBaseline:
    mode = "hybrid_rerank_rag"

    def __init__(
        self,
        rag_service: RAGService,
        llm_service: EvalLLM,
        prompt_builder: AnswerPromptBuilder | None = None,
        citation_grounder: CitationGrounder | None = None,
    ) -> None:
        self._rag_service = rag_service
        self._llm_service = llm_service
        self._prompt_builder = prompt_builder or AnswerPromptBuilder()
        self._citation_grounder = citation_grounder or CitationGrounder()

    async def run_case(self, case: EvalCase) -> BaselineResult:
        return await _run_with_latency(self.mode, case, self._answer)

    async def _answer(self, case: EvalCase) -> tuple[str, list[Citation], list[dict], list[dict]]:
        chunks = await self._rag_service.retrieve_context(
            case.question,
            paper_ids=case.paper_ids,
            top_k=5,
            score_threshold=0.65,
            chat_history=_chat_history(case),
        )
        return await _generate_answer(
            case,
            chunks,
            self._llm_service,
            self._prompt_builder,
            self._citation_grounder,
            trace=[{"stage": "hybrid_rerank_retrieve", "chunk_count": len(chunks)}],
        )


class FullAgenticRAGBaseline:
    mode = "full_agentic_rag"

    def __init__(self, workflow: AgenticChatWorkflow) -> None:
        self._workflow = workflow

    async def run_case(self, case: EvalCase) -> BaselineResult:
        started_at = time.perf_counter()
        try:
            result = await self._workflow.run(
                ChatWorkflowRequest(
                    question=case.question,
                    paper_ids=case.paper_ids,
                    chat_history=_chat_history(case),
                )
            )
            latency_ms = (time.perf_counter() - started_at) * 1000
            return BaselineResult(
                case_id=case.id,
                mode=self.mode,
                answer=result.answer,
                citation_chunk_ids=_citation_chunk_ids(result.citations),
                retrieved_chunk_ids=_retrieved_chunk_ids_from_trace(result.trace),
                trace=[dict(event) for event in result.trace],
                latency_ms=latency_ms,
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - started_at) * 1000
            return BaselineResult(
                case_id=case.id,
                mode=self.mode,
                answer="",
                citation_chunk_ids=[],
                retrieved_chunk_ids=[],
                latency_ms=latency_ms,
                error=str(exc),
            )


def build_default_baselines(modes: list[str]) -> list[EvalBaseline]:
    llm_service = LLMService()
    vector_store = ChromaVectorStore()
    hybrid_no_rerank_retriever = RetrieverService(
        vector_store=vector_store,
        reranker_service=NoopReranker(),
    )
    hybrid_rerank_retriever = RetrieverService(vector_store=vector_store)
    rag_service = RAGService(hybrid_rerank_retriever, llm_service)
    available: dict[str, Callable[[], EvalBaseline]] = {
        "vector_only_rag": lambda: VectorOnlyRAGBaseline(vector_store, llm_service),
        "hybrid_rag": lambda: HybridRAGBaseline(hybrid_no_rerank_retriever, llm_service),
        "hybrid_rerank_rag": lambda: HybridRerankRAGBaseline(rag_service, llm_service),
        "full_agentic_rag": lambda: FullAgenticRAGBaseline(AgenticChatWorkflow(rag_service, llm_service)),
    }
    return [available[mode]() for mode in modes]


def all_modes() -> list[str]:
    return ["vector_only_rag", "hybrid_rag", "hybrid_rerank_rag", "full_agentic_rag"]


async def _run_with_latency(
    mode: str,
    case: EvalCase,
    answer_fn: Callable[[EvalCase], Awaitable[tuple[str, list[Citation], list[dict], list[dict]]]],
) -> BaselineResult:
    started_at = time.perf_counter()
    try:
        answer, citations, retrieved_chunks, trace = await answer_fn(case)
        latency_ms = (time.perf_counter() - started_at) * 1000
        return BaselineResult(
            case_id=case.id,
            mode=mode,
            answer=answer,
            citation_chunk_ids=_citation_chunk_ids(citations),
            retrieved_chunk_ids=_retrieved_chunk_ids(retrieved_chunks),
            trace=trace,
            latency_ms=latency_ms,
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - started_at) * 1000
        return BaselineResult(
            case_id=case.id,
            mode=mode,
            answer="",
            citation_chunk_ids=[],
            retrieved_chunk_ids=[],
            latency_ms=latency_ms,
            error=str(exc),
        )


async def _generate_answer(
    case: EvalCase,
    chunks: list[dict],
    llm_service: EvalLLM,
    prompt_builder: AnswerPromptBuilder,
    citation_grounder: CitationGrounder,
    trace: list[dict],
) -> tuple[str, list[Citation], list[dict], list[dict]]:
    if not chunks:
        return "I don't know", [], [], trace
    citations = citation_grounder.build_citations(chunks, case.question)
    prompt = prompt_builder.build(case.question, chunks, _chat_history(case))
    answer = (await llm_service.complete(prompt)).strip() or "I don't know"
    display_answer = citation_grounder.display_answer_with_numbered_citations(answer, citations)
    return display_answer, citations, chunks, trace


def _chat_history(case: EvalCase) -> list[ChatHistoryMessage] | None:
    if not case.chat_history:
        return None
    return [ChatHistoryMessage.model_validate(message) for message in case.chat_history]


def _citation_chunk_ids(citations: list[Citation]) -> list[str]:
    return [citation.chunk_id for citation in citations if citation.chunk_id]


def _retrieved_chunk_ids(chunks: list[dict]) -> list[str]:
    chunk_ids = []
    for chunk in chunks:
        citation = chunk.get("citation") or {}
        metadata = chunk.get("metadata") or {}
        chunk_id = str(citation.get("chunk_id") or metadata.get("chunk_id") or chunk.get("id") or "")
        if chunk_id:
            chunk_ids.append(chunk_id)
    return chunk_ids


def _retrieved_chunk_ids_from_trace(trace: list[dict]) -> list[str]:
    chunk_ids = []
    for event in trace:
        tool_result = event.get("tool_result") or {}
        for chunk in tool_result.get("chunks") or []:
            chunk_id = str(chunk.get("id") or "")
            if chunk_id and chunk_id not in chunk_ids:
                chunk_ids.append(chunk_id)
    return chunk_ids
