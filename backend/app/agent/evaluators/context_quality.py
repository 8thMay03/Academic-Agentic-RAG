from app.agent.models import (
    ContextQuality,
    RetrievedChunk,
    optional_float,
    retrieved_chunk_ranking_score,
    retrieved_chunk_source_id,
    retrieved_chunk_text,
)
from app.services.llm_service import LLMService
from app.vectorstore.bm25 import tokenize


MIN_CONTEXT_CHUNKS = 2
MIN_CONTEXT_CHARS = 600
MIN_TOP_SCORE = 0.45
MIN_AVERAGE_SCORE = 0.35
MIN_QUERY_COVERAGE = 0.25
STRONG_TOP_SCORE = 0.75
STRONG_QUERY_COVERAGE = 0.5
LATEST_QUERY_TERMS = {
    "latest",
    "current",
    "recent",
    "today",
    "now",
    "newest",
    "state of the art",
    "sota",
    "mới nhất",
    "gan day",
    "gần đây",
    "hien tai",
    "hiện tại",
    "hom nay",
    "hôm nay",
}


class ContextQualityEvaluator:
    def __init__(self, llm_service: LLMService) -> None:
        self._llm_service = llm_service

    async def evaluate(self, question: str, chunks: list[RetrievedChunk]) -> ContextQuality:
        context_chars = sum(len(retrieved_chunk_text(chunk)) for chunk in chunks)
        top_score = self._top_score(chunks)
        average_score = self._average_score(chunks)
        source_count = self._source_count(chunks)
        query_coverage = self._query_coverage(question, chunks)
        base_quality = {
            "chunk_count": len(chunks),
            "context_chars": context_chars,
            "top_score": top_score,
            "average_score": average_score,
            "source_count": source_count,
            "query_coverage": query_coverage,
        }

        if self._requires_fresh_context(question):
            return ContextQuality(
                False,
                reason="latest_query_requires_web",
                **base_quality,
            )
        if not chunks:
            return ContextQuality(False, reason="no_local_context", **base_quality)
        if len(chunks) < MIN_CONTEXT_CHUNKS and context_chars < MIN_CONTEXT_CHARS:
            return ContextQuality(False, reason="insufficient_local_context", **base_quality)
        if top_score is not None and top_score < MIN_TOP_SCORE:
            return ContextQuality(False, reason="low_top_score", **base_quality)
        if average_score is not None and average_score < MIN_AVERAGE_SCORE:
            return ContextQuality(False, reason="low_average_score", **base_quality)
        if query_coverage < MIN_QUERY_COVERAGE:
            return ContextQuality(False, reason="low_query_coverage", **base_quality)

        if self._should_self_check(top_score, query_coverage):
            self_check_passed = await self._self_check_context(question, chunks)
            return ContextQuality(
                self_check_passed,
                reason="llm_self_check_passed" if self_check_passed else "llm_self_check_failed",
                self_check_used=True,
                self_check_passed=self_check_passed,
                **base_quality,
            )

        return ContextQuality(True, reason="strong_context", **base_quality)

    async def _self_check_context(self, question: str, chunks: list[RetrievedChunk]) -> bool:
        prompt = self._self_check_prompt(question, chunks)
        try:
            response = (await self._llm_service.complete(prompt)).strip().lower()
        except Exception:
            return False
        return response.startswith("yes") or '"sufficient": true' in response

    @staticmethod
    def _self_check_prompt(question: str, chunks: list[RetrievedChunk], max_chars: int = 4000) -> str:
        context = "\n\n".join(
            f"[{index}] {retrieved_chunk_text(chunk)}"
            for index, chunk in enumerate(chunks, start=1)
        )[:max_chars]
        return (
            "Decide whether the retrieved context is sufficient to answer the question without web search.\n"
            "Return only YES or NO.\n\n"
            f"Question: {question}\n\n"
            f"Retrieved context:\n{context}"
        )

    @staticmethod
    def _should_self_check(top_score: float | None, query_coverage: float) -> bool:
        if top_score is None:
            return True
        return top_score < STRONG_TOP_SCORE or query_coverage < STRONG_QUERY_COVERAGE

    @classmethod
    def _top_score(cls, chunks: list[RetrievedChunk]) -> float | None:
        scores = [
            score
            for chunk in chunks
            if (score := cls._ranking_score(chunk)) is not None
        ]
        return max(scores) if scores else None

    @classmethod
    def _average_score(cls, chunks: list[RetrievedChunk]) -> float | None:
        scores = [
            score
            for chunk in chunks
            if (score := cls._ranking_score(chunk)) is not None
        ]
        if not scores:
            return None
        return sum(scores) / len(scores)

    @classmethod
    def _ranking_score(cls, chunk: RetrievedChunk) -> float | None:
        return retrieved_chunk_ranking_score(chunk)

    @staticmethod
    def _source_count(chunks: list[RetrievedChunk]) -> int:
        sources = set()
        for chunk in chunks:
            source = retrieved_chunk_source_id(chunk)
            if source:
                sources.add(source)
        return len(sources)

    @staticmethod
    def _query_coverage(question: str, chunks: list[RetrievedChunk]) -> float:
        query_terms = set(tokenize(question))
        if not query_terms:
            return 1.0
        context_terms = set(tokenize(" ".join(retrieved_chunk_text(chunk) for chunk in chunks)))
        return len(query_terms & context_terms) / len(query_terms)

    @staticmethod
    def _requires_fresh_context(question: str) -> bool:
        normalized = " ".join(question.lower().split())
        return any(term in normalized for term in LATEST_QUERY_TERMS)

    @staticmethod
    def _optional_float(value: object) -> float | None:
        return optional_float(value)
