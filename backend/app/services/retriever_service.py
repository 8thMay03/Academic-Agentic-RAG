from app.config.settings import settings
from app.services.reranker_service import RerankerService
from app.vectorstore.chroma import ChromaVectorStore


class RetrieverService:
    def __init__(
        self,
        vector_store: ChromaVectorStore | None = None,
        reranker_service: RerankerService | None = None,
        vector_weight: float | None = None,
        keyword_weight: float | None = None,
        candidate_multiplier: int | None = None,
    ) -> None:
        self._vector_store = vector_store or ChromaVectorStore()
        self._reranker_service = reranker_service or RerankerService()
        self._vector_weight = (
            vector_weight
            if vector_weight is not None
            else settings.RETRIEVAL_VECTOR_WEIGHT
        )
        self._keyword_weight = (
            keyword_weight
            if keyword_weight is not None
            else settings.RETRIEVAL_KEYWORD_WEIGHT
        )
        self._candidate_multiplier = (
            candidate_multiplier
            if candidate_multiplier is not None
            else settings.RETRIEVAL_CANDIDATE_MULTIPLIER
        )

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float | None = None,
        paper_ids: list[str] | None = None,
    ) -> list[dict]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        if score_threshold is not None and not 0 <= score_threshold <= 1:
            raise ValueError("score_threshold must be between 0 and 1")

        candidate_count = max(top_k, top_k * self._candidate_multiplier)
        vector_results = await self._vector_store.similarity_search(
            query,
            top_k=candidate_count,
            score_threshold=None,
            paper_ids=paper_ids,
        )
        keyword_results = await self._vector_store.keyword_search(
            query,
            top_k=candidate_count,
            score_threshold=None,
            paper_ids=paper_ids,
        )

        merged_results = self._merge_results(vector_results, keyword_results)
        if score_threshold is not None:
            merged_results = [
                result
                for result in merged_results
                if float(result.get("score", 0.0)) >= score_threshold
            ]

        return self._reranker_service.rerank(query, merged_results)[:top_k]

    def _merge_results(self, vector_results: list[dict], keyword_results: list[dict]) -> list[dict]:
        merged_results: dict[str, dict] = {}

        for result in vector_results:
            result_key = self._result_key(result)
            merged_result = self._ensure_result(merged_results, result_key, result)
            merged_result["vector_score"] = float(result.get("score", 0.0))
            merged_result["retrieval_sources"].add("vector")

        for result in keyword_results:
            result_key = self._result_key(result)
            merged_result = self._ensure_result(merged_results, result_key, result)
            merged_result["keyword_score"] = float(result.get("score", 0.0))
            merged_result["retrieval_sources"].add("keyword")

        scored_results = []
        for result in merged_results.values():
            result["retrieval_sources"] = sorted(result["retrieval_sources"])
            result["score"] = self._hybrid_score(result)
            scored_results.append(result)

        return sorted(scored_results, key=lambda result: result["score"], reverse=True)

    @staticmethod
    def _result_key(result: dict) -> str:
        metadata = result.get("metadata") or {}
        return str(metadata.get("chunk_id") or result.get("id"))

    @staticmethod
    def _ensure_result(
        merged_results: dict[str, dict],
        result_key: str,
        result: dict,
    ) -> dict:
        if result_key not in merged_results:
            merged_results[result_key] = {
                **result,
                "vector_score": None,
                "keyword_score": None,
                "retrieval_sources": set(),
            }
        return merged_results[result_key]

    def _hybrid_score(self, result: dict) -> float:
        weighted_score = 0.0
        active_weight = 0.0

        if result.get("vector_score") is not None:
            weighted_score += self._vector_weight * float(result["vector_score"])
            active_weight += self._vector_weight
        if result.get("keyword_score") is not None:
            weighted_score += self._keyword_weight * float(result["keyword_score"])
            active_weight += self._keyword_weight

        if active_weight == 0:
            return 0.0
        return weighted_score / active_weight
