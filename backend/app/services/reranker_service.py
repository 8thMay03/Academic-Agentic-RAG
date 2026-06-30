import math
from typing import Any

from app.config.settings import settings
from app.vectorstore.bm25 import tokenize


class RerankerService:
    def __init__(
        self,
        model: Any | None = None,
        model_name: str | None = None,
        enabled: bool | None = None,
        fallback_to_heuristic: bool | None = None,
    ) -> None:
        self._model = model
        self._model_name = model_name or settings.CROSS_ENCODER_RERANKER_MODEL
        self._enabled = (
            settings.CROSS_ENCODER_RERANKER_ENABLED
            if enabled is None
            else enabled
        )
        self._fallback_to_heuristic = (
            settings.CROSS_ENCODER_FALLBACK_TO_HEURISTIC
            if fallback_to_heuristic is None
            else fallback_to_heuristic
        )

    def rerank(self, query: str, chunks: list[dict]) -> list[dict]:
        if not chunks:
            return chunks
        if not query.strip():
            return chunks

        if self._enabled:
            try:
                return self._cross_encoder_rerank(query, chunks)
            except Exception:
                if not self._fallback_to_heuristic:
                    raise

        return self._heuristic_rerank(query, chunks)

    def _cross_encoder_rerank(self, query: str, chunks: list[dict]) -> list[dict]:
        model = self._get_model()
        pairs = [(query, chunk.get("text", "")) for chunk in chunks]
        scores = model.predict(pairs)

        reranked_chunks = []
        for chunk, score in zip(chunks, scores, strict=True):
            raw_score = float(score)
            reranked_chunk = dict(chunk)
            reranked_chunk["cross_encoder_score"] = raw_score
            reranked_chunk["rerank_score"] = self._sigmoid(raw_score)
            reranked_chunk["reranker"] = self._model_name
            reranked_chunks.append(reranked_chunk)

        return sorted(
            reranked_chunks,
            key=lambda chunk: (
                chunk.get("cross_encoder_score", float("-inf")),
                chunk.get("score", 0.0),
            ),
            reverse=True,
        )

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model

        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(self._model_name)
        return self._model

    @staticmethod
    def _heuristic_rerank(query: str, chunks: list[dict]) -> list[dict]:
        query_terms = set(tokenize(query))
        if not query_terms:
            return chunks

        reranked_chunks = []
        for chunk in chunks:
            reranked_chunk = dict(chunk)
            text_terms = set(tokenize(reranked_chunk.get("text", "")))
            lexical_overlap = len(query_terms & text_terms) / len(query_terms)
            base_score = float(reranked_chunk.get("score", 0.0))
            reranked_chunk["rerank_score"] = (0.85 * base_score) + (0.15 * lexical_overlap)
            reranked_chunk["reranker"] = "heuristic"
            reranked_chunks.append(reranked_chunk)

        return sorted(
            reranked_chunks,
            key=lambda chunk: (chunk.get("rerank_score", 0.0), chunk.get("score", 0.0)),
            reverse=True,
        )

    @staticmethod
    def _sigmoid(score: float) -> float:
        if score >= 0:
            z = math.exp(-score)
            return 1 / (1 + z)
        z = math.exp(score)
        return z / (1 + z)
