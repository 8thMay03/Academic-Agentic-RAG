import pytest

from app.services.reranker_service import RerankerService


class FakeCrossEncoder:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.pairs = None

    def predict(self, pairs):
        self.pairs = pairs
        return self.scores


class FailingCrossEncoder:
    def predict(self, pairs):
        raise RuntimeError("model unavailable")


def test_reranker_service_uses_cross_encoder_scores() -> None:
    model = FakeCrossEncoder(scores=[-1.2, 3.5])
    service = RerankerService(model=model, model_name="fake-cross-encoder")

    results = service.rerank(
        "agentic rag planning",
        [
            {
                "id": "high-base-low-cross-encoder",
                "text": "A general retrieval system.",
                "score": 0.8,
            },
            {
                "id": "lower-base-high-cross-encoder",
                "text": "Agentic RAG uses planning before retrieval.",
                "score": 0.72,
            },
        ],
    )

    assert model.pairs == [
        ("agentic rag planning", "A general retrieval system."),
        ("agentic rag planning", "Agentic RAG uses planning before retrieval."),
    ]
    assert results[0]["id"] == "lower-base-high-cross-encoder"
    assert results[0]["cross_encoder_score"] == 3.5
    assert results[0]["rerank_score"] == pytest.approx(0.9706, abs=0.0001)
    assert results[0]["reranker"] == "fake-cross-encoder"


def test_reranker_service_falls_back_to_heuristic_when_cross_encoder_fails() -> None:
    service = RerankerService(model=FailingCrossEncoder(), fallback_to_heuristic=True)

    results = service.rerank(
        "agentic rag planning",
        [
            {
                "id": "high-base-low-overlap",
                "text": "A general retrieval system.",
                "score": 0.8,
            },
            {
                "id": "lower-base-high-overlap",
                "text": "Agentic RAG uses planning before retrieval.",
                "score": 0.72,
            },
        ],
    )

    assert results[0]["id"] == "lower-base-high-overlap"
    assert results[0]["rerank_score"] > results[1]["rerank_score"]
    assert results[0]["reranker"] == "heuristic"


def test_reranker_service_can_disable_fallback() -> None:
    service = RerankerService(model=FailingCrossEncoder(), fallback_to_heuristic=False)

    with pytest.raises(RuntimeError, match="model unavailable"):
        service.rerank("agentic rag planning", [{"text": "Agentic RAG.", "score": 0.5}])
