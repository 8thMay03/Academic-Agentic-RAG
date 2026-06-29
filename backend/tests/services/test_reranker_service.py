from app.services.reranker_service import RerankerService


def test_reranker_service_boosts_chunks_with_query_term_overlap() -> None:
    service = RerankerService()

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
