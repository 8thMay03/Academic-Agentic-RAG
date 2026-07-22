from evals.report import build_markdown_report


def test_build_markdown_report_includes_delta_and_error_analysis() -> None:
    report = build_markdown_report(
        {
            "dataset": "evals/datasets/agentic_rag_eval.jsonl",
            "case_count": 2,
            "modes": ["hybrid_rerank_rag", "full_agentic_rag"],
            "summary": [
                {
                    "mode": "full_agentic_rag",
                    "case_count": 2,
                    "answer_point_recall": 1.0,
                    "citation_precision": 1.0,
                    "citation_recall": 1.0,
                    "retrieval_recall": 1.0,
                    "abstention_accuracy": 1.0,
                    "p50_latency_ms": 600.0,
                    "p95_latency_ms": 800.0,
                    "avg_estimated_cost_usd": 0.002,
                    "error_count": 0,
                },
                {
                    "mode": "hybrid_rerank_rag",
                    "case_count": 2,
                    "answer_point_recall": 0.5,
                    "citation_precision": 0.5,
                    "citation_recall": 0.5,
                    "retrieval_recall": 0.5,
                    "abstention_accuracy": 0.5,
                    "p50_latency_ms": 200.0,
                    "p95_latency_ms": 300.0,
                    "avg_estimated_cost_usd": 0.001,
                    "error_count": 0,
                },
            ],
            "results": [
                {
                    "case_id": "q1",
                    "mode": "hybrid_rerank_rag",
                    "metrics": {
                        "answer_point_recall": 0.0,
                        "citation_precision": 0.0,
                        "citation_recall": 0.0,
                        "retrieval_recall": 0.0,
                        "abstention_correct": False,
                        "error": None,
                    },
                }
            ],
        }
    )

    assert "# Agentic RAG Evaluation Report" in report
    assert "| mode | case_count | answer_point_recall |" in report
    assert "- Dataset: `evals/datasets/agentic_rag_eval.jsonl`" in report
    assert "- answer point recall: 1.000 vs 0.500 (+0.500)" in report
    assert "- p95 latency: 800.000 ms vs 300.000 ms (+500.000 ms)" in report
    assert "- average estimated cost: $0.002000 vs $0.001000 (+0.001000)" in report
    assert "| `q1` | `hybrid_rerank_rag` |" in report
    assert "wrong abstention behavior" in report
