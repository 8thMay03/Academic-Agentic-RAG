from evals.metrics import (
    citation_scores,
    context_scores,
    evaluate_result,
    retrieval_scores,
    summarize_results,
)
from evals.models import BaselineResult, EvalCase


def test_citation_scores_measure_precision_recall_and_invalid_rate() -> None:
    scores = citation_scores(
        expected_ids=["paper-1:p1:c0", "paper-1:p2:c0"],
        actual_ids=["paper-1:p1:c0", "fake:c0"],
    )

    assert scores["citation_precision"] == 0.5
    assert scores["citation_recall"] == 0.5
    assert scores["invalid_citation_rate"] == 0.5


def test_retrieval_scores_compute_recall_mrr_and_ndcg() -> None:
    scores = retrieval_scores(
        expected_ids=["chunk-b"],
        retrieved_ids=["chunk-a", "chunk-b", "chunk-c"],
    )

    assert scores["retrieval_recall"] == 1.0
    assert scores["mrr"] == 0.5
    assert 0 < scores["ndcg"] < 1


def test_context_scores_measure_retrieved_context_precision() -> None:
    scores = context_scores(
        expected_ids=["chunk-b"],
        retrieved_ids=["chunk-a", "chunk-b", "chunk-c"],
    )

    assert scores["context_precision"] == 1 / 3


def test_evaluate_result_scores_answer_points_and_abstention() -> None:
    case = EvalCase(
        id="q1",
        question="How does Agentic RAG work?",
        expected_answer_points=["Agentic RAG uses planning to retrieve evidence"],
        expected_citation_chunk_ids=["paper-1:p3:c0"],
        is_answerable=True,
    )
    result = BaselineResult(
        case_id="q1",
        mode="hybrid_rerank_rag",
        answer="Agentic RAG uses planning to decide when to retrieve evidence [1].",
        citation_chunk_ids=["paper-1:p3:c0"],
        retrieved_chunk_ids=["paper-1:p3:c0"],
        latency_ms=12.5,
        trace=[
            {"stage": "execute_tool", "tool_name": "web_search", "success": True},
            {"stage": "plan_recovery"},
            {
                "stage": "verify_answer",
                "supported_claim_count": 1,
                "contradicted_claim_count": 0,
                "insufficient_claim_count": 1,
                "unsupported_claim_count": 1,
                "claim_citation_map": [
                    {
                        "claim": "Supported claim",
                        "supporting_chunk_ids": ["paper-1:p3:c0"],
                    },
                    {"claim": "Unsupported claim", "supporting_chunk_ids": []},
                ],
            },
        ],
    )

    evaluated = evaluate_result(case, result)

    assert evaluated.metrics["answer_point_recall"] == 1.0
    assert evaluated.metrics["abstention_correct"] is True
    assert evaluated.metrics["citation_precision"] == 1.0
    assert evaluated.metrics["retrieval_recall"] == 1.0
    assert evaluated.metrics["context_precision"] == 1.0
    assert evaluated.metrics["unsupported_claim_rate"] == 0.5
    assert evaluated.metrics["claim_citation_coverage"] == 0.5
    assert evaluated.metrics["verifier_available"] == 1.0
    assert evaluated.metrics["tool_call_count"] == 1
    assert evaluated.metrics["tool_success_rate"] == 1.0
    assert evaluated.metrics["web_arxiv_usage"] == 1.0
    assert evaluated.metrics["recovery_used"] == 1.0


def test_summarize_results_groups_by_mode() -> None:
    case = EvalCase(id="q1", question="What is missing?", is_answerable=False)
    evaluated = evaluate_result(
        case,
        BaselineResult(
            case_id="q1",
            mode="vector_only_rag",
            answer="I don't know",
            citation_chunk_ids=[],
            retrieved_chunk_ids=[],
            latency_ms=10,
        ),
    )

    summary = summarize_results([evaluated])

    assert summary == [
        {
            "mode": "vector_only_rag",
            "case_count": 1,
            "answer_point_recall": 1.0,
            "citation_precision": 1.0,
            "citation_recall": 1.0,
            "invalid_citation_rate": 0.0,
            "retrieval_recall": 1.0,
            "context_precision": 1.0,
            "mrr": 1.0,
            "ndcg": 1.0,
            "abstention_accuracy": 1.0,
            "unsupported_claim_rate": 0.0,
            "claim_citation_coverage": 0.0,
            "verifier_availability_rate": 0.0,
            "avg_tool_calls": 0.0,
            "tool_success_rate": 1.0,
            "web_arxiv_usage_rate": 0.0,
            "recovery_usage_rate": 0.0,
            "p50_latency_ms": 10.0,
            "p95_latency_ms": 10.0,
            "avg_input_tokens": 0.0,
            "avg_output_tokens": 0.0,
            "avg_estimated_cost_usd": 0.0,
            "error_count": 0,
        }
    ]
