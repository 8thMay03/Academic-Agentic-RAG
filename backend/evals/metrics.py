from __future__ import annotations

import math
import re
from collections import defaultdict
from statistics import median
from typing import Any

from evals.models import BaselineResult, EvalCase, EvaluatedResult


UNKNOWN_ANSWER = "I don't know"
TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")


def evaluate_result(case: EvalCase, result: BaselineResult) -> EvaluatedResult:
    citation_metrics = citation_scores(case.expected_citation_chunk_ids, result.citation_chunk_ids)
    retrieval_metrics = retrieval_scores(case.expected_citation_chunk_ids, result.retrieved_chunk_ids)
    context_metrics = context_scores(case.expected_citation_chunk_ids, result.retrieved_chunk_ids)
    answer_metrics = answer_scores(case, result.answer)
    verifier_metrics = verifier_scores(result.trace, result.answered_unknown)
    agent_metrics = agent_scores(result.trace)
    metrics: dict[str, float | int | str | bool | None] = {
        **citation_metrics,
        **retrieval_metrics,
        **context_metrics,
        **answer_metrics,
        **verifier_metrics,
        **agent_metrics,
        "latency_ms": result.latency_ms,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "estimated_cost_usd": result.estimated_cost_usd,
        "error": result.error,
    }
    return EvaluatedResult(case_id=case.id, mode=result.mode, metrics=metrics, result=result)


def citation_scores(expected_ids: list[str], actual_ids: list[str]) -> dict[str, float]:
    expected = set(expected_ids)
    actual = set(actual_ids)
    if not expected and not actual:
        return {
            "citation_precision": 1.0,
            "citation_recall": 1.0,
            "invalid_citation_rate": 0.0,
        }
    precision = len(expected & actual) / len(actual) if actual else 0.0
    recall = len(expected & actual) / len(expected) if expected else 1.0
    invalid_rate = len(actual - expected) / len(actual) if actual else 0.0
    return {
        "citation_precision": precision,
        "citation_recall": recall,
        "invalid_citation_rate": invalid_rate,
    }


def retrieval_scores(expected_ids: list[str], retrieved_ids: list[str]) -> dict[str, float]:
    expected = set(expected_ids)
    if not expected:
        return {
            "retrieval_recall": 1.0,
            "mrr": 1.0,
            "ndcg": 1.0,
        }
    hits = [1 if chunk_id in expected else 0 for chunk_id in retrieved_ids]
    recall = len(set(retrieved_ids) & expected) / len(expected)
    reciprocal_rank = 0.0
    for index, hit in enumerate(hits, start=1):
        if hit:
            reciprocal_rank = 1 / index
            break
    dcg = sum(hit / math.log2(index + 1) for index, hit in enumerate(hits, start=1))
    ideal_hits = [1] * min(len(expected), len(retrieved_ids))
    idcg = sum(hit / math.log2(index + 1) for index, hit in enumerate(ideal_hits, start=1))
    return {
        "retrieval_recall": recall,
        "mrr": reciprocal_rank,
        "ndcg": dcg / idcg if idcg else 0.0,
    }


def context_scores(expected_ids: list[str], retrieved_ids: list[str]) -> dict[str, float]:
    expected = set(expected_ids)
    retrieved = set(retrieved_ids)
    if not expected and not retrieved:
        return {"context_precision": 1.0}
    if not retrieved:
        return {"context_precision": 0.0 if expected else 1.0}
    return {"context_precision": len(expected & retrieved) / len(retrieved)}


def answer_scores(case: EvalCase, answer: str) -> dict[str, float | bool]:
    normalized_answer = normalize_text(answer)
    expected_points = [normalize_text(point) for point in case.expected_answer_points]
    covered_points = [
        point
        for point in expected_points
        if point and _point_is_covered(point, normalized_answer)
    ]
    answer_point_recall = len(covered_points) / len(expected_points) if expected_points else 1.0
    answered_unknown = answer.strip() == UNKNOWN_ANSWER
    abstention_correct = (not case.is_answerable and answered_unknown) or (case.is_answerable and not answered_unknown)
    return {
        "answer_point_recall": answer_point_recall,
        "abstention_correct": abstention_correct,
        "answered_unknown": answered_unknown,
    }


def verifier_scores(trace: list[dict[str, Any]], answered_unknown: bool) -> dict[str, float | None]:
    verification = _latest_trace_event(trace, "verify_answer")
    if not verification:
        return {
            "unsupported_claim_rate": None,
            "claim_citation_coverage": None,
            "verifier_available": 0.0,
        }

    supported = int(verification.get("supported_claim_count") or 0)
    contradicted = int(verification.get("contradicted_claim_count") or 0)
    insufficient = int(verification.get("insufficient_claim_count") or 0)
    unsupported = int(verification.get("unsupported_claim_count") or 0)
    checked_claim_count = supported + contradicted + insufficient
    claim_map = verification.get("claim_citation_map") or []
    mapped_claim_count = sum(
        1
        for item in claim_map
        if isinstance(item, dict) and item.get("supporting_chunk_ids")
    )
    unsupported_denominator = checked_claim_count or unsupported
    return {
        "unsupported_claim_rate": unsupported / unsupported_denominator if unsupported_denominator else 0.0,
        "claim_citation_coverage": mapped_claim_count / len(claim_map) if claim_map else 1.0,
        "verifier_available": 1.0,
    }


def agent_scores(trace: list[dict[str, Any]]) -> dict[str, float | int]:
    tool_events = [event for event in trace if event.get("stage") == "execute_tool"]
    successful_tools = sum(1 for event in tool_events if event.get("success") is not False)
    recovery_events = [event for event in trace if event.get("stage") == "plan_recovery"]
    return {
        "tool_call_count": len(tool_events),
        "tool_success_rate": successful_tools / len(tool_events) if tool_events else 1.0,
        "web_arxiv_usage": 1.0
        if any(event.get("tool_name") in {"web_search", "arxiv_search"} for event in tool_events)
        else 0.0,
        "recovery_used": 1.0 if recovery_events else 0.0,
    }


def summarize_results(results: list[EvaluatedResult]) -> list[dict[str, Any]]:
    grouped: dict[str, list[EvaluatedResult]] = defaultdict(list)
    for result in results:
        grouped[result.mode].append(result)

    summaries = []
    for mode, mode_results in sorted(grouped.items()):
        summaries.append(
            {
                "mode": mode,
                "case_count": len(mode_results),
                "answer_point_recall": average_metric(mode_results, "answer_point_recall"),
                "citation_precision": average_metric(mode_results, "citation_precision"),
                "citation_recall": average_metric(mode_results, "citation_recall"),
                "invalid_citation_rate": average_metric(mode_results, "invalid_citation_rate"),
                "retrieval_recall": average_metric(mode_results, "retrieval_recall"),
                "context_precision": average_metric(mode_results, "context_precision"),
                "mrr": average_metric(mode_results, "mrr"),
                "ndcg": average_metric(mode_results, "ndcg"),
                "abstention_accuracy": average_bool_metric(mode_results, "abstention_correct"),
                "unsupported_claim_rate": average_metric(mode_results, "unsupported_claim_rate"),
                "claim_citation_coverage": average_metric(mode_results, "claim_citation_coverage"),
                "verifier_availability_rate": average_metric(mode_results, "verifier_available"),
                "avg_tool_calls": average_metric(mode_results, "tool_call_count"),
                "tool_success_rate": average_metric(mode_results, "tool_success_rate"),
                "web_arxiv_usage_rate": average_metric(mode_results, "web_arxiv_usage"),
                "recovery_usage_rate": average_metric(mode_results, "recovery_used"),
                "p50_latency_ms": percentile_metric(mode_results, "latency_ms", 50),
                "p95_latency_ms": percentile_metric(mode_results, "latency_ms", 95),
                "avg_input_tokens": average_metric(mode_results, "input_tokens"),
                "avg_output_tokens": average_metric(mode_results, "output_tokens"),
                "avg_estimated_cost_usd": average_metric(mode_results, "estimated_cost_usd"),
                "error_count": sum(1 for result in mode_results if result.metrics.get("error")),
            }
        )
    return summaries


def _latest_trace_event(trace: list[dict[str, Any]], stage: str) -> dict[str, Any] | None:
    for event in reversed(trace):
        if event.get("stage") == stage:
            return event
    return None


def average_metric(results: list[EvaluatedResult], metric_name: str) -> float:
    values = [
        float(value)
        for result in results
        if isinstance((value := result.metrics.get(metric_name)), int | float)
    ]
    return sum(values) / len(values) if values else 0.0


def average_bool_metric(results: list[EvaluatedResult], metric_name: str) -> float:
    values = [
        bool(value)
        for result in results
        if isinstance((value := result.metrics.get(metric_name)), bool)
    ]
    return sum(1 for value in values if value) / len(values) if values else 0.0


def percentile_metric(results: list[EvaluatedResult], metric_name: str, percentile: int) -> float:
    values = sorted(
        float(value)
        for result in results
        if isinstance((value := result.metrics.get(metric_name)), int | float)
    )
    if not values:
        return 0.0
    if percentile == 50:
        return float(median(values))
    rank = math.ceil((percentile / 100) * len(values)) - 1
    return values[min(max(rank, 0), len(values) - 1)]


def normalize_text(value: str) -> str:
    return " ".join(TOKEN_PATTERN.findall(value.lower()))


def _point_is_covered(expected_point: str, normalized_answer: str) -> bool:
    if expected_point in normalized_answer:
        return True
    point_terms = set(expected_point.split())
    answer_terms = set(normalized_answer.split())
    if not point_terms:
        return True
    return len(point_terms & answer_terms) / len(point_terms) >= 0.65
