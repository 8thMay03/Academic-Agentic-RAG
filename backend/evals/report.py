from __future__ import annotations

from pathlib import Path
from typing import Any


PRIMARY_MODE = "full_agentic_rag"
COMPARISON_MODE = "hybrid_rerank_rag"


SUMMARY_COLUMNS = [
    "mode",
    "case_count",
    "answer_point_recall",
    "citation_precision",
    "citation_recall",
    "retrieval_recall",
    "context_precision",
    "abstention_accuracy",
    "unsupported_claim_rate",
    "claim_citation_coverage",
    "verifier_availability_rate",
    "avg_tool_calls",
    "web_arxiv_usage_rate",
    "p50_latency_ms",
    "p95_latency_ms",
    "avg_estimated_cost_usd",
    "error_count",
]


def build_markdown_report(payload: dict[str, Any]) -> str:
    summary = list(payload.get("summary") or [])
    results = list(payload.get("results") or [])
    lines = [
        "# Agentic RAG Evaluation Report",
        "",
        "## Run Metadata",
        "",
        f"- Dataset: `{_display_path(payload.get('dataset', ''))}`",
        f"- Cases: {payload.get('case_count', 0)}",
        f"- Modes: {', '.join(str(mode) for mode in payload.get('modes', []))}",
        "",
        "## Baseline Summary",
        "",
        _markdown_table(summary, SUMMARY_COLUMNS),
        "",
        "## Agentic Delta",
        "",
        *_agentic_delta_lines(summary),
        "",
        "## Error Analysis",
        "",
        *_error_analysis_lines(results),
        "",
        "## Interpretation Notes",
        "",
        "- Treat fixture profiles as harness/instrumentation checks, not live model-quality claims.",
        "- Use live profile results on an indexed paper corpus before making portfolio claims.",
        "- Compare `full_agentic_rag` against `hybrid_rerank_rag` to isolate the value of planning, tool recovery, and verification.",
        "- Interpret verifier-only metrics together with `verifier_availability_rate`; a zero availability means unsupported-claim and claim-citation metrics were not measured for that mode.",
        "",
    ]
    return "\n".join(lines)


def write_markdown_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_markdown_report(payload), encoding="utf-8")


def _agentic_delta_lines(summary: list[dict[str, Any]]) -> list[str]:
    by_mode = {str(row.get("mode")): row for row in summary}
    primary = by_mode.get(PRIMARY_MODE)
    comparison = by_mode.get(COMPARISON_MODE)
    if not primary or not comparison:
        return [
            f"- Delta unavailable because `{PRIMARY_MODE}` or `{COMPARISON_MODE}` is missing.",
        ]

    metrics = [
        ("answer_point_recall", "answer point recall"),
        ("citation_precision", "citation precision"),
        ("citation_recall", "citation recall"),
        ("retrieval_recall", "retrieval recall"),
        ("context_precision", "context precision"),
        ("abstention_accuracy", "abstention accuracy"),
        ("claim_citation_coverage", "claim citation coverage"),
    ]
    lines = []
    for key, label in metrics:
        lines.append(f"- {label}: {_delta(primary, comparison, key)}")
    lines.append(
        "- unsupported claim rate: "
        f"{_delta(primary, comparison, 'unsupported_claim_rate')}"
    )
    lines.append(f"- average tool calls: {_delta(primary, comparison, 'avg_tool_calls')}")
    lines.append(f"- p95 latency: {_delta(primary, comparison, 'p95_latency_ms', suffix=' ms')}")
    lines.append(
        "- average estimated cost: "
        f"{_delta(primary, comparison, 'avg_estimated_cost_usd', prefix='$', precision=6)}"
    )
    return lines


def _error_analysis_lines(results: list[dict[str, Any]], limit: int = 10) -> list[str]:
    failing = [
        result
        for result in results
        if _result_has_quality_issue(result)
    ]
    if not failing:
        return ["- No failing or low-quality cases were detected by the current metrics."]

    lines = [
        f"- Failing/low-quality cases detected: {len(failing)}",
        "",
        "| Case | Mode | Issues |",
        "| --- | --- | --- |",
    ]
    for result in failing[:limit]:
        metrics = result.get("metrics") or {}
        issues = _quality_issues(metrics)
        lines.append(
            "| "
            f"`{result.get('case_id', '')}` | "
            f"`{result.get('mode', '')}` | "
            f"{', '.join(issues)} |"
        )
    if len(failing) > limit:
        lines.append(f"- Additional cases omitted: {len(failing) - limit}")
    return lines


def _result_has_quality_issue(result: dict[str, Any]) -> bool:
    return bool(_quality_issues(result.get("metrics") or {}))


def _quality_issues(metrics: dict[str, Any]) -> list[str]:
    issues = []
    if metrics.get("error"):
        issues.append("runtime error")
    if float(metrics.get("answer_point_recall") or 0.0) < 1.0:
        issues.append("missing expected answer points")
    if float(metrics.get("citation_precision") or 0.0) < 1.0:
        issues.append("invalid citation")
    if float(metrics.get("citation_recall") or 0.0) < 1.0:
        issues.append("missing expected citation")
    if float(metrics.get("retrieval_recall") or 0.0) < 1.0:
        issues.append("retrieval miss")
    if metrics.get("abstention_correct") is False:
        issues.append("wrong abstention behavior")
    return issues


def _delta(
    primary: dict[str, Any],
    comparison: dict[str, Any],
    key: str,
    prefix: str = "",
    suffix: str = "",
    precision: int = 3,
) -> str:
    primary_value = float(primary.get(key) or 0.0)
    comparison_value = float(comparison.get(key) or 0.0)
    delta = primary_value - comparison_value
    sign = "+" if delta >= 0 else ""
    return (
        f"{prefix}{primary_value:.{precision}f}{suffix} vs "
        f"{prefix}{comparison_value:.{precision}f}{suffix} ({sign}{delta:.{precision}f}{suffix})"
    )


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No summary rows available._"
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(_format_cell(row.get(column), column) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def _format_cell(value: Any, column: str) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if column == "avg_estimated_cost_usd":
            return f"{value:.6f}"
        return f"{value:.3f}"
    return str(value)


def _display_path(value: Any) -> str:
    return str(value).replace("\\", "/")
