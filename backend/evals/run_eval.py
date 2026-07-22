from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from evals.baselines import all_modes, build_default_baselines  # noqa: E402
from evals.local_fixture import build_local_fixture_baselines  # noqa: E402
from evals.metrics import evaluate_result, summarize_results  # noqa: E402
from evals.models import BaselineResult, EvalCase, EvaluatedResult  # noqa: E402
from evals.offline_fixture import build_offline_fixture_baselines  # noqa: E402
from evals.report import write_markdown_report  # noqa: E402
from app.config.settings import settings  # noqa: E402


DEFAULT_DATASET = Path("evals/datasets/agentic_rag_eval.jsonl")
DEFAULT_OUTPUT = Path("evals/results/latest_results.json")


async def main() -> None:
    args = parse_args()
    modes = all_modes() if args.mode == "all" else [args.mode]
    cases = load_dataset(args.dataset)
    if args.limit:
        cases = cases[: args.limit]
    if args.profile == "live" and not args.skip_live_preflight:
        require_live_profile_ready(cases, modes)
    baselines = build_baselines(modes, args.profile)
    evaluated_results: list[EvaluatedResult] = []

    for baseline in baselines:
        for case in cases:
            result = await run_case_with_timeout(baseline, case, args.case_timeout_seconds)
            evaluated_results.append(evaluate_result(case, result))

    payload = {
        "dataset": str(args.dataset),
        "case_count": len(cases),
        "modes": modes,
        "summary": summarize_results(evaluated_results),
        "results": [result.to_json() for result in evaluated_results],
    }
    write_results(args.output, payload)
    if args.report_output:
        write_markdown_report(args.report_output, payload)
    print_summary(payload["summary"])
    print(f"\nSaved detailed results to {args.output}")
    if args.report_output:
        print(f"Saved markdown report to {args.report_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAG baseline evaluation.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=None)
    parser.add_argument("--mode", choices=[*all_modes(), "all"], default="all")
    parser.add_argument(
        "--profile",
        choices=["live", "offline_fixture", "local_fixture"],
        default="live",
        help="Use live services, deterministic offline fixtures, or a temp indexed local corpus fixture.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N cases when greater than 0.")
    parser.add_argument("--case-timeout-seconds", type=float, default=30.0)
    parser.add_argument(
        "--skip-live-preflight",
        action="store_true",
        help="Bypass live profile checks for custom CI or intentionally partial benchmark runs.",
    )
    return parser.parse_args()


def build_baselines(modes: list[str], profile: str):
    if profile == "offline_fixture":
        return build_offline_fixture_baselines(modes)
    if profile == "local_fixture":
        return build_local_fixture_baselines(modes)
    return build_default_baselines(modes)


def require_live_profile_ready(cases: list[EvalCase], modes: list[str]) -> None:
    errors = live_preflight_errors(cases, modes)
    if not errors:
        return
    details = "\n".join(f"- {error}" for error in errors)
    raise SystemExit(
        "Live evaluation preflight failed.\n"
        f"{details}\n"
        "Use --profile offline_fixture or --profile local_fixture for deterministic no-API runs, "
        "or pass --skip-live-preflight only when you intentionally want a partial live run."
    )


def live_preflight_errors(cases: list[EvalCase], modes: list[str]) -> list[str]:
    errors = []
    if not settings.OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY is required for live LLM and embedding calls.")

    chroma_dir = _backend_relative_path(Path(settings.CHROMA_DIR))
    if not chroma_dir.exists():
        errors.append(f"CHROMA_DIR does not exist: {chroma_dir}")
    elif not (chroma_dir / "chroma.sqlite3").exists():
        errors.append(f"CHROMA_DIR has no chroma.sqlite3 corpus: {chroma_dir}")

    fresh_case_count = sum(1 for case in cases if case.requires_fresh_context)
    if fresh_case_count and "full_agentic_rag" in modes and not settings.TAVILY_API_KEY:
        errors.append(
            f"{fresh_case_count} fresh-context cases will need external search; set TAVILY_API_KEY or run a non-live fixture profile."
        )
    return errors


def _backend_relative_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return BACKEND_ROOT / path


def load_dataset(path: Path) -> list[EvalCase]:
    cases = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped_line = line.strip()
            if not stripped_line or stripped_line.startswith("#"):
                continue
            try:
                cases.append(EvalCase.from_json(json.loads(stripped_line)))
            except Exception as exc:
                raise ValueError(f"Invalid eval case at {path}:{line_number}: {exc}") from exc
    return cases


async def run_case_with_timeout(baseline: Any, case: EvalCase, timeout_seconds: float) -> BaselineResult:
    try:
        return await asyncio.wait_for(baseline.run_case(case), timeout=timeout_seconds)
    except TimeoutError:
        return BaselineResult(
            case_id=case.id,
            mode=baseline.mode,
            answer="",
            citation_chunk_ids=[],
            retrieved_chunk_ids=[],
            error=f"case timed out after {timeout_seconds:g}s",
        )


def write_results(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def print_summary(summary: list[dict[str, Any]]) -> None:
    if not summary:
        print("No results.")
        return
    columns = [
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
        "p50_latency_ms",
        "p95_latency_ms",
        "error_count",
    ]
    widths = {
        column: max(len(column), *(len(format_value(row[column])) for row in summary))
        for column in columns
    }
    print(" | ".join(column.ljust(widths[column]) for column in columns))
    print("-+-".join("-" * widths[column] for column in columns))
    for row in summary:
        print(" | ".join(format_value(row[column]).ljust(widths[column]) for column in columns))


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


if __name__ == "__main__":
    asyncio.run(main())
