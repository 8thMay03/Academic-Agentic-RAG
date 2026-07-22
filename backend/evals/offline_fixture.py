from __future__ import annotations

from evals.models import BaselineResult, EvalCase


class OfflineFixtureBaseline:
    """Deterministic baseline used to validate the evaluation harness without external services."""

    def __init__(self, mode: str) -> None:
        self.mode = mode

    async def run_case(self, case: EvalCase) -> BaselineResult:
        profile = _mode_profile(self.mode, case)
        if not profile["can_answer"]:
            return BaselineResult(
                case_id=case.id,
                mode=self.mode,
                answer="I don't know",
                citation_chunk_ids=[],
                retrieved_chunk_ids=[],
                trace=[{"stage": "offline_fixture", "reason": profile["reason"]}],
                latency_ms=profile["latency_ms"],
            )

        expected_points = list(case.expected_answer_points)
        covered_points = expected_points[: profile["point_count"]]
        citation_ids = case.expected_citation_chunk_ids[: profile["citation_count"]]
        retrieved_ids = case.expected_citation_chunk_ids[: profile["retrieval_count"]]
        if profile["add_invalid_citation"]:
            citation_ids = [*citation_ids, f"invalid:{case.id}:c0"]
        if profile["add_distractor_retrieval"]:
            retrieved_ids = [f"distractor:{case.id}:c0", *retrieved_ids]

        answer = _answer_from_points(covered_points, citation_ids)
        return BaselineResult(
            case_id=case.id,
            mode=self.mode,
            answer=answer,
            citation_chunk_ids=citation_ids,
            retrieved_chunk_ids=retrieved_ids,
            trace=[
                {
                    "stage": "offline_fixture",
                    "answer_type": case.answer_type,
                    "requires_fresh_context": case.requires_fresh_context,
                    "requires_multi_hop": case.requires_multi_hop,
                }
            ],
            latency_ms=profile["latency_ms"],
            input_tokens=profile["input_tokens"],
            output_tokens=profile["output_tokens"],
            estimated_cost_usd=profile["estimated_cost_usd"],
        )


def build_offline_fixture_baselines(modes: list[str]) -> list[OfflineFixtureBaseline]:
    return [OfflineFixtureBaseline(mode) for mode in modes]


def _mode_profile(mode: str, case: EvalCase) -> dict:
    profiles = {
        "vector_only_rag": _vector_only_profile,
        "hybrid_rag": _hybrid_profile,
        "hybrid_rerank_rag": _hybrid_rerank_profile,
        "full_agentic_rag": _full_agentic_profile,
    }
    return profiles[mode](case)


def _vector_only_profile(case: EvalCase) -> dict:
    can_answer = case.is_answerable and not case.requires_fresh_context and not case.chat_history
    point_count = 1 if can_answer else 0
    if case.requires_multi_hop:
        point_count = min(point_count, 1)
    return _profile(
        can_answer=can_answer,
        point_count=point_count,
        citation_count=1 if can_answer else 0,
        retrieval_count=1 if can_answer else 0,
        add_invalid_citation=case.answer_type == "adversarial" and can_answer,
        add_distractor_retrieval=True,
        latency_ms=180,
        input_tokens=700,
        output_tokens=90,
        cost=0.00018,
        reason="vector_only_cannot_recover",
    )


def _hybrid_profile(case: EvalCase) -> dict:
    can_answer = case.is_answerable and not case.requires_fresh_context
    point_count = min(len(case.expected_answer_points), 1 if case.requires_multi_hop else 2)
    return _profile(
        can_answer=can_answer,
        point_count=point_count,
        citation_count=min(len(case.expected_citation_chunk_ids), 1 if case.requires_multi_hop else 2),
        retrieval_count=min(len(case.expected_citation_chunk_ids), 2),
        add_invalid_citation=False,
        add_distractor_retrieval=case.requires_multi_hop,
        latency_ms=260,
        input_tokens=950,
        output_tokens=120,
        cost=0.00026,
        reason="hybrid_cannot_handle_fresh_context",
    )


def _hybrid_rerank_profile(case: EvalCase) -> dict:
    can_answer = case.is_answerable and not case.requires_fresh_context
    point_count = len(case.expected_answer_points)
    if case.answer_type == "adversarial":
        point_count = max(1, len(case.expected_answer_points) - 1)
    return _profile(
        can_answer=can_answer,
        point_count=point_count,
        citation_count=len(case.expected_citation_chunk_ids),
        retrieval_count=len(case.expected_citation_chunk_ids),
        add_invalid_citation=case.answer_type == "adversarial",
        add_distractor_retrieval=False,
        latency_ms=430,
        input_tokens=1300,
        output_tokens=150,
        cost=0.00039,
        reason="hybrid_rerank_no_external_recovery",
    )


def _full_agentic_profile(case: EvalCase) -> dict:
    can_answer = case.is_answerable
    point_count = len(case.expected_answer_points)
    citation_count = len(case.expected_citation_chunk_ids)
    retrieval_count = len(case.expected_citation_chunk_ids)
    if case.answer_type == "adversarial" and _case_number(case.id) % 5 == 0:
        point_count = max(1, point_count - 1)
    if case.answer_type == "fresh_research" and _case_number(case.id) == 10:
        can_answer = False
        point_count = 0
        citation_count = 0
        retrieval_count = 0
    return _profile(
        can_answer=can_answer,
        point_count=point_count,
        citation_count=citation_count,
        retrieval_count=retrieval_count,
        add_invalid_citation=False,
        add_distractor_retrieval=False,
        latency_ms=780 if case.requires_fresh_context or case.requires_multi_hop else 620,
        input_tokens=1850 if case.requires_fresh_context else 1600,
        output_tokens=190,
        cost=0.00062 if case.requires_fresh_context else 0.00053,
        reason="agentic_abstains_or_recovers",
    )


def _profile(
    can_answer: bool,
    point_count: int,
    citation_count: int,
    retrieval_count: int,
    add_invalid_citation: bool,
    add_distractor_retrieval: bool,
    latency_ms: float,
    input_tokens: int,
    output_tokens: int,
    cost: float,
    reason: str,
) -> dict:
    return {
        "can_answer": can_answer,
        "point_count": point_count,
        "citation_count": citation_count,
        "retrieval_count": retrieval_count,
        "add_invalid_citation": add_invalid_citation,
        "add_distractor_retrieval": add_distractor_retrieval,
        "latency_ms": latency_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": cost,
        "reason": reason,
    }


def _answer_from_points(points: list[str], citation_ids: list[str]) -> str:
    if not points:
        return "I don't know"
    if not citation_ids:
        return " ".join(points)
    sentences = []
    for index, point in enumerate(points):
        citation_id = citation_ids[min(index, len(citation_ids) - 1)]
        sentences.append(f"{point} [{citation_id}].")
    return " ".join(sentences)


def _case_number(case_id: str) -> int:
    try:
        return int(case_id.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return 0
