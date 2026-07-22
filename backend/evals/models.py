from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvalCase:
    id: str
    question: str
    language: str = "en"
    paper_ids: list[str] | None = None
    answer_type: str = "factual"
    expected_answer_points: list[str] = field(default_factory=list)
    expected_citation_chunk_ids: list[str] = field(default_factory=list)
    is_answerable: bool = True
    requires_fresh_context: bool = False
    requires_multi_hop: bool = False
    chat_history: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "EvalCase":
        return cls(
            id=str(payload["id"]),
            question=str(payload["question"]),
            language=str(payload.get("language") or "en"),
            paper_ids=list(payload["paper_ids"]) if payload.get("paper_ids") else None,
            answer_type=str(payload.get("answer_type") or "factual"),
            expected_answer_points=[str(point) for point in payload.get("expected_answer_points", [])],
            expected_citation_chunk_ids=[
                str(chunk_id) for chunk_id in payload.get("expected_citation_chunk_ids", [])
            ],
            is_answerable=bool(payload.get("is_answerable", True)),
            requires_fresh_context=bool(payload.get("requires_fresh_context", False)),
            requires_multi_hop=bool(payload.get("requires_multi_hop", False)),
            chat_history=list(payload.get("chat_history") or []),
        )


@dataclass(frozen=True)
class BaselineResult:
    case_id: str
    mode: str
    answer: str
    citation_chunk_ids: list[str]
    retrieved_chunk_ids: list[str]
    trace: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    error: str | None = None

    @property
    def answered_unknown(self) -> bool:
        return self.answer.strip() == "I don't know"


@dataclass(frozen=True)
class EvaluatedResult:
    case_id: str
    mode: str
    metrics: dict[str, float | int | str | bool | None]
    result: BaselineResult

    def to_json(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "mode": self.mode,
            "metrics": self.metrics,
            "answer": self.result.answer,
            "citation_chunk_ids": self.result.citation_chunk_ids,
            "retrieved_chunk_ids": self.result.retrieved_chunk_ids,
            "latency_ms": self.result.latency_ms,
            "input_tokens": self.result.input_tokens,
            "output_tokens": self.result.output_tokens,
            "estimated_cost_usd": self.result.estimated_cost_usd,
            "error": self.result.error,
        }

