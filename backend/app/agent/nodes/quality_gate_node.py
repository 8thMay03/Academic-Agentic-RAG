from __future__ import annotations

from app.agent.models import ContextQuality, append_trace
from app.agent.state import AgenticRAGState


async def quality_gate_node(state: AgenticRAGState) -> AgenticRAGState:
    request = state["request"]
    local_chunks = state.get("local_chunks", [])
    evaluator = state["quality_evaluator"]
    quality: ContextQuality = await evaluator.evaluate(request.question, local_chunks)
    trace = append_trace(
        state.get("trace", []),
        "quality_gate",
        sufficient=quality.sufficient,
        reason=quality.reason,
        chunk_count=quality.chunk_count,
        context_chars=quality.context_chars,
        top_score=quality.top_score,
        average_score=quality.average_score,
        source_count=quality.source_count,
        query_coverage=quality.query_coverage,
        self_check_used=quality.self_check_used,
        self_check_passed=quality.self_check_passed,
    )
    return {
        **state,
        "quality": quality,
        "trace": trace,
    }
