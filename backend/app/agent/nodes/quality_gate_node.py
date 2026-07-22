from __future__ import annotations

from app.agent.models import ContextQuality, append_trace
from app.agent.models import retrieved_chunk_ranking_score, retrieved_chunk_source_id, retrieved_chunk_text
from app.agent.state import AgenticRAGState


async def quality_gate_node(state: AgenticRAGState) -> AgenticRAGState:
    request = state["request"]
    local_chunks = state.get("local_chunks", [])
    web_chunks = state.get("web_chunks", [])
    if web_chunks or state.get("refreshed_local_context"):
        quality = _external_context_quality(state.get("evidence", []) or [*local_chunks, *web_chunks])
    else:
        evaluator = state["quality_evaluator"]
        quality = await evaluator.evaluate(request.question, local_chunks)
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


def _external_context_quality(chunks: list[dict]) -> ContextQuality:
    scores = [
        score
        for chunk in chunks
        if (score := retrieved_chunk_ranking_score(chunk)) is not None
    ]
    source_ids = {
        source_id
        for chunk in chunks
        if (source_id := retrieved_chunk_source_id(chunk))
    }
    context_chars = sum(len(retrieved_chunk_text(chunk)) for chunk in chunks)
    return ContextQuality(
        sufficient=bool(chunks),
        chunk_count=len(chunks),
        context_chars=context_chars,
        reason="external_context_available" if chunks else "no_external_context",
        top_score=max(scores) if scores else None,
        average_score=sum(scores) / len(scores) if scores else None,
        source_count=len(source_ids),
        query_coverage=1.0 if chunks else 0.0,
    )
