from __future__ import annotations

from typing import TYPE_CHECKING

from app.agent.state import AgenticRAGState

if TYPE_CHECKING:
    from app.services.agentic_chat_workflow import ContextQuality


async def quality_gate_node(state: AgenticRAGState) -> AgenticRAGState:
    workflow = state["workflow"]
    request = state["request"]
    local_chunks = state.get("local_chunks", [])
    quality: ContextQuality = await workflow._evaluate_context(request, local_chunks)
    trace = [
        *state.get("trace", []),
        {
            "stage": "quality_gate",
            "sufficient": quality.sufficient,
            "reason": quality.reason,
            "chunk_count": quality.chunk_count,
            "context_chars": quality.context_chars,
            "top_score": quality.top_score,
            "average_score": quality.average_score,
            "source_count": quality.source_count,
            "query_coverage": quality.query_coverage,
            "self_check_used": quality.self_check_used,
            "self_check_passed": quality.self_check_passed,
        },
    ]
    return {
        **state,
        "quality": quality,
        "trace": trace,
    }
