from app.agent.models import append_trace
from app.agent.nodes.observer_node import compact_tool_result
from app.agent.state import AgenticRAGState
from app.agent.tools.local_retrieve_tool import local_retrieve_input


async def local_retrieve_node(state: AgenticRAGState) -> AgenticRAGState:
    request = state["request"]
    tool_registry = state["tool_registry"]
    result = await tool_registry.run(
        "local_retrieve",
        local_retrieve_input(
            question=request.question,
            chat_id=request.chat_id,
            paper_ids=request.paper_ids,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            chat_history=request.chat_history,
        ),
    )
    local_chunks = result.chunks or []
    trace_fields = {
        "chunk_count": len(local_chunks),
        "paper_ids": request.paper_ids,
        "success": result.success,
        "tool_result": compact_tool_result(result),
    }
    if result.error:
        trace_fields["reason"] = result.error
    trace = append_trace(
        state.get("trace", []),
        "local_retrieve",
        **trace_fields,
    )
    return {
        **state,
        "local_chunks": local_chunks,
        "evidence": local_chunks,
        "trace": trace,
    }
