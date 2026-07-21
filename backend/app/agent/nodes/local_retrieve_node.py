from app.agent.models import append_trace
from app.agent.nodes.observer_node import compact_tool_result
from app.agent.state import AgenticRAGState
from app.agent.tools.local_retrieve_tool import local_retrieve_input


async def local_retrieve_node(state: AgenticRAGState) -> AgenticRAGState:
    request = state["request"]
    tool_registry = state["tool_registry"]
    query_plan = state.get("query_plan")
    search_queries = query_plan.search_queries if query_plan else [request.question]
    results = []
    for query in search_queries:
        results.append(
            await tool_registry.run(
                "local_retrieve",
                local_retrieve_input(
                    question=query,
                    chat_id=request.chat_id,
                    paper_ids=request.paper_ids,
                    top_k=request.top_k,
                    score_threshold=request.score_threshold,
                    chat_history=request.chat_history if query == request.question else None,
                ),
            )
        )
    result = _merge_local_retrieve_results(results)
    local_chunks = result.chunks or []
    trace_fields = {
        "chunk_count": len(local_chunks),
        "paper_ids": request.paper_ids,
        "success": result.success,
        "query_count": len(search_queries),
        "queries": search_queries,
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


def _merge_local_retrieve_results(results):
    if len(results) == 1:
        return results[0]

    seen_chunk_ids = set()
    merged_chunks = []
    errors = []
    for result in results:
        if result.error:
            errors.append(result.error)
        for chunk in result.chunks or []:
            chunk_id = _chunk_id(chunk)
            if chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)
            merged_chunks.append(chunk)

    from app.agent.models import ToolResult

    return ToolResult(
        tool_name="local_retrieve",
        success=any(result.success for result in results),
        chunks=merged_chunks,
        error="; ".join(errors) if errors else None,
        metadata={
            "chunk_count": len(merged_chunks),
            "query_count": len(results),
        },
    )


def _chunk_id(chunk: dict) -> str:
    citation = chunk.get("citation") or {}
    metadata = chunk.get("metadata") or {}
    return str(citation.get("chunk_id") or metadata.get("chunk_id") or chunk.get("id") or "")
