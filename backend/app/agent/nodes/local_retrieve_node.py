from app.agent.models import append_trace
from app.agent.nodes.observer_node import compact_tool_result
from app.agent.state import AgenticRAGState
from app.agent.tools.local_retrieve_tool import local_retrieve_input


async def local_retrieve_node(state: AgenticRAGState) -> AgenticRAGState:
    request = state["request"]
    tool_registry = state["tool_registry"]
    query_plan = state.get("query_plan")
    search_queries = query_plan.search_queries if query_plan else [request.question]
    top_k = (
        getattr(query_plan, "per_query_top_k", None)
        if query_plan and getattr(query_plan, "per_query_top_k", None)
        else request.top_k
    )
    score_threshold = (
        getattr(query_plan, "score_threshold", None)
        if query_plan and getattr(query_plan, "score_threshold", None) is not None
        else request.score_threshold
    )
    results = []
    for query in search_queries:
        results.append(
            await tool_registry.run(
                "local_retrieve",
                local_retrieve_input(
                    question=query,
                    chat_id=request.chat_id,
                    paper_ids=request.paper_ids,
                    top_k=top_k,
                    score_threshold=score_threshold,
                    chat_history=request.chat_history if query == request.question else None,
                ),
            )
        )
    max_total_chunks = getattr(query_plan, "max_total_chunks", None) if query_plan else None
    result = _merge_local_retrieve_results(results, max_total_chunks=max_total_chunks)
    local_chunks = result.chunks or []
    trace_fields = {
        "chunk_count": len(local_chunks),
        "paper_ids": request.paper_ids,
        "success": result.success,
        "query_count": len(search_queries),
        "queries": search_queries,
        "retrieval_mode": getattr(query_plan, "retrieval_mode", "focused") if query_plan else "focused",
        "per_query_top_k": top_k,
        "score_threshold": score_threshold,
        "max_total_chunks": max_total_chunks,
        "tool_result": compact_tool_result(result),
    }
    metadata = result.metadata or {}
    if "embedding_model" in metadata:
        trace_fields["embedding_model"] = str(metadata["embedding_model"])
    if "embedding_input_count" in metadata:
        trace_fields["embedding_input_count"] = int(metadata["embedding_input_count"])
    if "embedding_tokens" in metadata:
        trace_fields["embedding_tokens"] = int(metadata["embedding_tokens"])
    if "embedding_estimated_cost_usd" in metadata:
        trace_fields["embedding_estimated_cost_usd"] = float(metadata["embedding_estimated_cost_usd"])
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


def _merge_local_retrieve_results(results, max_total_chunks: int | None = None):
    if len(results) == 1:
        return results[0]

    seen_chunk_ids = set()
    merged_chunks = []
    errors = []
    metadata = _merge_embedding_metadata(results)
    for result in results:
        if result.error:
            errors.append(result.error)
        for chunk in result.chunks or []:
            chunk_id = _chunk_id(chunk)
            if chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)
            merged_chunks.append(chunk)
            if max_total_chunks and len(merged_chunks) >= max_total_chunks:
                break
        if max_total_chunks and len(merged_chunks) >= max_total_chunks:
            break

    from app.agent.models import ToolResult

    return ToolResult(
        tool_name="local_retrieve",
        success=any(result.success for result in results),
        chunks=merged_chunks,
        error="; ".join(errors) if errors else None,
        metadata={
            **metadata,
            "chunk_count": len(merged_chunks),
            "query_count": len(results),
        },
    )


def _chunk_id(chunk: dict) -> str:
    citation = chunk.get("citation") or {}
    metadata = chunk.get("metadata") or {}
    return str(citation.get("chunk_id") or metadata.get("chunk_id") or chunk.get("id") or "")


def _merge_embedding_metadata(results) -> dict:
    metadata_items = [result.metadata or {} for result in results]
    models = []
    seen_models = set()
    for metadata in metadata_items:
        model = metadata.get("embedding_model")
        if model and model not in seen_models:
            seen_models.add(model)
            models.append(str(model))
    merged = {}
    if models:
        merged["embedding_model"] = ", ".join(models)
    embedding_input_count = sum(int(metadata.get("embedding_input_count") or 0) for metadata in metadata_items)
    embedding_tokens = sum(int(metadata.get("embedding_tokens") or 0) for metadata in metadata_items)
    embedding_cost = sum(float(metadata.get("embedding_estimated_cost_usd") or 0.0) for metadata in metadata_items)
    if embedding_input_count:
        merged["embedding_input_count"] = embedding_input_count
    if embedding_tokens:
        merged["embedding_tokens"] = embedding_tokens
    if embedding_cost:
        merged["embedding_estimated_cost_usd"] = embedding_cost
    return merged
