from app.agent.models import append_trace
from app.agent.state import AgenticRAGState


async def observer_node(state: AgenticRAGState) -> AgenticRAGState:
    result = state["current_tool_result"]
    current_step_index = state.get("current_step_index", 0)
    metadata = result.metadata or {}
    trace_fields = {
        "tool_name": result.tool_name,
        "step_index": current_step_index,
        "success": result.success,
    }
    if result.chunks is not None:
        trace_fields["chunk_count"] = len(result.chunks)
    if result.artifacts is not None:
        trace_fields["artifact_count"] = len(result.artifacts)
    if "paper_count" in metadata:
        trace_fields["paper_count"] = int(metadata["paper_count"])
    if "chunks_indexed" in metadata:
        trace_fields["chunks_indexed"] = int(metadata["chunks_indexed"])
    if "snippets_ingested" in metadata:
        trace_fields["snippets_ingested"] = int(metadata["snippets_ingested"])
    for key in (
        "source_type",
        "source_url",
        "pdf_url",
        "discovered_by_query",
        "trust_level",
        "ingestion_status",
    ):
        if metadata.get(key) is not None:
            trace_fields[key] = str(metadata[key])
    if result.error:
        trace_fields["reason"] = result.error

    return {
        **state,
        "current_step_index": current_step_index + 1,
        "tool_results": [*state.get("tool_results", []), result],
        "trace": append_trace(state.get("trace", []), "observe", **trace_fields),
    }
