from app.agent.models import append_trace
from app.agent.state import AgenticRAGState

MAX_RESULT_ITEMS = 3
MAX_TEXT_CHARS = 360


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
    trace_fields["tool_result"] = compact_tool_result(result)

    return {
        **state,
        "current_step_index": current_step_index + 1,
        "tool_results": [*state.get("tool_results", []), result],
        "trace": append_trace(state.get("trace", []), "observe", **trace_fields),
    }


def compact_tool_result(result) -> dict:
    payload = {
        "tool_name": result.tool_name,
        "success": result.success,
    }
    if result.error:
        payload["error"] = result.error
    if result.metadata:
        payload["metadata"] = _compact_value(result.metadata)
    if result.chunks is not None:
        payload["chunks"] = [_compact_chunk(chunk) for chunk in result.chunks[:MAX_RESULT_ITEMS]]
        payload["chunk_count"] = len(result.chunks)
    if result.artifacts is not None:
        payload["artifacts"] = [_compact_value(artifact) for artifact in result.artifacts[:MAX_RESULT_ITEMS]]
        payload["artifact_count"] = len(result.artifacts)
    return payload


def _compact_chunk(chunk: dict) -> dict:
    citation = chunk.get("citation") or {}
    metadata = chunk.get("metadata") or {}
    return {
        key: value
        for key, value in {
            "id": chunk.get("id") or citation.get("chunk_id") or metadata.get("chunk_id"),
            "title": citation.get("title") or metadata.get("title"),
            "url": citation.get("url") or metadata.get("url") or metadata.get("source_url"),
            "score": chunk.get("score"),
            "rerank_score": chunk.get("rerank_score"),
            "query_anchor_terms": chunk.get("query_anchor_terms"),
            "matched_anchor_terms": chunk.get("matched_anchor_terms"),
            "query_anchor_coverage": chunk.get("query_anchor_coverage"),
            "text": _compact_text(chunk.get("text") or citation.get("text") or ""),
        }.items()
        if value not in (None, "")
    }


def _compact_value(value):
    if isinstance(value, str):
        return _compact_text(value)
    if isinstance(value, list):
        return [_compact_value(item) for item in value[:MAX_RESULT_ITEMS]]
    if isinstance(value, dict):
        return {
            str(key): _compact_value(item)
            for key, item in value.items()
        }
    return value


def _compact_text(value: str) -> str:
    text = " ".join(str(value).split())
    if len(text) <= MAX_TEXT_CHARS:
        return text
    return f"{text[:MAX_TEXT_CHARS].rstrip()}..."
