from app.agent.models import append_trace
from app.agent.security import mark_suspicious_chunks
from app.agent.state import AgenticRAGState


async def draft_answer_node(state: AgenticRAGState) -> AgenticRAGState:
    request = state["request"]
    quality = state["quality"]
    citation_grounder = state["citation_grounder"]
    prompt_builder = state["prompt_builder"]
    evidence = state.get("evidence", [])
    local_chunks = state.get("local_chunks", [])
    web_chunks = state.get("web_chunks", [])

    if web_chunks or state.get("refreshed_local_context"):
        chunks = evidence or [*local_chunks, *web_chunks]
    elif quality.sufficient:
        chunks = evidence or local_chunks
    else:
        chunks = []

    if not chunks:
        trace = append_trace(state.get("trace", []), "draft_answer", status="no_context")
        return {
            **state,
            "prompt": None,
            "citations": [],
            "trace": trace,
        }

    chunks = mark_suspicious_chunks(chunks)
    suspicious_context_count = sum(
        1
        for chunk in chunks
        if (chunk.get("metadata") or {}).get("security_flag") == "suspicious_instruction"
    )
    citations = citation_grounder.build_citations(chunks, request.question)
    trace_fields = {
        "status": "ready",
        "context_count": len(chunks),
        "citation_count": len(citations),
    }
    if suspicious_context_count:
        trace_fields["suspicious_context_count"] = suspicious_context_count
    trace = append_trace(
        state.get("trace", []),
        "draft_answer",
        **trace_fields,
    )
    return {
        **state,
        "prompt": prompt_builder.build(request.question, chunks, request.chat_history),
        "citations": citations,
        "trace": trace,
    }
