from app.agent.models import append_trace
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

    citations = citation_grounder.build_citations(chunks, request.question)
    trace = append_trace(
        state.get("trace", []),
        "draft_answer",
        status="ready",
        context_count=len(chunks),
        citation_count=len(citations),
    )
    return {
        **state,
        "prompt": prompt_builder.build(request.question, chunks, request.chat_history),
        "citations": citations,
        "trace": trace,
    }
