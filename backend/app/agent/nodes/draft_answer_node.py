from app.agent.models import append_trace
from app.agent.state import AgenticRAGState


async def draft_answer_node(state: AgenticRAGState) -> AgenticRAGState:
    request = state["request"]
    quality = state["quality"]
    citation_grounder = state["citation_grounder"]
    prompt_builder = state["prompt_builder"]
    evidence = state.get("evidence", [])
    if evidence:
        chunks = evidence
    elif quality.sufficient:
        chunks = state.get("local_chunks", [])
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
