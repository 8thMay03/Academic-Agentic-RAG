from app.agent.state import AgenticRAGState


async def answer_node(state: AgenticRAGState) -> AgenticRAGState:
    workflow = state["workflow"]
    request = state["request"]
    quality = state["quality"]
    local_chunks = state.get("local_chunks", [])
    web_chunks = state.get("web_chunks", [])

    chunks = local_chunks if quality.sufficient else [*local_chunks, *web_chunks]
    if not quality.sufficient and not web_chunks:
        chunks = []

    if not chunks:
        trace = [
            *state.get("trace", []),
            {"stage": "answer", "status": "no_context"},
        ]
        return {
            **state,
            "prompt": None,
            "citations": [],
            "trace": trace,
        }

    citations = workflow._citations(chunks, request.question)
    trace = [
        *state.get("trace", []),
        {
            "stage": "answer",
            "status": "ready",
            "context_count": len(chunks),
            "citation_count": len(citations),
        },
    ]
    return {
        **state,
        "prompt": workflow._build_prompt(request.question, chunks, request.chat_history),
        "citations": citations,
        "trace": trace,
    }
