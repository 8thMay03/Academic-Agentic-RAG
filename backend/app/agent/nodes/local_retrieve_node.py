from app.agent.state import AgenticRAGState


async def local_retrieve_node(state: AgenticRAGState) -> AgenticRAGState:
    workflow = state["workflow"]
    request = state["request"]
    local_chunks = await workflow._retrieve_local(request)
    trace = [
        *state.get("trace", []),
        {
            "stage": "local_retrieve",
            "chunk_count": len(local_chunks),
            "paper_ids": request.paper_ids,
        },
    ]
    return {
        **state,
        "local_chunks": local_chunks,
        "trace": trace,
    }
