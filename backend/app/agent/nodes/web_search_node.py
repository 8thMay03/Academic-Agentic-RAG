from app.agent.state import AgenticRAGState


async def web_search_node(state: AgenticRAGState) -> AgenticRAGState:
    workflow = state["workflow"]
    request = state["request"]
    quality = state["quality"]
    web_chunks = await workflow._search_web(request)
    trace = [
        *state.get("trace", []),
        {
            "stage": "web_search",
            "chunk_count": len(web_chunks),
            "trigger": quality.reason,
        },
    ]
    return {
        **state,
        "web_chunks": web_chunks,
        "trace": trace,
    }
