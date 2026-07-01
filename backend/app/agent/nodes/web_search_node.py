from app.agent.state import AgenticRAGState


async def web_search_node(state: AgenticRAGState) -> AgenticRAGState:
    workflow = state["workflow"]
    request = state["request"]
    quality = state["quality"]
    web_chunks, web_papers = await workflow._search_web(request)
    trace = [
        *state.get("trace", []),
        {
            "stage": "web_search",
            "chunk_count": len(web_chunks),
            "paper_count": len(web_papers),
            "trigger": quality.reason,
        },
    ]
    return {
        **state,
        "web_chunks": web_chunks,
        "web_papers": web_papers,
        "trace": trace,
    }
