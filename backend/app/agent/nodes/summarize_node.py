from app.agent.state import ResearchState


async def summarize_node(state: ResearchState) -> ResearchState:
    return {**state, "summaries": []}

