from app.agent.state import ResearchState


async def compare_node(state: ResearchState) -> ResearchState:
    return {**state, "comparison": ""}

