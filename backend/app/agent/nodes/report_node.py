from app.agent.state import ResearchState


async def report_node(state: ResearchState) -> ResearchState:
    return {**state, "report": ""}

