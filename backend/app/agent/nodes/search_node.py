from app.agent.state import ResearchState
from app.services.search_service import SearchService


async def search_node(state: ResearchState) -> ResearchState:
    papers = await SearchService().search(state["query"], state["max_results"])
    return {**state, "papers": papers}

