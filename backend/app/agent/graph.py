from app.models.research import ResearchRequest, ResearchResponse
from app.services.search_service import SearchService


async def run_research_workflow(request: ResearchRequest) -> ResearchResponse:
    # TODO: Replace this linear stub with a LangGraph StateGraph.
    papers = await SearchService().search(request.query, request.max_results)
    return ResearchResponse(query=request.query, papers=papers)

