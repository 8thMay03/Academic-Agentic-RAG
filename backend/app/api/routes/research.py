from fastapi import APIRouter

from app.agent.graph import run_research_workflow
from app.models.research import ResearchRequest, ResearchResponse

router = APIRouter()


@router.post("", response_model=ResearchResponse)
async def run_research(request: ResearchRequest) -> ResearchResponse:
    return await run_research_workflow(request)

