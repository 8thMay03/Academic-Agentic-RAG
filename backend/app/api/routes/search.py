from fastapi import APIRouter, Depends

from app.api.dependencies import get_search_service
from app.models.paper import PaperSearchRequest, PaperSearchResponse
from app.services.search_service import SearchService

router = APIRouter()


@router.post("", response_model=PaperSearchResponse)
async def search_papers(
    request: PaperSearchRequest,
    search_service: SearchService = Depends(get_search_service),
) -> PaperSearchResponse:
    papers = await search_service.search(request.query, request.max_results)
    return PaperSearchResponse(query=request.query, papers=papers)

