from fastapi import APIRouter, Depends

from app.api.dependencies import get_compare_service
from app.models.compare import CompareRequest, CompareResponse
from app.services.compare_service import CompareService

router = APIRouter()


@router.post("", response_model=CompareResponse)
async def compare_papers(
    request: CompareRequest,
    compare_service: CompareService = Depends(get_compare_service),
) -> CompareResponse:
    result = await compare_service.compare(request.papers)
    return CompareResponse(comparison=result)

