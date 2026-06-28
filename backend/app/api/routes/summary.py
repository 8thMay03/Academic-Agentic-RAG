from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_summary_service
from app.models.summary import SummaryRequest, SummaryResponse
from app.services.summary_service import SummaryService

router = APIRouter()


@router.post("", response_model=SummaryResponse)
async def summarize_paper(
    request: SummaryRequest,
    summary_service: SummaryService = Depends(get_summary_service),
) -> SummaryResponse:
    try:
        summary = await summary_service.summarize(request.title, request.text)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return SummaryResponse(title=request.title, summary=summary)
