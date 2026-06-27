from fastapi import APIRouter

from app.models.report import ReportRequest, ReportResponse

router = APIRouter()


@router.post("", response_model=ReportResponse)
async def create_report(request: ReportRequest) -> ReportResponse:
    return ReportResponse(title=request.title, content="Report generation is not implemented yet.")

