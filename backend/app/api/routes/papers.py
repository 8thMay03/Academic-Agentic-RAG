from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_pdf_service
from app.config.settings import settings
from app.models.paper import Paper, PaperDownloadRequest, PaperDownloadResponse
from app.services.pdf_service import PDFDownloadError, PDFService

router = APIRouter()


@router.get("", response_model=list[Paper])
async def list_papers() -> list[Paper]:
    return []


@router.post("/download", response_model=PaperDownloadResponse)
async def download_papers(
    request: PaperDownloadRequest,
    pdf_service: PDFService = Depends(get_pdf_service),
) -> PaperDownloadResponse:
    destination_dir = Path(settings.DATA_DIR) / "pdfs"

    try:
        results = await pdf_service.download_pdf_results(
            [str(pdf_url) for pdf_url in request.pdf_urls],
            destination_dir,
        )
    except PDFDownloadError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    files = [result.path.as_posix() for result in results]
    cached_files = [result.path.as_posix() for result in results if result.cached]
    return PaperDownloadResponse(files=files, cached_files=cached_files)
