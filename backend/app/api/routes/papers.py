from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_pdf_index_service, get_pdf_service
from app.config.settings import settings
from app.models.paper import (
    DownloadedPDF,
    DownloadedPDFIndexRequest,
    DownloadedPDFIndexResponse,
    Paper,
    PaperDownloadFailure,
    PaperDownloadRequest,
    PaperDownloadResponse,
)
from app.services.pdf_index_service import PDFIndexService
from app.services.pdf_service import PDFDownloadError, PDFService

router = APIRouter()


@router.get("", response_model=list[Paper])
async def list_papers() -> list[Paper]:
    return []


@router.get("/pdfs", response_model=list[DownloadedPDF])
async def list_downloaded_pdfs() -> list[DownloadedPDF]:
    destination_dir = Path(settings.DATA_DIR) / "pdfs"
    if not destination_dir.exists():
        return []

    pdf_files = sorted(
        destination_dir.glob("*.pdf"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return [
        DownloadedPDF(
            filename=path.name,
            path=path.as_posix(),
            size_bytes=path.stat().st_size,
            modified_at=datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat(),
        )
        for path in pdf_files
        if path.is_file()
    ]


@router.post("/pdfs/index", response_model=DownloadedPDFIndexResponse)
async def index_downloaded_pdf(
    request: DownloadedPDFIndexRequest,
    pdf_index_service: PDFIndexService = Depends(get_pdf_index_service),
) -> DownloadedPDFIndexResponse:
    try:
        result = await pdf_index_service.index_downloaded_pdf(request.filename)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Downloaded PDF not found: {request.filename}",
        ) from exc

    return DownloadedPDFIndexResponse(
        paper_id=result.paper_id,
        filename=result.filename,
        chunks_indexed=result.chunks_indexed,
    )


@router.post("/download", response_model=PaperDownloadResponse)
async def download_papers(
    request: PaperDownloadRequest,
    pdf_service: PDFService = Depends(get_pdf_service),
) -> PaperDownloadResponse:
    destination_dir = Path(settings.DATA_DIR) / "pdfs"

    results = []
    errors: list[PaperDownloadFailure] = []

    for pdf_url in request.pdf_urls:
        pdf_url_value = str(pdf_url)
        try:
            results.append(await pdf_service.download_pdf_result(pdf_url_value, destination_dir))
        except PDFDownloadError as exc:
            errors.append(PaperDownloadFailure(pdf_url=pdf_url, error=str(exc)))

    if not results and errors:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=[error.model_dump(mode="json") for error in errors],
        )

    files = [result.path.as_posix() for result in results]
    cached_files = [result.path.as_posix() for result in results if result.cached]
    return PaperDownloadResponse(files=files, cached_files=cached_files, errors=errors)
