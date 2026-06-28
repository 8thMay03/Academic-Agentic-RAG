import asyncio
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_pdf_service
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
from app.parser.chunker import chunk_text_with_metadata
from app.parser.cleaner import PAGE_BREAK, clean_text
from app.parser.pdf_parser import extract_text_from_pdf
from app.services.pdf_service import PDFDownloadError, PDFService
from app.vectorstore.indexing import index_chunks

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
async def index_downloaded_pdf(request: DownloadedPDFIndexRequest) -> DownloadedPDFIndexResponse:
    destination_dir = Path(settings.DATA_DIR) / "pdfs"
    filename = Path(request.filename).name
    pdf_path = destination_dir / filename

    if not filename.lower().endswith(".pdf") or not pdf_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Downloaded PDF not found: {request.filename}",
        )

    paper_id = Path(filename).stem
    raw_text = await asyncio.to_thread(extract_text_from_pdf, pdf_path)
    cleaned_pages = [
        clean_text(page)
        for page in raw_text.split(PAGE_BREAK)
        if page.strip()
    ]
    cleaned_text = PAGE_BREAK.join(page for page in cleaned_pages if page)
    chunks = chunk_text_with_metadata(cleaned_text, paper_id=paper_id)
    for chunk in chunks:
        chunk.metadata.update(
            {
                "title": filename,
                "source_path": pdf_path.as_posix(),
            }
        )

    await index_chunks(chunks)
    return DownloadedPDFIndexResponse(
        paper_id=paper_id,
        filename=filename,
        chunks_indexed=len(chunks),
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
