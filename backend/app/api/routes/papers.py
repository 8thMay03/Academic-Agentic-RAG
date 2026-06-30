from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

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


@router.post("/pdfs/upload", response_model=list[DownloadedPDF], status_code=status.HTTP_201_CREATED)
async def upload_local_pdfs(files: list[UploadFile] = File(...)) -> list[DownloadedPDF]:
    destination_dir = Path(settings.DATA_DIR) / "pdfs"
    destination_dir.mkdir(parents=True, exist_ok=True)

    filenames = []
    for upload in files:
        filename = Path(upload.filename or "").name
        if not filename.lower().endswith(".pdf") or not Path(filename).stem:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"Only PDF files can be uploaded: {upload.filename}",
            )
        filenames.append(filename)

    saved_files: list[Path] = []
    for upload, filename in zip(files, filenames, strict=True):
        destination_path = _unique_pdf_path(destination_dir, filename)
        with destination_path.open("wb") as output_file:
            while chunk := await upload.read(1024 * 1024):
                output_file.write(chunk)
        saved_files.append(destination_path)

    return [
        DownloadedPDF(
            filename=path.name,
            path=path.as_posix(),
            size_bytes=path.stat().st_size,
            modified_at=datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat(),
        )
        for path in saved_files
    ]


@router.get("/pdfs/{filename}/content")
async def get_downloaded_pdf_content(filename: str) -> FileResponse:
    pdf_path = _downloaded_pdf_path(filename)
    if not pdf_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Downloaded PDF not found: {filename}",
        )

    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=pdf_path.name,
        content_disposition_type="inline",
    )


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


def _downloaded_pdf_path(filename: str) -> Path:
    normalized_filename = Path(filename).name
    if normalized_filename != filename or not normalized_filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Downloaded PDF not found: {filename}",
        )
    return Path(settings.DATA_DIR) / "pdfs" / normalized_filename


def _unique_pdf_path(destination_dir: Path, filename: str) -> Path:
    path = destination_dir / filename
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    counter = 2
    while True:
        candidate = destination_dir / f"{stem}-{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1
