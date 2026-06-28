from pathlib import Path

from app.services.pdf_service import PDFService


async def pdf_download_tool(pdf_url: str, destination: Path) -> Path:
    return await PDFService().download_pdf(pdf_url, destination)


async def pdf_download_many_tool(pdf_urls: list[str], destination_dir: Path) -> list[Path]:
    return await PDFService().download_pdfs(pdf_urls, destination_dir)


async def pdf_dowload_tool(
    pdf_url: str | None = None,
    pdf_urls: list[str] | None = None,
    destination: Path | None = None,
    destination_dir: Path | None = None,
) -> Path | list[Path]:
    service = PDFService()

    if pdf_urls is not None:
        if destination_dir is None:
            raise ValueError("destination_dir is required when downloading multiple PDFs.")
        return await service.download_pdfs(pdf_urls, destination_dir)

    if pdf_url is None or destination is None:
        raise ValueError("pdf_url and destination are required when downloading one PDF.")

    return await service.download_pdf(pdf_url, destination)
