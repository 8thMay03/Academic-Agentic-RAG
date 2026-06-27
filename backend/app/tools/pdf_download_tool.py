from pathlib import Path

from app.services.pdf_service import PDFService


async def pdf_download_tool(pdf_url: str, destination: Path) -> Path:
    return await PDFService().download_pdf(pdf_url, destination)

