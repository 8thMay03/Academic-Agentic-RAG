from pathlib import Path


class PDFService:
    async def download_pdf(self, pdf_url: str, destination: Path) -> Path:
        # TODO: Replace with async HTTP download.
        raise NotImplementedError("PDF download is not implemented yet.")

