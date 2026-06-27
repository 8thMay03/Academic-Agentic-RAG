from pathlib import Path


class ParserService:
    async def parse_pdf(self, pdf_path: Path) -> str:
        # TODO: Use PyMuPDF/pdfplumber to extract text.
        raise NotImplementedError("PDF parsing is not implemented yet.")

