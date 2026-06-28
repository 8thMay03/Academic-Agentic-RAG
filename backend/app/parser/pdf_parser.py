from pathlib import Path

import fitz


class PDFParseError(RuntimeError):
    pass


def extract_text_from_pdf(pdf_path: Path) -> str:
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")

    try:
        with fitz.open(pdf_path) as document:
            page_texts = [
                page.get_text("text", sort=True).strip()
                for page in document
            ]
    except fitz.FileDataError as exc:
        raise PDFParseError(f"Failed to parse PDF: {pdf_path}") from exc

    return "\n\n".join(page_text for page_text in page_texts if page_text)
