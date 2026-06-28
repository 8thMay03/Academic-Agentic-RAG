import asyncio
from pathlib import Path

from app.parser.cleaner import clean_text
from app.parser.pdf_parser import extract_text_from_pdf


class ParserService:
    async def parse_pdf(self, pdf_path: Path) -> str:
        text = await asyncio.to_thread(extract_text_from_pdf, pdf_path)
        return clean_text(text)
