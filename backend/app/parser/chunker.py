from langchain_text_splitters import RecursiveCharacterTextSplitter
import re

from app.config.constants import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from app.models.chunk import Chunk
from app.parser.cleaner import PAGE_BREAK

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
SECTION_HEADING_PATTERN = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?"
    r"(abstract|introduction|background|related work|method|methods|methodology|approach|"
    r"experiments?|experimental setup|results?|evaluation|discussion|limitations?|conclusion|"
    r"future work|references?)\s*$",
    re.IGNORECASE,
)
SECTION_TYPE_ALIASES = {
    "method": "method",
    "methods": "method",
    "methodology": "method",
    "approach": "method",
    "experiment": "experiments",
    "experiments": "experiments",
    "experimental setup": "experiments",
    "result": "results",
    "results": "results",
    "evaluation": "results",
    "limitation": "limitations",
    "limitations": "limitations",
    "reference": "references",
    "references": "references",
}


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    separators: list[str] | None = None,
) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap must be greater than or equal to 0")
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=separators or DEFAULT_SEPARATORS,
        length_function=len,
    )
    return splitter.split_text(text)


def chunk_text_with_metadata(
    text: str,
    paper_id: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    separators: list[str] | None = None,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    pages = _split_pages(text)
    current_section_title = ""
    current_section_type = ""

    for page_number, page_text in pages:
        page_section = detect_section_heading(page_text)
        if page_section:
            current_section_title, current_section_type = page_section

        page_chunks = chunk_text(
            page_text,
            chunk_size=chunk_size,
            overlap=overlap,
            separators=separators,
        )

        for chunk_index, chunk in enumerate(page_chunks):
            chunk_section = detect_section_heading(chunk)
            if chunk_section:
                current_section_title, current_section_type = chunk_section
            chunk_id = f"{paper_id}:p{page_number}:c{chunk_index}"
            metadata = {
                "chunk_id": chunk_id,
                "page_number": str(page_number),
                "chunk_index": str(chunk_index),
            }
            if current_section_title:
                metadata["section_title"] = current_section_title
                metadata["section_type"] = current_section_type
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    paper_id=paper_id,
                    text=chunk,
                    page_number=page_number,
                    page=page_number,
                    metadata=metadata,
                )
            )

    return chunks


def _split_pages(text: str) -> list[tuple[int, str]]:
    pages = text.split(PAGE_BREAK)
    return [
        (page_number, page_text.strip())
        for page_number, page_text in enumerate(pages, start=1)
        if page_text.strip()
    ]


def detect_section_heading(text: str) -> tuple[str, str] | None:
    for line in text.splitlines()[:6]:
        normalized_line = " ".join(line.strip().split())
        if not normalized_line:
            continue
        match = SECTION_HEADING_PATTERN.match(normalized_line)
        if not match:
            continue
        section_title = normalized_line
        raw_section_type = match.group(1).lower()
        section_type = SECTION_TYPE_ALIASES.get(raw_section_type, raw_section_type)
        return section_title, section_type
    return None

