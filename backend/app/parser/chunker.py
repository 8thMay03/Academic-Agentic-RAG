from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config.constants import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from app.models.chunk import Chunk
from app.parser.cleaner import PAGE_BREAK

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


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

    for page_number, page_text in pages:
        page_chunks = chunk_text(
            page_text,
            chunk_size=chunk_size,
            overlap=overlap,
            separators=separators,
        )

        for chunk_index, chunk in enumerate(page_chunks):
            chunk_id = f"{paper_id}:p{page_number}:c{chunk_index}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    paper_id=paper_id,
                    text=chunk,
                    page_number=page_number,
                    page=page_number,
                    metadata={
                        "chunk_id": chunk_id,
                        "page_number": str(page_number),
                        "chunk_index": str(chunk_index),
                    },
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

