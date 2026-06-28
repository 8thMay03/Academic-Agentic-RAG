from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config.constants import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE

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

