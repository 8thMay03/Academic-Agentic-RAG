import pytest

from app.parser.cleaner import PAGE_BREAK
from app.parser.chunker import chunk_text, chunk_text_with_metadata


def test_chunk_text_splits_text() -> None:
    chunks = chunk_text("abcdef", chunk_size=4, overlap=1)
    assert chunks == ["abcd", "def"]


def test_chunk_text_prefers_recursive_separators() -> None:
    text = "Introduction sentence.\n\nMethods sentence.\n\nConclusion sentence."

    chunks = chunk_text(text, chunk_size=35, overlap=5)

    assert chunks == [
        "Introduction sentence.",
        "Methods sentence.",
        "Conclusion sentence.",
    ]


def test_chunk_text_falls_back_to_character_splits_with_overlap() -> None:
    chunks = chunk_text("abcdefghij", chunk_size=4, overlap=2)

    assert chunks == ["abcd", "cdef", "efgh", "ghij"]
    assert chunks[0][-2:] == chunks[1][:2]


def test_chunk_text_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=2, overlap=2)


def test_chunk_text_rejects_negative_overlap() -> None:
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=2, overlap=-1)


def test_chunk_text_with_metadata_stores_chunk_id_and_page_number() -> None:
    text = PAGE_BREAK.join(["First page alpha beta", "Second page gamma delta"])

    chunks = chunk_text_with_metadata(
        text,
        paper_id="paper-1",
        chunk_size=12,
        overlap=2,
    )

    assert [chunk.page_number for chunk in chunks] == [1, 1, 2, 2]
    assert [chunk.chunk_id for chunk in chunks] == [
        "paper-1:p1:c0",
        "paper-1:p1:c1",
        "paper-1:p2:c0",
        "paper-1:p2:c1",
    ]
    assert chunks[0].metadata == {
        "chunk_id": "paper-1:p1:c0",
        "page_number": "1",
        "chunk_index": "0",
    }

