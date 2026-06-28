import pytest

from app.parser.chunker import chunk_text


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

