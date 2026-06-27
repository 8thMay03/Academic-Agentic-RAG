import pytest

from app.parser.chunker import chunk_text


def test_chunk_text_splits_text() -> None:
    chunks = chunk_text("abcdef", chunk_size=4, overlap=1)
    assert chunks == ["abcd", "def"]


def test_chunk_text_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=2, overlap=2)

