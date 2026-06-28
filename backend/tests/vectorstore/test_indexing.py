from app.models.chunk import Chunk
from app.vectorstore.indexing import chunks_to_documents, index_chunks


def test_chunks_to_documents_includes_chunk_metadata() -> None:
    chunk = Chunk(
        chunk_id="paper-1:p3:c0",
        paper_id="paper-1",
        text="Chunk text",
        page_number=3,
        metadata={"source": "arxiv"},
    )

    documents, metadatas = chunks_to_documents([chunk])

    assert documents == ["Chunk text"]
    assert metadatas == [
        {
            "chunk_id": "paper-1:p3:c0",
            "paper_id": "paper-1",
            "page_number": "3",
            "page": "3",
            "source": "arxiv",
        }
    ]


class FakeVectorStore:
    def __init__(self) -> None:
        self.documents = None
        self.metadatas = None

    async def add_documents(self, documents: list[str], metadatas: list[dict]) -> None:
        self.documents = documents
        self.metadatas = metadatas


async def test_index_chunks_adds_chunks_to_vector_store() -> None:
    chunk = Chunk(
        chunk_id="paper-1:p1:c0",
        paper_id="paper-1",
        text="Chunk text",
        page_number=1,
    )
    vector_store = FakeVectorStore()

    await index_chunks([chunk], vector_store)

    assert vector_store.documents == ["Chunk text"]
    assert vector_store.metadatas == [
        {
            "chunk_id": "paper-1:p1:c0",
            "paper_id": "paper-1",
            "page_number": "1",
            "page": "1",
        }
    ]
