import pytest

from app.vectorstore.chroma import ChromaVectorStore


class FakeEmbeddingService:
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[float(index), 1.0] for index, _ in enumerate(texts)]

    async def embed_query(self, query: str) -> list[float]:
        assert query == "agentic rag"
        return [0.5, 1.0]


class FakeCollection:
    def __init__(self) -> None:
        self.upsert_payload = None

    def upsert(self, ids, documents, metadatas, embeddings) -> None:
        self.upsert_payload = {
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
            "embeddings": embeddings,
        }

    def query(self, query_embeddings, n_results, include):
        assert query_embeddings == [[0.5, 1.0]]
        assert n_results == 2
        assert include == ["documents", "metadatas", "distances"]
        return {
            "ids": [["chunk-1"]],
            "documents": [["Relevant chunk"]],
            "metadatas": [[{"paper_id": "paper-1"}]],
            "distances": [[0.12]],
        }


class FakeChromaClient:
    def __init__(self) -> None:
        self.collection = FakeCollection()

    def get_or_create_collection(self, name: str) -> FakeCollection:
        assert name == "test_chunks"
        return self.collection


@pytest.mark.asyncio
async def test_chroma_vector_store_adds_documents_with_embeddings(tmp_path) -> None:
    client = FakeChromaClient()
    store = ChromaVectorStore(
        persist_dir=tmp_path,
        collection_name="test_chunks",
        embedding_service=FakeEmbeddingService(),
        client=client,
    )

    await store.add_documents(
        ["First chunk", "Second chunk"],
        [{"chunk_id": "paper-1:p1:c0"}, {"chunk_id": "paper-1:p1:c1"}],
    )

    assert client.collection.upsert_payload == {
        "ids": ["paper-1:p1:c0", "paper-1:p1:c1"],
        "documents": ["First chunk", "Second chunk"],
        "metadatas": [{"chunk_id": "paper-1:p1:c0"}, {"chunk_id": "paper-1:p1:c1"}],
        "embeddings": [[0.0, 1.0], [1.0, 1.0]],
    }


@pytest.mark.asyncio
async def test_chroma_vector_store_similarity_search_formats_results(tmp_path) -> None:
    store = ChromaVectorStore(
        persist_dir=tmp_path,
        collection_name="test_chunks",
        embedding_service=FakeEmbeddingService(),
        client=FakeChromaClient(),
    )

    results = await store.similarity_search("agentic rag", top_k=2)

    assert results == [
        {
            "id": "chunk-1",
            "text": "Relevant chunk",
            "metadata": {"paper_id": "paper-1"},
            "distance": 0.12,
        }
    ]


@pytest.mark.asyncio
async def test_chroma_vector_store_rejects_mismatched_documents_and_metadata(tmp_path) -> None:
    store = ChromaVectorStore(
        persist_dir=tmp_path,
        collection_name="test_chunks",
        embedding_service=FakeEmbeddingService(),
        client=FakeChromaClient(),
    )

    with pytest.raises(ValueError, match="same length"):
        await store.add_documents(["chunk"], [])
