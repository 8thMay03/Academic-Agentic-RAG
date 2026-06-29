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

    def query(self, query_embeddings, n_results, where, include):
        assert query_embeddings == [[0.5, 1.0]]
        assert n_results in {2, 8}
        assert where in (None, {"paper_id": "paper-1"})
        assert include == ["documents", "metadatas", "distances"]
        return {
            "ids": [["chunk-1", "chunk-2"]],
            "documents": [["Relevant chunk", "Weak chunk"]],
            "metadatas": [
                [
                    {
                        "chunk_id": "chunk-1",
                        "paper_id": "paper-1",
                        "title": "Agentic RAG",
                        "page_number": "7",
                    },
                    {
                        "chunk_id": "chunk-2",
                        "paper_id": "paper-2",
                        "title": "Other Paper",
                        "page_number": "9",
                    },
                ]
            ],
            "distances": [[0.12, 2.0]],
        }

    def get(self, where, include):
        assert where in (None, {"paper_id": "paper-1"})
        assert include == ["documents", "metadatas"]
        return {
            "ids": ["chunk-1", "chunk-2"],
            "documents": [
                "Agentic RAG planning retrieves relevant evidence.",
                "Vision transformer benchmark results.",
            ],
            "metadatas": [
                {
                    "chunk_id": "chunk-1",
                    "paper_id": "paper-1",
                    "title": "Agentic RAG",
                    "page_number": "7",
                },
                {
                    "chunk_id": "chunk-2",
                    "paper_id": "paper-2",
                    "title": "Other Paper",
                    "page_number": "9",
                },
            ],
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
            "metadata": {
                "chunk_id": "chunk-1",
                "paper_id": "paper-1",
                "title": "Agentic RAG",
                "page_number": "7",
            },
            "distance": 0.12,
            "score": 0.8928571428571428,
            "citation": {
                "paper_id": "paper-1",
                "title": "Agentic RAG",
                "page_number": 7,
                "page": 7,
                "chunk_id": "chunk-1",
                "text": "Relevant chunk",
            },
        },
        {
            "id": "chunk-2",
            "text": "Weak chunk",
            "metadata": {
                "chunk_id": "chunk-2",
                "paper_id": "paper-2",
                "title": "Other Paper",
                "page_number": "9",
            },
            "distance": 2.0,
            "score": 0.3333333333333333,
            "citation": {
                "paper_id": "paper-2",
                "title": "Other Paper",
                "page_number": 9,
                "page": 9,
                "chunk_id": "chunk-2",
                "text": "Weak chunk",
            },
        }
    ]


@pytest.mark.asyncio
async def test_chroma_vector_store_similarity_search_filters_by_paper_id(tmp_path) -> None:
    store = ChromaVectorStore(
        persist_dir=tmp_path,
        collection_name="test_chunks",
        embedding_service=FakeEmbeddingService(),
        client=FakeChromaClient(),
    )

    results = await store.similarity_search("agentic rag", top_k=2, paper_ids=["paper-1"])

    assert results[0]["metadata"]["paper_id"] == "paper-1"


@pytest.mark.asyncio
async def test_chroma_vector_store_similarity_search_filters_by_score_threshold(tmp_path) -> None:
    store = ChromaVectorStore(
        persist_dir=tmp_path,
        collection_name="test_chunks",
        embedding_service=FakeEmbeddingService(),
        client=FakeChromaClient(),
    )

    results = await store.similarity_search("agentic rag", top_k=2, score_threshold=0.8)

    assert [result["id"] for result in results] == ["chunk-1"]


@pytest.mark.asyncio
async def test_chroma_vector_store_keyword_search_uses_bm25_and_formats_results(tmp_path) -> None:
    store = ChromaVectorStore(
        persist_dir=tmp_path,
        collection_name="test_chunks",
        embedding_service=FakeEmbeddingService(),
        client=FakeChromaClient(),
    )

    results = await store.keyword_search("agentic rag planning", top_k=2, paper_ids=["paper-1"])

    assert results[0]["id"] == "chunk-1"
    assert results[0]["keyword_score"] == 1.0
    assert results[0]["citation"]["page_number"] == 7
    assert results[0]["metadata"]["paper_id"] == "paper-1"


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
