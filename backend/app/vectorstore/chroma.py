from pathlib import Path

import chromadb

from app.config.settings import settings
from app.services.embedding_service import EmbeddingService


class ChromaVectorStore:
    def __init__(
        self,
        persist_dir: str | Path | None = None,
        collection_name: str = "research_chunks",
        embedding_service: EmbeddingService | None = None,
        client: chromadb.ClientAPI | None = None,
    ) -> None:
        self._persist_dir = Path(persist_dir or settings.CHROMA_DIR)
        self._collection_name = collection_name
        self._embedding_service = embedding_service or EmbeddingService()
        self._client = client or chromadb.PersistentClient(path=str(self._persist_dir))
        self._collection = self._client.get_or_create_collection(name=self._collection_name)

    async def add_documents(self, documents: list[str], metadatas: list[dict]) -> None:
        if not documents:
            return
        if len(documents) != len(metadatas):
            raise ValueError("documents and metadatas must have the same length")

        embeddings = await self._embedding_service.embed_texts(documents)
        ids = [self._document_id(index, metadata) for index, metadata in enumerate(metadatas)]
        self._collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    async def similarity_search(self, query: str, top_k: int = 5) -> list[dict]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        query_embedding = await self._embedding_service.embed_query(query)
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        return self._format_query_results(results)

    @staticmethod
    def _document_id(index: int, metadata: dict) -> str:
        chunk_id = metadata.get("chunk_id")
        if chunk_id:
            return str(chunk_id)
        return f"chunk-{index}"

    @staticmethod
    def _format_query_results(results: dict) -> list[dict]:
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        formatted_results = []
        for index, document_id in enumerate(ids):
            formatted_results.append(
                {
                    "id": document_id,
                    "text": documents[index],
                    "metadata": metadatas[index] or {},
                    "distance": distances[index],
                }
            )

        return formatted_results
