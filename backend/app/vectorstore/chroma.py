from pathlib import Path

import chromadb

from app.config.settings import settings
from app.services.embedding_service import EmbeddingService
from app.vectorstore.bm25 import BM25Scorer


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

    async def similarity_search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float | None = None,
        paper_ids: list[str] | None = None,
    ) -> list[dict]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        if score_threshold is not None and not 0 <= score_threshold <= 1:
            raise ValueError("score_threshold must be between 0 and 1")

        query_embedding = await self._embedding_service.embed_query(query)
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=self._paper_filter(paper_ids),
            include=["documents", "metadatas", "distances"],
        )
        formatted_results = self._format_query_results(results)
        if score_threshold is None:
            return formatted_results
        return [result for result in formatted_results if result["score"] >= score_threshold]

    async def keyword_search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float | None = None,
        paper_ids: list[str] | None = None,
    ) -> list[dict]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        if score_threshold is not None and not 0 <= score_threshold <= 1:
            raise ValueError("score_threshold must be between 0 and 1")

        results = self._collection.get(
            where=self._paper_filter(paper_ids),
            include=["documents", "metadatas"],
        )
        formatted_results = self._format_get_results(results)
        scorer = BM25Scorer([result["text"] for result in formatted_results])
        ranked_results = []

        for document_index, score in scorer.rank(query, top_k):
            result = dict(formatted_results[document_index])
            result["score"] = score
            result["keyword_score"] = score
            ranked_results.append(result)

        if score_threshold is None:
            return ranked_results
        return [result for result in ranked_results if result["score"] >= score_threshold]

    @staticmethod
    def _document_id(index: int, metadata: dict) -> str:
        chunk_id = metadata.get("chunk_id")
        if chunk_id:
            return str(chunk_id)
        return f"chunk-{index}"

    @staticmethod
    def _paper_filter(paper_ids: list[str] | None) -> dict | None:
        if not paper_ids:
            return None
        if len(paper_ids) == 1:
            return {"paper_id": paper_ids[0]}
        return {"paper_id": {"$in": paper_ids}}

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
                    "score": ChromaVectorStore._distance_to_score(distances[index]),
                    "citation": ChromaVectorStore._citation(metadatas[index] or {}, documents[index]),
                }
            )

        return formatted_results

    @staticmethod
    def _format_get_results(results: dict) -> list[dict]:
        ids = results.get("ids", [])
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])

        formatted_results = []
        for index, document_id in enumerate(ids):
            metadata = metadatas[index] or {}
            document = documents[index]
            formatted_results.append(
                {
                    "id": document_id,
                    "text": document,
                    "metadata": metadata,
                    "score": 0.0,
                    "citation": ChromaVectorStore._citation(metadata, document),
                }
            )

        return formatted_results

    @staticmethod
    def _distance_to_score(distance: float) -> float:
        return 1 / (1 + max(distance, 0))

    @staticmethod
    def _citation(metadata: dict, text: str) -> dict:
        page_number = metadata.get("page_number") or metadata.get("page")
        try:
            page_number = int(page_number) if page_number not in {None, ""} else None
        except (TypeError, ValueError):
            page_number = None

        return {
            "paper_id": metadata.get("paper_id", ""),
            "title": metadata.get("title", ""),
            "page_number": page_number,
            "page": page_number,
            "chunk_id": metadata.get("chunk_id", ""),
            "text": text,
        }
