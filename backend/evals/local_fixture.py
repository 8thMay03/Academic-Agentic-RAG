from __future__ import annotations

import json
import tempfile
from uuid import uuid4
from pathlib import Path

from app.agent.workflow import AgenticChatWorkflow
from app.models.chunk import Chunk
from app.services.rag_service import RAGService
from app.services.retriever_service import RetrieverService
from app.services.web_search_service import WebSearchResult
from app.vectorstore.bm25 import tokenize
from app.vectorstore.chroma import ChromaVectorStore
from app.vectorstore.indexing import index_chunks
from evals.baselines import (
    FullAgenticRAGBaseline,
    HybridRAGBaseline,
    HybridRerankRAGBaseline,
    NoopReranker,
    VectorOnlyRAGBaseline,
)


EVALS_ROOT = Path(__file__).resolve().parent
LOCAL_FIXTURE_CORPUS = EVALS_ROOT / "fixtures" / "local_corpus_chunks.jsonl"


class LocalFixtureEmbeddingService:
    def __init__(self) -> None:
        self.last_usage = None

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    async def embed_query(self, query: str) -> list[float]:
        return self._embed(query)

    def _embed(self, text: str) -> list[float]:
        return [1.0 if tokenize(text) else 0.0]


class LocalFixtureLLM:
    async def complete(self, prompt: str) -> str:
        normalized_prompt = prompt.lower()
        if "decide whether the retrieved context is sufficient" in normalized_prompt:
            return "YES" if "plans retrieval actions" in normalized_prompt else "NO"
        if "generate up to 2 alternate retrieval queries" in normalized_prompt:
            return "[]"
        if "rewrite the current question" in normalized_prompt:
            return self._current_question(prompt)
        question = self._answer_question(prompt).lower()
        if "private benchmark" in question:
            return "I don't know"
        if "fixed rag pipeline" in question and "agentic_rag_fixture:p1:c0" in prompt:
            return (
                "Agentic RAG plans retrieval actions and verifies whether evidence is sufficient "
                "[agentic_rag_fixture:p1:c0]."
            )
        return "I don't know"

    @staticmethod
    def _current_question(prompt: str) -> str:
        marker = "Current question:"
        if marker not in prompt:
            return ""
        return prompt.rsplit(marker, 1)[-1].strip().splitlines()[0].strip()

    @staticmethod
    def _answer_question(prompt: str) -> str:
        marker = "Question:"
        if marker not in prompt:
            return ""
        return prompt.rsplit(marker, 1)[-1].split("Retrieved context:", 1)[0].strip()


class LocalFixtureWebSearchService:
    async def search(self, query: str, max_results: int = 5) -> WebSearchResult:
        return WebSearchResult(sources=[], skipped_reason="local_fixture_web_disabled")


class LocalFixtureEnvironment:
    def __init__(self, corpus_path: Path = LOCAL_FIXTURE_CORPUS) -> None:
        self._corpus_path = corpus_path
        self._persist_dir = Path(tempfile.gettempdir()) / f"agentic-rag-local-fixture-{uuid4().hex}"
        self._initialized = False
        self.llm_service = LocalFixtureLLM()
        self.vector_store = ChromaVectorStore(
            persist_dir=self._persist_dir,
            collection_name="local_fixture_chunks",
            embedding_service=LocalFixtureEmbeddingService(),
        )
        self.hybrid_no_rerank_retriever = RetrieverService(
            vector_store=self.vector_store,
            reranker_service=NoopReranker(),
        )
        self.hybrid_rerank_retriever = RetrieverService(
            vector_store=self.vector_store,
            reranker_service=NoopReranker(),
        )
        self.rag_service = RAGService(self.hybrid_rerank_retriever, self.llm_service)

    async def ensure_indexed(self) -> None:
        if self._initialized:
            return
        chunks = self._load_chunks()
        await index_chunks(chunks, vector_store=self.vector_store)
        self._initialized = True

    def _load_chunks(self) -> list[Chunk]:
        chunks = []
        with self._corpus_path.open(encoding="utf-8") as file:
            for line in file:
                payload = json.loads(line)
                chunks.append(Chunk.model_validate(payload))
        return chunks


class LocalFixtureBaselineWrapper:
    def __init__(self, baseline, environment: LocalFixtureEnvironment) -> None:
        self._baseline = baseline
        self._environment = environment
        self.mode = baseline.mode

    async def run_case(self, case):
        await self._environment.ensure_indexed()
        return await self._baseline.run_case(case)


def build_local_fixture_baselines(modes: list[str]) -> list[LocalFixtureBaselineWrapper]:
    environment = LocalFixtureEnvironment()
    available = {
        "vector_only_rag": lambda: VectorOnlyRAGBaseline(environment.vector_store, environment.llm_service),
        "hybrid_rag": lambda: HybridRAGBaseline(environment.hybrid_no_rerank_retriever, environment.llm_service),
        "hybrid_rerank_rag": lambda: HybridRerankRAGBaseline(environment.rag_service, environment.llm_service),
        "full_agentic_rag": lambda: FullAgenticRAGBaseline(
            AgenticChatWorkflow(
                environment.rag_service,
                environment.llm_service,
                web_search_service=LocalFixtureWebSearchService(),
            )
        ),
    }
    return [
        LocalFixtureBaselineWrapper(available[mode](), environment)
        for mode in modes
    ]
