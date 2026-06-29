from collections.abc import AsyncIterator

from app.models.citation import Citation
from app.services.llm_service import LLMService
from app.services.retriever_service import RetrieverService


UNKNOWN_ANSWER = "I don't know"


class ChatService:
    def __init__(
        self,
        retriever_service: RetrieverService,
        llm_service: LLMService,
    ) -> None:
        self._retriever_service = retriever_service
        self._llm_service = llm_service

    async def answer(
        self,
        question: str,
        paper_ids: list[str] | None = None,
        top_k: int = 5,
        score_threshold: float = 0.65,
    ) -> tuple[str, list[Citation]]:
        prompt, citations = await self._prepare_answer(question, paper_ids, top_k, score_threshold)
        if not prompt:
            return UNKNOWN_ANSWER, []

        answer = (await self._llm_service.complete(prompt)).strip()
        if not answer:
            return UNKNOWN_ANSWER, []

        return answer, citations

    async def stream_answer(
        self,
        question: str,
        paper_ids: list[str] | None = None,
        top_k: int = 5,
        score_threshold: float = 0.65,
    ) -> tuple[AsyncIterator[str], list[Citation]]:
        prompt, citations = await self._prepare_answer(question, paper_ids, top_k, score_threshold)
        if not prompt:
            return self._single_token_stream(UNKNOWN_ANSWER), []
        return self._llm_service.stream_complete(prompt), citations

    async def _prepare_answer(
        self,
        question: str,
        paper_ids: list[str] | None,
        top_k: int,
        score_threshold: float | None,
    ) -> tuple[str | None, list[Citation]]:
        retrieved_chunks = await self._retriever_service.retrieve(
            question,
            top_k=top_k,
            score_threshold=score_threshold,
            paper_ids=paper_ids,
        )
        filtered_chunks = self._filter_by_paper_ids(retrieved_chunks, paper_ids)

        if not filtered_chunks and score_threshold is not None and score_threshold > 0:
            retrieved_chunks = await self._retriever_service.retrieve(
                question,
                top_k=top_k,
                score_threshold=None,
                paper_ids=paper_ids,
            )
            filtered_chunks = self._filter_by_paper_ids(retrieved_chunks, paper_ids)

        if not filtered_chunks:
            return None, []

        return self._build_prompt(question, filtered_chunks), self._citations(filtered_chunks)

    @staticmethod
    async def _single_token_stream(token: str) -> AsyncIterator[str]:
        yield token

    @staticmethod
    def _filter_by_paper_ids(chunks: list[dict], paper_ids: list[str] | None) -> list[dict]:
        if not paper_ids:
            return chunks
        allowed_paper_ids = set(paper_ids)
        return [
            chunk
            for chunk in chunks
            if (chunk.get("metadata") or {}).get("paper_id") in allowed_paper_ids
        ]

    @staticmethod
    def _build_prompt(question: str, chunks: list[dict]) -> str:
        context_blocks = []
        for index, chunk in enumerate(chunks, start=1):
            citation = chunk.get("citation") or {}
            metadata = chunk.get("metadata") or {}
            paper_id = citation.get("paper_id") or metadata.get("paper_id", "")
            title = citation.get("title") or metadata.get("title", "")
            page_number = citation.get("page_number") or metadata.get("page_number", "")
            chunk_id = citation.get("chunk_id") or metadata.get("chunk_id") or chunk.get("id", "")
            context_blocks.append(
                "\n".join(
                    [
                        f"[Context {index}]",
                        f"paper_id: {paper_id}",
                        f"title: {title}",
                        f"page_number: {page_number}",
                        f"chunk_id: {chunk_id}",
                        f"text: {chunk.get('text', '')}",
                    ]
                )
            )

        context_text = "\n\n".join(context_blocks)
        return (
            "Answer the question using only the retrieved paper context below.\n"
            "If the context does not contain enough information to answer, respond exactly:\n"
            f"{UNKNOWN_ANSWER}\n\n"
            "Do not use outside knowledge. Do not guess. Keep the answer concise.\n"
            "When useful, cite context using page numbers in parentheses, e.g. (p. 3).\n\n"
            f"Question: {question}\n\n"
            "Retrieved context:\n"
            f"{context_text}"
        )

    @staticmethod
    def _citations(chunks: list[dict]) -> list[Citation]:
        citations: list[Citation] = []
        seen_chunk_ids: set[str] = set()

        for chunk in chunks:
            citation = chunk.get("citation") or {}
            metadata = chunk.get("metadata") or {}
            chunk_id = citation.get("chunk_id") or metadata.get("chunk_id") or chunk.get("id", "")
            if chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)

            page_number = citation.get("page_number") or metadata.get("page_number")
            try:
                page_number = int(page_number) if page_number not in {None, ""} else None
            except (TypeError, ValueError):
                page_number = None

            citations.append(
                Citation(
                    paper_id=citation.get("paper_id") or metadata.get("paper_id", ""),
                    title=citation.get("title") or metadata.get("title", ""),
                    page_number=page_number,
                    page=page_number,
                    chunk_id=chunk_id,
                    text=citation.get("text") or chunk.get("text"),
                )
            )

        return citations
