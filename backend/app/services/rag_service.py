import json
import re

from app.models.chat import ChatHistoryMessage
from app.services.llm_service import LLMService
from app.services.retriever_service import RetrieverService


MAX_RETRIEVAL_HISTORY_QUESTIONS = 3
MAX_RETRIEVAL_QUERIES = 3
JSON_ARRAY_PATTERN = re.compile(r"\[.*\]", re.DOTALL)


class RAGService:
    def __init__(
        self,
        retriever_service: RetrieverService,
        llm_service: LLMService,
    ) -> None:
        self._retriever_service = retriever_service
        self._llm_service = llm_service

    async def retrieve_context(
        self,
        question: str,
        paper_ids: list[str] | None = None,
        top_k: int = 5,
        score_threshold: float | None = 0.65,
        chat_history: list[ChatHistoryMessage] | None = None,
    ) -> list[dict]:
        retrieval_queries = await self._build_retrieval_queries(question, chat_history)
        chunks = await self._retrieve_with_queries(
            retrieval_queries,
            top_k,
            score_threshold=score_threshold,
            paper_ids=paper_ids,
        )

        if not chunks and score_threshold is not None and score_threshold > 0:
            chunks = await self._retrieve_with_queries(
                retrieval_queries,
                top_k,
                score_threshold=None,
                paper_ids=paper_ids,
            )

        return chunks

    async def _retrieve_with_queries(
        self,
        retrieval_queries: list[str],
        top_k: int,
        score_threshold: float | None,
        paper_ids: list[str] | None,
    ) -> list[dict]:
        retrieved_chunks = []
        for retrieval_query in retrieval_queries:
            chunks = await self._retriever_service.retrieve(
                retrieval_query,
                top_k=top_k,
                score_threshold=score_threshold,
                paper_ids=paper_ids,
            )
            retrieved_chunks.extend(self._filter_by_paper_ids(chunks, paper_ids))

        return self._merge_retrieved_chunks(retrieved_chunks)[:top_k]

    async def _build_retrieval_queries(
        self,
        question: str,
        chat_history: list[ChatHistoryMessage] | None,
    ) -> list[str]:
        rewritten_query = await self._rewrite_query(question, chat_history)
        multi_queries = await self._build_multi_queries(rewritten_query)
        return self._dedupe_queries([rewritten_query, *multi_queries])

    async def _rewrite_query(
        self,
        question: str,
        chat_history: list[ChatHistoryMessage] | None,
    ) -> str:
        fallback_query = self._build_retrieval_query(question, chat_history)
        if not chat_history:
            return fallback_query

        prompt = self._rewrite_query_prompt(question, chat_history)

        try:
            rewritten_query = self._clean_query(await self._llm_service.complete(prompt))
        except Exception:
            return fallback_query

        return rewritten_query or fallback_query

    async def _build_multi_queries(self, rewritten_query: str) -> list[str]:
        prompt = self._multi_query_prompt(rewritten_query)

        try:
            raw_queries = await self._llm_service.complete(prompt)
        except Exception:
            return []

        return self._parse_multi_queries(raw_queries)

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
    def _build_retrieval_query(question: str, chat_history: list[ChatHistoryMessage] | None) -> str:
        prior_questions = RAGService._recent_user_questions(chat_history)
        if not prior_questions:
            return question

        return (
            "Previous user questions for resolving follow-up references:\n"
            f"{prior_questions}\n\n"
            f"Current question: {question}"
        )

    @staticmethod
    def _rewrite_query_prompt(question: str, chat_history: list[ChatHistoryMessage] | None) -> str:
        prior_questions = RAGService._recent_user_questions(chat_history) or "None"
        return (
            "Rewrite the current question as a standalone retrieval query for question answering "
            "over the selected paper PDFs.\n"
            "Do not answer the question. Do not add facts that are not implied by the conversation.\n"
            "Return only the rewritten query, with no labels or explanation.\n\n"
            f"Previous user questions:\n{prior_questions}\n\n"
            f"Current question: {question}"
        )

    @staticmethod
    def _multi_query_prompt(rewritten_query: str) -> str:
        return (
            "Generate up to 2 alternate retrieval queries for searching academic paper chunks.\n"
            "Use concise academic wording, synonyms, or likely section terminology.\n"
            "Do not answer the question. Do not add facts that are not present in the query.\n"
            "Return JSON array only, for example: [\"query variant 1\", \"query variant 2\"]\n\n"
            f"Rewritten query: {rewritten_query}"
        )

    @staticmethod
    def _parse_multi_queries(raw_queries: str) -> list[str]:
        query_text = raw_queries.strip()
        json_match = JSON_ARRAY_PATTERN.search(query_text)
        if json_match:
            query_text = json_match.group(0)

        try:
            parsed_queries = json.loads(query_text)
        except json.JSONDecodeError:
            return []
        if not isinstance(parsed_queries, list):
            return []

        return [
            cleaned_query
            for query in parsed_queries[: MAX_RETRIEVAL_QUERIES - 1]
            if (cleaned_query := RAGService._clean_query(query))
        ]

    @staticmethod
    def _clean_query(query: object) -> str:
        if not isinstance(query, str):
            return ""
        return " ".join(query.split()).strip(" \t\n\r\"'")

    @staticmethod
    def _dedupe_queries(queries: list[str]) -> list[str]:
        deduped_queries = []
        seen_queries = set()

        for query in queries:
            normalized_query = query.lower()
            if not query or normalized_query in seen_queries:
                continue
            seen_queries.add(normalized_query)
            deduped_queries.append(query)
            if len(deduped_queries) >= MAX_RETRIEVAL_QUERIES:
                break

        return deduped_queries

    @staticmethod
    def _merge_retrieved_chunks(chunks: list[dict]) -> list[dict]:
        merged_chunks: dict[str, dict] = {}

        for chunk in chunks:
            chunk_id = RAGService._chunk_id_for(chunk)
            if not chunk_id:
                continue
            existing_chunk = merged_chunks.get(chunk_id)
            if (
                existing_chunk is None
                or RAGService._ranking_score(chunk) > RAGService._ranking_score(existing_chunk)
            ):
                merged_chunks[chunk_id] = chunk

        return sorted(
            merged_chunks.values(),
            key=RAGService._ranking_score,
            reverse=True,
        )

    @staticmethod
    def _ranking_score(chunk: dict) -> float:
        for score_key in ("rerank_score", "score", "vector_score", "keyword_score"):
            score = RAGService._optional_float(chunk.get(score_key))
            if score is not None:
                return score
        return 0.0

    @staticmethod
    def _optional_float(value: object) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _recent_user_questions(chat_history: list[ChatHistoryMessage] | None) -> str:
        if not chat_history:
            return ""

        questions = []
        for message in chat_history:
            if message.role != "user":
                continue
            content = " ".join(message.content.split())
            if content:
                questions.append(content)

        recent_questions = questions[-MAX_RETRIEVAL_HISTORY_QUESTIONS:]
        return "\n".join(f"- {question}" for question in recent_questions)

    @staticmethod
    def _chunk_id_for(chunk: dict) -> str:
        citation = chunk.get("citation") or {}
        metadata = chunk.get("metadata") or {}
        return str(citation.get("chunk_id") or metadata.get("chunk_id") or chunk.get("id") or "")
