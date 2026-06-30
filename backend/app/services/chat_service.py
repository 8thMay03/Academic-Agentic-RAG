import json
import re
from collections.abc import AsyncIterator

from app.models.chat import ChatHistoryMessage
from app.models.citation import Citation
from app.services.llm_service import LLMService
from app.services.retriever_service import RetrieverService
from app.vectorstore.bm25 import tokenize


UNKNOWN_ANSWER = "I don't know"
MAX_HISTORY_MESSAGES = 6
MAX_HISTORY_CHARS = 2000
MAX_RETRIEVAL_HISTORY_QUESTIONS = 3
MAX_RETRIEVAL_QUERIES = 3
CITATION_PATTERN = re.compile(r"\[([^\[\]]+)\]")
JSON_ARRAY_PATTERN = re.compile(r"\[.*\]", re.DOTALL)


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
        chat_history: list[ChatHistoryMessage] | None = None,
    ) -> tuple[str, list[Citation]]:
        prompt, citations = await self._prepare_answer(question, paper_ids, top_k, score_threshold, chat_history)
        if not prompt:
            return UNKNOWN_ANSWER, []

        answer = (await self._llm_service.complete(prompt)).strip()
        if not answer:
            return UNKNOWN_ANSWER, []

        answer = self._ground_answer_citations(answer, citations)
        return answer, self._citations_referenced_by_answer(citations, answer)

    async def stream_answer(
        self,
        question: str,
        paper_ids: list[str] | None = None,
        top_k: int = 5,
        score_threshold: float = 0.65,
        chat_history: list[ChatHistoryMessage] | None = None,
    ) -> tuple[AsyncIterator[str], list[Citation]]:
        prompt, citations = await self._prepare_answer(question, paper_ids, top_k, score_threshold, chat_history)
        if not prompt:
            return self._single_token_stream(UNKNOWN_ANSWER), []
        return self._llm_service.stream_complete(prompt), citations

    async def _prepare_answer(
        self,
        question: str,
        paper_ids: list[str] | None,
        top_k: int,
        score_threshold: float | None,
        chat_history: list[ChatHistoryMessage] | None = None,
    ) -> tuple[str | None, list[Citation]]:
        retrieval_queries = await self._build_retrieval_queries(question, chat_history)
        filtered_chunks = await self._retrieve_with_queries(
            retrieval_queries,
            top_k,
            score_threshold=score_threshold,
            paper_ids=paper_ids,
        )

        if not filtered_chunks and score_threshold is not None and score_threshold > 0:
            filtered_chunks = await self._retrieve_with_queries(
                retrieval_queries,
                top_k,
                score_threshold=None,
                paper_ids=paper_ids,
            )

        if not filtered_chunks:
            return None, []

        return self._build_prompt(question, filtered_chunks, chat_history), self._citations(filtered_chunks, question)

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

    @staticmethod
    def _build_retrieval_query(question: str, chat_history: list[ChatHistoryMessage] | None) -> str:
        prior_questions = ChatService._recent_user_questions(chat_history)
        if not prior_questions:
            return question

        return (
            "Previous user questions for resolving follow-up references:\n"
            f"{prior_questions}\n\n"
            f"Current question: {question}"
        )

    @staticmethod
    def _rewrite_query_prompt(question: str, chat_history: list[ChatHistoryMessage] | None) -> str:
        prior_questions = ChatService._recent_user_questions(chat_history) or "None"
        return (
            "Rewrite the current question as a standalone retrieval query for question answering "
            "over the selected paper PDFs.\n"
            "Do not answer the question. Do not add facts that are not implied by the conversation.\n"
            "Return only the rewritten query, with no labels or explanation.\n\n"
            f"Previous user questions:\n{prior_questions}\n\n"
            f"Current question: {question}"
        )

    async def _build_multi_queries(self, rewritten_query: str) -> list[str]:
        prompt = self._multi_query_prompt(rewritten_query)

        try:
            raw_queries = await self._llm_service.complete(prompt)
        except Exception:
            return []

        return self._parse_multi_queries(raw_queries)

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
            if (cleaned_query := ChatService._clean_query(query))
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
            chunk_id = ChatService._chunk_id_for(chunk)
            if not chunk_id:
                continue
            existing_chunk = merged_chunks.get(chunk_id)
            if (
                existing_chunk is None
                or ChatService._ranking_score(chunk) > ChatService._ranking_score(existing_chunk)
            ):
                merged_chunks[chunk_id] = chunk

        return sorted(
            merged_chunks.values(),
            key=ChatService._ranking_score,
            reverse=True,
        )

    @staticmethod
    def _ranking_score(chunk: dict) -> float:
        for score_key in ("rerank_score", "score", "vector_score", "keyword_score"):
            score = ChatService._optional_float(chunk.get(score_key))
            if score is not None:
                return score
        return 0.0

    @staticmethod
    def _conversation_context(chat_history: list[ChatHistoryMessage] | None) -> str:
        if not chat_history:
            return ""

        lines = []
        for message in chat_history[-MAX_HISTORY_MESSAGES:]:
            role = "User" if message.role == "user" else "Assistant"
            content = " ".join(message.content.split())
            if content:
                lines.append(f"{role}: {content}")

        context = "\n".join(lines)
        if len(context) <= MAX_HISTORY_CHARS:
            return context
        return context[-MAX_HISTORY_CHARS:].lstrip()

    @staticmethod
    def _build_prompt(
        question: str,
        chunks: list[dict],
        chat_history: list[ChatHistoryMessage] | None = None,
    ) -> str:
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
        available_chunk_ids = ", ".join(
            chunk_id
            for chunk in chunks
            if (chunk_id := ChatService._chunk_id_for(chunk))
        )
        conversation_context = ChatService._conversation_context(chat_history)
        conversation_section = (
            "Recent conversation:\n"
            f"{conversation_context}\n\n"
            if conversation_context
            else ""
        )
        return (
            "Answer the question using only the retrieved paper context below.\n"
            "Use the recent conversation only to resolve pronouns, ellipses, and follow-up references.\n"
            "If the context does not contain enough information to answer, respond exactly:\n"
            f"{UNKNOWN_ANSWER}\n\n"
            "Do not use outside knowledge. Do not guess. Keep the answer concise.\n"
            "Every factual claim supported by paper context must end with one or more exact chunk_id citations "
            "in square brackets, e.g. [paper-1:p3:c0].\n"
            "Use only chunk_id values that appear in the retrieved context. Do not invent citation ids.\n\n"
            f"Available chunk_ids: {available_chunk_ids}\n\n"
            f"{conversation_section}"
            f"Question: {question}\n\n"
            "Retrieved context:\n"
            f"{context_text}"
        )

    @staticmethod
    def _citations(chunks: list[dict], question: str = "") -> list[Citation]:
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
                    score=ChatService._optional_float(chunk.get("score")),
                    rerank_score=ChatService._optional_float(chunk.get("rerank_score")),
                    cross_encoder_score=ChatService._optional_float(chunk.get("cross_encoder_score")),
                    vector_score=ChatService._optional_float(chunk.get("vector_score")),
                    keyword_score=ChatService._optional_float(chunk.get("keyword_score")),
                    reranker=chunk.get("reranker"),
                    retrieval_sources=list(chunk.get("retrieval_sources") or []),
                    evidence_quality=ChatService._evidence_quality(chunk),
                    matched_terms=ChatService._matched_terms(question, citation.get("text") or chunk.get("text") or ""),
                )
            )

        return citations

    @staticmethod
    def _optional_float(value: object) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _evidence_quality(chunk: dict) -> str:
        score = ChatService._optional_float(chunk.get("rerank_score"))
        if score is None:
            score = ChatService._optional_float(chunk.get("score"))
        if score is None:
            return "unknown"
        if score >= 0.75:
            return "high"
        if score >= 0.5:
            return "medium"
        return "low"

    @staticmethod
    def _matched_terms(question: str, text: str) -> list[str]:
        query_terms = []
        seen_terms = set()
        for term in tokenize(question):
            if term not in seen_terms:
                seen_terms.add(term)
                query_terms.append(term)

        text_terms = set(tokenize(text))
        return [term for term in query_terms if term in text_terms][:8]

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

    @staticmethod
    def _ground_answer_citations(answer: str, citations: list[Citation]) -> str:
        if answer.strip() == UNKNOWN_ANSWER:
            return answer

        valid_chunk_ids = [citation.chunk_id for citation in citations if citation.chunk_id]
        if not valid_chunk_ids:
            return answer

        valid_chunk_id_set = set(valid_chunk_ids)

        def keep_valid_citations(match: re.Match[str]) -> str:
            raw_citation_ids = re.split(r"[,;\s]+", match.group(1).strip())
            supported_citation_ids = [
                citation_id
                for citation_id in raw_citation_ids
                if citation_id in valid_chunk_id_set
            ]
            if not supported_citation_ids:
                return ""
            return f"[{', '.join(supported_citation_ids)}]"

        grounded_answer = CITATION_PATTERN.sub(keep_valid_citations, answer).strip()
        if not ChatService._answer_references_any_chunk(grounded_answer, valid_chunk_id_set):
            grounded_answer = f"{grounded_answer} [{valid_chunk_ids[0]}]"
        grounded_answer = " ".join(grounded_answer.split())
        return re.sub(r"\s+([,.!?;:])", r"\1", grounded_answer)

    @staticmethod
    def _citations_referenced_by_answer(citations: list[Citation], answer: str) -> list[Citation]:
        referenced_chunk_ids = ChatService._referenced_chunk_ids(answer)
        if not referenced_chunk_ids:
            return citations

        citations_by_chunk_id = {
            citation.chunk_id: citation
            for citation in citations
            if citation.chunk_id
        }
        return [
            citations_by_chunk_id[chunk_id]
            for chunk_id in referenced_chunk_ids
            if chunk_id in citations_by_chunk_id
        ]

    @staticmethod
    def _answer_references_any_chunk(answer: str, valid_chunk_ids: set[str]) -> bool:
        return any(chunk_id in valid_chunk_ids for chunk_id in ChatService._referenced_chunk_ids(answer))

    @staticmethod
    def _referenced_chunk_ids(answer: str) -> list[str]:
        chunk_ids = []
        seen_chunk_ids = set()
        for match in CITATION_PATTERN.finditer(answer):
            raw_citation_ids = re.split(r"[,;\s]+", match.group(1).strip())
            for citation_id in raw_citation_ids:
                if citation_id and citation_id not in seen_chunk_ids:
                    seen_chunk_ids.add(citation_id)
                    chunk_ids.append(citation_id)
        return chunk_ids
