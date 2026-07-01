import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from app.models.chat import ChatHistoryMessage
from app.models.citation import Citation
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService
from app.services.web_search_service import WebSearchService
from app.vectorstore.bm25 import tokenize

UNKNOWN_ANSWER = "I don't know"
MAX_HISTORY_MESSAGES = 6
MAX_HISTORY_CHARS = 2000
MIN_CONTEXT_CHUNKS = 2
MIN_CONTEXT_CHARS = 600
CITATION_PATTERN = re.compile(r"\[([^\[\]]+)\]")


@dataclass(frozen=True)
class ChatWorkflowRequest:
    question: str
    paper_ids: list[str] | None = None
    top_k: int = 5
    score_threshold: float | None = 0.65
    chat_history: list[ChatHistoryMessage] | None = None


@dataclass(frozen=True)
class ContextQuality:
    sufficient: bool
    chunk_count: int
    context_chars: int
    reason: str


@dataclass(frozen=True)
class PreparedAnswer:
    prompt: str | None
    citations: list[Citation]
    trace: list[dict]


@dataclass(frozen=True)
class ChatWorkflowResult:
    answer: str
    citations: list[Citation]
    trace: list[dict] = field(default_factory=list)


class AgenticChatWorkflow:
    def __init__(
        self,
        rag_service: RAGService,
        llm_service: LLMService,
        web_search_service: WebSearchService | None = None,
    ) -> None:
        self._rag_service = rag_service
        self._llm_service = llm_service
        self._web_search_service = web_search_service or WebSearchService()

    async def run(self, request: ChatWorkflowRequest) -> ChatWorkflowResult:
        prepared = await self.prepare_answer(request)
        if not prepared.prompt:
            return ChatWorkflowResult(answer=UNKNOWN_ANSWER, citations=[], trace=prepared.trace)

        answer = (await self._llm_service.complete(prepared.prompt)).strip()
        if not answer:
            return ChatWorkflowResult(answer=UNKNOWN_ANSWER, citations=[], trace=prepared.trace)

        grounded_answer = self._ground_answer_citations(answer, prepared.citations)
        return ChatWorkflowResult(
            answer=grounded_answer,
            citations=self._citations_referenced_by_answer(prepared.citations, grounded_answer),
            trace=prepared.trace,
        )

    async def stream(self, request: ChatWorkflowRequest) -> tuple[AsyncIterator[str], list[Citation], list[dict]]:
        prepared = await self.prepare_answer(request)
        if not prepared.prompt:
            return self._single_token_stream(UNKNOWN_ANSWER), [], prepared.trace
        return self._llm_service.stream_complete(prepared.prompt), prepared.citations, prepared.trace

    async def prepare_answer(self, request: ChatWorkflowRequest) -> PreparedAnswer:
        trace: list[dict] = []

        local_chunks = await self._retrieve_local(request)
        trace.append(
            {
                "stage": "local_retrieve",
                "chunk_count": len(local_chunks),
                "paper_ids": request.paper_ids,
            }
        )

        quality = self._evaluate_context(local_chunks)
        trace.append(
            {
                "stage": "quality_gate",
                "sufficient": quality.sufficient,
                "reason": quality.reason,
                "chunk_count": quality.chunk_count,
                "context_chars": quality.context_chars,
            }
        )

        web_chunks: list[dict] = []
        if not quality.sufficient:
            web_chunks = await self._search_web(request)
            trace.append(
                {
                    "stage": "web_search",
                    "chunk_count": len(web_chunks),
                    "trigger": quality.reason,
                }
            )

        chunks = [*local_chunks, *web_chunks]
        if not chunks:
            trace.append({"stage": "answer", "status": "no_context"})
            return PreparedAnswer(prompt=None, citations=[], trace=trace)

        citations = self._citations(chunks, request.question)
        trace.append(
            {
                "stage": "answer",
                "status": "ready",
                "context_count": len(chunks),
                "citation_count": len(citations),
            }
        )
        return PreparedAnswer(
            prompt=self._build_prompt(request.question, chunks, request.chat_history),
            citations=citations,
            trace=trace,
        )

    async def _retrieve_local(self, request: ChatWorkflowRequest) -> list[dict]:
        return await self._rag_service.retrieve_context(
            question=request.question,
            paper_ids=request.paper_ids,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            chat_history=request.chat_history,
        )

    async def _search_web(self, request: ChatWorkflowRequest) -> list[dict]:
        result = await self._web_search_service.search_papers(
            request.question,
            max_results=request.top_k,
        )
        chunks = []
        for index, source in enumerate(result.sources[: request.top_k], start=1):
            text = " ".join(str(source.get("content") or "").split())
            if not text:
                continue
            title = str(source.get("title") or source.get("url") or f"Web source {index}")
            url = str(source.get("url") or "")
            chunk_id = f"web:{index}"
            chunks.append(
                {
                    "id": chunk_id,
                    "text": text,
                    "metadata": {
                        "paper_id": url or chunk_id,
                        "title": title,
                        "chunk_id": chunk_id,
                        "url": url,
                    },
                    "score": self._optional_float(source.get("score")),
                    "retrieval_sources": ["web"],
                    "citation": {
                        "paper_id": url or chunk_id,
                        "title": title,
                        "chunk_id": chunk_id,
                        "text": text,
                    },
                }
            )
        return chunks

    @staticmethod
    def _evaluate_context(chunks: list[dict]) -> ContextQuality:
        context_chars = sum(len(str(chunk.get("text") or "")) for chunk in chunks)
        if len(chunks) >= MIN_CONTEXT_CHUNKS:
            return ContextQuality(True, len(chunks), context_chars, "enough_chunks")
        if context_chars >= MIN_CONTEXT_CHARS:
            return ContextQuality(True, len(chunks), context_chars, "enough_context_chars")
        return ContextQuality(False, len(chunks), context_chars, "insufficient_local_context")

    @staticmethod
    async def _single_token_stream(token: str) -> AsyncIterator[str]:
        yield token

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
            if (chunk_id := AgenticChatWorkflow._chunk_id_for(chunk))
        )
        conversation_context = AgenticChatWorkflow._conversation_context(chat_history)
        conversation_section = (
            "Recent conversation:\n"
            f"{conversation_context}\n\n"
            if conversation_context
            else ""
        )
        return (
            "Answer the question using only the retrieved local paper and web context below.\n"
            "Use the recent conversation only to resolve pronouns, ellipses, and follow-up references.\n"
            "If the context does not contain enough information to answer, respond exactly:\n"
            f"{UNKNOWN_ANSWER}\n\n"
            "Do not use outside knowledge. Do not guess. Keep the answer concise.\n"
            "Every factual claim supported by retrieved context must end with one or more exact chunk_id citations "
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
                    score=AgenticChatWorkflow._optional_float(chunk.get("score")),
                    rerank_score=AgenticChatWorkflow._optional_float(chunk.get("rerank_score")),
                    cross_encoder_score=AgenticChatWorkflow._optional_float(chunk.get("cross_encoder_score")),
                    vector_score=AgenticChatWorkflow._optional_float(chunk.get("vector_score")),
                    keyword_score=AgenticChatWorkflow._optional_float(chunk.get("keyword_score")),
                    reranker=chunk.get("reranker"),
                    retrieval_sources=list(chunk.get("retrieval_sources") or []),
                    evidence_quality=AgenticChatWorkflow._evidence_quality(chunk),
                    matched_terms=AgenticChatWorkflow._matched_terms(
                        question,
                        citation.get("text") or chunk.get("text") or "",
                    ),
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
        if "web" in set(chunk.get("retrieval_sources") or []):
            return "web"
        score = AgenticChatWorkflow._optional_float(chunk.get("rerank_score"))
        if score is None:
            score = AgenticChatWorkflow._optional_float(chunk.get("score"))
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
        if not AgenticChatWorkflow._answer_references_any_chunk(grounded_answer, valid_chunk_id_set):
            grounded_answer = f"{grounded_answer} [{valid_chunk_ids[0]}]"
        grounded_answer = " ".join(grounded_answer.split())
        return re.sub(r"\s+([,.!?;:])", r"\1", grounded_answer)

    @staticmethod
    def _citations_referenced_by_answer(citations: list[Citation], answer: str) -> list[Citation]:
        if answer.strip() == UNKNOWN_ANSWER:
            return []
        referenced_chunk_ids = AgenticChatWorkflow._referenced_chunk_ids(answer)
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
        return any(
            chunk_id in valid_chunk_ids
            for chunk_id in AgenticChatWorkflow._referenced_chunk_ids(answer)
        )

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
