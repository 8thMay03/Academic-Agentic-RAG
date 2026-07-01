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
MIN_TOP_SCORE = 0.45
MIN_AVERAGE_SCORE = 0.35
MIN_QUERY_COVERAGE = 0.25
STRONG_TOP_SCORE = 0.75
STRONG_QUERY_COVERAGE = 0.5
CITATION_PATTERN = re.compile(r"\[([^\[\]]+)\]")
LATEST_QUERY_TERMS = {
    "latest",
    "current",
    "recent",
    "today",
    "now",
    "newest",
    "state of the art",
    "sota",
    "mới nhất",
    "gan day",
    "gần đây",
    "hien tai",
    "hiện tại",
    "hom nay",
    "hôm nay",
}


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
    top_score: float | None = None
    average_score: float | None = None
    source_count: int = 0
    query_coverage: float = 0.0
    self_check_used: bool = False
    self_check_passed: bool | None = None


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
        from app.agent.graph import run_agentic_rag_workflow

        return await run_agentic_rag_workflow(self, request)

    @staticmethod
    def _prepared_answer(
        prompt: str | None,
        citations: list[Citation],
        trace: list[dict],
    ) -> PreparedAnswer:
        return PreparedAnswer(prompt=prompt, citations=citations, trace=trace)

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

    async def _evaluate_context(
        self,
        request: ChatWorkflowRequest,
        chunks: list[dict],
    ) -> ContextQuality:
        context_chars = sum(len(str(chunk.get("text") or "")) for chunk in chunks)
        top_score = self._top_score(chunks)
        average_score = self._average_score(chunks)
        source_count = self._source_count(chunks)
        query_coverage = self._query_coverage(request.question, chunks)
        base_quality = {
            "chunk_count": len(chunks),
            "context_chars": context_chars,
            "top_score": top_score,
            "average_score": average_score,
            "source_count": source_count,
            "query_coverage": query_coverage,
        }

        if self._requires_fresh_context(request.question):
            return ContextQuality(
                False,
                reason="latest_query_requires_web",
                **base_quality,
            )
        if not chunks:
            return ContextQuality(False, reason="no_local_context", **base_quality)
        if len(chunks) < MIN_CONTEXT_CHUNKS and context_chars < MIN_CONTEXT_CHARS:
            return ContextQuality(False, reason="insufficient_local_context", **base_quality)
        if top_score is not None and top_score < MIN_TOP_SCORE:
            return ContextQuality(False, reason="low_top_score", **base_quality)
        if average_score is not None and average_score < MIN_AVERAGE_SCORE:
            return ContextQuality(False, reason="low_average_score", **base_quality)
        if query_coverage < MIN_QUERY_COVERAGE:
            return ContextQuality(False, reason="low_query_coverage", **base_quality)

        if self._should_self_check(top_score, query_coverage):
            self_check_passed = await self._self_check_context(request.question, chunks)
            return ContextQuality(
                self_check_passed,
                reason="llm_self_check_passed" if self_check_passed else "llm_self_check_failed",
                self_check_used=True,
                self_check_passed=self_check_passed,
                **base_quality,
            )

        return ContextQuality(True, reason="strong_context", **base_quality)

    async def _self_check_context(self, question: str, chunks: list[dict]) -> bool:
        prompt = self._self_check_prompt(question, chunks)
        try:
            response = (await self._llm_service.complete(prompt)).strip().lower()
        except Exception:
            return False
        return response.startswith("yes") or '"sufficient": true' in response

    @staticmethod
    def _self_check_prompt(question: str, chunks: list[dict], max_chars: int = 4000) -> str:
        context = "\n\n".join(
            f"[{index}] {str(chunk.get('text') or '')}"
            for index, chunk in enumerate(chunks, start=1)
        )[:max_chars]
        return (
            "Decide whether the retrieved context is sufficient to answer the question without web search.\n"
            "Return only YES or NO.\n\n"
            f"Question: {question}\n\n"
            f"Retrieved context:\n{context}"
        )

    @staticmethod
    def _should_self_check(top_score: float | None, query_coverage: float) -> bool:
        if top_score is None:
            return True
        return top_score < STRONG_TOP_SCORE or query_coverage < STRONG_QUERY_COVERAGE

    @staticmethod
    def _top_score(chunks: list[dict]) -> float | None:
        scores = [
            score
            for chunk in chunks
            if (score := AgenticChatWorkflow._ranking_score(chunk)) is not None
        ]
        return max(scores) if scores else None

    @staticmethod
    def _average_score(chunks: list[dict]) -> float | None:
        scores = [
            score
            for chunk in chunks
            if (score := AgenticChatWorkflow._ranking_score(chunk)) is not None
        ]
        if not scores:
            return None
        return sum(scores) / len(scores)

    @staticmethod
    def _ranking_score(chunk: dict) -> float | None:
        for key in ("rerank_score", "score", "vector_score", "keyword_score"):
            score = AgenticChatWorkflow._optional_float(chunk.get(key))
            if score is not None:
                return score
        return None

    @staticmethod
    def _source_count(chunks: list[dict]) -> int:
        sources = set()
        for chunk in chunks:
            citation = chunk.get("citation") or {}
            metadata = chunk.get("metadata") or {}
            source = citation.get("paper_id") or metadata.get("paper_id") or metadata.get("title")
            if source:
                sources.add(str(source))
        return len(sources)

    @staticmethod
    def _query_coverage(question: str, chunks: list[dict]) -> float:
        query_terms = set(tokenize(question))
        if not query_terms:
            return 1.0
        context_terms = set(tokenize(" ".join(str(chunk.get("text") or "") for chunk in chunks)))
        return len(query_terms & context_terms) / len(query_terms)

    @staticmethod
    def _requires_fresh_context(question: str) -> bool:
        normalized = " ".join(question.lower().split())
        return any(term in normalized for term in LATEST_QUERY_TERMS)

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
