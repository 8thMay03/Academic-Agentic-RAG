from dataclasses import dataclass, field
from typing import Any, Literal, NotRequired, TypedDict

from app.models.chat import ChatHistoryMessage
from app.models.citation import Citation


@dataclass(frozen=True)
class ChatWorkflowRequest:
    question: str
    chat_id: str | None = None
    paper_ids: list[str] | None = None
    top_k: int = 5
    score_threshold: float | None = 0.65
    chat_history: list[ChatHistoryMessage] | None = None
    max_agent_steps: int = 6
    enable_web_search: bool = True
    enable_research_ingest: bool = True
    auto_download_pdfs: bool = True


@dataclass(frozen=True)
class ChatWorkflowResult:
    answer: str
    citations: list[Citation]
    trace: list["AgentTraceEvent"] = field(default_factory=list)


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
class ToolResult:
    tool_name: str
    success: bool
    chunks: list["RetrievedChunk"] | None = None
    artifacts: list[dict[str, Any]] | None = None
    observations: list[str] | None = None
    error: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class ToolCall:
    tool_name: str
    input: dict[str, Any]
    reason: str
    step_index: int


@dataclass(frozen=True)
class ResearchPlanStep:
    tool_name: str
    reason: str
    input: dict[str, Any]


@dataclass(frozen=True)
class ResearchPlan:
    goal: str
    steps: list[ResearchPlanStep]


@dataclass(frozen=True)
class AgentLimits:
    max_steps: int = 6
    max_web_searches: int = 2
    max_arxiv_searches: int = 2
    max_pdf_downloads: int = 3
    max_retrieval_rounds: int = 3
    tool_timeout_seconds: float = 20.0


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    answer: str
    citations: list[Any]
    issues: list[str]
    unsupported_claims: list[str]
    suggested_action: Literal["finalize", "retrieve_more", "revise_answer", "answer_unknown"]


class RetrievedChunk(TypedDict, total=False):
    id: str
    text: str
    metadata: dict[str, Any]
    citation: dict[str, Any]
    score: float | None
    rerank_score: float | None
    cross_encoder_score: float | None
    vector_score: float | None
    keyword_score: float | None
    reranker: str | None
    retrieval_sources: list[str]


def normalize_retrieved_chunk(chunk: dict[str, Any]) -> RetrievedChunk:
    citation = dict(chunk.get("citation") or {})
    metadata = dict(chunk.get("metadata") or {})
    text = str(chunk.get("text") or citation.get("text") or "")
    chunk_id = str(citation.get("chunk_id") or metadata.get("chunk_id") or chunk.get("id") or "")
    if chunk_id:
        metadata.setdefault("chunk_id", chunk_id)
        citation.setdefault("chunk_id", chunk_id)

    paper_id = str(citation.get("paper_id") or metadata.get("paper_id") or "")
    title = str(citation.get("title") or metadata.get("title") or "")
    if paper_id:
        metadata.setdefault("paper_id", paper_id)
        citation.setdefault("paper_id", paper_id)
    if title:
        metadata.setdefault("title", title)
        citation.setdefault("title", title)
    if text:
        citation.setdefault("text", text)

    page_number = citation.get("page_number") or metadata.get("page_number")
    if page_number not in {None, ""}:
        metadata.setdefault("page_number", page_number)
        citation.setdefault("page_number", page_number)

    url = citation.get("url") or metadata.get("url")
    if url:
        metadata.setdefault("url", url)
        citation.setdefault("url", url)

    normalized: RetrievedChunk = {
        **chunk,
        "id": chunk_id or str(chunk.get("id") or ""),
        "text": text,
        "metadata": metadata,
        "citation": citation,
        "retrieval_sources": list(chunk.get("retrieval_sources") or []),
    }
    return normalized


def normalize_retrieved_chunks(chunks: list[dict[str, Any]]) -> list[RetrievedChunk]:
    return [normalize_retrieved_chunk(chunk) for chunk in chunks]


def retrieved_chunk_id(chunk: dict[str, Any]) -> str:
    citation = chunk.get("citation") or {}
    metadata = chunk.get("metadata") or {}
    return str(citation.get("chunk_id") or metadata.get("chunk_id") or chunk.get("id") or "")


def retrieved_chunk_text(chunk: dict[str, Any]) -> str:
    citation = chunk.get("citation") or {}
    return str(chunk.get("text") or citation.get("text") or "")


def retrieved_chunk_page_number(chunk: dict[str, Any]) -> int | None:
    citation = chunk.get("citation") or {}
    metadata = chunk.get("metadata") or {}
    value = citation.get("page_number") or metadata.get("page_number")
    try:
        return int(value) if value not in {None, ""} else None
    except (TypeError, ValueError):
        return None


def retrieved_chunk_source_id(chunk: dict[str, Any]) -> str:
    citation = chunk.get("citation") or {}
    metadata = chunk.get("metadata") or {}
    return str(citation.get("paper_id") or metadata.get("paper_id") or metadata.get("title") or "")


def retrieved_chunk_title(chunk: dict[str, Any]) -> str:
    citation = chunk.get("citation") or {}
    metadata = chunk.get("metadata") or {}
    return str(citation.get("title") or metadata.get("title") or "")


def optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def retrieved_chunk_ranking_score(chunk: dict[str, Any]) -> float | None:
    for key in ("rerank_score", "score", "vector_score", "keyword_score"):
        score = optional_float(chunk.get(key))
        if score is not None:
            return score
    return None


class AgentTraceEvent(TypedDict):
    stage: str
    chunk_count: NotRequired[int]
    paper_ids: NotRequired[list[str] | None]
    sufficient: NotRequired[bool]
    intent: NotRequired[str]
    reason: NotRequired[str]
    context_chars: NotRequired[int]
    top_score: NotRequired[float | None]
    average_score: NotRequired[float | None]
    source_count: NotRequired[int]
    query_coverage: NotRequired[float]
    self_check_used: NotRequired[bool]
    self_check_passed: NotRequired[bool | None]
    trigger: NotRequired[str]
    snippets_ingested: NotRequired[int]
    errors: NotRequired[list[str]]
    status: NotRequired[str]
    context_count: NotRequired[int]
    citation_count: NotRequired[int]
    step_count: NotRequired[int]
    step_index: NotRequired[int]
    tool_name: NotRequired[str]
    success: NotRequired[bool]
    artifact_count: NotRequired[int]
    paper_count: NotRequired[int]
    chunks_indexed: NotRequired[int]
    source_type: NotRequired[str]
    source_url: NotRequired[str]
    pdf_url: NotRequired[str]
    discovered_by_query: NotRequired[str]
    trust_level: NotRequired[str]
    ingestion_status: NotRequired[str]
    issue_count: NotRequired[int]
    unsupported_claim_count: NotRequired[int]
    suggested_action: NotRequired[str]
    answer_chars: NotRequired[int]
    tool_result: NotRequired[dict[str, Any]]


def trace_event(stage: str, **fields: Any) -> AgentTraceEvent:
    return {"stage": stage, **fields}


def append_trace(
    trace: list[AgentTraceEvent],
    stage: str,
    **fields: Any,
) -> list[AgentTraceEvent]:
    return [*trace, trace_event(stage, **fields)]
