from typing import Any, Mapping

from pydantic import BaseModel
from pydantic import Field

from app.models.citation import Citation


class ChatRequest(BaseModel):
    question: str
    chat_id: str | None = None
    paper_ids: list[str] | None = None
    top_k: int = Field(default=5, ge=1, le=20)
    score_threshold: float = Field(default=0.65, ge=0, le=1)
    max_agent_steps: int = Field(default=6, ge=1, le=10)
    enable_web_search: bool = True
    enable_research_ingest: bool = True
    auto_download_pdfs: bool = True


class AgentTraceEventResponse(BaseModel):
    stage: str
    chunk_count: int | None = None
    paper_ids: list[str] | None = None
    sufficient: bool | None = None
    intent: str | None = None
    reason: str | None = None
    context_chars: int | None = None
    top_score: float | None = None
    average_score: float | None = None
    source_count: int | None = None
    query_coverage: float | None = None
    self_check_used: bool | None = None
    self_check_passed: bool | None = None
    trigger: str | None = None
    snippets_ingested: int | None = None
    errors: list[str] | None = None
    status: str | None = None
    context_count: int | None = None
    citation_count: int | None = None
    step_count: int | None = None
    step_index: int | None = None
    tool_name: str | None = None
    success: bool | None = None
    artifact_count: int | None = None
    paper_count: int | None = None
    chunks_indexed: int | None = None
    source_type: str | None = None
    source_url: str | None = None
    pdf_url: str | None = None
    discovered_by_query: str | None = None
    trust_level: str | None = None
    ingestion_status: str | None = None
    issue_count: int | None = None
    unsupported_claim_count: int | None = None
    supported_claim_count: int | None = None
    contradicted_claim_count: int | None = None
    insufficient_claim_count: int | None = None
    claim_citation_map: list[dict[str, Any]] | None = None
    suggested_action: str | None = None
    answer_chars: int | None = None
    tool_result: dict[str, Any] | None = None
    query_type: str | None = None
    query_count: int | None = None
    queries: list[str] | None = None
    stop_reason: str | None = None
    selected_tools: list[str] | None = None
    stop_condition: str | None = None
    risk_notes: list[str] | None = None
    available_tools: list[str] | None = None
    planner_source: str | None = None
    latency_ms: float | None = None
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    estimated_cost_usd: float | None = None
    tool_estimated_cost_usd: float | None = None
    embedding_model: str | None = None
    embedding_input_count: int | None = None
    embedding_tokens: int | None = None
    embedding_estimated_cost_usd: float | None = None
    suspicious_context_count: int | None = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    trace: list[AgentTraceEventResponse] = Field(default_factory=list)
    stop_reason: str | None = None


def agent_trace_event_payload(event: Mapping[str, Any]) -> dict[str, Any]:
    return AgentTraceEventResponse.model_validate(event).model_dump(mode="json", exclude_none=True)


def agent_trace_payload(trace: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [agent_trace_event_payload(event) for event in trace]


class ChatHistoryMessage(BaseModel):
    role: str
    content: str
    citations: list[Citation] = Field(default_factory=list)
    trace: list[AgentTraceEventResponse] = Field(default_factory=list)
    stop_reason: str | None = None
    created_at: str


class ChatHistoryResponse(BaseModel):
    paper_id: str
    messages: list[ChatHistoryMessage] = Field(default_factory=list)


class ChatSource(BaseModel):
    paper_id: str
    title: str
    filename: str | None = None
    path: str | None = None


class ChatSession(BaseModel):
    chat_id: str
    title: str
    sources: list[ChatSource] = Field(default_factory=list)
    messages: list[ChatHistoryMessage] = Field(default_factory=list)
    created_at: str
    updated_at: str


class ChatSessionCreateRequest(BaseModel):
    title: str | None = None


class ChatSessionUpdateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=160)


class ChatSourceAddRequest(ChatSource):
    pass


class ChatThread(BaseModel):
    chat_id: str
    title: str
    last_message: str
    updated_at: str
    message_count: int
    source_count: int = 0


class ChatThreadListResponse(BaseModel):
    chats: list[ChatThread] = Field(default_factory=list)


class ResearchFinding(BaseModel):
    finding_id: str
    chat_id: str
    run_id: str
    question: str
    summary: str
    source_ids: list[str] = Field(default_factory=list)
    citation_ids: list[str] = Field(default_factory=list)
    confidence: str = "unknown"
    created_at: str


class ResearchFindingListResponse(BaseModel):
    findings: list[ResearchFinding] = Field(default_factory=list)


class AgentRunUsageSummary(BaseModel):
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    embedding_tokens: int = 0
    estimated_cost_usd: float = 0.0
    tool_call_count: int = 0
    models: list[str] = Field(default_factory=list)


class AgentRunRecord(BaseModel):
    run_id: str
    chat_id: str
    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    trace: list[AgentTraceEventResponse] = Field(default_factory=list)
    stop_reason: str | None = None
    usage: AgentRunUsageSummary = Field(default_factory=AgentRunUsageSummary)
    findings: list[ResearchFinding] = Field(default_factory=list)
    created_at: str


class AgentRunListResponse(BaseModel):
    runs: list[AgentRunRecord] = Field(default_factory=list)
