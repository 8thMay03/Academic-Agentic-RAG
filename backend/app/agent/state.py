from typing import Any, TypedDict

from app.agent.models import (
    AgentLimits,
    AgentTraceEvent,
    ChatWorkflowRequest,
    ContextQuality,
    PlannerDecision,
    QueryPlan,
    ResearchPlan,
    StopReason,
    ToolCall,
    ToolResult,
    RetrievedChunk,
    VerificationResult,
)
from app.models.citation import Citation


class AgenticRAGState(TypedDict, total=False):
    request: ChatWorkflowRequest
    intent: str
    local_chunks: list[RetrievedChunk]
    web_chunks: list[RetrievedChunk]
    evidence: list[RetrievedChunk]
    quality: ContextQuality
    query_plan: QueryPlan
    citations: list[Citation]
    answer: str
    verification: VerificationResult
    answer_verifier: Any
    citation_grounder: Any
    llm_service: Any
    prompt_builder: Any
    quality_evaluator: Any
    tool_registry: Any
    prompt: str | None
    trace: list[AgentTraceEvent]
    plan: ResearchPlan
    current_step_index: int
    tool_calls: list[ToolCall]
    current_tool_result: ToolResult
    tool_results: list[ToolResult]
    limits: AgentLimits
    refreshed_local_context: bool
    stop_reason: StopReason
    planner_decision: PlannerDecision
