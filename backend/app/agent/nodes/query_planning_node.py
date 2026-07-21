import re

from app.agent.models import QueryPlan, append_trace
from app.agent.state import AgenticRAGState


COMPARISON_TERMS = (
    "compare",
    "comparison",
    "differ",
    "different",
    "difference",
    "versus",
    " vs ",
    "khác",
    "so sánh",
    "so sanh",
    "khác gì",
)
MULTI_ASPECT_TERMS = (
    "ưu điểm",
    "nhược điểm",
    "hạn chế",
    "ứng dụng",
    "cách hoạt động",
    "nguyên lý",
    "advantages",
    "disadvantages",
    "limitations",
    "applications",
    "how it works",
)
PAPER_REVIEW_TERMS = (
    "tóm tắt paper",
    "đóng góp",
    "kết quả thực nghiệm",
    "phương pháp",
    "summary of the paper",
    "contribution",
    "experiment",
    "methodology",
)
ANCHOR_PATTERN = re.compile(r"\b[A-Z][A-Z0-9_-]{1,}\b")


async def query_planning_node(state: AgenticRAGState) -> AgenticRAGState:
    request = state["request"]
    question = " ".join(request.question.split())
    normalized_question = f" {question.lower()} "

    query_type = "simple_lookup"
    reason = "single_focus_question"
    if any(term in normalized_question for term in PAPER_REVIEW_TERMS):
        query_type = "paper_review"
        reason = "question_requests_paper_review_aspects"
    elif any(term in normalized_question for term in COMPARISON_TERMS):
        query_type = "comparison"
        reason = "question_compares_multiple_items"
    elif sum(1 for term in MULTI_ASPECT_TERMS if term in normalized_question) >= 2:
        query_type = "multi_aspect"
        reason = "question_requests_multiple_aspects"

    trace = append_trace(
        state.get("trace", []),
        "query_planning",
        query_type=query_type,
        reason=reason,
    )
    return {
        **state,
        "query_plan": QueryPlan(
            original_query=question,
            query_type=query_type,
            search_queries=[question],
            reason=reason,
        ),
        "trace": trace,
    }


async def query_decomposition_node(state: AgenticRAGState) -> AgenticRAGState:
    query_plan = state["query_plan"]
    search_queries = _decompose_queries(query_plan)
    decomposed_plan = QueryPlan(
        original_query=query_plan.original_query,
        query_type=query_plan.query_type,
        search_queries=search_queries,
        reason=query_plan.reason,
    )
    trace = append_trace(
        state.get("trace", []),
        "query_decomposition",
        query_type=decomposed_plan.query_type,
        query_count=len(search_queries),
        queries=search_queries,
        reason=decomposed_plan.reason,
    )
    return {
        **state,
        "query_plan": decomposed_plan,
        "trace": trace,
    }


async def retrieval_planning_node(state: AgenticRAGState) -> AgenticRAGState:
    query_plan = state["query_plan"]
    trace = append_trace(
        state.get("trace", []),
        "retrieval_planning",
        status="ready",
        query_type=query_plan.query_type,
        query_count=len(query_plan.search_queries),
        reason="local_retrieve_first_with_planned_queries",
    )
    return {**state, "trace": trace}


def _decompose_queries(query_plan: QueryPlan) -> list[str]:
    question = query_plan.original_query
    queries = [question]

    if query_plan.query_type == "comparison":
        anchors = _anchor_terms(question)
        queries.extend(anchors)
        if len(anchors) >= 2:
            queries.append(f"{' vs '.join(anchors)} comparison")
    elif query_plan.query_type == "multi_aspect":
        for aspect in ("definition", "mechanism", "advantages", "limitations", "applications"):
            queries.append(f"{question} {aspect}")
    elif query_plan.query_type == "paper_review":
        for aspect in ("method", "contribution", "experiments", "limitations", "results"):
            queries.append(f"{question} {aspect}")

    return _dedupe_queries(queries)[:6]


def _anchor_terms(question: str) -> list[str]:
    anchors = []
    seen = set()
    for match in ANCHOR_PATTERN.finditer(question):
        anchor = match.group(0)
        if anchor not in seen:
            seen.add(anchor)
            anchors.append(anchor)
    return anchors


def _dedupe_queries(queries: list[str]) -> list[str]:
    deduped = []
    seen = set()
    for query in queries:
        normalized = " ".join(query.split()).lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(" ".join(query.split()))
    return deduped
