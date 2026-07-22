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
    request = state["request"]
    query_plan = state["query_plan"]
    planned_query = _with_retrieval_strategy(
        query_plan,
        request_top_k=request.top_k,
        request_score_threshold=request.score_threshold,
    )
    trace = append_trace(
        state.get("trace", []),
        "retrieval_planning",
        status="ready",
        query_type=planned_query.query_type,
        query_count=len(planned_query.search_queries),
        retrieval_mode=planned_query.retrieval_mode,
        per_query_top_k=planned_query.per_query_top_k,
        score_threshold=planned_query.score_threshold,
        max_total_chunks=planned_query.max_total_chunks,
        reason=planned_query.retrieval_reason,
    )
    return {**state, "query_plan": planned_query, "trace": trace}


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


def _with_retrieval_strategy(
    query_plan: QueryPlan,
    *,
    request_top_k: int,
    request_score_threshold: float | None,
) -> QueryPlan:
    query_count = max(1, len(query_plan.search_queries))
    base_top_k = max(1, request_top_k)
    mode = _retrieval_mode(query_plan.query_type)
    threshold = _planned_score_threshold(
        query_plan.query_type,
        request_score_threshold,
    )
    per_query_top_k = _planned_per_query_top_k(
        query_plan.query_type,
        base_top_k,
        query_count,
    )
    max_total_chunks = _planned_max_total_chunks(
        query_plan.query_type,
        base_top_k,
        per_query_top_k,
        query_count,
    )
    return QueryPlan(
        original_query=query_plan.original_query,
        query_type=query_plan.query_type,
        search_queries=query_plan.search_queries,
        reason=query_plan.reason,
        retrieval_mode=mode,
        per_query_top_k=per_query_top_k,
        score_threshold=threshold,
        max_total_chunks=max_total_chunks,
        retrieval_reason=_retrieval_reason(query_plan.query_type, query_count),
    )


def _retrieval_mode(query_type: str) -> str:
    if query_type == "comparison":
        return "comparative"
    if query_type == "paper_review":
        return "paper_review"
    if query_type == "multi_aspect":
        return "expanded"
    return "focused"


def _planned_score_threshold(query_type: str, score_threshold: float | None) -> float | None:
    if score_threshold is None:
        return None
    if query_type in {"multi_aspect", "paper_review"}:
        return max(0.0, score_threshold - 0.05)
    return score_threshold


def _planned_per_query_top_k(query_type: str, base_top_k: int, query_count: int) -> int:
    if query_count == 1:
        return base_top_k
    if query_type == "comparison":
        return max(2, min(base_top_k, 4))
    if query_type in {"multi_aspect", "paper_review"}:
        return max(2, min(base_top_k, 3))
    return base_top_k


def _planned_max_total_chunks(
    query_type: str,
    base_top_k: int,
    per_query_top_k: int,
    query_count: int,
) -> int:
    if query_count == 1:
        return base_top_k
    cap = 12 if query_type == "comparison" else 15
    return max(base_top_k, min(cap, per_query_top_k * query_count))


def _retrieval_reason(query_type: str, query_count: int) -> str:
    if query_count == 1:
        return "focused_local_retrieve_first"
    if query_type == "comparison":
        return "balanced_per_entity_local_retrieve_with_chunk_budget"
    if query_type == "paper_review":
        return "paper_review_aspect_queries_with_lower_threshold"
    if query_type == "multi_aspect":
        return "multi_aspect_queries_with_lower_threshold"
    return "local_retrieve_first_with_planned_queries"
