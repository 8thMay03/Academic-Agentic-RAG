from typing import Any, TypedDict

from app.models.paper import Paper


class ResearchState(TypedDict, total=False):
    query: str
    max_results: int
    user_intent: str
    plan: dict[str, Any]
    search_queries: list[str]
    attempted_queries: list[str]
    search_iterations: int
    max_search_iterations: int
    papers: list[Paper]
    selected_papers: list[Paper]
    rejected_papers: list[dict[str, Any]]
    downloaded_files: list[dict[str, Any]]
    parsed_documents: list[dict[str, Any]]
    indexed_paper_ids: list[str]
    summaries: list[dict[str, Any]]
    summary: str
    comparison: str
    report: str
    critique: dict[str, Any]
    next_action: str
    reflection_iterations: int
    max_reflection_iterations: int
    errors: list[dict[str, Any]]
