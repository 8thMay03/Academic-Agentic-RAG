from typing import TypedDict

from app.models.paper import Paper


class ResearchState(TypedDict, total=False):
    query: str
    max_results: int
    papers: list[Paper]
    summaries: list[str]
    comparison: str
    report: str

