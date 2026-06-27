from pydantic import BaseModel, Field

from app.config.constants import DEFAULT_MAX_RESULTS
from app.models.paper import Paper


class ResearchRequest(BaseModel):
    query: str
    max_results: int = Field(default=DEFAULT_MAX_RESULTS, ge=1, le=20)


class ResearchResponse(BaseModel):
    query: str
    papers: list[Paper]
    summary: str | None = None
    comparison: str | None = None
    report: str | None = None

