from pydantic import BaseModel, Field, HttpUrl

from app.config.constants import DEFAULT_MAX_RESULTS


class Paper(BaseModel):
    paper_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    abstract: str | None = None
    published: str | None = None
    url: HttpUrl | None = None
    pdf_url: HttpUrl | None = None


class PaperSearchRequest(BaseModel):
    query: str
    max_results: int = Field(default=DEFAULT_MAX_RESULTS, ge=1, le=20)


class PaperSearchResponse(BaseModel):
    query: str
    papers: list[Paper]

