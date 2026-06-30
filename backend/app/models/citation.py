from pydantic import BaseModel, Field


class Citation(BaseModel):
    paper_id: str
    title: str
    page_number: int | None = None
    page: int | None = None
    chunk_id: str | None = None
    text: str | None = None
    score: float | None = None
    rerank_score: float | None = None
    vector_score: float | None = None
    keyword_score: float | None = None
    retrieval_sources: list[str] = Field(default_factory=list)
    evidence_quality: str | None = None
