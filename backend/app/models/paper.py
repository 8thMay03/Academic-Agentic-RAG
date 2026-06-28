from typing import Literal

from pydantic import BaseModel, Field, HttpUrl

from app.config.constants import DEFAULT_MAX_RESULTS


class Paper(BaseModel):
    paper_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    abstract: str | None = None
    published: str | None = None
    arxiv_url: HttpUrl | None = None
    url: HttpUrl | None = None
    pdf_url: HttpUrl | None = None


class PaperSearchRequest(BaseModel):
    query: str
    max_results: int = Field(default=DEFAULT_MAX_RESULTS, ge=1, le=20)
    sort_by: Literal["relevance", "lastUpdatedDate", "submittedDate"] = "submittedDate"


class PaperSearchResponse(BaseModel):
    query: str
    papers: list[Paper]


class PaperDownloadRequest(BaseModel):
    pdf_urls: list[HttpUrl] = Field(min_length=1, max_length=20)


class PaperDownloadFailure(BaseModel):
    pdf_url: HttpUrl
    error: str


class PaperDownloadResponse(BaseModel):
    files: list[str]
    cached_files: list[str] = Field(default_factory=list)
    errors: list[PaperDownloadFailure] = Field(default_factory=list)


class DownloadedPDF(BaseModel):
    filename: str
    path: str
    size_bytes: int
    modified_at: str


class DownloadedPDFIndexRequest(BaseModel):
    filename: str


class DownloadedPDFIndexResponse(BaseModel):
    paper_id: str
    filename: str
    chunks_indexed: int
