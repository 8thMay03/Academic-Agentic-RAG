from pydantic import BaseModel


class Citation(BaseModel):
    paper_id: str
    title: str
    page_number: int | None = None
    page: int | None = None
    chunk_id: str | None = None
    text: str | None = None
