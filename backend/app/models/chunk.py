from pydantic import BaseModel, Field


class Chunk(BaseModel):
    chunk_id: str
    paper_id: str
    text: str
    page_number: int | None = None
    page: int | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
