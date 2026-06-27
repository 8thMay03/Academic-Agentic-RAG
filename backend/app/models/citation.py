from pydantic import BaseModel


class Citation(BaseModel):
    paper_id: str
    title: str
    page: int | None = None
    text: str | None = None

