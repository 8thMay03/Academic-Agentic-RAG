from pydantic import BaseModel

from app.models.citation import Citation


class ChatRequest(BaseModel):
    question: str
    paper_ids: list[str] | None = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]

