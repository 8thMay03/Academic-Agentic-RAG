from pydantic import BaseModel
from pydantic import Field

from app.models.citation import Citation


class ChatRequest(BaseModel):
    question: str
    paper_ids: list[str] | None = None
    top_k: int = Field(default=5, ge=1, le=20)
    score_threshold: float = Field(default=0.65, ge=0, le=1)


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]


class ChatHistoryMessage(BaseModel):
    role: str
    content: str
    citations: list[Citation] = Field(default_factory=list)
    created_at: str


class ChatHistoryResponse(BaseModel):
    paper_id: str
    messages: list[ChatHistoryMessage] = Field(default_factory=list)
