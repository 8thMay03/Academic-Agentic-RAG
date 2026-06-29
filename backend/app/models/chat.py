from pydantic import BaseModel
from pydantic import Field

from app.models.citation import Citation


class ChatRequest(BaseModel):
    question: str
    chat_id: str | None = None
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


class ChatSource(BaseModel):
    paper_id: str
    title: str
    filename: str | None = None
    path: str | None = None


class ChatSession(BaseModel):
    chat_id: str
    title: str
    sources: list[ChatSource] = Field(default_factory=list)
    messages: list[ChatHistoryMessage] = Field(default_factory=list)
    created_at: str
    updated_at: str


class ChatSessionCreateRequest(BaseModel):
    title: str | None = None


class ChatSessionUpdateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=160)


class ChatSourceAddRequest(ChatSource):
    pass


class ChatThread(BaseModel):
    chat_id: str
    title: str
    last_message: str
    updated_at: str
    message_count: int
    source_count: int = 0


class ChatThreadListResponse(BaseModel):
    chats: list[ChatThread] = Field(default_factory=list)
