from pydantic import BaseModel


class SummaryRequest(BaseModel):
    title: str
    text: str


class SummaryResponse(BaseModel):
    title: str
    summary: str

