from pydantic import BaseModel


class ReportRequest(BaseModel):
    title: str
    paper_ids: list[str]


class ReportResponse(BaseModel):
    title: str
    content: str

