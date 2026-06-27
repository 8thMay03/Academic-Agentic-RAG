from pydantic import BaseModel

from app.models.paper import Paper


class CompareRequest(BaseModel):
    papers: list[Paper]


class CompareResponse(BaseModel):
    comparison: str

