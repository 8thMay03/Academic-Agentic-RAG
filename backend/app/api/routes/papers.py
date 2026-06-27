from fastapi import APIRouter

from app.models.paper import Paper

router = APIRouter()


@router.get("", response_model=list[Paper])
async def list_papers() -> list[Paper]:
    return []

