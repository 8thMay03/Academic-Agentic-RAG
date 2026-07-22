from pathlib import Path

from fastapi import APIRouter

from app.config.settings import settings

router = APIRouter()


@router.get("/health")
async def health_check() -> dict[str, str | bool]:
    data_dir = Path(settings.DATA_DIR)
    chroma_dir = Path(settings.CHROMA_DIR)
    return {
        "status": "ok",
        "version": settings.APP_VERSION,
        "data_dir_configured": bool(settings.DATA_DIR),
        "chroma_dir_configured": bool(settings.CHROMA_DIR),
        "data_dir_exists": data_dir.exists(),
        "chroma_dir_exists": chroma_dir.exists(),
    }
