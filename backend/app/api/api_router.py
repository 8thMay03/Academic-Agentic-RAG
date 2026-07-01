from fastapi import APIRouter

from app.api.routes import chat, health, papers

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
api_router.include_router(papers.router, prefix="/papers", tags=["papers"])
