from fastapi import APIRouter

from app.api.routes import chat, compare, health, papers, research, reports, search, summary

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(research.router, prefix="/research", tags=["research"])
api_router.include_router(search.router, prefix="/search", tags=["search"])
api_router.include_router(summary.router, prefix="/summary", tags=["summary"])
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
api_router.include_router(compare.router, prefix="/compare", tags=["compare"])
api_router.include_router(papers.router, prefix="/papers", tags=["papers"])
api_router.include_router(reports.router, prefix="/reports", tags=["reports"])

