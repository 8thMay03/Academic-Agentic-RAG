from fastapi import FastAPI

from app.api.api_router import api_router
from app.config.settings import settings


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
    )
    app.include_router(api_router, prefix=settings.API_PREFIX)
    return app


app = create_app()

