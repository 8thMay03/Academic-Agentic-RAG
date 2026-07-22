from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.api_router import api_router
from app.api.security import api_key_middleware
from app.config.logging import configure_logging
from app.middleware.request_id import request_id_middleware
from app.observability.tracing import configure_opentelemetry
from app.config.settings import settings
from app.services.pdf_index_service import PDFIndexService


async def index_local_pdfs_on_startup() -> None:
    if not settings.INDEX_LOCAL_PDFS_ON_STARTUP:
        return

    try:
        await PDFIndexService().index_all_downloaded_pdfs()
    except Exception as exc:
        print(f"Failed to index local PDFs on startup: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    await index_local_pdfs_on_startup()
    yield


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.middleware("http")(api_key_middleware)
    app.middleware("http")(request_id_middleware)
    configure_opentelemetry(app)
    app.include_router(api_router, prefix=settings.API_PREFIX)
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="localhost", port=8000, reload=True)
