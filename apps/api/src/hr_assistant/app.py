import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastmcp.utilities.lifespan import combine_lifespans

from hr_assistant.api.router import api_router
from hr_assistant.application.errors import ApplicationError
from hr_assistant.container import ServiceContainer
from hr_assistant.core.config import Settings, get_settings
from hr_assistant.mcp.server import create_mcp_server

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    *,
    container: ServiceContainer | None = None,
) -> FastAPI:
    """Build the ASGI app from explicit settings and services.

    Tests pass a container with fake LLM, vector store, tracer and sender, so
    the app can be exercised without network calls or Qdrant.
    """
    resolved_settings = settings or get_settings()
    resolved_container = container or ServiceContainer(resolved_settings)
    mcp_http_app = create_mcp_server(resolved_container).http_app(path="/")

    @asynccontextmanager
    async def application_lifespan(_app: FastAPI):
        await resolved_container.database.initialize()
        try:
            await resolved_container.vector_store.ensure_collection()
        except Exception:
            logger.warning(
                "Qdrant is not available at startup; vector indexing will retry on upload."
            )
        yield
        await resolved_container.close()

    app = FastAPI(
        title=resolved_settings.app_name,
        version="0.1.0",
        description="HR workflow API with hybrid RAG, SSE chat, and human approval gates.",
        lifespan=combine_lifespans(application_lifespan, mcp_http_app.lifespan),
    )
    app.state.container = resolved_container
    app.include_router(api_router, prefix=resolved_settings.api_v1_prefix)
    app.mount("/mcp", mcp_http_app, name="mcp")

    @app.middleware("http")
    async def protect_mcp(request: Request, call_next):
        if request.url.path.startswith("/mcp"):
            from hr_assistant.application.auth import authenticated_user

            try:
                await authenticated_user(request)
            except HTTPException as error:
                return JSONResponse(status_code=error.status_code, content={"detail": error.detail})
        return await call_next(request)

    @app.exception_handler(ApplicationError)
    async def application_error_handler(_request: Request, error: ApplicationError):
        return JSONResponse(status_code=error.status_code, content={"detail": str(error)})

    return app
