import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastmcp.utilities.lifespan import combine_lifespans

from hr_assistant.api.router import api_router
from hr_assistant.application.errors import ApplicationError
from hr_assistant.container import ServiceContainer
from hr_assistant.core.config import get_settings
from hr_assistant.mcp.server import create_mcp_server

settings = get_settings()
logging.basicConfig(level=settings.log_level.upper())
logger = logging.getLogger(__name__)
container = ServiceContainer(settings)
mcp_server = create_mcp_server(container)
mcp_http_app = mcp_server.http_app(path="/")


@asynccontextmanager
async def application_lifespan(app: FastAPI):
    await container.database.initialize()
    try:
        await container.vector_store.ensure_collection()
    except Exception:
        logger.warning("Qdrant is not available at startup; vector indexing will retry on upload.")
    yield
    await container.close()


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="HR workflow API with hybrid RAG, SSE chat, and human approval gates.",
    lifespan=combine_lifespans(application_lifespan, mcp_http_app.lifespan),
)
app.state.container = container
app.include_router(api_router, prefix=settings.api_v1_prefix)
app.mount("/mcp", mcp_http_app, name="mcp")


@app.exception_handler(ApplicationError)
async def application_error_handler(_request: Request, error: ApplicationError):
    return JSONResponse(status_code=error.status_code, content={"detail": str(error)})

