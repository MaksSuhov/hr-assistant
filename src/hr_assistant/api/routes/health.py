from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health", tags=["health"])
async def health(request: Request) -> dict:
    container = request.app.state.container
    return {
        "status": "ok",
        "openai_configured": container.llm.configured,
        "qdrant_url": container.settings.qdrant_url,
        "langfuse_configured": container.settings.langfuse_configured,
    }

