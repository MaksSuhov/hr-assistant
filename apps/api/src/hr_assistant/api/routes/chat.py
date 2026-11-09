import json
import logging

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import StreamingResponse

from hr_assistant.api.dependencies import get_actor_id, get_container
from hr_assistant.application.auth import authenticated_user
from hr_assistant.application.errors import ApplicationError
from hr_assistant.schemas.chat import ChatRequest

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(authenticated_user)])


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/stream")
async def stream_chat(
    data: ChatRequest,
    request: Request,
    actor_id: str = Depends(get_actor_id),
    include_eval_metrics: bool = Header(default=False, alias="X-Eval-Metrics"),
):
    container = get_container(request)
    if not container.chat.configured:
        raise HTTPException(status_code=503, detail="Set OPENAI_API_KEY to enable assistant chat.")

    async def event_stream():
        try:
            async for event, payload in container.chat.stream_chat(
                data.message,
                data.conversation_id,
                actor_id,
                include_eval_metrics=include_eval_metrics,
            ):
                yield _sse(event, payload)
        except ApplicationError as error:
            yield _sse("error", {"status": error.status_code, "message": str(error)})
        except Exception:
            logger.exception("Chat stream failed for actor %s", actor_id)
            yield _sse("error", {"status": 500, "message": "Chat request failed."})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
