from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from hr_assistant.api.dependencies import get_actor_id, get_container
from hr_assistant.application.errors import NotFoundError
from hr_assistant.schemas.candidate import CandidateOut

router = APIRouter()


@router.post("", response_model=CandidateOut, status_code=201)
async def upload_resume(
    request: Request,
    file: UploadFile = File(...),
    candidate_id: str | None = Form(default=None),
    candidate_name: str | None = Form(default=None),
    source: str = Form(default="HR upload"),
    actor_id: str = Depends(get_actor_id),
):
    container = get_container(request)
    content = await file.read(container.settings.max_upload_bytes + 1)
    try:
        return await container.candidates.upload_resume(
            content=content,
            filename=file.filename or "resume.pdf",
            actor_id=actor_id,
            candidate_id=candidate_id,
            candidate_name=candidate_name,
            source=source,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except NotFoundError:
        raise

