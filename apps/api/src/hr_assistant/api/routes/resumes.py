from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from hr_assistant.api.dependencies import get_actor_id, get_container
from hr_assistant.application.auth import authenticated_user
from hr_assistant.application.errors import NotFoundError
from hr_assistant.schemas.candidate import CandidateOut

router = APIRouter()
hr_router = APIRouter(dependencies=[Depends(authenticated_user)])


async def _upload_resume(
    request: Request,
    file: UploadFile,
    candidate_id: str | None,
    candidate_name: str | None,
    source: str,
    actor_id: str,
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


@hr_router.post("", response_model=CandidateOut, status_code=201)
async def upload_hr_resume(
    request: Request,
    file: UploadFile = File(...),  # noqa: B008
    candidate_id: str | None = Form(default=None),
    candidate_name: str | None = Form(default=None),
    source: str = Form(default="HR upload"),
    actor_id: str = Depends(get_actor_id),
):
    return await _upload_resume(request, file, candidate_id, candidate_name, source, actor_id)


@router.post("/public", response_model=CandidateOut, status_code=201)
async def upload_candidate_resume(
    request: Request,
    file: UploadFile = File(...),  # noqa: B008
    candidate_name: str | None = Form(default=None),
):
    return await _upload_resume(
        request, file, None, candidate_name, "Candidate portal", "candidate-portal"
    )
