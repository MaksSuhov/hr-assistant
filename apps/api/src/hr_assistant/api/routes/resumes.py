from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, Response, UploadFile

from hr_assistant.api.dependencies import get_actor_id, get_container
from hr_assistant.application.auth import authenticated_user
from hr_assistant.application.errors import NotFoundError
from hr_assistant.schemas.candidate import CandidateDocumentOut, CandidateOut

router = APIRouter()
hr_router = APIRouter(dependencies=[Depends(authenticated_user)])


async def _upload_resume(
    request: Request,
    file: UploadFile,
    candidate_id: str | None,
    candidate_name: str | None,
    source: str,
    actor_id: str,
    email: str | None = None,
    phone: str | None = None,
    vacancy_id: str | None = None,
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
            email=email,
            phone=phone,
            vacancy_id=vacancy_id,
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
    email: str | None = Form(default=None),
    phone: str | None = Form(default=None),
    vacancy_id: str | None = Form(default=None),
    source: str = Form(default="HR upload"),
    actor_id: str = Depends(get_actor_id),
):
    return await _upload_resume(
        request,
        file,
        candidate_id,
        candidate_name,
        source,
        actor_id,
        email=email,
        phone=phone,
        vacancy_id=vacancy_id,
    )


@hr_router.post("/{resume_id}/reindex", response_model=CandidateDocumentOut)
async def reindex_resume(
    resume_id: str,
    request: Request,
    actor_id: str = Depends(get_actor_id),
):
    return await get_container(request).candidates.reindex_resume(resume_id, actor_id)


@hr_router.delete("/{resume_id}", status_code=204)
async def delete_resume(
    resume_id: str,
    request: Request,
    actor_id: str = Depends(get_actor_id),
):
    await get_container(request).candidates.delete_resume(resume_id, actor_id)
    return Response(status_code=204)


@router.post("/public", response_model=CandidateOut, status_code=201)
async def upload_candidate_resume(
    request: Request,
    file: UploadFile = File(...),  # noqa: B008
    candidate_name: str | None = Form(default=None),
    email: str | None = Form(default=None),
    phone: str | None = Form(default=None),
    vacancy_id: str | None = Form(default=None),
):
    return await _upload_resume(
        request,
        file,
        None,
        candidate_name,
        "Candidate portal",
        "candidate-portal",
        email=email,
        phone=phone,
        vacancy_id=vacancy_id,
    )
