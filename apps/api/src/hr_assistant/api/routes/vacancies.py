from fastapi import APIRouter, Depends, Query, Request, Response
from pydantic import BaseModel, Field

from hr_assistant.api.dependencies import get_actor_id, get_container
from hr_assistant.application.auth import authenticated_user
from hr_assistant.container import ServiceContainer
from hr_assistant.schemas.application import (
    ApplicationCreate,
    ApplicationOut,
    StageChangeRequest,
)
from hr_assistant.schemas.approval import ApprovalOut
from hr_assistant.schemas.matching import CompareCandidatesResponse
from hr_assistant.schemas.vacancy import VacancyCreate, VacancyOut, VacancyUpdate

router = APIRouter(dependencies=[Depends(authenticated_user)])


class VacancyMatchRequest(BaseModel):
    candidate_ids: list[str] = Field(default_factory=list, max_length=10)


@router.post("", response_model=VacancyOut, status_code=201)
async def create_vacancy(
    data: VacancyCreate,
    request: Request,
    actor_id: str = Depends(get_actor_id),
):
    return await get_container(request).vacancies.create(data, actor_id)


@router.get("", response_model=list[VacancyOut])
async def list_vacancies(request: Request, limit: int = Query(default=100, ge=1, le=100)):
    return await get_container(request).vacancies.list_vacancies(limit)


@router.get("/{vacancy_id}", response_model=VacancyOut)
async def get_vacancy(vacancy_id: str, request: Request):
    return await get_container(request).vacancies.get(vacancy_id)


@router.patch("/{vacancy_id}", response_model=VacancyOut)
async def update_vacancy(
    vacancy_id: str,
    data: VacancyUpdate,
    request: Request,
    actor_id: str = Depends(get_actor_id),
):
    return await get_container(request).vacancies.update(vacancy_id, data, actor_id)


@router.post("/{vacancy_id}/matches", response_model=CompareCandidatesResponse)
async def match_vacancy(
    vacancy_id: str,
    data: VacancyMatchRequest,
    request: Request,
):
    container: ServiceContainer = get_container(request)
    await container.vacancies.get(vacancy_id)
    candidate_ids = data.candidate_ids
    if not candidate_ids:
        candidates = await container.candidates.list_candidates(limit=10)
        candidate_ids = [candidate.id for candidate in candidates]
    if not candidate_ids:
        return CompareCandidatesResponse(vacancy_id=vacancy_id, candidates=[])
    return await container.matching.compare(vacancy_id, candidate_ids)


@router.get("/{vacancy_id}/applications", response_model=list[ApplicationOut])
async def list_applications(
    vacancy_id: str,
    request: Request,
    limit: int = Query(default=200, ge=1, le=500),
):
    return await get_container(request).applications.list_for_vacancy(vacancy_id, limit)


@router.post("/{vacancy_id}/applications", response_model=ApplicationOut, status_code=201)
async def create_application(
    vacancy_id: str,
    data: ApplicationCreate,
    request: Request,
    actor_id: str = Depends(get_actor_id),
):
    return await get_container(request).applications.apply(
        candidate_id=data.candidate_id,
        vacancy_id=vacancy_id,
        actor_id=actor_id,
        stage=data.stage,
    )


@router.get("/{vacancy_id}/applications/{application_id}", response_model=ApplicationOut)
async def get_application(vacancy_id: str, application_id: str, request: Request):
    return await get_container(request).applications.get(vacancy_id, application_id)


@router.delete("/{vacancy_id}/applications/{application_id}", status_code=204)
async def delete_application(
    vacancy_id: str,
    application_id: str,
    request: Request,
    actor_id: str = Depends(get_actor_id),
):
    await get_container(request).applications.remove(vacancy_id, application_id, actor_id)
    return Response(status_code=204)


@router.post(
    "/{vacancy_id}/applications/{application_id}/stage-change/request",
    response_model=ApprovalOut,
    status_code=202,
)
async def request_stage_change(
    vacancy_id: str,
    application_id: str,
    data: StageChangeRequest,
    request: Request,
    actor_id: str = Depends(get_actor_id),
):
    container: ServiceContainer = get_container(request)
    # Validate the nested vacancy id before creating an approval.
    await container.applications.get(vacancy_id, application_id)
    return await container.approvals.request_stage_change(application_id, data, actor_id)
