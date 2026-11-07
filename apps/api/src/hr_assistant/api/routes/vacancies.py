from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field

from hr_assistant.api.dependencies import get_actor_id, get_container
from hr_assistant.application.auth import authenticated_user
from hr_assistant.container import ServiceContainer
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
    return await get_container(request).vacancies.list(limit)


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
        candidates = await container.candidates.list(limit=10)
        candidate_ids = [candidate.id for candidate in candidates]
    if not candidate_ids:
        return CompareCandidatesResponse(vacancy_id=vacancy_id, candidates=[])
    return await container.matching.compare(vacancy_id, candidate_ids)
