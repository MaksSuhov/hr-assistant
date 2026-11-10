from fastapi import APIRouter, Depends, Query, Request, Response

from hr_assistant.api.dependencies import get_actor_id, get_container
from hr_assistant.application.auth import authenticated_user
from hr_assistant.schemas.application import MergeCandidatesRequest
from hr_assistant.schemas.candidate import AuditEventOut, CandidateOut
from hr_assistant.schemas.matching import (
    CandidateSearchHit,
    CompareCandidatesRequest,
    CompareCandidatesResponse,
    SearchRequest,
)

router = APIRouter(dependencies=[Depends(authenticated_user)])


@router.get("", response_model=list[CandidateOut])
async def list_candidates(request: Request, limit: int = Query(default=100, ge=1, le=100)):
    return await get_container(request).candidates.list_candidates(limit)


@router.get("/{candidate_id}", response_model=CandidateOut)
async def get_candidate(candidate_id: str, request: Request):
    return await get_container(request).candidates.get(candidate_id)


@router.delete("/{candidate_id}", status_code=204)
async def delete_candidate(
    candidate_id: str,
    request: Request,
    actor_id: str = Depends(get_actor_id),
):
    await get_container(request).candidates.delete_candidate(candidate_id, actor_id)
    return Response(status_code=204)


@router.get("/{candidate_id}/history", response_model=list[AuditEventOut])
async def get_candidate_history(candidate_id: str, request: Request):
    return await get_container(request).candidates.history(candidate_id)


@router.post("/{candidate_id}/merge", response_model=CandidateOut)
async def merge_candidate(
    candidate_id: str,
    data: MergeCandidatesRequest,
    request: Request,
    actor_id: str = Depends(get_actor_id),
):
    return await get_container(request).candidates.merge_candidates(
        target_id=candidate_id,
        source_id=data.source_candidate_id,
        actor_id=actor_id,
    )


@router.post("/search", response_model=list[CandidateSearchHit])
async def search_candidates(data: SearchRequest, request: Request):
    return await get_container(request).search.search(data.query, data.limit)


@router.post("/compare", response_model=CompareCandidatesResponse)
async def compare_candidates(data: CompareCandidatesRequest, request: Request):
    return await get_container(request).matching.compare(data.vacancy_id, data.candidate_ids)
