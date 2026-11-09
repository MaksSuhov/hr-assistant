from fastapi import APIRouter, Depends, Query, Request

from hr_assistant.api.dependencies import get_actor_id, get_container
from hr_assistant.application.auth import authenticated_user
from hr_assistant.schemas.approval import (
    ApprovalDecision,
    ApprovalOut,
    ApprovalResult,
    MessageDraftRequest,
    OutboxMessageOut,
    StatusChangeRequest,
)

router = APIRouter(dependencies=[Depends(authenticated_user)])


@router.get("/pending", response_model=list[ApprovalOut])
async def pending_approvals(request: Request, limit: int = Query(default=100, ge=1, le=100)):
    return await get_container(request).approvals.list_pending(limit)


@router.get("/outbox", response_model=list[OutboxMessageOut])
async def message_outbox(request: Request, limit: int = Query(default=100, ge=1, le=100)):
    return await get_container(request).approvals.list_outbox(limit)


@router.post(
    "/candidates/{candidate_id}/status-change/request", response_model=ApprovalOut, status_code=202
)
async def request_candidate_status_change(
    candidate_id: str,
    data: StatusChangeRequest,
    request: Request,
    actor_id: str = Depends(get_actor_id),
):
    return await get_container(request).approvals.request_status_change(
        candidate_id, data, actor_id
    )


@router.post("/messages/drafts", response_model=ApprovalOut, status_code=202)
async def request_message_approval(
    data: MessageDraftRequest,
    request: Request,
    actor_id: str = Depends(get_actor_id),
):
    return await get_container(request).approvals.request_message(data, actor_id)


@router.post("/{approval_id}/approve", response_model=ApprovalResult)
async def approve_action(
    approval_id: str,
    data: ApprovalDecision,
    request: Request,
    actor_id: str = Depends(get_actor_id),
):
    return await get_container(request).approvals.approve(approval_id, actor_id, data.note)


@router.post("/{approval_id}/reject", response_model=ApprovalResult)
async def reject_action(
    approval_id: str,
    data: ApprovalDecision,
    request: Request,
    actor_id: str = Depends(get_actor_id),
):
    return await get_container(request).approvals.reject(approval_id, actor_id, data.note)
