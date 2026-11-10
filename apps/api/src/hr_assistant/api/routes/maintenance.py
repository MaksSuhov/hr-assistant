from fastapi import APIRouter, Depends, Query, Request

from hr_assistant.api.dependencies import get_admin_actor_id, get_container
from hr_assistant.application.auth import authenticated_user
from hr_assistant.schemas.application import RetentionPurgeResult

router = APIRouter(dependencies=[Depends(authenticated_user)])


@router.post("/retention/purge", response_model=RetentionPurgeResult)
async def purge_expired_records(
    request: Request,
    days: int | None = Query(default=None, ge=0),
    audit_days: int | None = Query(default=None, ge=0),
    actor_id: str = Depends(get_admin_actor_id),
):
    """Delete candidates older than the retention window.

    Defaults come from ``RESUME_RETENTION_DAYS`` and ``AUDIT_RETENTION_DAYS``.
    When both are zero the endpoint is a no-op, so an accidental call cannot
    erase data that has no configured policy.
    """

    return await get_container(request).candidates.purge_expired(
        days=days,
        audit_days=audit_days,
        actor_id=actor_id,
    )
