from typing import Annotated

from fastapi import Depends, HTTPException, Request

from hr_assistant.application.auth import authenticated_user
from hr_assistant.container import ServiceContainer
from hr_assistant.infrastructure.db.models import UserRecord


def get_container(request: Request) -> ServiceContainer:
    return request.app.state.container


async def get_actor_id(user: Annotated[UserRecord, Depends(authenticated_user)]) -> str:
    return user.email


async def require_admin(user: Annotated[UserRecord, Depends(authenticated_user)]) -> UserRecord:
    if user.role != "admin":
        raise HTTPException(403, "Administrator access is required.")
    return user


async def get_admin_actor_id(user: Annotated[UserRecord, Depends(require_admin)]) -> str:
    return user.email
