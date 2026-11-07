from typing import Annotated

from fastapi import Depends, Request

from hr_assistant.application.auth import authenticated_user
from hr_assistant.container import ServiceContainer
from hr_assistant.infrastructure.db.models import UserRecord


def get_container(request: Request) -> ServiceContainer:
    return request.app.state.container


async def get_actor_id(user: Annotated[UserRecord, Depends(authenticated_user)]) -> str:
    return user.email
