from fastapi import Header, Request

from hr_assistant.container import ServiceContainer


def get_container(request: Request) -> ServiceContainer:
    return request.app.state.container


async def get_actor_id(x_actor_id: str = Header(default="local-hr")) -> str:
    return x_actor_id.strip() or "local-hr"

