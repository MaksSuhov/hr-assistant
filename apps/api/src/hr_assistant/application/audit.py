import uuid
from typing import Any

from hr_assistant.application.ports import UnitOfWork
from hr_assistant.infrastructure.db.models import AuditEventRecord


async def record_event(
    uow: UnitOfWork,
    *,
    actor_id: str,
    action: str,
    entity_type: str,
    entity_id: str,
    details: dict[str, Any],
) -> None:
    await uow.audit.add(
        AuditEventRecord(
            id=str(uuid.uuid4()),
            actor_id=actor_id,
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            details=details,
        )
    )
