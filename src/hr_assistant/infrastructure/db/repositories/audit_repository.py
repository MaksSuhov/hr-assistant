from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from hr_assistant.infrastructure.db.models import AuditEventRecord


class AuditRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(self, event: AuditEventRecord) -> None:
        self.session.add(event)
        await self.session.flush()

    async def list_for_entity(self, entity_type: str, entity_id: str) -> list[AuditEventRecord]:
        result = await self.session.scalars(
            select(AuditEventRecord)
            .where(
                AuditEventRecord.entity_type == entity_type,
                AuditEventRecord.entity_id == entity_id,
            )
            .order_by(AuditEventRecord.created_at.desc())
        )
        return list(result)

