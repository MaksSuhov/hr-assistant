from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from hr_assistant.infrastructure.db.models import ApprovalActionRecord, OutboxMessageRecord


class ApprovalRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(self, approval: ApprovalActionRecord) -> None:
        self.session.add(approval)
        await self.session.flush()

    async def get(self, approval_id: str) -> ApprovalActionRecord | None:
        return await self.session.get(ApprovalActionRecord, approval_id)

    async def list_pending(self, limit: int = 100) -> list[ApprovalActionRecord]:
        result = await self.session.scalars(
            select(ApprovalActionRecord)
            .where(ApprovalActionRecord.status == "pending")
            .order_by(ApprovalActionRecord.created_at)
            .limit(limit)
        )
        return list(result)

    async def enqueue_message(self, message: OutboxMessageRecord) -> None:
        self.session.add(message)
        await self.session.flush()

    async def list_outbox(self, limit: int = 100) -> list[OutboxMessageRecord]:
        result = await self.session.scalars(
            select(OutboxMessageRecord).order_by(OutboxMessageRecord.queued_at.desc()).limit(limit)
        )
        return list(result)

    async def update_outbox_status(
        self, approval_id: str, status: str, last_error: str | None = None
    ) -> None:
        result = await self.session.scalars(
            select(OutboxMessageRecord).where(OutboxMessageRecord.approval_id == approval_id)
        )
        message = result.first()
        if message:
            message.status = status
            message.last_error = last_error
            await self.session.flush()

    async def reassign_outbox(self, source_id: str, target_id: str) -> int:
        result = await self.session.scalars(
            select(OutboxMessageRecord).where(OutboxMessageRecord.candidate_id == source_id)
        )
        messages = list(result)
        for message in messages:
            message.candidate_id = target_id
        await self.session.flush()
        return len(messages)
