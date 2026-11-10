from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from hr_assistant.infrastructure.db.models import CandidateRecord


class CandidateRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(self, candidate: CandidateRecord) -> None:
        self.session.add(candidate)
        await self.session.flush()

    async def get(self, candidate_id: str) -> CandidateRecord | None:
        return await self.session.get(CandidateRecord, candidate_id)

    async def list_all(self, limit: int = 100) -> list[CandidateRecord]:
        result = await self.session.scalars(
            select(CandidateRecord).order_by(CandidateRecord.created_at.desc()).limit(limit)
        )
        return list(result)

    async def find_by_email(self, email: str | None) -> CandidateRecord | None:
        if not email:
            return None
        return await self.session.scalar(
            select(CandidateRecord).where(CandidateRecord.email_normalized == email)
        )

    async def find_by_phone(self, phone: str | None) -> CandidateRecord | None:
        if not phone:
            return None
        return await self.session.scalar(
            select(CandidateRecord).where(CandidateRecord.phone_normalized == phone)
        )

    async def delete(self, candidate: CandidateRecord) -> None:
        await self.session.delete(candidate)
        await self.session.flush()
