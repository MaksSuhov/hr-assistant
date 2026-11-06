from __future__ import annotations

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from hr_assistant.infrastructure.db.models import RequirementRecord, VacancyRecord


class VacancyRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(self, vacancy: VacancyRecord) -> None:
        self.session.add(vacancy)
        await self.session.flush()

    async def get(self, vacancy_id: str) -> VacancyRecord | None:
        return await self.session.get(VacancyRecord, vacancy_id)

    async def list(self, limit: int = 100) -> list[VacancyRecord]:
        result = await self.session.scalars(
            select(VacancyRecord).order_by(VacancyRecord.created_at.desc()).limit(limit)
        )
        return list(result)

    async def requirements(self, vacancy_id: str) -> list[RequirementRecord]:
        result = await self.session.scalars(
            select(RequirementRecord)
            .where(RequirementRecord.vacancy_id == vacancy_id)
            .order_by(RequirementRecord.id)
        )
        return list(result)

    async def add_requirements(self, requirements: list[RequirementRecord]) -> None:
        self.session.add_all(requirements)
        await self.session.flush()

    async def replace_requirements(
        self, vacancy_id: str, requirements: list[RequirementRecord]
    ) -> None:
        await self.session.execute(
            delete(RequirementRecord).where(RequirementRecord.vacancy_id == vacancy_id)
        )
        self.session.add_all(requirements)
        await self.session.flush()

