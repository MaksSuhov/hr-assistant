from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from hr_assistant.infrastructure.db.models import (
    ApplicationRecord,
    CandidateRecord,
    VacancyRecord,
)


class ApplicationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(self, application: ApplicationRecord) -> None:
        self.session.add(application)
        await self.session.flush()

    async def get(self, application_id: str) -> ApplicationRecord | None:
        return await self.session.get(ApplicationRecord, application_id)

    async def get_for_pair(self, candidate_id: str, vacancy_id: str) -> ApplicationRecord | None:
        return await self.session.scalar(
            select(ApplicationRecord).where(
                ApplicationRecord.candidate_id == candidate_id,
                ApplicationRecord.vacancy_id == vacancy_id,
            )
        )

    async def list_for_candidate_with_vacancy(
        self, candidate_id: str
    ) -> list[tuple[ApplicationRecord, VacancyRecord]]:
        result = await self.session.execute(
            select(ApplicationRecord, VacancyRecord)
            .join(VacancyRecord, VacancyRecord.id == ApplicationRecord.vacancy_id)
            .where(ApplicationRecord.candidate_id == candidate_id)
            .order_by(ApplicationRecord.created_at)
        )
        return [(application, vacancy) for application, vacancy in result.all()]

    async def list_for_vacancy_with_candidate(
        self, vacancy_id: str, limit: int = 200
    ) -> list[tuple[ApplicationRecord, CandidateRecord]]:
        result = await self.session.execute(
            select(ApplicationRecord, CandidateRecord)
            .join(CandidateRecord, CandidateRecord.id == ApplicationRecord.candidate_id)
            .where(ApplicationRecord.vacancy_id == vacancy_id)
            .order_by(ApplicationRecord.created_at)
            .limit(limit)
        )
        return [(application, candidate) for application, candidate in result.all()]

    async def list_for_candidate(self, candidate_id: str) -> list[ApplicationRecord]:
        result = await self.session.scalars(
            select(ApplicationRecord)
            .where(ApplicationRecord.candidate_id == candidate_id)
            .order_by(ApplicationRecord.created_at)
        )
        return list(result)

    async def count_for_vacancy(self, vacancy_id: str) -> int:
        return int(
            await self.session.scalar(
                select(func.count())
                .select_from(ApplicationRecord)
                .where(ApplicationRecord.vacancy_id == vacancy_id)
            )
            or 0
        )

    async def delete(self, application: ApplicationRecord) -> None:
        await self.session.delete(application)
        await self.session.flush()
