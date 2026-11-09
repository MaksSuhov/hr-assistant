from __future__ import annotations

import uuid
from typing import Any

from hr_assistant.application.audit import record_event
from hr_assistant.application.errors import NotFoundError
from hr_assistant.application.ports import UnitOfWorkProvider
from hr_assistant.domain.enums import VacancyStatus
from hr_assistant.infrastructure.db.models import RequirementRecord, VacancyRecord
from hr_assistant.schemas.vacancy import (
    RequirementInput,
    RequirementOut,
    VacancyCreate,
    VacancyOut,
    VacancyUpdate,
)


class VacancyService:
    def __init__(self, uow_provider: UnitOfWorkProvider) -> None:
        self.uow_provider = uow_provider

    async def create(self, data: VacancyCreate, actor_id: str) -> VacancyOut:
        vacancy_id = str(uuid.uuid4())
        async with self.uow_provider() as uow:
            record = VacancyRecord(
                id=vacancy_id,
                title=data.title,
                department=data.department,
                description=data.description,
                status="open",
            )
            await uow.vacancies.add(record)
            requirements = self._make_requirements(vacancy_id, data.requirements)
            await uow.vacancies.add_requirements(requirements)
            await record_event(
                uow,
                actor_id=actor_id,
                action="vacancy_created",
                entity_type="vacancy",
                entity_id=vacancy_id,
                details={"title": data.title, "requirements": [r.text for r in data.requirements]},
            )
            await uow.commit()
            return await self._to_output(uow, record)

    async def update(self, vacancy_id: str, data: VacancyUpdate, actor_id: str) -> VacancyOut:
        async with self.uow_provider() as uow:
            record = await uow.vacancies.get(vacancy_id)
            if not record:
                raise NotFoundError(f"Vacancy {vacancy_id} was not found.")
            before: dict[str, Any] = {
                "title": record.title,
                "department": record.department,
                "description": record.description,
                "status": record.status,
                "requirements": [r.text for r in await uow.vacancies.requirements(vacancy_id)],
            }
            updates = data.model_dump(exclude_unset=True)
            requirement_inputs = updates.pop("requirements", None)
            for field, value in updates.items():
                if value is not None or field == "department":
                    setattr(record, field, value.value if hasattr(value, "value") else value)
            if requirement_inputs is not None:
                parsed_requirements = [
                    RequirementInput.model_validate(item) for item in requirement_inputs
                ]
                await uow.vacancies.replace_requirements(
                    vacancy_id, self._make_requirements(vacancy_id, parsed_requirements)
                )
            after: dict[str, Any] = {
                "title": record.title,
                "department": record.department,
                "description": record.description,
                "status": record.status,
                "requirements": [r.text for r in await uow.vacancies.requirements(vacancy_id)],
            }
            await record_event(
                uow,
                actor_id=actor_id,
                action="vacancy_updated",
                entity_type="vacancy",
                entity_id=vacancy_id,
                details={"before": before, "after": after},
            )
            await uow.commit()
            return await self._to_output(uow, record)

    async def get(self, vacancy_id: str) -> VacancyOut:
        async with self.uow_provider() as uow:
            record = await uow.vacancies.get(vacancy_id)
            if not record:
                raise NotFoundError(f"Vacancy {vacancy_id} was not found.")
            return await self._to_output(uow, record)

    async def list_vacancies(self, limit: int = 100) -> list[VacancyOut]:
        async with self.uow_provider() as uow:
            records = await uow.vacancies.list_all(limit)
            return [await self._to_output(uow, record) for record in records]

    @staticmethod
    def _make_requirements(
        vacancy_id: str, requirements: list[RequirementInput]
    ) -> list[RequirementRecord]:
        return [
            RequirementRecord(
                id=str(uuid.uuid4()),
                vacancy_id=vacancy_id,
                text=requirement.text,
                is_must_have=requirement.is_must_have,
                weight=requirement.weight,
            )
            for requirement in requirements
        ]

    @staticmethod
    async def _to_output(uow: Any, record: VacancyRecord) -> VacancyOut:
        requirements = await uow.vacancies.requirements(record.id)
        return VacancyOut(
            id=record.id,
            title=record.title,
            department=record.department,
            description=record.description,
            status=VacancyStatus(record.status),
            requirements=[RequirementOut.model_validate(item) for item in requirements],
            created_at=record.created_at,
            updated_at=record.updated_at,
        )
