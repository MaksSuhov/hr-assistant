from __future__ import annotations

import uuid

from hr_assistant.application.audit import record_event
from hr_assistant.application.errors import ConflictError, NotFoundError
from hr_assistant.application.ports import UnitOfWorkProvider
from hr_assistant.domain.enums import ApplicationStage
from hr_assistant.infrastructure.db.models import (
    ApplicationRecord,
    CandidateRecord,
    VacancyRecord,
)
from hr_assistant.schemas.application import ApplicationOut


class ApplicationService:
    """Create, list and remove candidate↔vacancy links.

    Stage transitions go through ``ApprovalService`` so the existing human
    approval gate still applies to hiring decisions.
    """

    def __init__(self, uow_provider: UnitOfWorkProvider) -> None:
        self.uow_provider = uow_provider

    async def apply(
        self,
        *,
        candidate_id: str,
        vacancy_id: str,
        actor_id: str,
        stage: ApplicationStage = ApplicationStage.NEW,
        idempotent: bool = False,
    ) -> ApplicationOut:
        async with self.uow_provider() as uow:
            candidate = await uow.candidates.get(candidate_id)
            if not candidate:
                raise NotFoundError(f"Candidate {candidate_id} was not found.")
            vacancy = await uow.vacancies.get(vacancy_id)
            if not vacancy:
                raise NotFoundError(f"Vacancy {vacancy_id} was not found.")
            existing = await uow.applications.get_for_pair(candidate_id, vacancy_id)
            if existing:
                if not idempotent:
                    raise ConflictError(
                        f"Candidate {candidate_id} already has application {existing.id} "
                        f"for vacancy {vacancy_id}."
                    )
                return self._to_output(existing, vacancy, candidate)
            application = ApplicationRecord(
                id=str(uuid.uuid4()),
                candidate_id=candidate_id,
                vacancy_id=vacancy_id,
                stage=stage.value,
            )
            await uow.applications.add(application)
            await record_event(
                uow,
                actor_id=actor_id,
                action="application_created",
                entity_type="candidate",
                entity_id=candidate_id,
                details={
                    "application_id": application.id,
                    "vacancy_id": vacancy_id,
                    "stage": stage.value,
                },
            )
            await uow.commit()
            return self._to_output(application, vacancy, candidate)

    async def get(self, vacancy_id: str, application_id: str) -> ApplicationOut:
        async with self.uow_provider() as uow:
            application = await uow.applications.get(application_id)
            if not application or application.vacancy_id != vacancy_id:
                raise NotFoundError(
                    f"Application {application_id} was not found for vacancy {vacancy_id}."
                )
            vacancy = await uow.vacancies.get(vacancy_id)
            candidate = await uow.candidates.get(application.candidate_id)
            if not vacancy or not candidate:
                raise NotFoundError(f"Application {application_id} points at a missing record.")
            return self._to_output(application, vacancy, candidate)

    async def list_for_vacancy(self, vacancy_id: str, limit: int = 200) -> list[ApplicationOut]:
        async with self.uow_provider() as uow:
            if not await uow.vacancies.get(vacancy_id):
                raise NotFoundError(f"Vacancy {vacancy_id} was not found.")
            rows = await uow.applications.list_for_vacancy_with_candidate(vacancy_id, limit)
            vacancy = await uow.vacancies.get(vacancy_id)
            if not vacancy:
                raise NotFoundError(f"Vacancy {vacancy_id} was not found.")
            return [
                self._to_output(application, vacancy, candidate) for application, candidate in rows
            ]

    async def list_for_candidate(self, candidate_id: str) -> list[ApplicationOut]:
        async with self.uow_provider() as uow:
            candidate = await uow.candidates.get(candidate_id)
            if not candidate:
                raise NotFoundError(f"Candidate {candidate_id} was not found.")
            rows = await uow.applications.list_for_candidate_with_vacancy(candidate_id)
            return [
                self._to_output(application, vacancy, candidate) for application, vacancy in rows
            ]

    async def remove(self, vacancy_id: str, application_id: str, actor_id: str) -> None:
        async with self.uow_provider() as uow:
            application = await uow.applications.get(application_id)
            if not application or application.vacancy_id != vacancy_id:
                raise NotFoundError(
                    f"Application {application_id} was not found for vacancy {vacancy_id}."
                )
            await uow.applications.delete(application)
            await record_event(
                uow,
                actor_id=actor_id,
                action="application_removed",
                entity_type="candidate",
                entity_id=application.candidate_id,
                details={
                    "application_id": application.id,
                    "vacancy_id": vacancy_id,
                    "stage": application.stage,
                },
            )
            await uow.commit()

    @staticmethod
    def _to_output(
        application: ApplicationRecord,
        vacancy: VacancyRecord,
        candidate: CandidateRecord,
    ) -> ApplicationOut:
        return ApplicationOut(
            id=application.id,
            candidate_id=application.candidate_id,
            candidate_name=candidate.full_name,
            vacancy_id=application.vacancy_id,
            vacancy_title=vacancy.title,
            stage=ApplicationStage(application.stage),
            stage_changed_at=application.stage_changed_at,
            created_at=application.created_at,
            updated_at=application.updated_at,
        )
