from hr_assistant.container import ServiceContainer
from hr_assistant.domain.enums import EvidenceStatus
from hr_assistant.infrastructure.db.models import (
    CandidateRecord,
    RequirementRecord,
    VacancyRecord,
)
from tests.support import json_response


async def _seed_vacancy(
    container: ServiceContainer,
    *,
    vacancy_id: str = "v1",
    requirements: list[tuple[str, str]] | None = None,
) -> list[tuple[str, str]]:
    requirement_rows = requirements or []
    async with container.database.unit_of_work() as uow:
        await uow.vacancies.add(
            VacancyRecord(id=vacancy_id, title="Backend developer", description="", status="open")
        )
        await uow.vacancies.add_requirements(
            [
                RequirementRecord(
                    id=requirement_id,
                    vacancy_id=vacancy_id,
                    text=text,
                    is_must_have=True,
                    weight=5,
                )
                for requirement_id, text in requirement_rows
            ]
        )
        await uow.commit()
    return requirement_rows


async def _seed_candidate(container: ServiceContainer, candidate_id: str = "c1") -> None:
    async with container.database.unit_of_work() as uow:
        await uow.candidates.add(
            CandidateRecord(id=candidate_id, full_name="Candidate", status="new")
        )
        await uow.commit()


async def test_no_requirements_returns_clarification(container):
    await _seed_vacancy(container)
    await _seed_candidate(container)

    analysis = await container.matching.match_candidate("v1", "c1")

    assert analysis.requirement_evidence == []
    assert "не заданы требования" in analysis.summary
    assert analysis.clarification


async def test_llm_analysis_selects_valid_evidence(container, fake_vector_store, fake_llm):
    await _seed_vacancy(container, requirements=[("req1", "Python experience")])
    await _seed_candidate(container)
    await fake_vector_store.index_resume(
        resume_id="r1",
        candidate_id="c1",
        source_filename="resume.pdf",
        chunks=["Python developer with FastAPI"],
    )
    fake_llm.configured = True
    fake_llm.completions.responses.append(
        json_response(
            {
                "summary": "Соответствует требованию.",
                "requirements": [
                    {
                        "requirement_id": "req1",
                        "status": "supported",
                        "evidence_indices": [0],
                        "gap_or_question": "Уточнить масштаб проектов.",
                    }
                ],
            }
        )
    )

    analysis = await container.matching.match_candidate("v1", "c1")

    assert analysis.summary == "Соответствует требованию."
    evidence = analysis.requirement_evidence[0]
    assert evidence.status is EvidenceStatus.SUPPORTED
    assert len(evidence.evidence) == 1
    assert "Python" in evidence.evidence[0].text


async def test_invalid_status_and_indices_are_sanitized(container, fake_vector_store, fake_llm):
    await _seed_vacancy(container, requirements=[("req1", "Python experience")])
    await _seed_candidate(container)
    await fake_vector_store.index_resume(
        resume_id="r1",
        candidate_id="c1",
        source_filename="resume.pdf",
        chunks=["Python developer"],
    )
    fake_llm.configured = True
    fake_llm.completions.responses.append(
        json_response(
            {
                "summary": "Ответ модели.",
                "requirements": [
                    {
                        "requirement_id": "req1",
                        "status": "invented",
                        "evidence_indices": [99, 0],
                    }
                ],
            }
        )
    )

    analysis = await container.matching.match_candidate("v1", "c1")

    evidence = analysis.requirement_evidence[0]
    assert evidence.status is EvidenceStatus.UNKNOWN
    assert len(evidence.evidence) == 1


async def test_llm_failure_falls_back_to_evidence_only(container, fake_vector_store, fake_llm):
    await _seed_vacancy(container, requirements=[("req1", "Python experience")])
    await _seed_candidate(container)
    await fake_vector_store.index_resume(
        resume_id="r1",
        candidate_id="c1",
        source_filename="resume.pdf",
        chunks=["Python developer"],
    )
    fake_llm.configured = True

    analysis = await container.matching.match_candidate("v1", "c1")

    assert analysis.clarification
    assert analysis.requirement_evidence[0].status is EvidenceStatus.UNKNOWN


async def test_compare_returns_one_analysis_per_candidate(container):
    await _seed_vacancy(container)
    await _seed_candidate(container, "c1")
    await _seed_candidate(container, "c2")

    result = await container.matching.compare("v1", ["c1", "c2"])

    assert [item.candidate_id for item in result.candidates] == ["c1", "c2"]
