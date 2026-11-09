import asyncio
import json
import logging
from typing import Any

from hr_assistant.application.errors import NotFoundError
from hr_assistant.application.ports import ResumeSearch, UnitOfWorkProvider
from hr_assistant.core.config import Settings
from hr_assistant.core.prompt_safety import RESUME_DATA_RULE, wrap_resume_text
from hr_assistant.domain.enums import EvidenceStatus
from hr_assistant.infrastructure.llm.openai_provider import OpenAIProvider
from hr_assistant.schemas.matching import (
    CompareCandidatesResponse,
    EvidenceSnippet,
    MatchAnalysis,
    RequirementEvidence,
)

logger = logging.getLogger(__name__)


class MatchingService:
    def __init__(
        self,
        uow_provider: UnitOfWorkProvider,
        search_service: ResumeSearch,
        llm: OpenAIProvider,
        settings: Settings,
    ) -> None:
        self.uow_provider = uow_provider
        self.search_service = search_service
        self.llm = llm
        self.settings = settings

    async def match_candidate(self, vacancy_id: str, candidate_id: str) -> MatchAnalysis:
        async with self.uow_provider() as uow:
            vacancy = await uow.vacancies.get(vacancy_id)
            if not vacancy:
                raise NotFoundError(f"Vacancy {vacancy_id} was not found.")
            candidate = await uow.candidates.get(candidate_id)
            if not candidate:
                raise NotFoundError(f"Candidate {candidate_id} was not found.")
            requirements = await uow.vacancies.requirements(vacancy_id)
        if not requirements:
            return MatchAnalysis(
                candidate_id=candidate_id,
                vacancy_id=vacancy_id,
                summary="Для вакансии пока не заданы требования.",
                requirement_evidence=[],
                clarification="Добавьте требования, чтобы сопоставить кандидата.",
            )

        evidence_by_requirement: dict[str, list[EvidenceSnippet]] = {}
        for requirement in requirements:
            hits = await self.search_service.search_raw(
                requirement.text, limit=4, candidate_ids=[candidate_id]
            )
            evidence_by_requirement[requirement.id] = [
                EvidenceSnippet(
                    text=str(hit.get("text", ""))[:900],
                    source_filename=str(hit.get("source_filename", "")),
                    resume_id=str(hit.get("resume_id", "")),
                    score=hit.get("score"),
                )
                for hit in hits
            ]

        if self.llm.configured:
            try:
                return await self._analyze_with_llm(
                    vacancy_id, candidate_id, requirements, evidence_by_requirement
                )
            except Exception:
                logger.exception(
                    "LLM requirement analysis failed; returning evidence-only analysis"
                )
        return self._evidence_only(vacancy_id, candidate_id, requirements, evidence_by_requirement)

    async def compare(self, vacancy_id: str, candidate_ids: list[str]) -> CompareCandidatesResponse:
        analyses = await asyncio.gather(
            *(self.match_candidate(vacancy_id, candidate_id) for candidate_id in candidate_ids)
        )
        return CompareCandidatesResponse(vacancy_id=vacancy_id, candidates=analyses)

    async def _analyze_with_llm(
        self, vacancy_id, candidate_id, requirements, evidence_by_requirement
    ) -> MatchAnalysis:
        evidence_payload = []
        for requirement in requirements:
            evidence_payload.append(
                {
                    "requirement_id": requirement.id,
                    "requirement": requirement.text,
                    "is_must_have": requirement.is_must_have,
                    "weight": requirement.weight,
                    "evidence": [
                        {
                            "index": index,
                            "text": wrap_resume_text(item.text, source=item.source_filename),
                            "source": item.source_filename,
                        }
                        for index, item in enumerate(evidence_by_requirement[requirement.id])
                    ],
                }
            )
        response = await self.llm.require_client().chat.completions.create(
            model=self.settings.openai_chat_model,
            temperature=0.1,
            response_format={"type": "json_object"},
            extra_body=self.llm.chat_extra_body,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты помогаешь HR сопоставить требования с резюме. "
                        "Не делай выводов о найме. Используй только переданные фрагменты. "
                        "Не считай отсутствие фрагмента доказательством отсутствия навыка. "
                        + RESUME_DATA_RULE
                        + " "
                        "Верни JSON с ключами summary и requirements. "
                        "Для каждого требования верни requirement_id, status (supported, partial, "
                        "unknown, not_found), evidence_indices и gap_or_question. "
                        "Индексы выбирай только из переданных фрагментов. "
                        "Если фрагменты не подтверждают требование, "
                        "укажи unknown или not_found и сформулируй вопрос для интервью."
                    ),
                },
                {"role": "user", "content": json.dumps(evidence_payload, ensure_ascii=False)},
            ],
        )
        raw = response.choices[0].message.content or "{}"
        parsed = json.loads(raw)
        analyses = {item.get("requirement_id"): item for item in parsed.get("requirements", [])}
        requirement_evidence: list[RequirementEvidence] = []
        for requirement in requirements:
            item = analyses.get(requirement.id, {})
            source_evidence = evidence_by_requirement[requirement.id]
            chosen: list[EvidenceSnippet] = []
            for index in item.get("evidence_indices", []):
                if isinstance(index, int) and 0 <= index < len(source_evidence):
                    chosen.append(source_evidence[index])
            try:
                status = EvidenceStatus(item.get("status", EvidenceStatus.UNKNOWN.value))
            except ValueError:
                status = EvidenceStatus.UNKNOWN
            requirement_evidence.append(
                RequirementEvidence(
                    requirement_id=requirement.id,
                    requirement=requirement.text,
                    evidence=chosen,
                    gap_or_question=str(item.get("gap_or_question", "Уточнить на интервью.")),
                    status=status,
                )
            )
        return MatchAnalysis(
            candidate_id=candidate_id,
            vacancy_id=vacancy_id,
            summary=str(
                parsed.get("summary", "Сопоставление подготовлено по найденным фрагментам.")
            ),
            requirement_evidence=requirement_evidence,
        )

    @staticmethod
    def _evidence_only(
        vacancy_id: str,
        candidate_id: str,
        requirements: list[Any],
        evidence_by_requirement: dict[str, list[EvidenceSnippet]],
    ) -> MatchAnalysis:
        items = []
        for requirement in requirements:
            evidence = evidence_by_requirement[requirement.id]
            items.append(
                RequirementEvidence(
                    requirement_id=requirement.id,
                    requirement=requirement.text,
                    evidence=evidence[:2],
                    gap_or_question=(
                        "Попросить привести конкретный пример и уточнить личный вклад."
                        if evidence
                        else "Подтверждение не найдено в доступном тексте; уточнить на интервью."
                    ),
                    status=EvidenceStatus.UNKNOWN if evidence else EvidenceStatus.NOT_FOUND,
                )
            )
        return MatchAnalysis(
            candidate_id=candidate_id,
            vacancy_id=vacancy_id,
            summary="Найдены текстовые совпадения; вывод о соответствии требует проверки HR.",
            requirement_evidence=items,
            clarification="Оценка по тексту резюме не заменяет интервью и проверку фактов.",
        )
