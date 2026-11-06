from pydantic import BaseModel, Field

from hr_assistant.domain.enums import EvidenceStatus


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=1000)
    limit: int = Field(default=10, ge=1, le=50)


class EvidenceSnippet(BaseModel):
    text: str
    source_filename: str
    resume_id: str
    score: float | None = None


class RequirementEvidence(BaseModel):
    requirement_id: str
    requirement: str
    evidence: list[EvidenceSnippet] = Field(default_factory=list)
    gap_or_question: str
    status: EvidenceStatus


class MatchAnalysis(BaseModel):
    candidate_id: str
    vacancy_id: str
    summary: str
    requirement_evidence: list[RequirementEvidence]
    clarification: str | None = None


class CandidateSearchHit(BaseModel):
    candidate_id: str
    candidate_name: str | None = None
    document_id: str
    source_filename: str
    snippet: str
    score: float | None = None


class CompareCandidatesRequest(BaseModel):
    vacancy_id: str
    candidate_ids: list[str] = Field(min_length=2, max_length=10)


class CompareCandidatesResponse(BaseModel):
    vacancy_id: str
    candidates: list[MatchAnalysis]

