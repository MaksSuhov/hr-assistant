from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from hr_assistant.domain.enums import ApplicationStage


class ApplicationCreate(BaseModel):
    candidate_id: str
    stage: ApplicationStage = ApplicationStage.NEW


class ApplicationOut(BaseModel):
    id: str
    candidate_id: str
    candidate_name: str | None = None
    vacancy_id: str
    vacancy_title: str
    stage: ApplicationStage
    stage_changed_at: datetime
    created_at: datetime
    updated_at: datetime


class StageChangeRequest(BaseModel):
    stage: ApplicationStage
    reason: str | None = Field(default=None, max_length=1000)


class MergeCandidatesRequest(BaseModel):
    source_candidate_id: str


class RetentionPurgeResult(BaseModel):
    deleted_candidates: int
    deleted_resumes: int
    failed_candidates: int
    deleted_audit_events: int = 0
