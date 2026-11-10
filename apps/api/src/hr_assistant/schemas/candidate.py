from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from hr_assistant.schemas.application import ApplicationOut


class CandidateDocumentOut(BaseModel):
    id: str
    filename: str
    source: str
    uploaded_by: str
    uploaded_at: datetime
    indexing_status: str
    indexing_error: str | None = None
    extraction_method: str = "pdf-text"
    model_config = ConfigDict(from_attributes=True)


class CandidateOut(BaseModel):
    id: str
    full_name: str | None
    email: str | None
    phone: str | None
    created_at: datetime
    updated_at: datetime
    sources: list[CandidateDocumentOut] = Field(default_factory=list)
    applications: list[ApplicationOut] = Field(default_factory=list)


class AuditEventOut(BaseModel):
    id: str
    actor_id: str
    action: str
    entity_type: str
    entity_id: str
    details: dict
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)
