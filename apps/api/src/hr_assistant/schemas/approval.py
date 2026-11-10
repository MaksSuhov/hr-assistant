from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from hr_assistant.domain.enums import ApprovalActionType, ApprovalStatus


class MessageDraftRequest(BaseModel):
    candidate_id: str
    recipient: EmailStr
    subject: str = Field(min_length=1, max_length=300)
    body: str = Field(min_length=1, max_length=10000)


class ApprovalOut(BaseModel):
    id: str
    action_type: ApprovalActionType
    status: ApprovalStatus
    payload: dict
    requested_by: str
    resolved_by: str | None
    resolution_note: str | None
    created_at: datetime
    resolved_at: datetime | None
    model_config = ConfigDict(from_attributes=True)


class ApprovalDecision(BaseModel):
    note: str | None = Field(default=None, max_length=1000)


class ApprovalResult(BaseModel):
    approval: ApprovalOut
    outcome: str


class OutboxMessageOut(BaseModel):
    id: str
    approval_id: str
    candidate_id: str
    recipient: EmailStr
    subject: str
    body: str
    status: str
    last_error: str | None
    queued_at: datetime
    model_config = ConfigDict(from_attributes=True)
