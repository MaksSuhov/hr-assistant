from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from hr_assistant.domain.enums import VacancyStatus


class RequirementInput(BaseModel):
    text: str = Field(min_length=2, max_length=1000)
    is_must_have: bool = True
    weight: int = Field(default=1, ge=1, le=5)


class VacancyCreate(BaseModel):
    title: str = Field(min_length=2, max_length=200)
    department: str | None = Field(default=None, max_length=200)
    description: str = Field(default="", max_length=10000)
    requirements: list[RequirementInput] = Field(default_factory=list, max_length=100)


class VacancyUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=2, max_length=200)
    department: str | None = Field(default=None, max_length=200)
    description: str | None = Field(default=None, max_length=10000)
    status: VacancyStatus | None = None
    requirements: list[RequirementInput] | None = Field(default=None, max_length=100)


class RequirementOut(RequirementInput):
    id: str
    model_config = ConfigDict(from_attributes=True)


class VacancyOut(BaseModel):
    id: str
    title: str
    department: str | None
    description: str
    status: VacancyStatus
    requirements: list[RequirementOut]
    created_at: datetime
    updated_at: datetime

