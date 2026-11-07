from enum import StrEnum


class VacancyStatus(StrEnum):
    OPEN = "open"
    PAUSED = "paused"
    CLOSED = "closed"


class CandidateStatus(StrEnum):
    NEW = "new"
    REVIEW = "review"
    INTERVIEW = "interview"
    REJECTED = "rejected"
    HIRED = "hired"


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class ApprovalActionType(StrEnum):
    CANDIDATE_STATUS_CHANGE = "candidate_status_change"
    SEND_MESSAGE = "send_message"


class EvidenceStatus(StrEnum):
    SUPPORTED = "supported"
    PARTIAL = "partial"
    UNKNOWN = "unknown"
    NOT_FOUND = "not_found"

