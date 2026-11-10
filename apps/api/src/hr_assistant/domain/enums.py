from enum import StrEnum


class VacancyStatus(StrEnum):
    OPEN = "open"
    PAUSED = "paused"
    CLOSED = "closed"


class ApplicationStage(StrEnum):
    """Pipeline position of one candidate in one vacancy.

    A candidate has an independent stage per vacancy, so "interview" for the
    backend role does not hide an "archived" application for another role.
    """

    NEW = "new"
    SCREENING = "screening"
    INTERVIEW = "interview"
    OFFER = "offer"
    HIRED = "hired"
    REJECTED = "rejected"


# Ordered from the beginning of the funnel to its successful end. Used when two
# applications are merged and when reporting progress.
APPLICATION_STAGE_ORDER: tuple[ApplicationStage, ...] = (
    ApplicationStage.NEW,
    ApplicationStage.SCREENING,
    ApplicationStage.INTERVIEW,
    ApplicationStage.OFFER,
    ApplicationStage.HIRED,
)


def later_stage(first: ApplicationStage, second: ApplicationStage) -> ApplicationStage:
    """Return the stage further along the funnel, keeping ``rejected`` sticky.

    ``rejected`` is an archive state rather than a funnel step: merging an
    archived application with an active one keeps the active one.
    """

    if first is ApplicationStage.REJECTED:
        return second
    if second is ApplicationStage.REJECTED:
        return first
    first_index = APPLICATION_STAGE_ORDER.index(first)
    second_index = APPLICATION_STAGE_ORDER.index(second)
    return first if first_index >= second_index else second


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class ApprovalActionType(StrEnum):
    APPLICATION_STAGE_CHANGE = "application_stage_change"
    SEND_MESSAGE = "send_message"


class EvidenceStatus(StrEnum):
    SUPPORTED = "supported"
    PARTIAL = "partial"
    UNKNOWN = "unknown"
    NOT_FOUND = "not_found"
