import uuid
from datetime import UTC, datetime

from hr_assistant.application.audit import record_event
from hr_assistant.application.errors import ConflictError, NotFoundError
from hr_assistant.application.ports import MessageSender, UnitOfWorkProvider
from hr_assistant.domain.enums import ApprovalActionType, ApprovalStatus
from hr_assistant.infrastructure.db.models import ApprovalActionRecord, OutboxMessageRecord
from hr_assistant.schemas.approval import (
    ApprovalOut,
    ApprovalResult,
    MessageDraftRequest,
    OutboxMessageOut,
)
from hr_assistant.schemas.approval import StatusChangeRequest


class ApprovalService:
    def __init__(self, uow_provider: UnitOfWorkProvider, message_sender: MessageSender) -> None:
        self.uow_provider = uow_provider
        self.message_sender = message_sender

    async def request_status_change(
        self, candidate_id: str, data: StatusChangeRequest, actor_id: str
    ) -> ApprovalOut:
        async with self.uow_provider() as uow:
            candidate = await uow.candidates.get(candidate_id)
            if not candidate:
                raise NotFoundError(f"Candidate {candidate_id} was not found.")
            approval = ApprovalActionRecord(
                id=str(uuid.uuid4()),
                action_type=ApprovalActionType.CANDIDATE_STATUS_CHANGE.value,
                status=ApprovalStatus.PENDING.value,
                payload={
                    "candidate_id": candidate_id,
                    "from_status": candidate.status,
                    "to_status": data.status.value,
                    "reason": data.reason,
                },
                requested_by=actor_id,
            )
            await uow.approvals.add(approval)
            await record_event(
                uow,
                actor_id=actor_id,
                action="candidate_status_change_requested",
                entity_type="candidate",
                entity_id=candidate_id,
                details={"approval_id": approval.id, **approval.payload},
            )
            await uow.commit()
            return ApprovalOut.model_validate(approval)

    async def request_message(self, data: MessageDraftRequest, actor_id: str) -> ApprovalOut:
        async with self.uow_provider() as uow:
            candidate = await uow.candidates.get(data.candidate_id)
            if not candidate:
                raise NotFoundError(f"Candidate {data.candidate_id} was not found.")
            approval = ApprovalActionRecord(
                id=str(uuid.uuid4()),
                action_type=ApprovalActionType.SEND_MESSAGE.value,
                status=ApprovalStatus.PENDING.value,
                payload=data.model_dump(mode="json"),
                requested_by=actor_id,
            )
            await uow.approvals.add(approval)
            await record_event(
                uow,
                actor_id=actor_id,
                action="candidate_message_approval_requested",
                entity_type="candidate",
                entity_id=data.candidate_id,
                details={
                    "approval_id": approval.id,
                    "recipient": data.recipient,
                    "subject": data.subject,
                },
            )
            await uow.commit()
            return ApprovalOut.model_validate(approval)

    async def list_pending(self, limit: int = 100) -> list[ApprovalOut]:
        async with self.uow_provider() as uow:
            records = await uow.approvals.list_pending(limit)
            return [ApprovalOut.model_validate(record) for record in records]

    async def list_outbox(self, limit: int = 100) -> list[OutboxMessageOut]:
        async with self.uow_provider() as uow:
            records = await uow.approvals.list_outbox(limit)
            return [OutboxMessageOut.model_validate(record) for record in records]

    async def approve(self, approval_id: str, actor_id: str, note: str | None) -> ApprovalResult:
        delivery: tuple[str, str, str, str, str] | None = None
        async with self.uow_provider() as uow:
            approval = await uow.approvals.get(approval_id)
            if not approval:
                raise NotFoundError(f"Approval {approval_id} was not found.")
            if approval.status != ApprovalStatus.PENDING.value:
                raise ConflictError("This action has already been resolved.")

            outcome: str
            if approval.action_type == ApprovalActionType.CANDIDATE_STATUS_CHANGE.value:
                candidate_id = approval.payload["candidate_id"]
                candidate = await uow.candidates.get(candidate_id)
                if not candidate:
                    raise NotFoundError(f"Candidate {candidate_id} was not found.")
                before = candidate.status
                candidate.status = approval.payload["to_status"]
                outcome = "candidate_status_updated"
                await record_event(
                    uow,
                    actor_id=actor_id,
                    action="candidate_status_changed",
                    entity_type="candidate",
                    entity_id=candidate_id,
                    details={
                        "approval_id": approval.id,
                        "before": before,
                        "after": candidate.status,
                        "reason": approval.payload.get("reason"),
                    },
                )
            elif approval.action_type == ApprovalActionType.SEND_MESSAGE.value:
                payload = approval.payload
                candidate_id = payload["candidate_id"]
                if not await uow.candidates.get(candidate_id):
                    raise NotFoundError(f"Candidate {candidate_id} was not found.")
                await uow.approvals.enqueue_message(
                    OutboxMessageRecord(
                        id=str(uuid.uuid4()),
                        approval_id=approval.id,
                        candidate_id=candidate_id,
                        recipient=payload["recipient"],
                        subject=payload["subject"],
                        body=payload["body"],
                        status="queued",
                    )
                )
                outcome = "message_queued_for_delivery"
                if self.message_sender.configured:
                    delivery = (
                        approval.id,
                        candidate_id,
                        payload["recipient"],
                        payload["subject"],
                        payload["body"],
                    )
                await record_event(
                    uow,
                    actor_id=actor_id,
                    action="candidate_message_approved_and_queued",
                    entity_type="candidate",
                    entity_id=candidate_id,
                    details={
                        "approval_id": approval.id,
                        "recipient": payload["recipient"],
                        "subject": payload["subject"],
                    },
                )
            else:
                raise ConflictError(f"Unsupported approval action: {approval.action_type}")

            approval.status = ApprovalStatus.APPROVED.value
            approval.resolved_by = actor_id
            approval.resolution_note = note
            approval.resolved_at = datetime.now(UTC)
            await record_event(
                uow,
                actor_id=actor_id,
                action="approval_approved",
                entity_type="approval",
                entity_id=approval.id,
                details={"action_type": approval.action_type, "outcome": outcome, "note": note},
            )
            await uow.commit()
            approval_output = ApprovalOut.model_validate(approval)

        if delivery:
            current_approval_id, candidate_id, recipient, subject, body = delivery
            try:
                await self.message_sender.send(recipient, subject, body)
                delivery_status = "sent"
                outcome = "message_sent"
                delivery_error = None
                audit_action = "candidate_message_sent"
            except Exception as error:
                delivery_status = "failed"
                outcome = "message_delivery_failed"
                delivery_error = type(error).__name__
                audit_action = "candidate_message_delivery_failed"
            async with self.uow_provider() as uow:
                await uow.approvals.update_outbox_status(
                    current_approval_id, delivery_status, delivery_error
                )
                await record_event(
                    uow,
                    actor_id=actor_id,
                    action=audit_action,
                    entity_type="candidate",
                    entity_id=candidate_id,
                    details={
                        "approval_id": current_approval_id,
                        "recipient": recipient,
                        "subject": subject,
                        "outbox_status": delivery_status,
                        "delivery_error": delivery_error,
                    },
                )
                await uow.commit()
        return ApprovalResult(approval=approval_output, outcome=outcome)

    async def reject(self, approval_id: str, actor_id: str, note: str | None) -> ApprovalResult:
        async with self.uow_provider() as uow:
            approval = await uow.approvals.get(approval_id)
            if not approval:
                raise NotFoundError(f"Approval {approval_id} was not found.")
            if approval.status != ApprovalStatus.PENDING.value:
                raise ConflictError("This action has already been resolved.")
            approval.status = ApprovalStatus.REJECTED.value
            approval.resolved_by = actor_id
            approval.resolution_note = note
            approval.resolved_at = datetime.now(UTC)
            candidate_id = approval.payload.get("candidate_id")
            await record_event(
                uow,
                actor_id=actor_id,
                action="approval_rejected",
                entity_type="candidate" if candidate_id else "approval",
                entity_id=candidate_id or approval.id,
                details={"approval_id": approval.id, "action_type": approval.action_type, "note": note},
            )
            await uow.commit()
            return ApprovalResult(
                approval=ApprovalOut.model_validate(approval), outcome="action_rejected"
            )
