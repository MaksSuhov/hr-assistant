from tests.support import upload_resume


async def _candidate(auth_client, make_pdf) -> str:
    response = await upload_resume(auth_client, make_pdf("Python developer"))
    assert response.status_code == 201
    return response.json()["id"]


async def test_status_change_requires_approval_then_applies_it(auth_client, make_pdf):
    candidate_id = await _candidate(auth_client, make_pdf)

    requested = await auth_client.post(
        f"/api/v1/approvals/candidates/{candidate_id}/status-change/request",
        json={"status": "interview", "reason": "Resume reviewed"},
    )
    assert requested.status_code == 202
    approval = requested.json()
    assert approval["status"] == "pending"
    assert approval["action_type"] == "candidate_status_change"
    assert approval["payload"]["to_status"] == "interview"

    pending = await auth_client.get("/api/v1/approvals/pending")
    assert [item["id"] for item in pending.json()] == [approval["id"]]

    candidate_before = await auth_client.get(f"/api/v1/candidates/{candidate_id}")
    assert candidate_before.json()["status"] == "new"

    approved = await auth_client.post(
        f"/api/v1/approvals/{approval['id']}/approve",
        json={"note": "Approved"},
    )
    assert approved.status_code == 200, approved.text
    result = approved.json()
    assert result["approval"]["status"] == "approved"
    assert result["outcome"] == "candidate_status_updated"

    candidate_after = await auth_client.get(f"/api/v1/candidates/{candidate_id}")
    assert candidate_after.json()["status"] == "interview"


async def test_double_approval_returns_conflict(auth_client, make_pdf):
    candidate_id = await _candidate(auth_client, make_pdf)
    requested = await auth_client.post(
        f"/api/v1/approvals/candidates/{candidate_id}/status-change/request",
        json={"status": "review"},
    )
    approval_id = requested.json()["id"]

    first = await auth_client.post(f"/api/v1/approvals/{approval_id}/approve", json={})
    second = await auth_client.post(f"/api/v1/approvals/{approval_id}/approve", json={})

    assert first.status_code == 200
    assert second.status_code == 409


async def test_reject_leaves_candidate_unchanged(auth_client, make_pdf):
    candidate_id = await _candidate(auth_client, make_pdf)
    requested = await auth_client.post(
        f"/api/v1/approvals/candidates/{candidate_id}/status-change/request",
        json={"status": "hired"},
    )
    approval_id = requested.json()["id"]

    rejected = await auth_client.post(
        f"/api/v1/approvals/{approval_id}/reject",
        json={"note": "Not yet"},
    )

    assert rejected.status_code == 200
    assert rejected.json()["outcome"] == "action_rejected"
    candidate = await auth_client.get(f"/api/v1/candidates/{candidate_id}")
    assert candidate.json()["status"] == "new"

    after_reject = await auth_client.post(f"/api/v1/approvals/{approval_id}/approve", json={})
    assert after_reject.status_code == 409


async def test_approved_message_is_queued_in_outbox(auth_client, make_pdf):
    candidate_id = await _candidate(auth_client, make_pdf)

    draft = await auth_client.post(
        "/api/v1/approvals/messages/drafts",
        json={
            "candidate_id": candidate_id,
            "recipient": "candidate@example.org",
            "subject": "Interview invitation",
            "body": "Would you like to join an interview?",
        },
    )
    assert draft.status_code == 202
    approval_id = draft.json()["id"]

    approved = await auth_client.post(f"/api/v1/approvals/{approval_id}/approve", json={})
    assert approved.status_code == 200
    assert approved.json()["outcome"] == "message_queued_for_delivery"

    outbox = await auth_client.get("/api/v1/approvals/outbox")
    assert outbox.status_code == 200
    messages = outbox.json()
    assert len(messages) == 1
    assert messages[0]["status"] == "queued"
    assert messages[0]["recipient"] == "candidate@example.org"


async def test_configured_smtp_marks_message_as_sent(auth_client, make_pdf, fake_sender):
    candidate_id = await _candidate(auth_client, make_pdf)
    draft = await auth_client.post(
        "/api/v1/approvals/messages/drafts",
        json={
            "candidate_id": candidate_id,
            "recipient": "candidate@example.org",
            "subject": "Hello",
            "body": "Body",
        },
    )
    fake_sender.configured = True

    approved = await auth_client.post(f"/api/v1/approvals/{draft.json()['id']}/approve", json={})

    assert approved.status_code == 200
    assert approved.json()["outcome"] == "message_sent"
    assert fake_sender.sent == [("candidate@example.org", "Hello", "Body")]
    outbox = await auth_client.get("/api/v1/approvals/outbox")
    assert outbox.json()[0]["status"] == "sent"


async def test_approval_requests_validate_targets(auth_client):
    missing_candidate = await auth_client.post(
        "/api/v1/approvals/candidates/missing/status-change/request",
        json={"status": "review"},
    )
    assert missing_candidate.status_code == 404

    missing_message_candidate = await auth_client.post(
        "/api/v1/approvals/messages/drafts",
        json={
            "candidate_id": "missing",
            "recipient": "candidate@example.org",
            "subject": "Hello",
            "body": "Body",
        },
    )
    assert missing_message_candidate.status_code == 404

    missing_approval = await auth_client.post("/api/v1/approvals/missing/approve", json={})
    assert missing_approval.status_code == 404


async def test_approval_payload_validation(auth_client, make_pdf):
    candidate_id = await _candidate(auth_client, make_pdf)

    bad_status = await auth_client.post(
        f"/api/v1/approvals/candidates/{candidate_id}/status-change/request",
        json={"status": "not-a-status"},
    )
    assert bad_status.status_code == 422

    bad_recipient = await auth_client.post(
        "/api/v1/approvals/messages/drafts",
        json={
            "candidate_id": candidate_id,
            "recipient": "not-an-email",
            "subject": "Hello",
            "body": "Body",
        },
    )
    assert bad_recipient.status_code == 422
