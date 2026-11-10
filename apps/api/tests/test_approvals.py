from tests.support import upload_resume


async def _application(auth_client, make_pdf) -> tuple[str, str, str]:
    vacancy = await auth_client.post(
        "/api/v1/vacancies",
        json={"title": "Backend developer", "requirements": []},
    )
    vacancy_id = vacancy.json()["id"]
    uploaded = await upload_resume(auth_client, make_pdf("Python developer"), vacancy_id=vacancy_id)
    assert uploaded.status_code == 201, uploaded.text
    candidate = uploaded.json()
    return candidate["id"], candidate["applications"][0]["id"], vacancy_id


async def test_stage_change_requires_approval_then_applies_it(auth_client, make_pdf):
    candidate_id, application_id, vacancy_id = await _application(auth_client, make_pdf)

    requested = await auth_client.post(
        f"/api/v1/vacancies/{vacancy_id}/applications/{application_id}/stage-change/request",
        json={"stage": "interview", "reason": "Resume reviewed"},
    )
    assert requested.status_code == 202, requested.text
    approval = requested.json()
    assert approval["status"] == "pending"
    assert approval["action_type"] == "application_stage_change"
    assert approval["payload"]["to_stage"] == "interview"
    assert approval["payload"]["application_id"] == application_id

    pending = await auth_client.get("/api/v1/approvals/pending")
    assert [item["id"] for item in pending.json()] == [approval["id"]]

    candidate_before = await auth_client.get(f"/api/v1/candidates/{candidate_id}")
    assert candidate_before.json()["applications"][0]["stage"] == "new"

    approved = await auth_client.post(
        f"/api/v1/approvals/{approval['id']}/approve",
        json={"note": "Approved"},
    )
    assert approved.status_code == 200, approved.text
    result = approved.json()
    assert result["approval"]["status"] == "approved"
    assert result["outcome"] == "application_stage_updated"

    candidate_after = await auth_client.get(f"/api/v1/candidates/{candidate_id}")
    assert candidate_after.json()["applications"][0]["stage"] == "interview"


async def test_double_approval_returns_conflict(auth_client, make_pdf):
    candidate_id, application_id, vacancy_id = await _application(auth_client, make_pdf)
    requested = await auth_client.post(
        f"/api/v1/vacancies/{vacancy_id}/applications/{application_id}/stage-change/request",
        json={"stage": "screening"},
    )
    approval_id = requested.json()["id"]

    first = await auth_client.post(f"/api/v1/approvals/{approval_id}/approve", json={})
    second = await auth_client.post(f"/api/v1/approvals/{approval_id}/approve", json={})

    assert first.status_code == 200
    assert second.status_code == 409


async def test_reject_leaves_application_unchanged(auth_client, make_pdf):
    candidate_id, application_id, vacancy_id = await _application(auth_client, make_pdf)
    requested = await auth_client.post(
        f"/api/v1/vacancies/{vacancy_id}/applications/{application_id}/stage-change/request",
        json={"stage": "hired"},
    )
    approval_id = requested.json()["id"]

    rejected = await auth_client.post(
        f"/api/v1/approvals/{approval_id}/reject",
        json={"note": "Not yet"},
    )

    assert rejected.status_code == 200
    assert rejected.json()["outcome"] == "action_rejected"
    candidate = await auth_client.get(f"/api/v1/candidates/{candidate_id}")
    assert candidate.json()["applications"][0]["stage"] == "new"

    after_reject = await auth_client.post(f"/api/v1/approvals/{approval_id}/approve", json={})
    assert after_reject.status_code == 409


async def test_approved_message_is_queued_in_outbox(auth_client, make_pdf):
    candidate_id, _, _ = await _application(auth_client, make_pdf)

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

    history = await auth_client.get(f"/api/v1/candidates/{candidate_id}/history")
    requested = [
        event
        for event in history.json()
        if event["action"] == "candidate_message_approval_requested"
    ]
    assert requested[0]["details"]["recipient"] == "c***@example.org"

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
    candidate_id, _, _ = await _application(auth_client, make_pdf)
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


async def test_approval_requests_validate_targets(auth_client, make_pdf):
    candidate_id, application_id, vacancy_id = await _application(auth_client, make_pdf)

    missing_application = await auth_client.post(
        f"/api/v1/vacancies/{vacancy_id}/applications/missing/stage-change/request",
        json={"stage": "screening"},
    )
    assert missing_application.status_code == 404

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
    candidate_id, application_id, vacancy_id = await _application(auth_client, make_pdf)

    bad_stage = await auth_client.post(
        f"/api/v1/vacancies/{vacancy_id}/applications/{application_id}/stage-change/request",
        json={"stage": "not-a-stage"},
    )
    assert bad_stage.status_code == 422

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
