from datetime import UTC, datetime, timedelta

from sqlalchemy import update

from hr_assistant.container import ServiceContainer
from hr_assistant.infrastructure.db.models import CandidateRecord, ResumeDocumentRecord
from tests.support import build_docx, upload_resume

RESUME_TEXT = "Python developer with FastAPI experience and PostgreSQL"


async def _vacancy(auth_client, title: str = "Backend developer") -> str:
    response = await auth_client.post(
        "/api/v1/vacancies", json={"title": title, "requirements": []}
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


async def test_upload_for_vacancy_creates_application(auth_client, make_pdf):
    vacancy_id = await _vacancy(auth_client)

    response = await upload_resume(auth_client, make_pdf(RESUME_TEXT), vacancy_id=vacancy_id)

    assert response.status_code == 201, response.text
    candidate = response.json()
    assert len(candidate["applications"]) == 1
    application = candidate["applications"][0]
    assert application["vacancy_id"] == vacancy_id
    assert application["vacancy_title"] == "Backend developer"
    assert application["stage"] == "new"
    assert application["candidate_name"] == "Алексей Иванов"

    vacancy = await auth_client.get(f"/api/v1/vacancies/{vacancy_id}")
    assert vacancy.json()["applications_count"] == 1

    history = await auth_client.get(f"/api/v1/candidates/{candidate['id']}/history")
    actions = [event["action"] for event in history.json()]
    assert "application_created" in actions


async def test_application_crud_and_duplicate_conflict(auth_client, make_pdf):
    candidate_id = (await upload_resume(auth_client, make_pdf(RESUME_TEXT))).json()["id"]
    vacancy_id = await _vacancy(auth_client)

    created = await auth_client.post(
        f"/api/v1/vacancies/{vacancy_id}/applications",
        json={"candidate_id": candidate_id},
    )
    assert created.status_code == 201, created.text
    application_id = created.json()["id"]

    duplicate = await auth_client.post(
        f"/api/v1/vacancies/{vacancy_id}/applications",
        json={"candidate_id": candidate_id},
    )
    assert duplicate.status_code == 409

    listing = await auth_client.get(f"/api/v1/vacancies/{vacancy_id}/applications")
    assert [item["id"] for item in listing.json()] == [application_id]

    fetched = await auth_client.get(f"/api/v1/vacancies/{vacancy_id}/applications/{application_id}")
    assert fetched.status_code == 200
    assert fetched.json()["stage"] == "new"

    removed = await auth_client.delete(
        f"/api/v1/vacancies/{vacancy_id}/applications/{application_id}"
    )
    assert removed.status_code == 204
    assert (await auth_client.get(f"/api/v1/vacancies/{vacancy_id}/applications")).json() == []


async def test_candidate_stage_is_independent_per_vacancy(auth_client, make_pdf):
    candidate_id = (await upload_resume(auth_client, make_pdf(RESUME_TEXT))).json()["id"]
    first_vacancy = await _vacancy(auth_client, "Backend developer")
    second_vacancy = await _vacancy(auth_client, "Data engineer")

    first_application = await auth_client.post(
        f"/api/v1/vacancies/{first_vacancy}/applications",
        json={"candidate_id": candidate_id},
    )
    second_application = await auth_client.post(
        f"/api/v1/vacancies/{second_vacancy}/applications",
        json={"candidate_id": candidate_id},
    )

    requested = await auth_client.post(
        f"/api/v1/vacancies/{first_vacancy}/applications/"
        f"{first_application.json()['id']}/stage-change/request",
        json={"stage": "interview"},
    )
    approved = await auth_client.post(
        f"/api/v1/approvals/{requested.json()['id']}/approve", json={}
    )
    assert approved.status_code == 200, approved.text

    candidate = (await auth_client.get(f"/api/v1/candidates/{candidate_id}")).json()
    stages = {item["vacancy_id"]: item["stage"] for item in candidate["applications"]}
    assert stages[first_vacancy] == "interview"
    assert stages[second_vacancy] == "new"
    assert second_application.status_code == 201


async def test_upload_deduplicates_by_email(auth_client, make_pdf):
    first = await upload_resume(
        auth_client,
        make_pdf("Python developer"),
        candidate_name=None,
        email="Anna.Petrova@Example.com",
    )
    assert first.status_code == 201, first.text
    candidate_id = first.json()["id"]
    assert first.json()["email"] == "anna.petrova@example.com"

    second = await upload_resume(
        auth_client,
        make_pdf("Project manager with Agile experience"),
        candidate_name="Анна Петрова",
        email="anna.petrova@example.com",
        filename="second.pdf",
    )

    assert second.status_code == 201, second.text
    body = second.json()
    assert body["id"] == candidate_id
    assert body["full_name"] == "Анна Петрова"
    assert len(body["sources"]) == 2

    history = await auth_client.get(f"/api/v1/candidates/{candidate_id}/history")
    dedup_events = [
        event for event in history.json() if event["action"] == "candidate_deduplicated"
    ]
    assert dedup_events
    assert dedup_events[0]["details"]["matched_by"] == "email"
    assert "full_name" in dedup_events[0]["details"]["merged_fields"]


async def test_upload_deduplicates_by_phone_extracted_from_text(auth_client, make_pdf):
    first = await upload_resume(
        auth_client,
        make_pdf("Ivan Petrov\n+7 (999) 123-45-67\nPython developer"),
        candidate_name=None,
    )
    assert first.status_code == 201, first.text
    assert first.json()["phone"] == "79991234567"
    candidate_id = first.json()["id"]

    second = await upload_resume(
        auth_client,
        make_pdf("Ivan Petrov\n8 999 123-45-67\nJava developer"),
        candidate_name="Ivan Petrov",
        filename="second.pdf",
    )

    assert second.status_code == 201, second.text
    assert second.json()["id"] == candidate_id
    assert len(second.json()["sources"]) == 2


async def test_duplicate_upload_is_rejected_and_reindex_repairs_index(
    auth_client, make_pdf, fake_vector_store
):
    pdf = make_pdf(RESUME_TEXT)
    first = await upload_resume(auth_client, pdf)
    candidate_id = first.json()["id"]
    resume_id = first.json()["sources"][0]["id"]
    original_points = len(fake_vector_store.points)
    assert original_points > 0

    duplicate = await upload_resume(auth_client, pdf)
    assert duplicate.status_code == 409
    assert resume_id in duplicate.json()["detail"]

    fake_vector_store.fail_index = True
    failed = await auth_client.post(f"/api/v1/resumes/{resume_id}/reindex")
    assert failed.status_code == 200, failed.text
    assert failed.json()["indexing_status"] == "failed"
    assert failed.json()["indexing_error"] == "RuntimeError"
    assert resume_id in fake_vector_store.deleted_resumes
    assert fake_vector_store.points == []

    fake_vector_store.fail_index = False
    repaired = await auth_client.post(f"/api/v1/resumes/{resume_id}/reindex")
    assert repaired.status_code == 200, repaired.text
    assert repaired.json()["indexing_status"] == "indexed"
    assert len(fake_vector_store.points) == original_points

    candidate = (await auth_client.get(f"/api/v1/candidates/{candidate_id}")).json()
    assert candidate["sources"][0]["indexing_status"] == "indexed"


async def test_delete_resume_removes_file_record_and_vectors(
    auth_client, make_pdf, fake_vector_store, container: ServiceContainer
):
    uploaded = await upload_resume(auth_client, make_pdf(RESUME_TEXT))
    candidate_id = uploaded.json()["id"]
    resume_id = uploaded.json()["sources"][0]["id"]
    stored = list(container.settings.storage_dir.glob("*.pdf"))
    assert len(stored) == 1

    deleted = await auth_client.delete(f"/api/v1/resumes/{resume_id}")
    assert deleted.status_code == 204
    assert not stored[0].exists()
    assert resume_id in fake_vector_store.deleted_resumes
    assert fake_vector_store.points == []

    candidate = (await auth_client.get(f"/api/v1/candidates/{candidate_id}")).json()
    assert candidate["sources"] == []

    history = await auth_client.get(f"/api/v1/candidates/{candidate_id}/history")
    assert "resume_deleted" in [event["action"] for event in history.json()]


async def test_delete_is_blocked_when_vector_cleanup_fails(
    auth_client, make_pdf, fake_vector_store, container: ServiceContainer
):
    uploaded = await upload_resume(auth_client, make_pdf(RESUME_TEXT))
    resume_id = uploaded.json()["sources"][0]["id"]
    fake_vector_store.fail_delete = True

    response = await auth_client.delete(f"/api/v1/resumes/{resume_id}")

    assert response.status_code == 503
    assert len(list(container.settings.storage_dir.glob("*.pdf"))) == 1
    candidate = (await auth_client.get(f"/api/v1/candidates/{uploaded.json()['id']}")).json()
    assert len(candidate["sources"]) == 1


async def test_delete_candidate_cascades_applications_and_files(
    auth_client, make_pdf, fake_vector_store, container: ServiceContainer
):
    vacancy_id = await _vacancy(auth_client)
    uploaded = await upload_resume(auth_client, make_pdf(RESUME_TEXT), vacancy_id=vacancy_id)
    candidate_id = uploaded.json()["id"]

    deleted = await auth_client.delete(f"/api/v1/candidates/{candidate_id}")

    assert deleted.status_code == 204
    assert (await auth_client.get(f"/api/v1/candidates/{candidate_id}")).status_code == 404
    applications = await auth_client.get(f"/api/v1/vacancies/{vacancy_id}/applications")
    assert applications.json() == []
    assert candidate_id in fake_vector_store.deleted_candidates
    assert list(container.settings.storage_dir.glob("*.pdf")) == []


async def test_merge_candidates_moves_sources_and_applications(
    auth_client, make_pdf, fake_vector_store
):
    first = await upload_resume(auth_client, make_pdf("Python developer"))
    second = await upload_resume(auth_client, make_pdf("Project manager with Agile"))
    target_id = first.json()["id"]
    source_id = second.json()["id"]
    first_vacancy = await _vacancy(auth_client, "Backend")
    second_vacancy = await _vacancy(auth_client, "Delivery")
    await auth_client.post(
        f"/api/v1/vacancies/{first_vacancy}/applications",
        json={"candidate_id": target_id},
    )
    await auth_client.post(
        f"/api/v1/vacancies/{second_vacancy}/applications",
        json={"candidate_id": source_id},
    )

    merged = await auth_client.post(
        f"/api/v1/candidates/{target_id}/merge",
        json={"source_candidate_id": source_id},
    )

    assert merged.status_code == 200, merged.text
    body = merged.json()
    assert body["id"] == target_id
    assert len(body["sources"]) == 2
    assert {item["vacancy_id"] for item in body["applications"]} == {
        first_vacancy,
        second_vacancy,
    }
    assert (await auth_client.get(f"/api/v1/candidates/{source_id}")).status_code == 404
    assert fake_vector_store.reassignments == [(source_id, target_id)]
    assert {point["candidate_id"] for point in fake_vector_store.points} == {target_id}


async def test_merge_keeps_the_furthest_stage_for_the_same_vacancy(auth_client, make_pdf):
    vacancy_id = await _vacancy(auth_client)
    target = await upload_resume(auth_client, make_pdf("Python developer"), vacancy_id=vacancy_id)
    source = await upload_resume(
        auth_client, make_pdf("Senior Python developer"), vacancy_id=vacancy_id
    )
    source_application_id = source.json()["applications"][0]["id"]

    requested = await auth_client.post(
        f"/api/v1/vacancies/{vacancy_id}/applications/{source_application_id}/stage-change/request",
        json={"stage": "interview"},
    )
    await auth_client.post(f"/api/v1/approvals/{requested.json()['id']}/approve", json={})

    merged = await auth_client.post(
        f"/api/v1/candidates/{target.json()['id']}/merge",
        json={"source_candidate_id": source.json()["id"]},
    )

    assert merged.status_code == 200, merged.text
    applications = merged.json()["applications"]
    assert len(applications) == 1
    assert applications[0]["stage"] == "interview"


async def test_retention_purge_deletes_stale_candidates(
    auth_client, make_pdf, container: ServiceContainer
):
    uploaded = await upload_resume(auth_client, make_pdf(RESUME_TEXT))
    candidate_id = uploaded.json()["id"]
    old = datetime.now(UTC) - timedelta(days=90)
    async with container.database.session() as session:
        await session.execute(
            update(ResumeDocumentRecord).values(uploaded_at=old, indexing_status="indexed")
        )
        await session.execute(update(CandidateRecord).values(created_at=old, updated_at=old))
        await session.commit()

    response = await auth_client.post("/api/v1/maintenance/retention/purge?days=30")

    assert response.status_code == 200, response.text
    result = response.json()
    assert result["deleted_candidates"] == 1
    assert result["failed_candidates"] == 0
    assert (await auth_client.get(f"/api/v1/candidates/{candidate_id}")).status_code == 404

    missing = await auth_client.post("/api/v1/maintenance/retention/purge?days=0")
    assert missing.json()["deleted_candidates"] == 0


async def test_upload_validates_explicit_contacts(auth_client, make_pdf):
    bad_email = await upload_resume(auth_client, make_pdf(RESUME_TEXT), email="not-an-email")
    assert bad_email.status_code == 422
    assert "valid email" in bad_email.json()["detail"]

    bad_phone = await upload_resume(auth_client, make_pdf(RESUME_TEXT), phone="12-34")
    assert bad_phone.status_code == 422
    assert "10 to 15 digits" in bad_phone.json()["detail"]


async def test_docx_upload_is_supported(auth_client):
    content = build_docx("Анна Петрова\nPython developer", table=[["2019", "Яндекс"]])

    response = await upload_resume(
        auth_client,
        content,
        filename="resume.docx",
        candidate_name="Анна Петрова",
    )

    assert response.status_code == 201, response.text
    source = response.json()["sources"][0]
    assert source["filename"] == "resume.docx"
    assert source["extraction_method"] == "docx"
