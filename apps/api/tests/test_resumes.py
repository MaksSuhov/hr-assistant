from hr_assistant.container import ServiceContainer
from tests.support import json_response, upload_resume

RESUME_TEXT = "Python developer with FastAPI experience and PostgreSQL"


async def test_upload_extracts_text_indexes_and_creates_candidate(
    auth_client, make_pdf, fake_vector_store, container: ServiceContainer
):
    response = await upload_resume(auth_client, make_pdf(RESUME_TEXT))

    assert response.status_code == 201, response.text
    candidate = response.json()
    assert candidate["full_name"] == "Алексей Иванов"
    assert candidate["applications"] == []
    assert len(candidate["sources"]) == 1
    assert candidate["sources"][0]["filename"] == "resume.pdf"
    assert candidate["sources"][0]["indexing_status"] == "indexed"
    assert candidate["sources"][0]["source"] == "HR upload"

    assert len(fake_vector_store.points) >= 1
    stored = list(container.settings.storage_dir.glob("*.pdf"))
    assert len(stored) == 1


async def test_second_upload_attaches_to_existing_candidate(auth_client, make_pdf):
    first = await upload_resume(auth_client, make_pdf("Python developer"))
    candidate_id = first.json()["id"]

    second = await upload_resume(
        auth_client,
        make_pdf("Project manager with Agile experience"),
        candidate_id=candidate_id,
        filename="second.pdf",
    )

    assert second.status_code == 201
    assert second.json()["id"] == candidate_id
    assert len(second.json()["sources"]) == 2


async def test_upload_to_unknown_candidate_returns_404(auth_client, make_pdf):
    response = await upload_resume(
        auth_client,
        make_pdf(RESUME_TEXT),
        candidate_id="missing-candidate",
    )

    assert response.status_code == 404


async def test_upload_rejects_bad_files(auth_client, make_pdf):
    wrong_extension = await upload_resume(auth_client, make_pdf(RESUME_TEXT), filename="resume.txt")
    assert wrong_extension.status_code == 422
    assert "PDF or DOCX" in wrong_extension.json()["detail"]

    not_a_pdf = await upload_resume(auth_client, b"definitely not a pdf")
    assert not_a_pdf.status_code == 422

    blank = await upload_resume(auth_client, make_pdf(pages=[""]))
    assert blank.status_code == 422
    assert "No readable text" in blank.json()["detail"]


async def test_failed_indexing_is_recorded_but_upload_survives(
    auth_client, make_pdf, fake_vector_store
):
    fake_vector_store.fail_index = True

    response = await upload_resume(auth_client, make_pdf(RESUME_TEXT))

    assert response.status_code == 201
    source = response.json()["sources"][0]
    assert source["indexing_status"] == "failed"
    candidate_id = response.json()["id"]

    history = await auth_client.get(f"/api/v1/candidates/{candidate_id}/history")
    actions = [event["action"] for event in history.json()]
    assert "resume_uploaded" in actions
    details = next(
        event["details"] for event in history.json() if event["action"] == "resume_uploaded"
    )
    assert details["indexing_status"] == "failed"
    assert details["indexing_error"] == "RuntimeError"
    assert "text_preview" not in details
    assert "filename" not in details


async def test_public_upload_needs_no_session(client, make_pdf):
    response = await upload_resume(
        client,
        make_pdf("Candidate portal upload"),
        candidate_name="Публичный Кандидат",
        endpoint="/api/v1/resumes/public",
        authenticated=False,
    )

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["full_name"] == "Публичный Кандидат"
    assert body["sources"][0]["source"] == "Candidate portal"


async def test_candidate_search_uses_vector_store(auth_client, make_pdf):
    uploaded = await upload_resume(auth_client, make_pdf(RESUME_TEXT))
    candidate_id = uploaded.json()["id"]

    response = await auth_client.post(
        "/api/v1/candidates/search",
        json={"query": "FastAPI", "limit": 5},
    )

    assert response.status_code == 200
    hits = response.json()
    assert hits
    assert hits[0]["candidate_id"] == candidate_id
    assert "FastAPI" in hits[0]["snippet"]
    assert hits[0]["candidate_name"] == "Алексей Иванов"


async def test_search_falls_back_to_sqlite_when_vector_store_fails(
    auth_client, make_pdf, fake_vector_store
):
    await upload_resume(auth_client, make_pdf(RESUME_TEXT))
    fake_vector_store.fail_search = True

    response = await auth_client.post(
        "/api/v1/candidates/search",
        json={"query": "FastAPI PostgreSQL", "limit": 5},
    )

    assert response.status_code == 200
    hits = response.json()
    assert hits
    assert hits[0]["score"] is not None


async def test_candidate_history_and_listing(auth_client, make_pdf):
    uploaded = await upload_resume(auth_client, make_pdf(RESUME_TEXT))
    candidate_id = uploaded.json()["id"]

    listing = await auth_client.get("/api/v1/candidates")
    assert listing.status_code == 200
    assert [item["id"] for item in listing.json()] == [candidate_id]

    history = await auth_client.get(f"/api/v1/candidates/{candidate_id}/history")
    assert history.status_code == 200
    assert history.json()[0]["action"] == "resume_uploaded"

    missing = await auth_client.get("/api/v1/candidates/missing")
    assert missing.status_code == 404


async def test_matching_without_llm_returns_evidence_only(auth_client, make_pdf):
    vacancy = await auth_client.post(
        "/api/v1/vacancies",
        json={
            "title": "Backend developer",
            "requirements": [{"text": "Python experience", "is_must_have": True, "weight": 5}],
        },
    )
    vacancy_id = vacancy.json()["id"]
    candidate_id = (await upload_resume(auth_client, make_pdf(RESUME_TEXT))).json()["id"]

    response = await auth_client.post(
        f"/api/v1/vacancies/{vacancy_id}/matches",
        json={"candidate_ids": [candidate_id]},
    )

    assert response.status_code == 200, response.text
    analysis = response.json()["candidates"][0]
    assert analysis["candidate_id"] == candidate_id
    assert analysis["requirement_evidence"][0]["status"] in {"unknown", "not_found"}
    assert analysis["clarification"]


async def test_matching_with_llm_uses_selected_evidence(auth_client, make_pdf, fake_llm):
    vacancy = await auth_client.post(
        "/api/v1/vacancies",
        json={
            "title": "Backend developer",
            "requirements": [{"text": "Python experience", "is_must_have": True, "weight": 5}],
        },
    )
    vacancy_body = vacancy.json()
    requirement_id = vacancy_body["requirements"][0]["id"]
    candidate_id = (await upload_resume(auth_client, make_pdf(RESUME_TEXT))).json()["id"]

    fake_llm.configured = True
    fake_llm.completions.responses.append(
        json_response(
            {
                "summary": "Python подтверждён опытом.",
                "requirements": [
                    {
                        "requirement_id": requirement_id,
                        "status": "supported",
                        "evidence_indices": [0],
                        "gap_or_question": "Уточнить глубину экспертизы.",
                    }
                ],
            }
        )
    )

    response = await auth_client.post(
        f"/api/v1/vacancies/{vacancy_body['id']}/matches",
        json={"candidate_ids": [candidate_id]},
    )

    assert response.status_code == 200, response.text
    analysis = response.json()["candidates"][0]
    assert analysis["summary"] == "Python подтверждён опытом."
    evidence = analysis["requirement_evidence"][0]
    assert evidence["status"] == "supported"
    assert evidence["evidence"]


async def test_compare_requires_two_candidates(auth_client, make_pdf):
    vacancy = await auth_client.post(
        "/api/v1/vacancies",
        json={"title": "Backend developer", "requirements": []},
    )
    candidate_id = (await upload_resume(auth_client, make_pdf(RESUME_TEXT))).json()["id"]

    response = await auth_client.post(
        "/api/v1/candidates/compare",
        json={"vacancy_id": vacancy.json()["id"], "candidate_ids": [candidate_id]},
    )

    assert response.status_code == 422
