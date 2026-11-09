VACANCY = {
    "title": "Backend developer",
    "department": "Platform",
    "description": "Internal services",
    "requirements": [
        {"text": "Python experience of 3+ years", "is_must_have": True, "weight": 5},
        {"text": "PostgreSQL", "is_must_have": True, "weight": 3},
    ],
}


async def test_create_list_and_get_vacancy(auth_client):
    created = await auth_client.post("/api/v1/vacancies", json=VACANCY)

    assert created.status_code == 201
    vacancy = created.json()
    assert vacancy["title"] == VACANCY["title"]
    assert vacancy["status"] == "open"
    assert len(vacancy["requirements"]) == 2

    listing = await auth_client.get("/api/v1/vacancies")
    assert listing.status_code == 200
    assert [item["id"] for item in listing.json()] == [vacancy["id"]]

    fetched = await auth_client.get(f"/api/v1/vacancies/{vacancy['id']}")
    assert fetched.status_code == 200
    assert fetched.json()["department"] == "Platform"


async def test_patch_updates_fields_and_replaces_requirements(auth_client):
    created = await auth_client.post("/api/v1/vacancies", json=VACANCY)
    vacancy = created.json()

    response = await auth_client.patch(
        f"/api/v1/vacancies/{vacancy['id']}",
        json={
            "title": "Senior backend developer",
            "status": "paused",
            "department": None,
            "requirements": [{"text": "Distributed systems", "is_must_have": False, "weight": 2}],
        },
    )

    assert response.status_code == 200, response.text
    updated = response.json()
    assert updated["title"] == "Senior backend developer"
    assert updated["status"] == "paused"
    assert updated["department"] is None
    assert [item["text"] for item in updated["requirements"]] == ["Distributed systems"]
    assert updated["created_at"] <= updated["updated_at"]


async def test_patch_of_unknown_vacancy_returns_404(auth_client):
    response = await auth_client.patch("/api/v1/vacancies/missing", json={"title": "Another"})

    assert response.status_code == 404


async def test_vacancy_payload_validation(auth_client):
    short_title = await auth_client.post("/api/v1/vacancies", json={"title": "x"})
    assert short_title.status_code == 422

    bad_status = await auth_client.post("/api/v1/vacancies", json=VACANCY)
    invalid = await auth_client.patch(
        f"/api/v1/vacancies/{bad_status.json()['id']}",
        json={"status": "bogus"},
    )
    assert invalid.status_code == 422

    bad_limit = await auth_client.get("/api/v1/vacancies?limit=0")
    assert bad_limit.status_code == 422


async def test_vacancy_routes_require_authentication(client):
    response = await client.get("/api/v1/vacancies")

    assert response.status_code == 401
