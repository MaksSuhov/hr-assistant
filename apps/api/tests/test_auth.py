import httpx

from hr_assistant.app import create_app
from tests.support import ADMIN_CREDENTIALS, register_admin, sync_csrf

RECRUITER = {
    "email": "recruiter@example.org",
    "password": "another-long-password",
    "full_name": "Recruiter User",
}


async def test_first_registration_creates_admin_and_session(client):
    response = await client.post("/api/v1/auth/register", json=ADMIN_CREDENTIALS)

    assert response.status_code == 201
    body = response.json()
    assert body["role"] == "admin"
    assert body["email"] == ADMIN_CREDENTIALS["email"]
    assert "password" not in body
    assert client.cookies["access_token"]
    assert client.cookies["refresh_token"]
    assert client.cookies["csrf_token"]


async def test_registration_closes_after_first_account(client):
    await register_admin(client)

    response = await client.post("/api/v1/auth/register", json=RECRUITER)

    assert response.status_code == 403
    assert "Registration is closed" in response.json()["detail"]


async def test_login_rejects_wrong_password_and_accepts_valid_one(client):
    await register_admin(client)

    wrong = await client.post(
        "/api/v1/auth/login",
        json={"email": ADMIN_CREDENTIALS["email"], "password": "wrong-password"},
    )
    assert wrong.status_code == 401

    valid = await client.post(
        "/api/v1/auth/login",
        json={
            "email": ADMIN_CREDENTIALS["email"].upper(),
            "password": ADMIN_CREDENTIALS["password"],
        },
    )
    assert valid.status_code == 200
    assert valid.json()["role"] == "admin"
    sync_csrf(client)


async def test_me_requires_session(client):
    unauthenticated = await client.get("/api/v1/auth/me")
    assert unauthenticated.status_code == 401

    await register_admin(client)
    authenticated = await client.get("/api/v1/auth/me")
    assert authenticated.status_code == 200
    assert authenticated.json()["email"] == ADMIN_CREDENTIALS["email"]


async def test_state_changing_request_without_csrf_is_rejected(auth_client):
    auth_client.headers.pop("x-csrf-token")

    response = await auth_client.post(
        "/api/v1/vacancies",
        json={"title": "Backend developer"},
    )

    assert response.status_code == 403
    assert "CSRF" in response.json()["detail"]


async def test_refresh_rotates_tokens(auth_client):
    old_refresh = auth_client.cookies["refresh_token"]
    sync_csrf(auth_client)

    response = await auth_client.post("/api/v1/auth/refresh")

    assert response.status_code == 200
    assert auth_client.cookies["refresh_token"] != old_refresh
    assert auth_client.cookies["access_token"]


async def test_logout_clears_session(auth_client):
    sync_csrf(auth_client)

    response = await auth_client.post("/api/v1/auth/logout")

    assert response.status_code == 204
    assert (await auth_client.get("/api/v1/auth/me")).status_code == 401


async def test_only_admin_can_provision_recruiters(client):
    await register_admin(client)

    created = await client.post("/api/v1/auth/users", json=RECRUITER)
    assert created.status_code == 201
    assert created.json()["role"] == "recruiter"

    login = await client.post(
        "/api/v1/auth/login",
        json={"email": RECRUITER["email"], "password": RECRUITER["password"]},
    )
    assert login.status_code == 200
    sync_csrf(client)

    forbidden = await client.post(
        "/api/v1/auth/users",
        json={
            "email": "third@example.org",
            "password": "yet-another-long-password",
            "full_name": "Third User",
        },
    )
    assert forbidden.status_code == 403


async def test_registration_validates_payload(client):
    response = await client.post(
        "/api/v1/auth/register",
        json={"email": "not-an-email", "password": "short", "full_name": "A"},
    )

    assert response.status_code == 422


def _session_cookie(headers: httpx.Headers) -> str:
    return next(
        header for header in headers.get_list("set-cookie") if header.startswith("access_token=")
    )


async def test_session_cookies_are_secure_outside_localhost(settings, container):
    settings.auth_cookie_secure = None
    app = create_app(settings, container=container)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as browser:
        remote = await browser.post(
            "/api/v1/auth/register",
            json=ADMIN_CREDENTIALS,
            headers={"host": "hr.example.org"},
        )
        local = await browser.post(
            "/api/v1/auth/login",
            json=ADMIN_CREDENTIALS,
            headers={"host": "localhost"},
        )

    assert remote.status_code == 201
    assert "Secure" in _session_cookie(remote.headers)
    assert local.status_code == 200
    assert "Secure" not in _session_cookie(local.headers)
