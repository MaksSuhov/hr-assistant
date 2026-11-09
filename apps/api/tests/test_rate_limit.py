import httpx

from hr_assistant.api.rate_limit import SlidingWindowLimiter
from hr_assistant.app import create_app
from tests.support import ADMIN_CREDENTIALS


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver")


def test_sliding_window_limiter_forgets_hits_after_the_window(monkeypatch):
    clock = {"now": 1_000.0}
    monkeypatch.setattr("hr_assistant.api.rate_limit.time.monotonic", lambda: clock["now"])
    limiter = SlidingWindowLimiter()

    assert limiter.hit("client", 1, 60) is None
    assert limiter.hit("client", 1, 60) is not None

    clock["now"] += 61
    assert limiter.hit("client", 1, 60) is None


def test_sliding_window_limiter_prunes_beyond_capacity():
    limiter = SlidingWindowLimiter(max_keys=1)

    assert limiter.hit("a", 1, 60) is None
    assert limiter.hit("b", 1, 60) is None
    assert limiter.hit("c", 1, 60) is None

    assert len(limiter._hits) <= 2


async def test_login_is_rate_limited_per_client(settings, container):
    settings.rate_limit_enabled = True
    settings.rate_limit_login_per_minute = 2
    app = create_app(settings, container=container)

    async with _client(app) as client:
        payload = {"email": ADMIN_CREDENTIALS["email"], "password": "wrong-password"}
        first = await client.post("/api/v1/auth/login", json=payload)
        second = await client.post("/api/v1/auth/login", json=payload)
        third = await client.post("/api/v1/auth/login", json=payload)

    assert first.status_code == 401
    assert second.status_code == 401
    assert third.status_code == 429
    assert third.headers["Retry-After"]
    assert third.json()["detail"] == "Too many requests. Try again later."


async def test_public_upload_is_rate_limited_per_client(settings, container):
    settings.rate_limit_enabled = True
    settings.rate_limit_public_upload_per_hour = 1
    app = create_app(settings, container=container)

    async with _client(app) as client:
        files = {"file": ("resume.pdf", b"not a pdf", "application/pdf")}
        first = await client.post("/api/v1/resumes/public", files=files)
        second = await client.post("/api/v1/resumes/public", files=files)

    assert first.status_code == 422
    assert second.status_code == 429


async def test_global_limit_applies_to_every_route(settings, container):
    settings.rate_limit_enabled = True
    settings.rate_limit_per_minute = 2
    app = create_app(settings, container=container)

    async with _client(app) as client:
        first = await client.get("/api/v1/auth/me")
        second = await client.get("/api/v1/auth/me")
        third = await client.get("/api/v1/auth/me")

    assert first.status_code == 401
    assert second.status_code == 401
    assert third.status_code == 429


async def test_health_checks_are_never_throttled(settings, container):
    settings.rate_limit_enabled = True
    settings.rate_limit_per_minute = 1
    app = create_app(settings, container=container)

    async with _client(app) as client:
        responses = [await client.get("/api/v1/health") for _ in range(3)]

    assert [response.status_code for response in responses] == [200, 200, 200]
