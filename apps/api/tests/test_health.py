async def test_health_reports_configuration(client):
    response = await client.get("/api/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["openai_configured"] is False
    assert body["langfuse_configured"] is False
    assert body["qdrant_url"] == "http://127.0.0.1:1"


async def test_openapi_schema_is_served(client):
    response = await client.get("/openapi.json")

    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/api/v1/vacancies" in paths
    assert "/api/v1/chat/stream" in paths
