import json

import httpx
from fastapi import FastAPI

from tests.support import (
    sse_events,
    stream_response,
    sync_csrf,
    text_chunk,
    tool_call_chunk,
    upload_resume,
    usage_chunk,
)


async def test_chat_returns_503_without_api_key(auth_client):
    response = await auth_client.post("/api/v1/chat/stream", json={"message": "Привет"})

    assert response.status_code == 503
    assert "OPENAI_API_KEY" in response.json()["detail"]


async def test_chat_streams_tokens_and_persists_messages(auth_client, fake_llm):
    fake_llm.configured = True
    fake_llm.completions.responses.append(
        stream_response(text_chunk("Здравствуйте"), text_chunk("!"), usage_chunk())
    )

    response = await auth_client.post(
        "/api/v1/chat/stream",
        json={"message": "Привет"},
    )

    assert response.status_code == 200
    events = sse_events(response.text)
    names = [name for name, _ in events]
    assert names[0] == "conversation"
    assert names[-1] == "done"
    tokens = [json.loads(data)["text"] for name, data in events if name == "token"]
    assert "".join(tokens) == "Здравствуйте!"
    conversation_id = json.loads(events[0][1])["conversation_id"]
    assert json.loads(events[-1][1])["conversation_id"] == conversation_id


async def test_chat_executes_tool_calls_and_answers_again(auth_client, fake_llm, make_pdf):
    candidate_id = (await upload_resume(auth_client, make_pdf("Python developer"))).json()["id"]
    fake_llm.configured = True
    fake_llm.completions.responses.extend(
        [
            stream_response(
                tool_call_chunk(
                    "get_candidate_profile",
                    json.dumps({"candidate_id": candidate_id}),
                )
            ),
            stream_response(text_chunk("Кандидат найден."), usage_chunk()),
        ]
    )

    response = await auth_client.post(
        "/api/v1/chat/stream",
        json={"message": "Покажи кандидата"},
    )

    assert response.status_code == 200
    events = sse_events(response.text)
    tokens = [json.loads(data)["text"] for name, data in events if name == "token"]
    assert "".join(tokens) == "Кандидат найден."

    first_call, final_call = fake_llm.completions.calls
    assert first_call["tools"]
    assert first_call["tool_choice"] == "auto"
    assert final_call["tool_choice"] == "none"
    tool_messages = [message for message in final_call["messages"] if message["role"] == "tool"]
    assert tool_messages[0]["content"].startswith("{")


async def test_chat_emits_eval_metrics_on_request(auth_client, fake_llm):
    fake_llm.configured = True
    fake_llm.completions.responses.append(stream_response(text_chunk("ok"), usage_chunk()))

    response = await auth_client.post(
        "/api/v1/chat/stream",
        json={"message": "Привет"},
        headers={"X-Eval-Metrics": "true"},
    )

    events = sse_events(response.text)
    metrics = [json.loads(data) for name, data in events if name == "eval_metrics"]
    assert len(metrics) == 1
    assert metrics[0]["latency_ms"] is not None


async def test_chat_rejects_foreign_conversation(auth_client, fake_llm, app: FastAPI):
    fake_llm.configured = True
    fake_llm.completions.responses.append(stream_response(text_chunk("ok"), usage_chunk()))
    first = await auth_client.post("/api/v1/chat/stream", json={"message": "Привет"})
    conversation_id = json.loads(sse_events(first.text)[0][1])["conversation_id"]

    recruiter = {
        "email": "recruiter@example.org",
        "password": "another-long-password",
        "full_name": "Recruiter",
    }
    assert (await auth_client.post("/api/v1/auth/users", json=recruiter)).status_code == 201

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as other:
        login = await other.post(
            "/api/v1/auth/login",
            json={"email": recruiter["email"], "password": recruiter["password"]},
        )
        assert login.status_code == 200
        sync_csrf(other)

        response = await other.post(
            "/api/v1/chat/stream",
            json={"message": "Привет", "conversation_id": conversation_id},
        )

    events = sse_events(response.text)
    errors = [json.loads(data) for name, data in events if name == "error"]
    assert errors[0]["status"] == 404


async def test_chat_survives_tool_failure(auth_client, fake_llm):
    fake_llm.configured = True
    fake_llm.completions.responses.extend(
        [
            stream_response(tool_call_chunk("get_candidate_profile", '{"candidate_id":"missing"}')),
            stream_response(text_chunk("Не удалось."), usage_chunk()),
        ]
    )

    response = await auth_client.post(
        "/api/v1/chat/stream",
        json={"message": "Покажи кандидата"},
    )

    assert response.status_code == 200
    events = sse_events(response.text)
    tokens = [json.loads(data)["text"] for name, data in events if name == "token"]
    assert "".join(tokens) == "Не удалось."
    final_messages = fake_llm.completions.calls[1]["messages"]
    tool_message = next(message for message in final_messages if message["role"] == "tool")
    assert "Tool failed: NotFoundError" in tool_message["content"]
