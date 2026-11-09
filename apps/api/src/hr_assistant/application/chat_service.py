import json
import time
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

from hr_assistant.application.errors import NotFoundError
from hr_assistant.application.ports import ResumeSearch, UnitOfWorkProvider
from hr_assistant.application.tool_executor import ToolExecutor
from hr_assistant.core.config import Settings
from hr_assistant.core.prompt_safety import RESUME_DATA_RULE, wrap_resume_text
from hr_assistant.infrastructure.db.models import ChatMessageRecord, ChatSessionRecord
from hr_assistant.infrastructure.llm.openai_provider import OpenAIProvider
from hr_assistant.infrastructure.observability.langfuse_tracer import LangfuseTracer
from hr_assistant.tools.definitions import HR_ASSISTANT_TOOLS


def _usage_details(usage: Any) -> dict[str, int] | None:
    """Map an OpenAI usage object onto Langfuse's usage_details keys."""
    if usage is None:
        return None
    details = {
        key: value
        for key, value in (
            ("input", getattr(usage, "prompt_tokens", None)),
            ("output", getattr(usage, "completion_tokens", None)),
            ("total", getattr(usage, "total_tokens", None)),
        )
        if isinstance(value, int)
    }
    return details or None


class ChatService:
    SYSTEM_PROMPT = (
        "Ты — помощник HR. Отвечай по-русски, кратко и конкретно. "
        "Опирайся на найденные фрагменты резюме и называй источник. " + RESUME_DATA_RULE + " "
        "Не утверждай, что открыл ссылку, сайт или вложение, если их содержимое не передано тебе. "
        "Если пользователь прислал только внешний URL, не связывай его с фрагментами "
        "другого кандидата, найденными по совпадению слов. "
        "Для ссылок hh.ru и Telegram попроси загрузить PDF "
        "или вставить текст профиля, если содержимое не передано в контексте. "
        "Если данных не хватает, прямо обозначай это и предлагай вопрос для интервью. "
        "Не принимай решение о найме, не меняй статусы и не отправляй сообщения. "
        "Действия, требующие подтверждения HR, доступны отдельно через API подтверждений."
    )

    def __init__(
        self,
        uow_provider: UnitOfWorkProvider,
        search: ResumeSearch,
        tools: ToolExecutor,
        llm: OpenAIProvider,
        tracer: LangfuseTracer,
        settings: Settings,
    ) -> None:
        self.uow_provider = uow_provider
        self.search = search
        self.tools = tools
        self.llm = llm
        self.tracer = tracer
        self.settings = settings

    @property
    def configured(self) -> bool:
        return self.llm.configured

    async def stream_chat(
        self,
        message: str,
        conversation_id: str | None,
        actor_id: str,
        include_eval_metrics: bool = False,
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        request_started = time.perf_counter()
        client = self.llm.require_client()
        session_id, history = await self._save_user_message(message, conversation_id, actor_id)
        yield "conversation", {"conversation_id": session_id}

        answer_parts: list[str] = []
        tool_calls: dict[int, dict[str, Any]] = {}
        ttft_ms: float | None = None
        ttfrt_ms: float | None = None
        completion_started_at: datetime | None = None
        final_completion_started_at: datetime | None = None
        with self.tracer.trace(
            "hr-assistant-chat",
            {"conversation_id": session_id, "message": message},
            as_type="agent",
            session_id=session_id,
            user_id=actor_id,
        ) as observation:
            with self.tracer.observation(
                "resume-search", {"query": message, "limit": 5}, as_type="retriever"
            ) as search_span:
                retrieved = await self.search.search_raw(message, limit=5)
                if search_span:
                    search_span.update(
                        output=[
                            {
                                "candidate_id": hit.get("candidate_id"),
                                "resume_id": hit.get("resume_id"),
                                "source_filename": hit.get("source_filename"),
                                "score": hit.get("score"),
                            }
                            for hit in retrieved
                        ]
                    )
            rag_context = (
                "\n\n".join(
                    wrap_resume_text(
                        str(hit.get("text", ""))[:1000],
                        source=str(hit.get("source_filename", "резюме")),
                        candidate_id=str(hit.get("candidate_id", "")),
                    )
                    for hit in retrieved
                )
                or "Подходящие фрагменты резюме не найдены."
            )
            messages: list[Any] = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {
                    "role": "system",
                    "content": f"Контекст из поиска резюме:\n{rag_context}",
                },
                *history,
            ]

            usage: dict[str, int] | None = None
            with self.tracer.generation(
                "chat-completion", model=self.settings.openai_chat_model, input_data=messages
            ) as generation:
                first_stream = await client.chat.completions.create(
                    model=self.settings.openai_chat_model,
                    messages=messages,
                    tools=HR_ASSISTANT_TOOLS,
                    tool_choice="auto",
                    stream=True,
                    extra_body=self.llm.chat_extra_body,
                )
                async for chunk in first_stream:
                    # RouterAI ends every stream with a usage-only chunk whose
                    # `choices` list is empty; there is no delta to read there.
                    if not chunk.choices:
                        usage = _usage_details(getattr(chunk, "usage", None)) or usage
                        continue
                    delta = chunk.choices[0].delta
                    if ttft_ms is None and (delta.content or delta.tool_calls):
                        ttft_ms = (time.perf_counter() - request_started) * 1000
                        completion_started_at = datetime.now(UTC)
                    if delta.content:
                        if ttfrt_ms is None:
                            ttfrt_ms = (time.perf_counter() - request_started) * 1000
                        answer_parts.append(delta.content)
                        yield "token", {"text": delta.content}
                    for tool_delta in delta.tool_calls or []:
                        call = tool_calls.setdefault(
                            tool_delta.index,
                            {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            },
                        )
                        if tool_delta.id:
                            call["id"] = tool_delta.id
                        if tool_delta.function:
                            if tool_delta.function.name:
                                call["function"]["name"] += tool_delta.function.name
                            if tool_delta.function.arguments:
                                call["function"]["arguments"] += tool_delta.function.arguments
                if generation:
                    generation.update(
                        output="".join(answer_parts) or None,
                        usage_details=usage,
                        completion_start_time=completion_started_at,
                        metadata={"requested_tool_calls": len(tool_calls)},
                    )

            if tool_calls:
                ordered_calls = [tool_calls[index] for index in sorted(tool_calls)]
                messages.append({"role": "assistant", "content": None, "tool_calls": ordered_calls})
                for call in ordered_calls:
                    tool_name = call["function"]["name"]
                    with self.tracer.observation(
                        f"tool-{tool_name}",
                        {"arguments": call["function"]["arguments"]},
                        as_type="tool",
                    ) as tool_span:
                        try:
                            tool_result = await self.tools.execute(
                                tool_name, call["function"]["arguments"]
                            )
                        except Exception as error:
                            if tool_span:
                                tool_span.update(
                                    level="ERROR",
                                    status_message=f"{type(error).__name__}: {error}",
                                )
                            tool_result = json.dumps(
                                {"error": f"Tool failed: {type(error).__name__}"},
                                ensure_ascii=False,
                            )
                        else:
                            if tool_span:
                                tool_span.update(output=tool_result[:2000])
                    messages.append(
                        {"role": "tool", "tool_call_id": call["id"], "content": tool_result}
                    )
                final_usage: dict[str, int] | None = None
                with self.tracer.generation(
                    "chat-completion-final",
                    model=self.settings.openai_chat_model,
                    input_data=messages,
                ) as generation:
                    final_stream = await client.chat.completions.create(
                        model=self.settings.openai_chat_model,
                        messages=messages,
                        tool_choice="none",
                        stream=True,
                        extra_body=self.llm.chat_extra_body,
                    )
                    async for chunk in final_stream:
                        if not chunk.choices:
                            final_usage = (
                                _usage_details(getattr(chunk, "usage", None)) or final_usage
                            )
                            continue
                        text = chunk.choices[0].delta.content
                        if text:
                            if ttfrt_ms is None:
                                ttfrt_ms = (time.perf_counter() - request_started) * 1000
                            if final_completion_started_at is None:
                                final_completion_started_at = datetime.now(UTC)
                            answer_parts.append(text)
                            yield "token", {"text": text}
                    if generation:
                        generation.update(
                            output="".join(answer_parts) or None,
                            usage_details=final_usage,
                            completion_start_time=final_completion_started_at,
                        )

            answer = "".join(answer_parts)
            if observation:
                observation.update(output=answer)
        message_id = await self._save_assistant_message(session_id, answer)
        if include_eval_metrics:
            yield (
                "eval_metrics",
                {
                    "ttft_ms": round(ttft_ms, 3) if ttft_ms is not None else None,
                    "ttfrt_ms": round(ttfrt_ms, 3) if ttfrt_ms is not None else None,
                    "latency_ms": round((time.perf_counter() - request_started) * 1000, 3),
                },
            )
        yield "done", {"conversation_id": session_id, "message_id": message_id}

    async def _save_user_message(
        self, content: str, conversation_id: str | None, actor_id: str
    ) -> tuple[str, list[dict[str, Any]]]:
        async with self.uow_provider() as uow:
            if conversation_id:
                chat = await uow.chats.get_session(conversation_id)
                if not chat or chat.actor_id != actor_id:
                    raise NotFoundError("Chat session was not found for this actor.")
                session_id = conversation_id
            else:
                session_id = str(uuid.uuid4())
                await uow.chats.add_session(ChatSessionRecord(id=session_id, actor_id=actor_id))
            await uow.chats.add_message(
                ChatMessageRecord(
                    id=str(uuid.uuid4()), session_id=session_id, role="user", content=content
                )
            )
            await uow.commit()
            history_records = await uow.chats.history(session_id)
            history = [{"role": item.role, "content": item.content} for item in history_records]
            return session_id, history

    async def _save_assistant_message(self, session_id: str, content: str) -> str:
        message_id = str(uuid.uuid4())
        async with self.uow_provider() as uow:
            await uow.chats.add_message(
                ChatMessageRecord(
                    id=message_id, session_id=session_id, role="assistant", content=content
                )
            )
            await uow.commit()
        return message_id
