import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

from hr_assistant.application.errors import NotFoundError
from hr_assistant.application.ports import ResumeSearch, UnitOfWorkProvider
from hr_assistant.application.tool_executor import ToolExecutor
from hr_assistant.core.config import Settings
from hr_assistant.infrastructure.db.models import ChatMessageRecord, ChatSessionRecord
from hr_assistant.infrastructure.llm.openai_provider import OpenAIProvider
from hr_assistant.infrastructure.observability.langfuse_tracer import LangfuseTracer
from hr_assistant.tools.definitions import HR_ASSISTANT_TOOLS


class ChatService:
    SYSTEM_PROMPT = (
        "Ты — помощник HR. Отвечай по-русски, кратко и конкретно. "
        "Опирайся на найденные фрагменты резюме и называй источник. "
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
        self, message: str, conversation_id: str | None, actor_id: str
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        session_id, history = await self._save_user_message(message, conversation_id, actor_id)
        yield "conversation", {"conversation_id": session_id}

        retrieved = await self.search.search_raw(message, limit=5)
        rag_context = "\n\n".join(
            f"[{hit.get('source_filename', 'резюме')} | candidate_id={hit.get('candidate_id')}] "
            f"{str(hit.get('text', ''))[:1000]}"
            for hit in retrieved
        ) or "Подходящие фрагменты резюме не найдены."
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "system",
                "content": f"Контекст из поиска резюме:\n{rag_context}",
            },
            *history,
        ]

        answer_parts: list[str] = []
        tool_calls: dict[int, dict[str, Any]] = {}
        with self.tracer.observation(
            "hr-assistant-chat", {"conversation_id": session_id, "message": message}
        ) as observation:
            first_stream = await self.llm.client.chat.completions.create(
                model=self.settings.openai_chat_model,
                messages=messages,
                tools=HR_ASSISTANT_TOOLS,
                tool_choice="auto",
                stream=True,
            )
            async for chunk in first_stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    answer_parts.append(delta.content)
                    yield "token", {"text": delta.content}
                for tool_delta in delta.tool_calls or []:
                    call = tool_calls.setdefault(
                        tool_delta.index,
                        {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                    )
                    if tool_delta.id:
                        call["id"] = tool_delta.id
                    if tool_delta.function:
                        if tool_delta.function.name:
                            call["function"]["name"] += tool_delta.function.name
                        if tool_delta.function.arguments:
                            call["function"]["arguments"] += tool_delta.function.arguments

            if tool_calls:
                ordered_calls = [tool_calls[index] for index in sorted(tool_calls)]
                messages.append({"role": "assistant", "content": None, "tool_calls": ordered_calls})
                for call in ordered_calls:
                    try:
                        tool_result = await self.tools.execute(
                            call["function"]["name"], call["function"]["arguments"]
                        )
                    except Exception as error:
                        tool_result = json.dumps(
                            {"error": f"Tool failed: {type(error).__name__}"}, ensure_ascii=False
                        )
                    messages.append(
                        {"role": "tool", "tool_call_id": call["id"], "content": tool_result}
                    )
                final_stream = await self.llm.client.chat.completions.create(
                    model=self.settings.openai_chat_model,
                    messages=messages,
                    tool_choice="none",
                    stream=True,
                )
                async for chunk in final_stream:
                    text = chunk.choices[0].delta.content
                    if text:
                        answer_parts.append(text)
                        yield "token", {"text": text}

            answer = "".join(answer_parts)
            if observation:
                observation.update(output=answer)
        message_id = await self._save_assistant_message(session_id, answer)
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
