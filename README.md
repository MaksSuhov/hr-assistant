# HR Assistant Backend

Backend первой версии HR-помощника: вакансии, резюме PDF, карточки кандидатов, гибридный поиск по резюме, сопоставление с требованиями вакансии, чат с SSE и действия с ручным подтверждением HR.

## Стек и архитектура

- **FastAPI + Pydantic + asyncio + Uvicorn** — REST API и SSE чат.
- **SQLite + SQLAlchemy async** — вакансии, требования, кандидаты, документы, история, диалоги, подтверждения и очередь сообщений.
- **Qdrant** — текстовые фрагменты резюме; dense-вектора OpenAI + BM25 FastEmbed объединяются Reciprocal Rank Fusion (RRF).
- **LangChain Text Splitters** — разбиение извлечённого текста на фрагменты.
- **OpenAI SDK** — embeddings, чат, tool calling и анализ подтверждений по требованиям; работает через любой OpenAI-совместимый шлюз, например RouterAI.
- **Langfuse** — трассировка чата при наличии ключей.
- **FastMCP** — read-only инструменты для поиска, просмотра карточки и сравнения кандидатов по адресу `/mcp`.

Код разделён на `api`, `application`, `domain`, `schemas`, `infrastructure` и `mcp`. Вынесены сервисы вакансий, кандидатов, поиска, сопоставления и подтверждений; доступ к SQLite идёт через репозитории и Unit of Work.

## Запуск локально

Нужны Python 3.11+ и Docker с Compose.

```bash
cp .env.example .env
# Впишите OPENAI_API_KEY в .env для embeddings, tool calling и чата.
docker compose up --build
```

После запуска:

- API и OpenAPI: [http://localhost:8000/docs](http://localhost:8000/docs)
- Проверка состояния: [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health)
- MCP Streamable HTTP: `http://localhost:8000/mcp`
- Qdrant Dashboard: [http://localhost:6333/dashboard](http://localhost:6333/dashboard)

Без `OPENAI_API_KEY` API и загрузка PDF работают, но индексация резюме и чат будут недоступны; поиск использует текстовый fallback по сохранённым документам. После добавления ключа перезапустите сервис. Повторная индексация старых резюме пока выполняется повторной загрузкой документа.

## Провайдер моделей

Модели подключаются через OpenAI-совместимый endpoint. Если `OPENAI_BASE_URL` пуст, запросы идут в `api.openai.com`. Для RouterAI в `.env` задаются:

```dotenv
OPENAI_API_KEY=<ключ RouterAI>
OPENAI_BASE_URL=https://routerai.ru/api/v1
OPENAI_CHAT_MODEL=openai/gpt-4.1-mini
OPENAI_EMBEDDING_MODEL=openai/text-embedding-3-small
```

Идентификаторы моделей у шлюза идут с префиксом провайдера (`openai/…`, `anthropic/…`, `google/…`), актуальный список — на [routerai.ru/models](https://routerai.ru/models). Модель чата должна поддерживать `tools` и `response_format`, иначе чат и анализ требований деградируют. `OPENAI_EMBEDDING_DIMENSIONS` должен совпадать с размерностью выбранной модели и с коллекцией Qdrant: при смене модели на другую размерность коллекцию нужно пересоздать.

Qdrant поднимается сервисом `qdrant` из `docker-compose.yml`; `QDRANT_URL` в `.env` указывает на него, а коллекция `QDRANT_COLLECTION` создаётся при старте приложения.

## Основные маршруты

Все REST-маршруты находятся под `/api/v1`. Для журнала действий API принимает `X-Actor-Id`; без заголовка используется `local-hr`. Это идентификатор автора события, а не механизм аутентификации.

| Метод | Маршрут | Назначение |
|---|---|---|
| `POST` | `/vacancies` | Создать вакансию и требования |
| `PATCH` | `/vacancies/{id}` | Изменить поля и заменить список требований |
| `GET` | `/vacancies`, `/vacancies/{id}` | Список и карточка вакансии |
| `POST` | `/resumes` | Загрузить PDF (`multipart/form-data`), извлечь текст и проиндексировать |
| `GET` | `/candidates`, `/candidates/{id}` | Список и карточка кандидата с источниками |
| `GET` | `/candidates/{id}/history` | История загрузок и изменений HR |
| `POST` | `/candidates/search` | Гибридный поиск по резюме |
| `POST` | `/vacancies/{id}/matches` | Сопоставить кандидатов с требованиями вакансии |
| `POST` | `/candidates/compare` | Сравнить 2–10 кандидатов |
| `POST` | `/approvals/candidates/{id}/status-change/request` | Создать запрос на смену статуса |
| `POST` | `/approvals/messages/drafts` | Создать черновик сообщения для подтверждения |
| `GET` | `/approvals/pending` | Показать ожидающие подтверждения |
| `GET` | `/approvals/outbox` | Посмотреть подтверждённые сообщения в очереди |
| `POST` | `/approvals/{id}/approve` или `/reject` | Подтвердить или отклонить действие |
| `POST` | `/chat/stream` | Чат с SSE |

Смена статуса происходит только при вызове `approve`. Черновик сообщения ничего не отправляет. После подтверждения сообщение отправится через SMTP, если заданы `SMTP_HOST` и `SMTP_FROM`; иначе оно останется в SQLite outbox со статусом `queued`. Статус и ошибки доставки доступны через `/approvals/outbox`.

## Примеры

Создать вакансию:

```bash
curl -X POST http://localhost:8000/api/v1/vacancies \
  -H 'Content-Type: application/json' \
  -H 'X-Actor-Id: hr@example.org' \
  -d '{
    "title":"Backend-разработчик",
    "department":"Платформа",
    "description":"Разработка внутренних сервисов",
    "requirements":[
      {"text":"Опыт Python от 3 лет","is_must_have":true,"weight":5},
      {"text":"Опыт с PostgreSQL","is_must_have":true,"weight":3}
    ]
  }'
```

Загрузить резюме. `candidate_id` можно не передавать — тогда будет создана новая карточка:

```bash
curl -X POST http://localhost:8000/api/v1/resumes \
  -H 'X-Actor-Id: hr@example.org' \
  -F 'candidate_name=Алексей Иванов' \
  -F 'source=Почта отдела кадров' \
  -F 'file=@./resume.pdf'
```

Попросить сменить статус и затем подтвердить действие:

```bash
curl -X POST http://localhost:8000/api/v1/approvals/candidates/CANDIDATE_ID/status-change/request \
  -H 'Content-Type: application/json' -H 'X-Actor-Id: hr@example.org' \
  -d '{"status":"interview","reason":"HR просмотрел резюме"}'

curl -X POST http://localhost:8000/api/v1/approvals/APPROVAL_ID/approve \
  -H 'Content-Type: application/json' -H 'X-Actor-Id: hr@example.org' \
  -d '{"note":"Подтверждаю приглашение на интервью"}'
```

SSE чат:

```bash
curl -N -X POST http://localhost:8000/api/v1/chat/stream \
  -H 'Content-Type: application/json' -H 'X-Actor-Id: hr@example.org' \
  -d '{"message":"Найди кандидатов с опытом FastAPI и сравни их по вакансии VACANCY_ID"}'
```

События SSE: `conversation` (ID диалога), `token` (часть ответа), `done` (ID сообщения) и `error`. Для продолжения передайте полученный `conversation_id` в следующем запросе.

## Первое приближение и границы

- При сканированных PDF без текстового слоя извлечение может вернуть пустой текст; OCR не включён.
- Если Qdrant или OpenAI временно недоступны, исходный документ остаётся в SQLite и локальном хранилище, а состояние индекса фиксируется в карточке и журнале.
- История чата сохраняется в SQLite; в Langfuse передаётся текст запроса для наблюдаемости.
- Автор события берётся из `X-Actor-Id`. Перед внешним развёртыванием добавьте аутентификацию и контроль доступа на уровне API и MCP.
- Схема SQLite создаётся при запуске через SQLAlchemy `create_all`; миграции Alembic и OCR не входят в первую версию.
