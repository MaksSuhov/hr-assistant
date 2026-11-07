# HR Assistant Backend

Эта часть монорепозитория расположена в `apps/api/`. Из корня проекта можно запустить весь стек через Docker Compose; локальные Python-команды ниже выполняются из `apps/api/`.

Backend первой версии HR-помощника: вакансии, резюме PDF, карточки кандидатов, гибридный поиск по резюме, сопоставление с требованиями вакансии, чат с SSE и действия с ручным подтверждением HR.

## Стек и архитектура

- **FastAPI + Pydantic + asyncio + Uvicorn** — REST API и SSE чат.
- **SQLite + SQLAlchemy async** — вакансии, требования, кандидаты, документы, история, диалоги, подтверждения и очередь сообщений.
- **Qdrant** — текстовые фрагменты резюме; dense-вектора OpenAI + BM25 FastEmbed объединяются Reciprocal Rank Fusion (RRF).
- **HyDE + reranking** — перед поиском `gpt-4.1-mini` формирует гипотетический фрагмент резюме, а после hybrid retrieval оценивает до 30 фрагментов по исходному запросу и возвращает самые релевантные.
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

Модели-«рассуждалки» (например `qwen/qwen3.7-flash`) по умолчанию генерируют скрытый блок рассуждений, поэтому первый видимый токен может прийти через 10–30 с. Чат и анализ требований по умолчанию отправляют `reasoning: {"enabled": false}` в `extra_body` запроса и отвечают в разы быстрее; вернуть режим рассуждений можно через `CHAT_REASONING_ENABLED=true`. Шлюз может игнорировать управление степенью (`reasoning.effort`), поэтому используется только бинарный переключатель.

Поиск резюме использует HyDE и реранкинг через `SEARCH_ENHANCEMENT_MODEL` (по умолчанию `gpt-4.1-mini`). HyDE добавляет к запросу гипотетическое описание релевантного опыта; оно не считается фактом о кандидате. Реранкер сравнивает найденные фрагменты с исходным запросом. Этапы можно независимо отключить через `HYDE_ENABLED=false` и `RERANKING_ENABLED=false`; глубина пула задаётся `RERANKING_CANDIDATE_POOL` (по умолчанию 30). Если LLM недоступна или этап завершается ошибкой, поиск использует доступный hybrid/текстовый fallback.

Qdrant поднимается сервисом `qdrant` из `docker-compose.yml`; `QDRANT_URL` в `.env` указывает на него, а коллекция `QDRANT_COLLECTION` создаётся при старте приложения.

## Основные маршруты

Все REST-маршруты находятся под `/api/v1`. Кадровые маршруты и MCP требуют входа. Первый созданный аккаунт получает роль администратора; публичная регистрация затем закрыта. Изменяющие запросы требуют CSRF-заголовок `X-CSRF-Token`, который веб-интерфейс добавляет автоматически. Портал `/candidate` остаётся публичным и использует отдельный маршрут загрузки.

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
| `POST` | `/auth/register` | Создать первый аккаунт администратора |
| `POST` | `/auth/users` | Создать аккаунт рекрутера (только администратор) |
| `POST` | `/auth/login` | Войти по почте и паролю |
| `POST` | `/auth/refresh` | Обновить access cookie и ротировать refresh cookie |
| `POST` | `/auth/logout` | Завершить сессию и отозвать refresh токен |
| `GET` | `/auth/me` | Текущий пользователь и роль |
| `GET` | `/auth/oauth/google` | Начать вход Google OAuth 2.0 (если настроен) |
| `POST` | `/resumes/public` | Публичная загрузка резюме из портала кандидата |

Access и refresh токены хранятся в `HttpOnly` cookies. Настройте `AUTH_SECRET_KEY`, а в production включите `AUTH_COOKIE_SECURE=true`. Для Google укажите OAuth client ID/secret и callback URI из `.env.example`. Первый аккаунт получает роль администратора; администратор может создать учётные записи рекрутеров через `POST /auth/users`.

Смена статуса происходит только при вызове `approve`. Черновик сообщения ничего не отправляет. После подтверждения сообщение отправится через SMTP, если заданы `SMTP_HOST` и `SMTP_FROM`; иначе оно останется в SQLite outbox со статусом `queued`. Статус и ошибки доставки доступны через `/approvals/outbox`.

## Примеры

Сначала создайте первого администратора или войдите и сохраните cookies. Для изменяющих запросов передайте CSRF cookie в заголовке:

```bash
curl -c cookies.txt -b cookies.txt -X POST http://localhost:8000/api/v1/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"email":"hr@example.org","password":"your-password"}'
CSRF=$(awk '$5 == "csrf_token" {print $6}' cookies.txt)
```

Создать вакансию:

```bash
curl -X POST http://localhost:8000/api/v1/vacancies \
  -b cookies.txt -c cookies.txt -H "X-CSRF-Token: $CSRF" \
  -H 'Content-Type: application/json' \
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
  -b cookies.txt -c cookies.txt -H "X-CSRF-Token: $CSRF" \
  -F 'candidate_name=Алексей Иванов' \
  -F 'source=Почта отдела кадров' \
  -F 'file=@./resume.pdf'
```

Попросить сменить статус и затем подтвердить действие:

```bash
curl -X POST http://localhost:8000/api/v1/approvals/candidates/CANDIDATE_ID/status-change/request \
  -b cookies.txt -c cookies.txt -H "X-CSRF-Token: $CSRF" -H 'Content-Type: application/json' \
  -d '{"status":"interview","reason":"HR просмотрел резюме"}'

curl -X POST http://localhost:8000/api/v1/approvals/APPROVAL_ID/approve \
  -b cookies.txt -c cookies.txt -H "X-CSRF-Token: $CSRF" -H 'Content-Type: application/json' \
  -d '{"note":"Подтверждаю приглашение на интервью"}'
```

SSE чат:

```bash
curl -N -X POST http://localhost:8000/api/v1/chat/stream \
  -b cookies.txt -c cookies.txt -H "X-CSRF-Token: $CSRF" -H 'Content-Type: application/json' \
  -d '{"message":"Найди кандидатов с опытом FastAPI и сравни их по вакансии VACANCY_ID"}'
```

События SSE: `conversation` (ID диалога), `token` (часть ответа), `done` (ID сообщения) и `error`. Для продолжения передайте полученный `conversation_id` в следующем запросе. Группа `performance` по заголовку `X-Eval-Metrics: true` дополнительно получает диагностическое событие `eval_metrics`.

## Оценка качества (`eval`)

Инструменты оценивания находятся в `eval/`. Наборы запускаются раздельно, чтобы, например, проверить только ранжирование без вызовов LLM-судьи или только скорость API:

```bash
# Один раз установить проект в активное Python-окружение
python -m pip install -e .

# LLM-as-a-judge: качество ответа и контекста
python -m eval judge eval/data/example.jsonl --output reports/judge.json

# Retrieval: context precision/recall, hit-rate, MRR, MAP и F1
python -m eval retrieval eval/data/example.jsonl --output reports/retrieval.json

# Live retrieval: запрашивает текущий индекс через /candidates/search
python -m eval retrieval eval/data/example.jsonl --base-url http://localhost:8000 \
  --output reports/retrieval-live.json

# Классификация кандидатов: micro/macro/weighted F1
python -m eval classification eval/data/example.jsonl --output reports/classification.json

# Нагрузочные замеры SSE API: TTFT, TTFRT, latency
python -m eval performance eval/data/example.jsonl --base-url http://localhost:8000 \
  --output reports/performance.json
```

Для `judge` нужен `OPENAI_API_KEY`; endpoint и модель берутся из настроек приложения. Для отдельной модели судьи укажите `EVAL_JUDGE_MODEL` или передайте `--model`. `retrieval` и `classification` считают метрики локально. `performance` вызывает работающий `/api/v1/chat/stream` и запускается с конкурентностью `1` по умолчанию; `--concurrency` позволяет явно задать нагрузку.

Датасет может быть JSON-массивом или JSONL: одна запись на строку. В [`eval/data/golden_set.jsonl`](eval/data/golden_set.jsonl) подготовлен стартовый набор из 20 синтетических вопросов с эталонными ответами и evidence-фрагментами. ID там демонстрационные: для оценки реального индекса их нужно заменить на размеченные ID кандидатов из ваших резюме. Поля, используемые группами, независимы:

- `judge`: `question` (или `query`), `answer`, `contexts` (строки либо объекты `{ "id", "text" }`) и необязательный `reference_answer`. Без reference answer метрики judge `context_precision` и `context_recall` пропускаются.
- `retrieval`: офлайн-режим использует `relevant_context_ids` и упорядоченный `retrieved_context_ids` (или `contexts` с ID). Live-режим `--base-url` вызывает `/api/v1/candidates/search` и требует `query`/`question` и `relevant_candidate_ids`. В обоих режимах precision/recall основаны на эталонных ID, MRR — reciprocal rank первого релевантного результата, MAP — mean average precision.
- `classification`: `expected_labels` и `predicted_labels` (также поддерживаются `expected_candidate_ids` / `predicted_candidate_ids`). F1 считается по наборам меток в разрезе всех записей.
- `performance`: `question` (или `query`/`message`); для замеров метрики передаются отдельным SSE-событием только при заголовке `X-Eval-Metrics: true`. TTFT — от начала запроса до первого чанка модели, TTFRT — до первого текстового токена ответа, latency — до сохранения ответа и завершения обработки. Отдельно выводится клиентская latency.

В стандартных метриках поиска используется **MRR** (mean reciprocal rank). **MMR** (maximal marginal relevance) — способ переранжирования с учётом разнообразия результатов, а не самостоятельная агрегатная метрика; в набор оценки включены MRR и MAP.

## Первое приближение и границы

- При сканированных PDF без текстового слоя извлечение может вернуть пустой текст; OCR не включён.
- Если Qdrant или OpenAI временно недоступны, исходный документ остаётся в SQLite и локальном хранилище, а состояние индекса фиксируется в карточке и журнале.
- История чата сохраняется в SQLite; в Langfuse передаётся текст запроса для наблюдаемости.
- Для пользователя применяются роли `admin` и `recruiter`; кадровые API и MCP закрыты без активной сессии.
- Схема SQLite создаётся при запуске через SQLAlchemy `create_all`; миграции Alembic и OCR не входят в первую версию.
