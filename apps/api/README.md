# HR Assistant Backend

Эта часть монорепозитория расположена в `apps/api/`. Из корня проекта можно запустить весь стек через Docker Compose; локальные Python-команды ниже выполняются из `apps/api/`.

Backend HR-помощника с ATS-моделью: вакансии, резюме PDF/DOCX с OCR-фолбэком, карточки кандидатов, отклики «кандидат ↔ вакансия» со стадиями, гибридный поиск по резюме, сопоставление с требованиями вакансии, чат с SSE и действия с ручным подтверждением HR.

## Стек и архитектура

- **FastAPI + Pydantic + asyncio + Uvicorn** — REST API и SSE чат.
- **SQLite + SQLAlchemy async** — вакансии, требования, кандидаты, отклики (`applications`), документы, история, диалоги, подтверждения и очередь сообщений.
- **Qdrant** — текстовые фрагменты резюме; dense-вектора OpenAI + BM25 FastEmbed объединяются Reciprocal Rank Fusion (RRF).
- **PDF/DOCX + OCR** — `pypdf` и `python-docx` извлекают текстовый слой; для сканированных PDF страницы рендерятся через `pypdfium2` и распознаются `tesseract` (`rus+eng`).
- **HyDE + локальный реранкинг** — перед поиском LLM (`SEARCH_ENHANCEMENT_MODEL`) формирует гипотетический фрагмент резюме, а после hybrid retrieval локальный ONNX cross-encoder `BAAI/bge-reranker-v2-m3-int8` (через FastEmbed) оценивает до 30 фрагментов по исходному запросу и возвращает самые релевантные. Реранкинг не тратит токены LLM.
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

Без `OPENAI_API_KEY` API и загрузка PDF/DOCX работают, но индексация резюме и чат будут недоступны; поиск использует текстовый fallback по сохранённым документам. После добавления ключа перезапустите сервис. Повторная индексация выполняется через `POST /resumes/{id}/reindex`, повторно загружать документ не нужно.

## Провайдер моделей

Модели подключаются через OpenAI-совместимый endpoint. Если `OPENAI_BASE_URL` пуст, запросы идут в `api.openai.com`. Для RouterAI в `.env` задаются:

```dotenv
OPENAI_API_KEY=<ключ RouterAI>
OPENAI_BASE_URL=https://routerai.ru/api/v1
OPENAI_CHAT_MODEL=qwen/qwen3.7-flash
SEARCH_ENHANCEMENT_MODEL=qwen/qwen3.7-flash
OPENAI_EMBEDDING_MODEL=openai/text-embedding-3-small
```

Идентификаторы моделей у шлюза идут с префиксом провайдера (`openai/…`, `anthropic/…`, `google/…`), актуальный список — на [routerai.ru/models](https://routerai.ru/models). Модель чата должна поддерживать `tools` и `response_format`, иначе чат и анализ требований деградируют. `OPENAI_EMBEDDING_DIMENSIONS` должен совпадать с размерностью выбранной модели и с коллекцией Qdrant: при смене модели на другую размерность коллекцию нужно пересоздать.

Модели-«рассуждалки» (например базовая по умолчанию `qwen/qwen3.7-flash`) по умолчанию генерируют скрытый блок рассуждений, поэтому первый видимый токен может прийти через 10–30 с. Чат и анализ требований по умолчанию отправляют `reasoning: {"enabled": false}` в `extra_body` запроса и отвечают в разы быстрее; вернуть режим рассуждений можно через `CHAT_REASONING_ENABLED=true`. Шлюз может игнорировать управление степенью (`reasoning.effort`), поэтому используется только бинарный переключатель.

Поиск резюме использует HyDE и локальный реранкинг. HyDE (`SEARCH_ENHANCEMENT_MODEL`, по умолчанию `qwen/qwen3.7-flash`) добавляет к запросу гипотетическое описание релевантного опыта; оно не считается фактом о кандидате. После hybrid retrieval ONNX cross-encoder `BAAI/bge-reranker-v2-m3-int8` пересчитывает релевантность найденных фрагментов по исходному (не HyDE) запросу: модель работает локально на CPU через `fastembed`/`onnxruntime`, поэтому поиск не расходует токены LLM и не ждёт сеть. Модель загружается при первом поиске: первый запуск скачивает ~0.57 ГБ в `RERANKING_CACHE_DIR` или в кеш FastEmbed (в Docker-образе — `/app/data/fastembed`, внутри уже смонтированного тома `hr_data`, поэтому загрузка переживает пересборку и перезапуск контейнера). Присвоить `RERANKING_MODEL` другую модель из списка `TextCrossEncoder.list_supported_models()`; для русскоязычных резюме мультиязычная `bge-reranker-v2-m3-int8` точнее англоцентричной `BAAI/bge-reranker-base` (~1.04 ГБ). Глубина пула задаётся `RERANKING_CANDIDATE_POOL` (по умолчанию 30), `RERANKING_DEVICE` (`auto`/`cpu`/`cuda`) и `RERANKING_BATCH_SIZE` управляют инференсом. Этапы отключаются независимо: `HYDE_ENABLED=false` и `RERANKING_ENABLED=false`. Если модель не загрузилась, поиск сохраняет порядок hybrid-поиска, а HyDE при ошибке LLM откатывается к исходному запросу. Матчинг кандидата с вакансией делает по одному поиску на требование параллельно и без HyDE и реранка (`search_raw(..., deep=False)`): при десятках требований это убирает десятки LLM-вызовов и лишние проходы cross-encoder, а порядок задаёт hybrid RRF.

Qdrant поднимается сервисом `qdrant` из `docker-compose.yml`; `QDRANT_URL` в `.env` указывает на него, а коллекция `QDRANT_COLLECTION` создаётся при старте приложения.

## Основные маршруты

Все REST-маршруты находятся под `/api/v1`. Кадровые маршруты и MCP требуют входа. Первый созданный аккаунт получает роль администратора; публичная регистрация затем закрыта. Изменяющие запросы требуют CSRF-заголовок `X-CSRF-Token`, который веб-интерфейс добавляет автоматически. Портал `/candidate` остаётся публичным и использует отдельный маршрут загрузки.

| Метод | Маршрут | Назначение |
|---|---|---|
| `POST` | `/vacancies` | Создать вакансию и требования |
| `PATCH` | `/vacancies/{id}` | Изменить поля и заменить список требований |
| `GET` | `/vacancies`, `/vacancies/{id}` | Список и карточка вакансии (с числом откликов) |
| `GET` | `/vacancies/{id}/applications` | Отклики на вакансию со стадиями и именами кандидатов |
| `POST` | `/vacancies/{id}/applications` | Привязать кандидата к вакансии (`candidate_id`, `stage`) |
| `GET` | `/vacancies/{id}/applications/{application_id}` | Один отклик |
| `DELETE` | `/vacancies/{id}/applications/{application_id}` | Убрать отклик из вакансии |
| `POST` | `/vacancies/{id}/applications/{application_id}/stage-change/request` | Запросить смену стадии (подтверждение HR) |
| `POST` | `/resumes` | Загрузить PDF/DOCX (`multipart/form-data`), извлечь текст и проиндексировать; `vacancy_id` сразу создаёт отклик |
| `DELETE` | `/resumes/{id}` | Удалить файл, запись и векторы резюме, записать аудит |
| `POST` | `/resumes/{id}/reindex` | Переиндексировать сохранённый текст с предварительным удалением старых точек |
| `GET` | `/candidates`, `/candidates/{id}` | Список и карточка кандидата с источниками и откликами |
| `DELETE` | `/candidates/{id}` | Каскадно удалить кандидата, файлы, отклики и векторы |
| `POST` | `/candidates/{id}/merge` | Слить второго кандидата в текущего (источники, отклики, outbox) |
| `GET` | `/candidates/{id}/history` | История загрузок, откликов и изменений HR |
| `POST` | `/candidates/search` | Гибридный поиск по резюме |
| `POST` | `/vacancies/{id}/matches` | Сопоставить кандидатов с требованиями вакансии |
| `POST` | `/candidates/compare` | Сравнить 2–10 кандидатов |
| `POST` | `/approvals/messages/drafts` | Создать черновик сообщения для подтверждения |
| `GET` | `/approvals/pending` | Показать ожидающие подтверждения |
| `GET` | `/approvals/outbox` | Посмотреть подтверждённые сообщения в очереди |
| `POST` | `/approvals/{id}/approve` или `/reject` | Подтвердить или отклонить действие |
| `POST` | `/maintenance/retention/purge` | Применить политику хранения (только администратор) |
| `POST` | `/chat/stream` | Чат с SSE |
| `POST` | `/auth/register` | Создать первый аккаунт администратора |
| `POST` | `/auth/users` | Создать аккаунт рекрутера (только администратор) |
| `POST` | `/auth/login` | Войти по почте и паролю |
| `POST` | `/auth/refresh` | Обновить access cookie и ротировать refresh cookie |
| `POST` | `/auth/logout` | Завершить сессию и отозвать refresh токен |
| `GET` | `/auth/me` | Текущий пользователь и роль |
| `GET` | `/auth/oauth/google` | Начать вход Google OAuth 2.0 (если настроен) |
| `POST` | `/resumes/public` | Публичная загрузка резюме из портала кандидата (PDF/DOCX) |

Access и refresh токены хранятся в `HttpOnly` cookies. Для Google укажите OAuth client ID/secret и callback URI из `.env.example`. Первый аккаунт получает роль администратора; администратор может создать учётные записи рекрутеров через `POST /auth/users`.

## ATS-модель: отклики, дубликаты и удаление

**Отклики.** Стадия хранится не у кандидата, а в таблице `applications` (`candidate_id`, `vacancy_id`, `stage`) с уникальностью пары. Один человек может быть «на интервью» по backend-вакансии и «в архиве» по другой; смена стадии по одной вакансии не затрагивает остальные. Стадии: `new`, `screening`, `interview`, `offer`, `hired`, `rejected`. Смена стадии, как и раньше, проходит через подтверждение HR: `POST /vacancies/{id}/applications/{application_id}/stage-change/request`, затем `POST /approvals/{approval_id}/approve`.

**Дедупликация.** При загрузке без явного `candidate_id` API ищет существующего кандидата по нормализованному email, затем по телефону (только цифры, `8…` приводится к `7…`), затем по SHA-256 нормализованного текста. Контакты берутся из форм `email`/`phone` или распознаются в тексте резюме. Найденная карточка дополняется недостающими ФИО/почтой/телефоном; факт склейки пишется в аудит как `candidate_deduplicated`. Повторная загрузка того же текста для того же кандидата возвращает `409` со ссылкой на `POST /resumes/{id}/reindex`. Явная склейка двух карточек: `POST /candidates/{id}/merge` с `source_candidate_id` — переносятся резюме, отклики, outbox и payload векторов; для одной и той же вакансии сохраняется более поздняя стадия.

**Удаление.** `DELETE /resumes/{id}` удаляет файл, запись и точки Qdrant (`delete` по фильтру `resume_id`). `DELETE /candidates/{id}` каскадно удаляет все резюме, отклики и сообщения outbox. Если Qdrant недоступен, удаление возвращает `503`, а файл и запись остаются — так поиск не начнёт ссылаться на отсутствующие данные; повторите запрос позже. Каждое удаление фиксируется в аудите (`resume_deleted`, `candidate_deleted`).

**Политика хранения.** `RESUME_RETENTION_DAYS` и `AUDIT_RETENTION_DAYS` (0 — выключено) задают окно хранения. При старте и через `POST /maintenance/retention/purge?days=…&audit_days=…` (только администратор) удаляются кандидаты, у которых последнее резюме старше окна. Ошибки Qdrant по отдельным кандидатам не прерывают весь прогон: они попадают в `failed_candidates`, и следующий запуск завершит удаление.

**Переиндексация.** `POST /resumes/{id}/reindex` берёт сохранённый текст, удаляет старые точки Qdrant по `resume_id`, заново режет текст и пишет векторы. Статус и ошибка (`indexing_status`, `indexing_error`) видны в карточке кандидата. Повторная загрузка документа для этого не нужна.

**PII в аудите.** События загрузки больше не хранят `text_preview` и имя файла — только тип, размер, число страниц/символов/фрагментов и результат индексации. Получатели писем в аудите маскируются (`a***@example.com`); полный адрес остаётся только в outbox, который нужен для отправки.

## Безопасность

- **Секрет подписи.** `APP_ENV=production` (значение по умолчанию) не даёт приложению стартовать, если `AUTH_SECRET_KEY` равен публичному значению из `.env.example` или короче 32 символов. Для локальной работы укажите `APP_ENV=development`, но и там лучше сгенерировать настоящий секрет: `python -c "import secrets; print(secrets.token_urlsafe(48))"`.
- **Secure cookies.** `AUTH_COOKIE_SECURE` по умолчанию не задан и включает `Secure` для всех хостов, кроме `localhost`, `127.0.0.1` и `::1`. Явное `true`/`false` переопределяет это поведение (например, `false` для LAN-машины без TLS).
- **Rate limiting.** Лимиты по IP включаются через `RATE_LIMIT_ENABLED` и отдельно настраиваются для входа, регистрации, refresh, публичной загрузки резюме и чата (`RATE_LIMIT_*`). При превышении API отвечает `429` с заголовком `Retry-After`; `/health` не ограничивается. Счётчики хранятся в памяти процесса: запускайте один воркер или перенесите их в Redis. За обратным прокси включите `RATE_LIMIT_TRUST_PROXY=true`, только если прокси перезаписывает `X-Forwarded-For`.
- **Prompt injection.** Текст резюме — недоверенные данные. При передаче модели он оборачивается в `<resume_text>…</resume_text>`, поддельные теги внутри текста нейтрализуются, а системный промпт запрещает выполнять инструкции из этого блока.

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

Загрузить резюме. `candidate_id` можно не передавать — тогда API сам найдёт дубликат по email/телефону/тексту или создаст новую карточку. `vacancy_id` сразу создаёт отклик:

```bash
curl -X POST http://localhost:8000/api/v1/resumes \
  -b cookies.txt -c cookies.txt -H "X-CSRF-Token: $CSRF" \
  -F 'candidate_name=Алексей Иванов' \
  -F 'email=alexey@example.org' \
  -F 'vacancy_id=VACANCY_ID' \
  -F 'source=Почта отдела кадров' \
  -F 'file=@./resume.pdf'
```

Для сканированного PDF достаточно загрузить его как обычно: при отсутствии текстового слоя сработает OCR. DOCX принимается наравне с PDF.

Попросить сменить стадию отклика и затем подтвердить действие:

```bash
curl -X POST "http://localhost:8000/api/v1/vacancies/VACANCY_ID/applications/APPLICATION_ID/stage-change/request" \
  -b cookies.txt -c cookies.txt -H "X-CSRF-Token: $CSRF" -H 'Content-Type: application/json' \
  -d '{"stage":"interview","reason":"HR просмотрел резюме"}'

curl -X POST http://localhost:8000/api/v1/approvals/APPROVAL_ID/approve \
  -b cookies.txt -c cookies.txt -H "X-CSRF-Token: $CSRF" -H 'Content-Type: application/json' \
  -d '{"note":"Подтверждаю приглашение на интервью"}'
```

Удалить резюме или кандидата целиком (файлы, записи, векторы и аудит):

```bash
curl -X DELETE -b cookies.txt -c cookies.txt -H "X-CSRF-Token: $CSRF" \
  http://localhost:8000/api/v1/resumes/RESUME_ID

curl -X DELETE -b cookies.txt -c cookies.txt -H "X-CSRF-Token: $CSRF" \
  http://localhost:8000/api/v1/candidates/CANDIDATE_ID
```

Переиндексировать сохранённое резюме без повторной загрузки:

```bash
curl -X POST -b cookies.txt -c cookies.txt -H "X-CSRF-Token: $CSRF" \
  http://localhost:8000/api/v1/resumes/RESUME_ID/reindex
```

SSE чат:

```bash
curl -N -X POST http://localhost:8000/api/v1/chat/stream \
  -b cookies.txt -c cookies.txt -H "X-CSRF-Token: $CSRF" -H 'Content-Type: application/json' \
  -d '{"message":"Найди кандидатов с опытом FastAPI и сравни их по вакансии VACANCY_ID"}'
```

События SSE: `conversation` (ID диалога), `token` (часть ответа), `done` (ID сообщения) и `error`. Для продолжения передайте полученный `conversation_id` в следующем запросе. Группа `performance` по заголовку `X-Eval-Metrics: true` дополнительно получает диагностическое событие `eval_metrics`.

## Тесты, линтеры и типы

```bash
python -m pip install -e ".[dev]"

ruff check .
ruff format --check .
mypy
pytest --cov=hr_assistant --cov=eval --cov-report=term-missing
```

Тесты не обращаются к OpenAI, Qdrant и SMTP: LLM, векторное хранилище, трейсер и отправитель сообщений подменяются детерминированными фейками, а SQLite создаётся во временном каталоге. Покрыты вход и CSRF, вакансии, загрузка PDF и индексация, поиск с лексическим fallback, подтверждения и outbox, SSE-чат с tool calls, HyDE/реранкинг и метрики `eval`. Prerequisite: Python 3.12 (как в Docker-образе); MyPy проверяет `src` и `eval`.

Из корня проекта те же шаги доступны через `make check`. GitHub Actions (`.github/workflows/ci.yml`) запускает `ruff`, `mypy`, `pytest` с покрытием, сборку фронтенда и Docker-образов.

## Оценка качества (`eval`)

Инструменты оценивания находятся в `eval/`. Наборы запускаются раздельно, чтобы, например, проверить только ранжирование без вызовов LLM-судьи или только скорость API:

```bash
# Один раз установить проект в активное Python-окружение
python -m pip install -e .

# Полный конвеер: свежие 20 кандидатов из БД -> вопросы -> ответы -> судья
docker compose exec -T api python -m eval pipeline --limit 20 --output /tmp/report.json

# Только сгенерировать и посмотреть свежие кейсы, без вызова ассистента и судьи
docker compose exec -T api python -m eval pipeline --limit 20 --dry-run

# LLM-as-a-judge по сохранённым прогонам (судья видит всю базу кандидатов)
python -m eval judge eval/data/example.jsonl --output reports/judge.json

# Без доступа к БД, только по переданным contexts
python -m eval judge eval/data/example.jsonl --corpus none --output reports/judge.json

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

`pipeline` (и `judge` без датасета) читает свежие резюме из той же SQLite-базы, что и API, по одному резюме на кандидата, и передаёт судье корпус целиком. Поэтому его нужно запускать там, где доступна база, — внутри контейнера `api`. Локально на хосте база из docker-тома недоступна, и `judge --corpus auto` честно предупредит и продолжит без корпуса; `--corpus full` в этом случае завершится ошибкой, а `--corpus none` отключит корпус явно.

Для `judge` нужен `OPENAI_API_KEY`; endpoint и модель берутся из настроек приложения. Для отдельной модели судьи укажите `EVAL_JUDGE_MODEL` или передайте `--model`. `retrieval` и `classification` считают метрики локально. `performance` вызывает работающий `/api/v1/chat/stream` и запускается с конкурентностью `1` по умолчанию; `--concurrency` позволяет явно задать нагрузку.

Датасет может быть JSON-массивом или JSONL: одна запись на строку. Золотой набор в проекте один — [`eval/data/golden_set.jsonl`](eval/data/golden_set.jsonl): 20 синтетических вопросов с эталонными ответами и evidence-фрагментами. Это замороженная регрессия на фикстурах: ID `demo-candidate-*` демонстрационные, а не копия текущего индекса. Для живой базы не нужно делать ID-копию набора — `pipeline` сам берёт актуальные UUID из БД. Поля, используемые группами, независимы:

- `judge`: `question` (или `query`), `answer`, `contexts` (строки либо объекты `{ "id", "text" }`) и необязательный `reference_answer`. Без reference answer метрики `context_precision` и `context_recall` равны null. При доступе к корпусу (`--corpus auto` по умолчанию) судья дополнительно считает `corpus_faithfulness` (совпадает ли ответ с резюме в БД) и `corpus_completeness` (учтены ли все подходящие кандидаты для вопросов-списков и сравнений). Это ловит противоречия с базой и пропущенных кандидатов, которые по одним лишь найденным фрагментам не видны.
- `retrieval`: офлайн-режим использует `relevant_context_ids` и упорядоченный `retrieved_context_ids` (или `contexts` с ID). Live-режим `--base-url` вызывает `/api/v1/candidates/search` и требует `query`/`question` и `relevant_candidate_ids`. В обоих режимах precision/recall основаны на эталонных ID, MRR — reciprocal rank первого релевантного результата, MAP — mean average precision.
- `classification`: `expected_labels` и `predicted_labels` (также поддерживаются `expected_candidate_ids` / `predicted_candidate_ids`). F1 считается по наборам меток в разрезе всех записей.
- `performance`: `question` (или `query`/`message`); для замеров метрики передаются отдельным SSE-событием только при заголовке `X-Eval-Metrics: true`. TTFT — от начала запроса до первого чанка модели, TTFRT — до первого текстового токена ответа, latency — до сохранения ответа и завершения обработки. Отдельно выводится клиентская latency.

В стандартных метриках поиска используется **MRR** (mean reciprocal rank). **MMR** (maximal marginal relevance) — способ переранжирования с учётом разнообразия результатов, а не самостоятельная агрегатная метрика; в набор оценки включены MRR и MAP.

## Первое приближение и границы

- Сканированные PDF проходят через OCR (`tesseract`, языки `rus+eng`). OCR зависит от системного бинарника: Docker-образ устанавливает его сам, при локальном запуске нужен `tesseract` с языковыми пакетами. При недоступности OCR документ сохраняется, но текст может остаться пустым — API сообщит об этом.
- DOCX извлекается по абзацам и таблицам; фиксированной пагинации у формата нет, поэтому число страниц для DOCX равно 1.
- Если Qdrant или OpenAI временно недоступны, исходный документ остаётся в SQLite и локальном хранилище, а состояние индекса (`indexing_status`, `indexing_error`) фиксируется в карточке и журнале. Повторная индексация — `POST /resumes/{id}/reindex`.
- Удаление резюме и кандидата требует доступного Qdrant: иначе оно возвращает `503` и не оставляет «половину» данных. Политика хранения по умолчанию выключена (`*_RETENTION_DAYS=0`).
- История чата сохраняется в SQLite; в Langfuse передаётся текст запроса для наблюдаемости.
- Для пользователя применяются роли `admin` и `recruiter`; кадровые API и MCP закрыты без активной сессии. Применение политики хранения доступно только администратору.
- Схема SQLite создаётся при запуске через SQLAlchemy `create_all`; небольшие аддитивные изменения (новые колонки, `applications`, удаление старого `candidates.status`) выполняются идемпотентной миграцией при старте. Alembic в проект не входит.
