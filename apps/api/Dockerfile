FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

COPY apps/api/pyproject.toml apps/api/README.md ./
COPY apps/api/src ./src
COPY apps/api/eval ./eval
RUN pip install --no-cache-dir .

RUN mkdir -p /app/data/resumes

EXPOSE 8000
CMD ["uvicorn", "hr_assistant.main:app", "--host", "0.0.0.0", "--port", "8000"]
