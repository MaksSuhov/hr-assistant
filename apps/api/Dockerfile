FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

# Tesseract provides the OCR fallback for scanned PDFs. The Russian and English
# language packs cover the resumes this project is built for.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-rus \
        tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY apps/api/pyproject.toml apps/api/README.md ./
COPY apps/api/src ./src
COPY apps/api/eval ./eval
RUN pip install --no-cache-dir .

RUN mkdir -p /app/data/resumes

EXPOSE 8000
CMD ["uvicorn", "hr_assistant.main:app", "--host", "0.0.0.0", "--port", "8000"]
