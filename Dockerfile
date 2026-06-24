FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps (psycopg2 build, tzdata).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Railway provides $PORT at runtime.
ENV PORT=8000
EXPOSE 8000

# Start the server directly — tables are auto-created on startup.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
