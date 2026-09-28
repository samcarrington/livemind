FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --no-cache --frozen

COPY app.py settings.py db.py stt_worker.py reconciler.py ./
COPY routes/ ./routes/
COPY services/ ./services/
COPY static/ ./static/

# SQLite DB lives here — mount an Azure Files share at this path in production
RUN mkdir -p /data

EXPOSE 8765

CMD ["uv", "run", "python", "app.py", "--host", "0.0.0.0", "--port", "8765"]
