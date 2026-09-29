# Backend API image (ADR-0002: every component runs as a container on the VMs).
#   docker build -t xcrs-api backend
#   docker compose --profile app up -d          (see docker-compose.yml)
# The same image runs migrations and admin jobs:
#   docker compose run --rm api alembic upgrade head
#   docker compose run --rm api xcrs embed-catalog qwen3-embedding:0.6b

FROM python:3.13-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# Dependencies first, so code changes don't invalidate this layer.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --no-install-project

COPY xcrs ./xcrs
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --no-editable


FROM python:3.13-slim
RUN useradd --system --uid 10001 --no-create-home xcrs
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY alembic.ini ./
COPY migrations ./migrations
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
USER xcrs
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=2)"]
CMD ["uvicorn", "xcrs.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
