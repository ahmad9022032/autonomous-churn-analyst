# ChurnSight web — single container: React build + FastAPI + the agent.
# Build context is the REPO ROOT:  docker build -f webapp/Dockerfile .
FROM node:22-slim AS ui
WORKDIR /ui
COPY webapp/frontend/package.json webapp/frontend/package-lock.json ./
RUN npm ci
COPY webapp/frontend/ ./
RUN npm run build

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
RUN pip install -e . fastapi "uvicorn[standard]"

COPY data ./data
COPY artifacts ./artifacts
COPY webapp/__init__.py ./webapp/__init__.py
COPY webapp/backend ./webapp/backend
COPY --from=ui /ui/dist ./webapp/frontend/dist

EXPOSE 8000
# Render injects PORT; default to 8000 locally
CMD ["sh", "-c", "uvicorn webapp.backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
