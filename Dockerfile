# Authored without local Docker (not installed on the dev machine) — verified by
# inspection; noted honestly in the README. get_model() retrains transparently if
# the committed artifact doesn't load under this image's library versions.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
RUN pip install -e ".[app]"

COPY data ./data
COPY artifacts ./artifacts
COPY app ./app
COPY .env.example ./

EXPOSE 8501

# LLM_API_KEY etc. come from --env-file .env (never baked into the image)
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
