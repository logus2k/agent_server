# syntax=docker/dockerfile:1
#
# agent_server v2 — SLIM forwarding-only image.
# No CUDA toolchain, no llama-cpp-python source build. The model lives
# in a sibling `llama-server` container; this image just forwards LLM
# calls over HTTP via httpx (see app/llm_engine_server.py).
#
# Build (~30 s):  docker build -t agent_server_v2:1.0 -f Dockerfile.v2 .
# Image size:     ~500 MB (vs ~8.5 GB for the previous Dockerfile.v2)
#
# In-process LlamaCppEngine fallback is still importable but raises a
# clear runtime error if instantiated (llama-cpp-python is not in this
# image). To roll back to in-process mode, rebuild from Dockerfile.v2.fat.

FROM python:3.12-slim

ARG DEBIAN_FRONTEND=noninteractive

# Minimal runtime: just curl for healthchecks + ca-certificates for HTTPS.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN pip install --upgrade pip

# Python deps (forwarding mode only — no llama-cpp-python).
# Pinned to match the previous image's versions for behavioural parity.
RUN pip install --no-cache-dir \
    aiohappyeyeballs==2.6.1 \
    aiohttp==3.12.15 \
    aiosignal==1.4.0 \
    annotated-types==0.7.0 \
    anyio==4.10.0 \
    async-timeout==5.0.1 \
    attrs==25.3.0 \
    bidict==0.23.1 \
    click==8.3.0 \
    diskcache==5.6.3 \
    exceptiongroup==1.3.0 \
    fastapi==0.116.2 \
    frozenlist==1.7.0 \
    h11==0.16.0 \
    httpx==0.28.0 \
    idna==3.10 \
    Jinja2==3.1.6 \
    MarkupSafe==3.0.2 \
    multidict==6.6.4 \
    pydantic==2.11.9 \
    pydantic_core==2.33.2 \
    python-engineio==4.12.2 \
    python-socketio==5.13.0 \
    simple-websocket==1.1.0 \
    sniffio==1.3.1 \
    starlette==0.48.0 \
    typing_extensions==4.15.0 \
    typing-inspection==0.4.1 \
    uvicorn==0.35.0 \
    wsproto==1.2.0 \
    yarl==1.20.1

# Application layer (rebuilds fast on code changes — Docker cache reuses
# the dep layer above).
WORKDIR /agent_server

COPY ./app/ ./app/
COPY ./agent_config.json ./agent_config.json

ENV PYTHONPATH=/agent_server

EXPOSE 7701

CMD ["uvicorn", "app.main:asgi_app", "--host", "0.0.0.0", "--port", "7701"]
