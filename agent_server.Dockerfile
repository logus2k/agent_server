FROM agent_server-server:1.0

USER root
WORKDIR /agent_server

COPY ./app/ ./app/
COPY ./agent_config.json ./agent_config.json

# Make sure Python can import "app"
ENV PYTHONPATH=/agent_server
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

EXPOSE 7701

CMD ["uvicorn", "app.main:asgi_app", "--host", "0.0.0.0", "--port", "7701"]
