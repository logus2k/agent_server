FROM agent_server-server:1.0

USER root

COPY *.py /tts_server

EXPOSE 7700

WORKDIR /tts_server

CMD ["python", "-u", "tts_server.py"]
