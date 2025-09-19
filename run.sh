# =========================
# run.sh
# =========================

export MODEL_PATH=./models/menlo/jan-nano-128k-Q4_K_M.gguf
uvicorn app.main:asgi_app --host 0.0.0.0 --port 7701
