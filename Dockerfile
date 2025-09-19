# =========================
# Dockerfile (CPU baseline)
# =========================
# Build a small CPU-only image. For GPU/cuBLAS builds, see notes below.
FROM 3.12.10-slim-bookworm


# System deps for llama-cpp build (CPU)
RUN apt-get update && apt-get install -y --no-install-recommends \
build-essential cmake && \
rm -rf /var/lib/apt/lists/*


WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt


COPY app ./app


ENV PORT=7701 \
MODEL_PATH=/models/llama-2-7b.Q4_K_M.gguf \
USE_FAKE=false \
N_CTX=4096 \
N_GPU_LAYERS=0


EXPOSE 7701
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7701"]


# --- GPU/cuBLAS Note ---
# For NVIDIA GPUs, rebuild llama-cpp-python with cuBLAS:
# docker build --build-arg CMAKE_ARGS="-DLLAMA_CUBLAS=on" ...
# or inside the image:
# pip uninstall -y llama-cpp-python && \
# CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install --no-cache-dir llama-cpp-python
# Also set N_GPU_LAYERS>0 and use an appropriate CUDA base image.
