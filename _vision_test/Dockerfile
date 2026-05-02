# syntax=docker/dockerfile:1
#
# Gemma 4 vision smoke-test container.
#
# Builds llama-cpp-python from source with its vendored llama.cpp
# submodule pinned to a specific tag (default b9006). Mirrors
# agent_server.server.Dockerfile's CUDA + Python toolchain so any
# behavioural difference is isolated to the llama.cpp version and
# the Python wrapper's handling of Gemma 4 vision.
#
# Build (10-15 min):
#   docker build --build-arg LLAMACPP_TAG=b9006 \
#     -t vision-test:b9006 \
#     -f _vision_test/Dockerfile _vision_test/
#
# Run (needs GPU, mounts the model files + cat.png):
#   docker run --rm --gpus all \
#     -v /home/logus/env/assets/agent_server/data/models:/models:ro \
#     -v /home/logus/env/assets/noted/frontend/images/cat.png:/test/cat.png:ro \
#     -v /home/logus/env/assets/agent_server/_vision_test/test_gemma4_vision.py:/test/test_gemma4_vision.py:ro \
#     vision-test:b9006

ARG CUDA_TAG=13.2.1-cudnn-devel-ubuntu24.04
FROM nvidia/cuda:${CUDA_TAG}

ARG DEBIAN_FRONTEND=noninteractive
ARG LLAMACPP_TAG=b9006

# System deps. Ubuntu 24.04 ships Python 3.12 (vs production base
# agent_server.server.Dockerfile on 22.04 / Python 3.10). Once
# vision is proven to work and the production base is rebuilt to
# match, both will sit on the same toolchain.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3-pip \
    build-essential cmake git curl ca-certificates \
    gcc-12 g++-12 \
 && rm -rf /var/lib/apt/lists/*

# Venv
RUN python3.12 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN pip install --upgrade pip && \
    pip install --no-cache-dir wheel setuptools 'scikit-build-core>=0.10' ninja packaging

# CUDA toolchain env (matches production base)
ENV CC=/usr/bin/gcc-12 \
    CXX=/usr/bin/g++-12 \
    CUDAHOSTCXX=/usr/bin/g++-12

# Build for RTX 4090 only (sm_89) — much faster than the default
# multi-arch build, and we know our target hardware.
# NB: deliberately NOT passing -DLLAMA_BUILD_TOOLS=OFF here, unlike
# the production base, because that flag also strips mtmd which is
# what carries the multimodal/vision support we are testing.
ENV CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=89" \
    FORCE_CMAKE=1

# CUDA driver stubs (real driver attaches at runtime)
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LIBRARY_PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH} \
    LDFLAGS="-L/usr/local/cuda/lib64/stubs -Wl,-rpath-link,/usr/local/cuda/lib64/stubs"
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so \
 && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so.1 \
 && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1

# Clone llama-cpp-python and pin its vendored llama.cpp to LLAMACPP_TAG
WORKDIR /opt
RUN git clone https://github.com/abetlen/llama-cpp-python.git \
 && cd llama-cpp-python \
 && git submodule update --init --recursive

WORKDIR /opt/llama-cpp-python/vendor/llama.cpp
# When LLAMACPP_TAG is non-empty, pin the bundled llama.cpp submodule
# to that tag. When empty, leave it at whatever commit
# llama-cpp-python's master currently points at (test-default behaviour).
RUN if [ -n "${LLAMACPP_TAG}" ]; then \
      git fetch --depth=1 origin refs/tags/${LLAMACPP_TAG}:refs/tags/${LLAMACPP_TAG} \
      && git checkout ${LLAMACPP_TAG} \
      && git submodule update --init --recursive ; \
    else \
      echo "LLAMACPP_TAG empty; using llama-cpp-python master's pinned llama.cpp commit." ; \
    fi

# Build llama-cpp-python from local source against the pinned llama.cpp
WORKDIR /opt/llama-cpp-python
RUN pip install --no-cache-dir --no-build-isolation -v .

WORKDIR /test
CMD ["python3", "test_gemma4_vision.py"]
