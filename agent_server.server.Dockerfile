# syntax=docker/dockerfile:1

#############################
# Builder: CUDA + toolchain #
#############################
FROM nvidia/cuda:12.9.0-devel-ubuntu22.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive

# System deps (Ubuntu 22.04 ships Python 3.10)
RUN apt-get update && apt-get install -y --no-install-recommends \
	python3.10 python3.10-venv python3-pip \
	build-essential cmake git curl ca-certificates \
	gcc-12 g++-12 \
	&& rm -rf /var/lib/apt/lists/*

# Create and activate virtualenv
RUN python3.10 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
	PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1

# Pip up-to-date
RUN pip install --upgrade pip

# --- Python dependencies (pinned; excluding llama-cpp-python) ---
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
	idna==3.10 \
	Jinja2==3.1.6 \
	MarkupSafe==3.0.2 \
	multidict==6.6.4 \
	numpy==2.2.6 \
	propcache==0.3.2 \
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

# --- Build llama-cpp-python with CUDA ---
# Use GCC-12 toolchain
ENV CC=/usr/bin/gcc-12 \
	CXX=/usr/bin/g++-12 \
	CUDAHOSTCXX=/usr/bin/g++-12

# Tell CMake to build CUDA, and skip tools/tests/examples (prevents mtmd/CLI)
ENV CMAKE_ARGS="-DGGML_CUDA=on -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF" \
	FORCE_CMAKE=1

# Make CUDA driver STUB available at link time (real driver is provided at runtime)
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LIBRARY_PATH} \
	LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH} \
	LDFLAGS="-L/usr/local/cuda/lib64/stubs -Wl,-rpath-link,/usr/local/cuda/lib64/stubs"
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so

# Build backend + helpers for no-build-isolation
RUN pip install --no-cache-dir \
	scikit-build-core>=0.10 \
	ninja \
	setuptools \
	wheel \
	packaging

# Make CUDA driver stubs visible *and* satisfy the exact SONAME libcuda.so.1
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LIBRARY_PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH} \
    LDFLAGS="-L/usr/local/cuda/lib64/stubs -Wl,-rpath-link,/usr/local/cuda/lib64/stubs"

RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so \
 && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so.1 \
 && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
 
# Fresh build of llama-cpp-python (wheel)
# --no-build-isolation uses the toolchain & backend we just installed
RUN pip uninstall -y llama-cpp-python || true && \
	pip cache purge && \
	pip install --no-cache-dir --no-build-isolation --force-reinstall llama-cpp-python

WORKDIR /agent_server
# COPY your app here if you want to bake it in:
# COPY . /agent_server


###########################
# Runtime: CUDA + Python  #
###########################
FROM nvidia/cuda:12.9.0-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# Minimal runtime deps (Python 3.10 and OpenMP runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
	python3.10 python3-pip libgomp1 \
	&& rm -rf /var/lib/apt/lists/*

# Bring in the venv from builder
COPY --from=builder /opt/venv /opt/venv

# Runtime env — IMPORTANT: do NOT include stubs path here
ENV PATH=/opt/venv/bin:$PATH \
	PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

WORKDIR /agent_server
# If you didn’t COPY your app in builder, do it here:
# COPY . /agent_server

# Health check / smoke test (optional)
# CMD ["python", "-c", "import llama_cpp; print('llama-cpp-python with CUDA is ready')"]

# For your server, replace with your actual entrypoint, e.g.:
# CMD ["uvicorn", "agent_server.main:app", "--host", "0.0.0.0", "--port", "8000"]
