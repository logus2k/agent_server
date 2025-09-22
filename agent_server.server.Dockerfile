FROM python:3.10.18-slim AS builder

USER root

RUN pip install --upgrade pip

# Install basic dependencies first
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install build dependencies including gcc-12/g++-12  
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake gcc-12 g++-12 git wget \
    && rm -rf /var/lib/apt/lists/*

# Install CUDA 12.9.0 via apt repository (using proven working method)
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin && \
    mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/3bf863cc.pub | gpg --dearmor -o /usr/share/keyrings/cuda-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ /" > /etc/apt/sources.list.d/cuda.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends cuda-toolkit-12-9 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies (excluding llama-cpp-python which we build separately)
RUN pip install \
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

# Build llama-cpp-python with CUDA support using exact working script
ENV CC=/usr/bin/gcc-12
ENV CXX=/usr/bin/g++-12
ENV CUDAHOSTCXX=/usr/bin/g++-12
ENV CMAKE_ARGS="-DGGML_CUDA=on"
ENV FORCE_CMAKE=1

RUN pip uninstall -y llama-cpp-python || true && \
    pip cache purge && \
    pip install --no-cache-dir --force-reinstall llama-cpp-python

WORKDIR /agent_server

# Final runtime stage
FROM python:3.10.18-slim

USER root

RUN pip install --upgrade pip

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy CUDA libraries from builder
COPY --from=builder /usr/local/cuda/lib64 /usr/local/cuda/lib64
COPY --from=builder /usr/local/cuda/include /usr/local/cuda/include

# Copy Python environment from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# Set environment variables for CUDA runtime
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64
ENV CUDA_HOME=/usr/local/cuda

WORKDIR /agent_server
