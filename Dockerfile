# Base image - using NVIDIA vLLM image for GB10/Blackwell compatibility
FROM nvcr.io/nvidia/vllm:25.11-py3

# Prevents prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install additional system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    sudo \
    ssh \
    tmux \
    vim \
    htop \
    unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv for dependency management
RUN pip install uv

# Set up the working directory
WORKDIR /app

# Copy the entire project
COPY . .

# Set fake versions for setuptools_scm packages (they need git history for versioning)
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_EVALPLUS=0.1.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LIVECODEBENCH=0.1.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_CRFM_HELM=0.5.7
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_EVALSCOPE=2.0.0

# Install dependencies during build - self-contained and reproducible
RUN pip install --no-deps --editable third-party/evalplus && \
    pip install --no-deps --editable third-party/LiveCodeBench && \
    pip install --no-deps --editable third-party/helm && \
    pip install --no-deps --editable third-party/evalscope && \
    pip install --no-deps --editable . && \
    pip install \
    accelerate \
    lm-eval \
    seaborn \
    matplotlib \
    python-dotenv \
    jupyter \
    trl \
    hatchling \
    wandb \
    umap-learn \
    fire \
    termcolor \
    multipledispatch \
    appdirs \
    tempdir \
    wget \
    cohere \
    google-genai \
    google-generativeai \
    mistralai \
    pebble \
    together \
    anthropic \
    boto3 \
    psutil \
    tree-sitter \
    tree-sitter-python

# Create a cache directory for uv
RUN mkdir -p /app/.uv_cache

# Set UV_CACHE_DIR environment variable
ENV UV_CACHE_DIR="/app/.uv_cache"

# Default command to run if no other command is specified
CMD ["/bin/bash"]