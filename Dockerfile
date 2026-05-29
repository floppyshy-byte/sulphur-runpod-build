# =============================================================================
# Sulphur-2 — RunPod Serverless Worker
# =============================================================================
# Uses Floppyshy/sulphur-2-runpod custom HF repo via custom loader.
# Repo contains Sulphur-2 distilled checkpoint + Gemma text encoder.
# Model is cached via RunPod's HF model cache, not baked into the image.
# Add this HF repo ID when creating the endpoint:
#   Floppyshy/sulphur-2-runpod
# =============================================================================

FROM nvidia/cuda:12.6.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/root/.local/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget ffmpeg \
    python3.11 python3.11-venv \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app

RUN uv venv --python python3.11
ENV PATH="/app/.venv/bin:${PATH}"

# Install PyTorch with CUDA 12.6
RUN uv pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126

# Install diffusers (0.38.0 for LTX2 support) and media deps
RUN uv pip install diffusers==0.38.0 \
    transformers accelerate \
    imageio[ffmpeg] pillow safetensors bitsandbytes

# Install runpod
RUN uv pip install runpod

# ---------------------------------------------------------------------------
# Stage 2: Runtime
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg python3.11 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local/bin/uv /root/.local/bin/uv
COPY --from=builder /app /app

WORKDIR /app

ENV PATH="/app/.venv/bin:/root/.local/bin:${PATH}"

# Offline mode — model must be pre-cached via RunPod Model Cache
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HOME=/runpod-volume/huggingface-cache/hub

ENV PYTHONFAULTHANDLER=1
ENV CUDA_VISIBLE_DEVICES=0

COPY handler.py /app/handler.py
COPY loader.py /app/loader.py

CMD ["python3", "-u", "/app/handler.py"]
