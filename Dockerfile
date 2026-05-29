# =============================================================================
# Sulphur-2 — RunPod Serverless Worker (minimal)
# =============================================================================
# Single-stage build on python:3.11-slim.
# PyTorch CUDA wheel bundles its own CUDA runtime libs;
# NVIDIA container runtime mounts GPU drivers from host.
# =============================================================================

FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch with CUDA 12.6
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126

# Install diffusers (0.38.0 for LTX2 support) and media deps
RUN pip install --no-cache-dir diffusers==0.38.0 \
    transformers accelerate \
    imageio[ffmpeg] pillow safetensors bitsandbytes runpod

# Offline mode — model must be pre-cached via RunPod Model Cache
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HOME=/runpod-volume/huggingface-cache/hub

ENV PYTHONFAULTHANDLER=1
ENV CUDA_VISIBLE_DEVICES=0

COPY handler.py /app/handler.py
COPY loader.py /app/loader.py

CMD ["python3", "-u", "/app/handler.py"]
