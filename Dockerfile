# =============================================================================
# Sulphur-2 — RunPod Serverless Worker
# =============================================================================
# Uses RunPod's PyTorch base image with CUDA 12.8.1 + torch 2.8.0 pre-installed.
# Model weights are cached via RunPod Model Cache, not baked into the image.
# Add this HF repo ID when creating the endpoint:
#   Floppyshy/sulphur-2-runpod
# =============================================================================

FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install diffusers (0.38.0 for LTX2 support) and remaining deps
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
