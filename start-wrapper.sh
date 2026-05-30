#!/bin/bash
# ---------------------------------------------------------------------------
# Sulphur-2 GGUF — Model symlink wrapper for RunPod Serverless
# ---------------------------------------------------------------------------
# Network volume mounts at /runpod-volume on serverless workers.
# We symlink the persistent models into ComfyUI's expected directories
# so the base image's start.sh and handler.py work without modification.
# ---------------------------------------------------------------------------

set -e

NV="/runpod-volume"
COMFY="/comfyui"

echo "[sulphur-gguf] Setting up model symlinks from network volume..."

# --- UNet / diffusion model (GGUF) ---
if [ -d "$NV/models/unet" ]; then
    mkdir -p "$COMFY/models/unet"
    for f in "$NV/models/unet"/*.gguf; do
        [ -e "$f" ] || continue
        bn=$(basename "$f")
        if [ ! -e "$COMFY/models/unet/$bn" ]; then
            ln -s "$f" "$COMFY/models/unet/$bn"
            echo "[sulphur-gguf]   unet: $bn"
        fi
    done
else
    echo "[sulphur-gguf] WARNING: no models/unet on network volume"
fi

# --- Text encoders (Gemma 3 12B as HF snapshot) ---
if [ -d "$NV/models/text_encoders" ]; then
    mkdir -p "$COMFY/models/text_encoders"
    for d in "$NV/models/text_encoders"/*/; do
        [ -d "$d" ] || continue
        bn=$(basename "$d")
        if [ ! -e "$COMFY/models/text_encoders/$bn" ]; then
            ln -s "$d" "$COMFY/models/text_encoders/$bn"
            echo "[sulphur-gguf]   text_encoder: $bn"
        fi
    done
fi

# --- VAE ---
if [ -d "$NV/models/vae" ]; then
    mkdir -p "$COMFY/models/vae"
    for f in "$NV/models/vae"/*; do
        [ -e "$f" ] || continue
        bn=$(basename "$f")
        if [ ! -e "$COMFY/models/vae/$bn" ]; then
            ln -s "$f" "$COMFY/models/vae/$bn"
            echo "[sulphur-gguf]   vae: $bn"
        fi
    done
fi

# --- CLIP ---
if [ -d "$NV/models/clip" ]; then
    mkdir -p "$COMFY/models/clip"
    for f in "$NV/models/clip"/*; do
        [ -e "$f" ] || continue
        bn=$(basename "$f")
        if [ ! -e "$COMFY/models/clip/$bn" ]; then
            ln -s "$f" "$COMFY/models/clip/$bn"
            echo "[sulphur-gguf]   clip: $bn"
        fi
    done
fi

echo "[sulphur-gguf] Done. Handing off to ComfyUI..."

# Chain to the base image's start.sh
exec /start.sh
