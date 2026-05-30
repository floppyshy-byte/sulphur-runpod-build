#!/bin/bash
# ---------------------------------------------------------------------------
# Sulphur-2 GGUF — Model symlink wrapper for RunPod Serverless
# ---------------------------------------------------------------------------
# Models are stored on a RunPod Network Volume mounted at /runpod-volume.
# We symlink from the network volume into ComfyUI's expected directories.
#
# Network volume layout:
#   /runpod-volume/models/unet/
#   /runpod-volume/models/loras/
#   /runpod-volume/models/vae/
#   /runpod-volume/models/text_encoders/
#   /runpod-volume/models/prompt_enhancer/
# ---------------------------------------------------------------------------

set -e

MODELS="/runpod-volume/models"
COMFY="/comfyui"

echo "[sulphur-gguf] Checking for models on network volume..."

if [ ! -d "$MODELS" ]; then
    echo "[sulphur-gguf] ERROR: /runpod-volume/models not found"
    echo "[sulphur-gguf] Network volume not attached? Mount path wrong?"
    exit 1
fi

echo "[sulphur-gguf] /runpod-volume/models contents:"
ls -la "$MODELS/" 2>/dev/null || true

# --- UNet / diffusion model (GGUF) ---
echo "[sulphur-gguf] Symlinking UNet..."
mkdir -p "$COMFY/models/unet"
UNET_COUNT=0
for f in "$MODELS"/unet/*.gguf; do
    [ -e "$f" ] || continue
    bn=$(basename "$f")
    ln -sf "$f" "$COMFY/models/unet/$bn"
    echo "[sulphur-gguf]   unet: $bn"
    UNET_COUNT=$((UNET_COUNT + 1))
done
if [ "$UNET_COUNT" -eq 0 ]; then
    echo "[sulphur-gguf] ERROR: no GGUF file found in $MODELS/unet/"
    exit 1
fi

# --- LoRAs ---
echo "[sulphur-gguf] Symlinking LoRAs..."
mkdir -p "$COMFY/models/loras"
LORA_COUNT=0
for f in "$MODELS"/loras/*.safetensors; do
    [ -e "$f" ] || continue
    bn=$(basename "$f")
    ln -sf "$f" "$COMFY/models/loras/$bn"
    echo "[sulphur-gguf]   lora: $bn"
    LORA_COUNT=$((LORA_COUNT + 1))
done
if [ "$LORA_COUNT" -eq 0 ]; then
    echo "[sulphur-gguf] WARNING: no LoRA files found in $MODELS/loras/"
fi

# --- VAE ---
echo "[sulphur-gguf] Symlinking VAEs..."
mkdir -p "$COMFY/models/vae"
VAE_COUNT=0
for f in "$MODELS"/vae/*.safetensors; do
    [ -e "$f" ] || continue
    bn=$(basename "$f")
    ln -sf "$f" "$COMFY/models/vae/$bn"
    echo "[sulphur-gguf]   vae: $bn"
    VAE_COUNT=$((VAE_COUNT + 1))
done
if [ "$VAE_COUNT" -eq 0 ]; then
    echo "[sulphur-gguf] ERROR: no VAE files found in $MODELS/vae/"
    exit 1
fi

# --- Text encoder ---
echo "[sulphur-gguf] Symlinking text encoder..."
if [ -d "$MODELS/text_encoders" ]; then
    mkdir -p "$COMFY/models/text_encoders"
    # Find the first text_encoder subdirectory
    for d in "$MODELS"/text_encoders/*/; do
        [ -d "$d" ] || continue
        bn=$(basename "$d")
        ln -sfn "$d" "$COMFY/models/text_encoders/$bn"
        echo "[sulphur-gguf]   text_encoder: $bn"
        break
    done
else
    echo "[sulphur-gguf] ERROR: text_encoders directory not found in $MODELS/"
    exit 1
fi

# --- Prompt enhancer (optional) ---
if [ -d "$MODELS/prompt_enhancer" ]; then
    echo "[sulphur-gguf] Symlinking prompt enhancer..."
    mkdir -p "$COMFY/models/prompt_enhancer"
    for f in "$MODELS"/prompt_enhancer/*.gguf; do
        [ -e "$f" ] 2>/dev/null || continue
        bn=$(basename "$f")
        ln -sf "$f" "$COMFY/models/prompt_enhancer/$bn"
        echo "[sulphur-gguf]   prompt_enhancer: $bn"
    done
fi

echo "[sulphur-gguf] Done. Handing off to ComfyUI..."

# Chain to the base image's start.sh
exec /start.sh
