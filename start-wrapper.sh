#!/bin/bash
# ---------------------------------------------------------------------------
# Sulphur-2 GGUF — Model symlink wrapper for RunPod Serverless
# ---------------------------------------------------------------------------
# Models are cached via RunPod Model Cache from HuggingFace repo
# "Floppyshy/sulphur-2-runpod". The cache lands at:
#   /runpod-volume/huggingface-cache/hub/models--Floppyshy--sulphur-2-runpod/
#
# We symlink from the HF cache into ComfyUI's expected directories.
# ---------------------------------------------------------------------------

set -e

HF_CACHE="/runpod-volume/huggingface-cache/hub"
REPO="models--Floppyshy--sulphur-2-runpod"
COMFY="/comfyui"

echo "[sulphur-gguf] Looking for HuggingFace cache..."

# Find the snapshot directory (hash-named subfolder)
SNAPSHOT_DIR="$HF_CACHE/$REPO/snapshots"
if [ ! -d "$SNAPSHOT_DIR" ]; then
    echo "[sulphur-gguf] ERROR: HF cache not found at $SNAPSHOT_DIR"
    echo "[sulphur-gguf] Is the RunPod Model Cache configured with Floppyshy/sulphur-2-runpod?"
    exit 1
fi

SNAP=$(ls -d "$SNAPSHOT_DIR"/*/ 2>/dev/null | head -1)
if [ -z "$SNAP" ]; then
    echo "[sulphur-gguf] ERROR: no snapshot found in $SNAPSHOT_DIR"
    exit 1
fi
SNAP="${SNAP%/}"
echo "[sulphur-gguf] Cache snapshot: $SNAP"

# --- UNet / diffusion model (GGUF) ---
echo "[sulphur-gguf] Symlinking UNet..."
mkdir -p "$COMFY/models/unet"
for f in "$SNAP"/*.gguf; do
    [ -e "$f" ] || continue
    bn=$(basename "$f")
    ln -sf "$f" "$COMFY/models/unet/$bn"
    echo "[sulphur-gguf]   unet: $bn"
done

# --- LoRAs ---
echo "[sulphur-gguf] Symlinking LoRAs..."
mkdir -p "$COMFY/models/loras"
for f in "$SNAP"/*lora*.safetensors "$SNAP"/*distill*.safetensors; do
    [ -e "$f" ] || continue
    bn=$(basename "$f")
    ln -sf "$f" "$COMFY/models/loras/$bn"
    echo "[sulphur-gguf]   lora: $bn"
done

# --- VAE ---
echo "[sulphur-gguf] Symlinking VAEs..."
mkdir -p "$COMFY/models/vae"
for f in "$SNAP"/*vae*.safetensors "$SNAP"/vae/*.safetensors; do
    [ -e "$f" ] 2>/dev/null || continue
    bn=$(basename "$f")
    ln -sf "$f" "$COMFY/models/vae/$bn"
    echo "[sulphur-gguf]   vae: $bn"
done

# --- Text encoder ---
echo "[sulphur-gguf] Symlinking text encoder..."
if [ -d "$SNAP/text_encoder" ]; then
    mkdir -p "$COMFY/models/text_encoders"
    ln -sfn "$SNAP/text_encoder" "$COMFY/models/text_encoders/gemma-3-12b-it"
    echo "[sulphur-gguf]   text_encoder: gemma-3-12b-it"
fi

# --- Prompt enhancer (future) ---
if [ -d "$SNAP/prompt_enhancer" ]; then
    echo "[sulphur-gguf] Symlinking prompt enhancer..."
    mkdir -p "$COMFY/models/prompt_enhancer"
    for f in "$SNAP/prompt_enhancer"/*.gguf; do
        [ -e "$f" ] 2>/dev/null || continue
        bn=$(basename "$f")
        ln -sf "$f" "$COMFY/models/prompt_enhancer/$bn"
        echo "[sulphur-gguf]   prompt_enhancer: $bn"
    done
fi

echo "[sulphur-gguf] Done. Handing off to ComfyUI..."

# Chain to the base image's start.sh
exec /start.sh
