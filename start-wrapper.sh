#!/bin/bash
# ---------------------------------------------------------------------------
# Sulphur-2 GGUF — Model symlink wrapper for RunPod Serverless
# ---------------------------------------------------------------------------
# Models are cached via RunPod Model Cache from HuggingFace repo
# "Floppyshy/sulphur-2-runpod". The cache lands at:
#   /runpod-volume/huggingface-cache/hub/models--floppyshy--sulphur-2-runpod/
#
# We symlink from the HF cache into ComfyUI's expected directories.
# If the cache is missing (e.g. smoke test without Model Cache), we warn
# and continue — ComfyUI will still boot, just without models.
# ---------------------------------------------------------------------------

# No set -e — we want to continue even if models are missing (smoke test)

HF_CACHE="/runpod-volume/huggingface-cache/hub"
REPO="models--floppyshy--sulphur-2-runpod"
COMFY="/comfyui"
MISSING=0

echo "[sulphur-gguf] Looking for HuggingFace cache..."

# Full diagnostic dump — matches model-cache-test handler output
echo "[sulphur-gguf] === CACHE DIAGNOSTIC ==="
echo "[sulphur-gguf] /runpod-volume:"
ls -laR /runpod-volume/ 2>/dev/null | head -80 || echo "[sulphur-gguf]   (empty or missing)"
echo "[sulphur-gguf] === END DIAGNOSTIC ==="

# Check for the HF cache repo directory
REPO_DIR="$HF_CACHE/$REPO"

# Show refs (git references — which commit is cached)
if [ -d "$REPO_DIR/refs" ]; then
    echo "[sulphur-gguf] refs:"
    cat "$REPO_DIR/refs/"* 2>/dev/null || echo "[sulphur-gguf]   (empty)"
fi

# Show snapshots
SNAPSHOT_DIR="$REPO_DIR/snapshots"
if [ -d "$SNAPSHOT_DIR" ]; then
    echo "[sulphur-gguf] snapshots:"
    ls -la "$SNAPSHOT_DIR/" 2>/dev/null
else
    echo "[sulphur-gguf] WARNING: no snapshots at $SNAPSHOT_DIR"
    echo "[sulphur-gguf] RunPod Model Cache not configured or still downloading?"
    MISSING=1
fi

# Find the snapshot directory following RunPod docs approach:
# 1. Read refs/main to get the canonical hash
# 2. Fallback: first snapshot directory
if [ "$MISSING" -eq 0 ]; then
    SNAP=""
    if [ -f "$REPO_DIR/refs/main" ]; then
        HASH=$(cat "$REPO_DIR/refs/main" 2>/dev/null)
        if [ -n "$HASH" ] && [ -d "$SNAPSHOT_DIR/$HASH" ]; then
            SNAP="$SNAPSHOT_DIR/$HASH"
            echo "[sulphur-gguf] Using refs/main: $HASH"
        fi
    fi
    if [ -z "$SNAP" ]; then
        SNAP=$(ls -d "$SNAPSHOT_DIR"/*/ 2>/dev/null | head -1)
        [ -n "$SNAP" ] && echo "[sulphur-gguf] Fallback: first snapshot dir"
    fi
    if [ -z "$SNAP" ]; then
        echo "[sulphur-gguf] WARNING: no snapshot hash dirs in $SNAPSHOT_DIR"
        MISSING=1
    else
        SNAP="${SNAP%/}"
        echo "[sulphur-gguf] Cache snapshot: $SNAP"
        echo "[sulphur-gguf] Snapshot files:"
        ls -la "$SNAP/" 2>/dev/null | head -20
    fi
fi

if [ "$MISSING" -eq 0 ]; then
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
    for f in "$SNAP"/*vae*.safetensors "$SNAP"/vae/*.safetensors "$SNAP"/tae*.safetensors; do
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
    else
        echo "[sulphur-gguf]   text_encoder: not found (skipped)"
    fi

    # --- Checkpoints ---
    echo "[sulphur-gguf] Symlinking checkpoints..."
    mkdir -p "$COMFY/models/checkpoints"
    for f in "$SNAP"/*ltx*.safetensors; do
        [ -e "$f" ] 2>/dev/null || continue
        bn=$(basename "$f")
        # skip VAE, TAE, and lora files already handled above
        case "$bn" in
            *vae*|*tae*|*lora*|*distill*) continue ;;
        esac
        ln -sf "$f" "$COMFY/models/checkpoints/$bn"
        echo "[sulphur-gguf]   checkpoint: $bn"
    done

    # --- GGUF folder ---
    echo "[sulphur-gguf] Symlinking GGUF folder..."
    mkdir -p "$COMFY/models/gguf"
    for f in "$SNAP"/*.gguf; do
        [ -e "$f" ] || continue
        bn=$(basename "$f")
        ln -sf "$f" "$COMFY/models/gguf/$bn"
        echo "[sulphur-gguf]   gguf: $bn"
    done

    # --- Prompt enhancer ---
    if [ -d "$SNAP/prompt_enhancer" ]; then
        echo "[sulphur-gguf] Symlinking prompt enhancer..."
        mkdir -p "$COMFY/models/prompt_enhancer"
        mkdir -p "$COMFY/models/LLM"
        for f in "$SNAP/prompt_enhancer"/*.gguf; do
            [ -e "$f" ] 2>/dev/null || continue
            bn=$(basename "$f")
            ln -sf "$f" "$COMFY/models/prompt_enhancer/$bn"
            ln -sf "$f" "$COMFY/models/LLM/$bn"
            echo "[sulphur-gguf]   prompt_enhancer: $bn"
        done
    fi
fi

# --- CLIP folder (for DualCLIPLoader GGUF) ---
# DualCLIPLoaderGGUF scans models/clip/ for both clip_name1 and clip_name2.
# We need the Gemma text encoder AND the LTX embeddings connector here.
echo "[sulphur-gguf] Symlinking CLIP folder..."
mkdir -p "$COMFY/models/clip"

# Gemma text encoder weights
if [ -d "$SNAP/text_encoder" ]; then
    for f in "$SNAP/text_encoder"/*.safetensors; do
        [ -e "$f" ] || continue
        bn=$(basename "$f")
        ln -sf "$f" "$COMFY/models/clip/$bn"
        echo "[sulphur-gguf]   clip: $bn"
    done
fi

# LTX embeddings connector (for DualCLIPLoaderGGUF clip_name2)
for f in "$SNAP"/*connector*.safetensors; do
    [ -e "$f" ] 2>/dev/null || continue
    bn=$(basename "$f")
    ln -sf "$f" "$COMFY/models/clip/$bn"
    echo "[sulphur-gguf]   clip: $bn"
done

echo "[sulphur-gguf] Done. Handing off to ComfyUI..."

# Chain to the base image's start.sh
exec /start.sh
