#!/bin/bash
# ---------------------------------------------------------------------------
# Download Sulphur-2 ComfyUI model files and upload to HF repo
# ---------------------------------------------------------------------------
# Run on a RunPod CPU pod (Ubuntu, 50 GB disk, ~$0.20/hr).
# Processes one file at a time to minimize disk usage.
#
# Setup:
#   apt-get update && apt-get install -y python3-pip
#   pip install huggingface_hub
#   huggingface-cli login  (use a write token for Floppyshy/sulphur-2-runpod)
# ---------------------------------------------------------------------------

set -e

HF_REPO="Floppyshy/sulphur-2-runpod"
WORKDIR="/tmp/sulphur-dl"
mkdir -p "$WORKDIR"

# Each entry: "source_repo  source_path  dest_filename"
# Files are processed one at a time: download → upload → cleanup
FILES=(
    "Abiray/Sulphur-2-base-GGUF        sulphur_dev-Q4_K_M.gguf              sulphur-2-base-Q4_K_M.gguf"
    "SulphurAI/Sulphur-2-base           sulphur_lora_rank_768.safetensors    sulphur_lora_rank_768.safetensors"
    "Kijai/LTX2.3_comfy                 LTX23_video_vae_bf16.safetensors        LTX23_video_vae_bf16.safetensors"
    "Kijai/LTX2.3_comfy                 LTX23_audio_vae_bf16.safetensors        LTX23_audio_vae_bf16.safetensors"
)

DIRS=(
    "SulphurAI/Sulphur-2-base           prompt_enhancer/"
)

echo "=== Sulphur-2 ComfyUI Model Setup ==="
echo "Target repo: $HF_REPO"
echo "Working dir: $WORKDIR"
echo "Disk free: $(df -h /tmp | tail -1 | awk '{print $4}')"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Download and upload individual files
# ---------------------------------------------------------------------------
for entry in "${FILES[@]}"; do
    read -r SRC_REPO SRC_PATH DEST_NAME <<< "$entry"
    LOCAL_PATH="$WORKDIR/$DEST_NAME"

    echo ">>> $DEST_NAME"
    echo "    Source: $SRC_REPO :: $SRC_PATH"

    # Download
    huggingface-cli download "$SRC_REPO" "$SRC_PATH" --local-dir "$WORKDIR" --quiet
    # huggingface-cli download renames to the original filename, fix if needed
    if [ -f "$WORKDIR/$SRC_PATH" ] && [ "$SRC_PATH" != "$DEST_NAME" ]; then
        mv "$WORKDIR/$SRC_PATH" "$LOCAL_PATH"
    fi
    [ -f "$LOCAL_PATH" ] || { echo "    ERROR: download failed"; exit 1; }

    SIZE=$(du -h "$LOCAL_PATH" | cut -f1)
    echo "    Downloaded: $SIZE"

    # Upload
    echo "    Uploading..."
    huggingface-cli upload "$HF_REPO" "$LOCAL_PATH" "$DEST_NAME" --quiet
    echo "    Uploaded."

    # Cleanup
    rm -f "$LOCAL_PATH" "$WORKDIR/$SRC_PATH"
    echo "    Done."
    echo ""
done

# ---------------------------------------------------------------------------
# Step 2: Download and upload directories
# ---------------------------------------------------------------------------
for entry in "${DIRS[@]}"; do
    read -r SRC_REPO SRC_DIR <<< "$entry"
    echo ">>> $SRC_DIR (directory)"
    echo "    Source: $SRC_REPO"

    huggingface-cli download "$SRC_REPO" "$SRC_DIR" --local-dir "$WORKDIR" --quiet

    echo "    Uploading..."
    huggingface-cli upload "$HF_REPO" "$WORKDIR/$SRC_DIR" "$SRC_DIR" --quiet
    echo "    Uploaded."

    rm -rf "$WORKDIR/${SRC_DIR%/}"
    echo "    Done."
    echo ""
done

# ---------------------------------------------------------------------------
# Step 3: Remove old monolithic safetensors
# ---------------------------------------------------------------------------
echo "=== Removing old monolithic checkpoint ==="
huggingface-cli delete "$HF_REPO" sulphur_distil_fp8mixed.safetensors 2>/dev/null || {
    echo "Could not delete via CLI."
    echo "Delete manually: https://huggingface.co/$HF_REPO/tree/main"
    echo "File to remove: sulphur_distil_fp8mixed.safetensors"
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "=== Done ==="
echo "Added to $HF_REPO:"
echo "  sulphur-2-base-Q4_K_M.gguf"
echo "  sulphur_lora_rank_768.safetensors"
echo "  LTX23_video_vae_bf16.safetensors"
echo "  LTX23_audio_vae_bf16.safetensors"
echo "  prompt_enhancer/"
echo ""
echo "Existing (untouched):"
echo "  text_encoder/"
echo "  tokenizer/"
echo ""
echo "Removed: sulphur_distil_fp8mixed.safetensors"
echo ""
echo "Verify: https://huggingface.co/$HF_REPO/tree/main"
