#!/bin/bash
# ---------------------------------------------------------------------------
# Download Sulphur-2 ComfyUI model files and upload to HF repo
# ---------------------------------------------------------------------------
# Run this on a machine with good bandwidth and ~60 GB free disk space.
# A temporary RunPod GPU Pod (RTX 4090, 100 GB disk) works well.
#
# Prerequisites:
#   pip install huggingface_hub
#   huggingface-cli login  (use a token with write access to Floppyshy/sulphur-2-runpod)
# ---------------------------------------------------------------------------

set -e

HF_REPO="Floppyshy/sulphur-2-runpod"
TMPDIR="${TMPDIR:-/tmp/sulphur-models}"
mkdir -p "$TMPDIR"

echo "=== Step 1: Download model files ==="

# --- Q4_K_M GGUF (from Abiray) ---
echo ""
echo ">>> Downloading sulphur-2-base-Q4_K_M.gguf (14 GB)..."
huggingface-cli download Abiray/Sulphur-2-base-GGUF \
    sulphur_dev-Q4_K_M.gguf \
    --local-dir "$TMPDIR"

# --- Distill LoRA (from SulphurAI) ---
echo ""
echo ">>> Downloading Sulphur-2-distill-lora.safetensors (10 GB)..."
huggingface-cli download SulphurAI/Sulphur-2-base \
    sulphur_lora_rank_768.safetensors \
    --local-dir "$TMPDIR"

# --- LTX-2.3 VAEs (from Kijai) ---
echo ""
echo ">>> Downloading ltx_2.3_video_vae.safetensors (1.5 GB)..."
huggingface-cli download Kijai/LTX2.3_comfy \
    ltx_2.3_video_vae.safetensors \
    --local-dir "$TMPDIR"

echo ""
echo ">>> Downloading ltx_2.3_audio_vae.safetensors (500 MB)..."
huggingface-cli download Kijai/LTX2.3_comfy \
    ltx_2.3_audio_vae.safetensors \
    --local-dir "$TMPDIR"

# --- Prompt enhancer (from SulphurAI) ---
echo ""
echo ">>> Downloading prompt enhancer files (12 GB)..."
huggingface-cli download SulphurAI/Sulphur-2-base \
    prompt_enhancer/ \
    --local-dir "$TMPDIR"

echo ""
echo "=== Downloads complete ==="
du -sh "$TMPDIR"/*

echo ""
echo "=== Step 2: Upload to $HF_REPO ==="

# Upload each file individually (huggingface-cli handles LFS automatically)
for f in "$TMPDIR"/*.gguf "$TMPDIR"/*.safetensors; do
    [ -e "$f" ] || continue
    echo ""
    echo ">>> Uploading $(basename "$f")..."
    huggingface-cli upload "$HF_REPO" "$f" "$(basename "$f")"
done

# Upload prompt_enhancer directory
if [ -d "$TMPDIR/prompt_enhancer" ]; then
    echo ""
    echo ">>> Uploading prompt_enhancer/..."
    huggingface-cli upload "$HF_REPO" "$TMPDIR/prompt_enhancer" prompt_enhancer/
fi

echo ""
echo "=== Step 3: Remove old monolithic safetensors ==="
echo ">>> Deleting sulphur_distil_fp8mixed.safetensors from $HF_REPO..."
huggingface-cli delete "$HF_REPO" sulphur_distil_fp8mixed.safetensors || {
    echo "WARNING: Could not delete via CLI. You may need to delete it manually in the HF web UI."
    echo "  https://huggingface.co/$HF_REPO/tree/main"
}

echo ""
echo "=== Done ==="
echo "Verify at: https://huggingface.co/$HF_REPO/tree/main"
echo "Expected files:"
echo "  sulphur_dev-Q4_K_M.gguf"
echo "  sulphur_lora_rank_768.safetensors"
echo "  ltx_2.3_video_vae.safetensors"
echo "  ltx_2.3_audio_vae.safetensors"
echo "  prompt_enhancer/"
echo "  text_encoder/"
echo "  tokenizer/"
echo ""
echo "Removed: sulphur_distil_fp8mixed.safetensors"
