#!/bin/bash
# ---------------------------------------------------------------------------
# Sulphur-2 FP8 — Model symlink wrapper for RunPod Serverless
# ---------------------------------------------------------------------------
# Models are cached via RunPod Model Cache from HuggingFace repos.
# The cache lands under /runpod-volume/huggingface-cache/hub/ as e.g.:
#   models--<org>--<repo-name>/
# Override via SULPHUR_HF_REPO env var. If unset, we scan HF_CACHE for all
# models--* directories and symlink files from each.
#
# We symlink from the HF cache into ComfyUI's expected directories.
# If the cache is missing (e.g. smoke test without Model Cache), we warn
# and continue — ComfyUI will still boot, just without models.
# ---------------------------------------------------------------------------

# No set -e — we want to continue even if models are missing (smoke test)

HF_CACHE="/runpod-volume/huggingface-cache/hub"
COMFY="/comfyui"

# --- Discover repo directories ---
# Prefer explicit env var, then scan HF_CACHE subdirectories for models--*
REPOS=""
if [ -n "${SULPHUR_HF_REPO:-}" ]; then
    REPO_DIR="$HF_CACHE/$SULPHUR_HF_REPO"
    if [ -d "$REPO_DIR" ]; then
        REPOS="$REPO_DIR"
        echo "[sulphur] Using SULPHUR_HF_REPO env var: $SULPHUR_HF_REPO"
    else
        echo "[sulphur] WARNING: SULPHUR_HF_REPO=$SULPHUR_HF_REPO not found in cache" >&2
    fi
else
    # Find all models--* dirs under HF_CACHE
    REPOS=$(find "$HF_CACHE" -maxdepth 1 -type d -name 'models--*' 2>/dev/null)
    REPO_COUNT=$(echo "$REPOS" | grep -c '^' || echo 0)

    if [ "$REPO_COUNT" -eq 0 ]; then
        echo "[sulphur] WARNING: No HF cache repos found in $HF_CACHE." >&2
        echo "[sulphur] Set SULPHUR_HF_REPO or ensure the model cache is mounted." >&2
    else
        echo "[sulphur] Found $REPO_COUNT HF cache repo(s):"
        echo "$REPOS" | sed 's/^/[sulphur]   /'
    fi
fi

echo "[sulphur] Looking for HuggingFace cache..."

# Full diagnostic dump — matches model-cache-test handler output
echo "[sulphur] === CACHE DIAGNOSTIC ==="
echo "[sulphur] /runpod-volume:"
ls -laR /runpod-volume/ 2>/dev/null | head -80 || echo "[sulphur]   (empty or missing)"
echo "[sulphur] === END DIAGNOSTIC ==="

# Pre-create all target directories once
mkdir -p "$COMFY/models/unet"
mkdir -p "$COMFY/models/gguf"
mkdir -p "$COMFY/models/loras"
mkdir -p "$COMFY/models/vae"
mkdir -p "$COMFY/models/checkpoints"
mkdir -p "$COMFY/models/text_encoders"
mkdir -p "$COMFY/models/clip"

# --- Process each repo ---
process_repo() {
    local REPO_DIR="$1"
    local REPO=$(basename "$REPO_DIR")
    local MISSING=0

    echo "[sulphur] --- Processing repo: $REPO ---"

    # Show refs (git references — which commit is cached)
    if [ -d "$REPO_DIR/refs" ]; then
        echo "[sulphur] refs:"
        cat "$REPO_DIR/refs/"* 2>/dev/null || echo "[sulphur]   (empty)"
    fi

    # Show snapshots
    local SNAPSHOT_DIR="$REPO_DIR/snapshots"
    if [ -d "$SNAPSHOT_DIR" ]; then
        echo "[sulphur] snapshots:"
        ls -la "$SNAPSHOT_DIR/" 2>/dev/null
    else
        echo "[sulphur] WARNING: no snapshots at $SNAPSHOT_DIR"
        MISSING=1
    fi

    # Find the snapshot directory
    local SNAP=""
    if [ "$MISSING" -eq 0 ]; then
        if [ -f "$REPO_DIR/refs/main" ]; then
            local HASH=$(cat "$REPO_DIR/refs/main" 2>/dev/null)
            if [ -n "$HASH" ] && [ -d "$SNAPSHOT_DIR/$HASH" ]; then
                SNAP="$SNAPSHOT_DIR/$HASH"
                echo "[sulphur] Using refs/main: $HASH"
            fi
        fi
        if [ -z "$SNAP" ]; then
            SNAP=$(ls -d "$SNAPSHOT_DIR"/*/ 2>/dev/null | head -1)
            [ -n "$SNAP" ] && echo "[sulphur] Fallback: first snapshot dir"
        fi
        if [ -z "$SNAP" ]; then
            echo "[sulphur] WARNING: no snapshot hash dirs in $SNAPSHOT_DIR"
            return
        else
            SNAP="${SNAP%/}"
            echo "[sulphur] Cache snapshot: $SNAP"
            echo "[sulphur] Snapshot files:"
            ls -la "$SNAP/" 2>/dev/null | head -20
        fi
    fi

    if [ -z "$SNAP" ]; then
        return
    fi

    echo "[sulphur] Scanning snapshot and symlinking all model files..."

    # Recursively walk the snapshot and symlink every model file we find
    find "$SNAP" -type f \( \
        -name '*.gguf' -o \
        -name '*.safetensors' -o \
        -name '*.bin' -o \
        -name '*.pt' -o \
        -name '*.ckpt' \
    \) | sort | while read -r filepath; do
        bn=$(basename "$filepath")
        relpath="${filepath#$SNAP/}"

        # Skip non-model junk
        case "$bn" in
            .git*|README*|*.json|*.txt|*.md|*.py) continue ;;
        esac

        # Route by relative path first, then by filename pattern
        case "$relpath" in
            text_encoder/*)
                ln -sf "$filepath" "$COMFY/models/text_encoders/$bn" 2>/dev/null
                ln -sf "$filepath" "$COMFY/models/clip/$bn" 2>/dev/null
                echo "[sulphur]   text_encoders + clip: $bn"
                ;;
            vae/*)
                ln -sf "$filepath" "$COMFY/models/vae/$bn" 2>/dev/null
                echo "[sulphur]   vae: $bn"
                ;;
            *)
                # Root-level files — categorize by extension + filename heuristics
                case "$bn" in
                    *.gguf)
                        ln -sf "$filepath" "$COMFY/models/unet/$bn" 2>/dev/null
                        ln -sf "$filepath" "$COMFY/models/gguf/$bn" 2>/dev/null
                        echo "[sulphur]   unet + gguf: $bn"
                        ;;
                    *lora*.safetensors|*distill*.safetensors)
                        ln -sf "$filepath" "$COMFY/models/loras/$bn" 2>/dev/null
                        echo "[sulphur]   lora: $bn"
                        ;;
                    *vae*.safetensors|tae*.safetensors)
                        ln -sf "$filepath" "$COMFY/models/vae/$bn" 2>/dev/null
                        echo "[sulphur]   vae: $bn"
                        ;;
                    *connector*.safetensors)
                        ln -sf "$filepath" "$COMFY/models/clip/$bn" 2>/dev/null
                        ln -sf "$filepath" "$COMFY/models/text_encoders/$bn" 2>/dev/null
                        echo "[sulphur]   clip + text_encoders: $bn"
                        ;;
                    *ltx*.safetensors)
                        ln -sf "$filepath" "$COMFY/models/checkpoints/$bn" 2>/dev/null
                        echo "[sulphur]   checkpoint: $bn"
                        ;;
                    *.safetensors)
                        # Unknown safetensors → checkpoints as a sensible default
                        ln -sf "$filepath" "$COMFY/models/checkpoints/$bn" 2>/dev/null
                        echo "[sulphur]   checkpoint (fallback): $bn"
                        ;;
                    *)
                        # Any other model file → unet as ultimate fallback
                        ln -sf "$filepath" "$COMFY/models/unet/$bn" 2>/dev/null
                        echo "[sulphur]   unet (fallback): $bn"
                        ;;
                esac
                ;;
        esac
    done

    # Directory-level symlinks (some ComfyUI nodes expect a folder, not files)
    if [ -d "$SNAP/text_encoder" ]; then
        ln -sfn "$SNAP/text_encoder" "$COMFY/models/text_encoders/gemma-3-12b-it"
        echo "[sulphur]   text_encoder dir: gemma-3-12b-it"
    fi
}

# Process each discovered repo
if [ -n "$REPOS" ]; then
    echo "$REPOS" | while read -r repo_dir; do
        [ -n "$repo_dir" ] && process_repo "$repo_dir"
    done
else
    echo "[sulphur] No repos to process."
fi

echo "[sulphur] Done. Handing off to ComfyUI..."

# Chain to the base image's start.sh
exec /start.sh
