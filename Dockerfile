# =============================================================================
# Sulphur-2 GGUF — RunPod Serverless Worker (ComfyUI)
# =============================================================================
# Extends RunPod's official ComfyUI worker with GGUF and LTX-Video support.
# Models are stored in HuggingFace repo "Floppyshy/sulphur-2-runpod" and
# cached to the worker via RunPod Model Cache. start-wrapper.sh symlinks
# from the HF cache into ComfyUI's expected model directories.
#
# HF repo layout (Floppyshy/sulphur-2-runpod):
#   sulphur-2-base-Q4_K_M.gguf                  (~13 GB)
#   sulphur_lora_rank_768.safetensors            (~10 GB)
#   LTX23_video_vae_bf16.safetensors             (~1.4 GB)
#   LTX23_audio_vae_bf16.safetensors             (~348 MB)
#   taeltx2_3.safetensors                        (~22 MB)
#   text_encoder/gemma_3_12B_it_fp8_scaled.safetensors  (~12 GB)
#   tokenizer/                                   (configs)
#   prompt_enhancer/                             (~6 GB, Q4_K_M GGUF)
# =============================================================================

ARG COMFYUI_VERSION=5.8.5
FROM runpod/worker-comfyui:${COMFYUI_VERSION}-base

# ---------------------------------------------------------------------------
# Custom nodes
# ---------------------------------------------------------------------------

# ComfyUI-GGUF — loads quantized UNet/diffusion models (supports ltxv arch)
RUN git clone https://github.com/city96/ComfyUI-GGUF.git \
    /comfyui/custom_nodes/ComfyUI-GGUF \
    && cd /comfyui/custom_nodes/ComfyUI-GGUF \
    && pip install --no-cache-dir -r requirements.txt 2>/dev/null || true

# ComfyUI-LTXVideo — LTX video pipeline nodes (scheduler, sampler, VAE loader)
RUN git clone https://github.com/Lightricks/ComfyUI-LTXVideo.git \
    /comfyui/custom_nodes/ComfyUI-LTXVideo \
    && cd /comfyui/custom_nodes/ComfyUI-LTXVideo \
    && pip install --no-cache-dir -r requirements.txt 2>/dev/null || true

# KJNodes — LTX-2.3 VAE loader, audio VAE, and utility nodes
RUN git clone https://github.com/kijai/ComfyUI-KJNodes.git \
    /comfyui/custom_nodes/ComfyUI-KJNodes \
    && cd /comfyui/custom_nodes/ComfyUI-KJNodes \
    && pip install --no-cache-dir -r requirements.txt 2>/dev/null || true

# VideoHelperSuite (VHS) — video I/O: combine frames to MP4, load video
RUN git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git \
    /comfyui/custom_nodes/ComfyUI-VideoHelperSuite \
    && cd /comfyui/custom_nodes/ComfyUI-VideoHelperSuite \
    && pip install --no-cache-dir -r requirements.txt 2>/dev/null || true

# VideoOutputBridge — exposes VHS video files as standard image outputs for RunPod handler
# Repo is a Python package (video_output_bridge/__init__.py), so we create a root __init__.py
# that ComfyUI's loader can discover.
RUN git clone https://github.com/ComfyNodePRs/PR-ComfyUI-VideoOutputBridge-2f67df8b.git \
    /comfyui/custom_nodes/ComfyUI-VideoOutputBridge \
    && echo "from .video_output_bridge import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS" \
    > /comfyui/custom_nodes/ComfyUI-VideoOutputBridge/__init__.py

# huggingface_hub — used by start-wrapper.sh for runtime model downloads
RUN pip install --no-cache-dir huggingface-hub

# ---------------------------------------------------------------------------
# Model symlink setup (runs at container boot, before ComfyUI starts)
# ---------------------------------------------------------------------------

COPY start-wrapper.sh /start-wrapper.sh
RUN chmod +x /start-wrapper.sh

# Override the base image CMD to symlink models first, then start normally
CMD ["/start-wrapper.sh"]
