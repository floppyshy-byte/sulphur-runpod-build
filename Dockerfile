# =============================================================================
# Sulphur-2 GGUF — RunPod Serverless Worker (ComfyUI)
# =============================================================================
# Extends RunPod's official ComfyUI worker with GGUF and LTX-Video support.
# Models are cached to the worker via RunPod Model Cache from one or more
# HuggingFace repos. start-wrapper.sh symlinks from the HF cache into
# ComfyUI's expected model directories.
#
# HF repo layout (Floppyshy/sulphur-2-runpod):
#   sulphur-2-base-Q4_K_M.gguf                  (~13 GB)
#   sulphur_lora_rank_768.safetensors            (~10 GB)
#   LTX23_video_vae_bf16.safetensors             (~1.4 GB)
#   LTX23_audio_vae_bf16.safetensors             (~348 MB)
#   taeltx2_3.safetensors                        (~22 MB)
#   text_encoder/gemma_3_12B_it_fp8_scaled.safetensors  (~12 GB)
#   tokenizer/                                   (configs)
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

# Comfyui_TTP_Toolset — video continuation, first/last/middle frame control
RUN git clone https://github.com/TTPlanetPig/Comfyui_TTP_Toolset.git \
    /comfyui/custom_nodes/Comfyui_TTP_Toolset \
    && cd /comfyui/custom_nodes/Comfyui_TTP_Toolset \
    && pip install --no-cache-dir -r requirements.txt 2>/dev/null || true

# ComfyUI-Llama — GGUF text LLM loader for prompt enhancement
# Installs CPU-only; avoids CUDA compilation issues and keeps VRAM free for video generation
RUN apt-get update && apt-get install -y --no-install-recommends build-essential python3-dev \
    && git clone https://github.com/sebagallo/comfyui-sg-llama-cpp.git \
    /comfyui/custom_nodes/comfyui-sg-llama-cpp \
    && cd /comfyui/custom_nodes/comfyui-sg-llama-cpp \
    && sed -i 's/present_penalty/presence_penalty/g' nodes.py \
    && pip install --no-cache-dir -r requirements.txt \
    && rm -rf /var/lib/apt/lists/*

# RunpodVideoBridge — custom node that bridges VHS video outputs to standard image
# outputs so RunPod's handler S3 uploader picks them up.
COPY custom_nodes/RunpodVideoBridge /comfyui/custom_nodes/RunpodVideoBridge

# Install cryptography for the worker handler's input decryption.
# The base image uses /opt/venv.
RUN /opt/venv/bin/pip install --no-cache-dir cryptography

# Replace the base worker handler with our wrapper that decrypts encrypted input.
# The original handler is kept as /handler_base.py so our wrapper can delegate to it.
RUN mv /handler.py /handler_base.py
COPY handler.py /handler.py

# ---------------------------------------------------------------------------
# Model symlink setup (runs at container boot, before ComfyUI starts)
# ---------------------------------------------------------------------------

COPY start-wrapper.sh /start-wrapper.sh
RUN chmod +x /start-wrapper.sh

# Override the base image CMD to symlink models first, then start normally
CMD ["/start-wrapper.sh"]
