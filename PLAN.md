# Sulphur-2 ComfyUI + GGUF — Deployment Plan

## Architecture

```
RunPod Serverless Container (stateless)
├── ComfyUI + ComfyUI-GGUF + ComfyUI-LTXVideo + KJNodes
├── start-wrapper.sh → symlinks models from HF cache into ComfyUI dirs
└── handler.py → receives ComfyUI workflow JSON, returns video frames

HuggingFace Repo (Floppyshy/sulphur-2-runpod)
├── sulphur-2-base-Q4_K_M.gguf           ~14 GB
├── Sulphur-2-distill-lora.safetensors   ~10 GB
├── LTX23_video_vae_bf16.safetensors        ~1.35 GB
├── LTX23_audio_vae_bf16.safetensors        ~348 MB
├── text_encoder/                         ~24 GB (Gemma-3-12B)
└── tokenizer/                            ~4 MB

RunPod Model Cache →
  /runpod-volume/huggingface-cache/hub/
    models--Floppyshy--sulphur-2-runpod/snapshots/<hash>/
```

No network volume — everything is in the HF repo, cached to workers via RunPod Model Cache.

## Docker Image

Base: `runpod/worker-comfyui:5.8.5-base`

Custom nodes installed at build time:
- `city96/ComfyUI-GGUF` — UnetLoaderGGUF, DualCLIPLoaderGGUF
- `Lightricks/ComfyUI-LTXVideo` — LTX pipeline nodes (scheduler, sampler)
- `Kijai/ComfyUI-KJNodes` — LTX-2.3 VAE loader, audio VAE support

`start-wrapper.sh` runs before `start.sh` to symlink models from the HF cache into ComfyUI directories.

## HF Repo Changes

`Floppyshy/sulphur-2-runpod`:
- REMOVE: `sulphur_distil_fp8mixed.safetensors` (29 GB, monolithic — not needed for ComfyUI)
- ADD: `sulphur-2-base-Q4_K_M.gguf` (~14 GB, from Abiray/Sulphur-2-base-GGUF)
- ADD: `Sulphur-2-distill-lora.safetensors` (~10 GB, from SulphurAI/Sulphur-2-base)
- ADD: `LTX23_video_vae_bf16.safetensors` (~1.35 GB, from Kijai/LTX2.3_comfy)
- ADD: `LTX23_audio_vae_bf16.safetensors` (~348 MB, from Kijai/LTX2.3_comfy)
- KEEP: `text_encoder/` (Gemma-3-12B safetensors)
- KEEP: `tokenizer/` (tokenizer configs)

Total: ~50 GB (vs 53 GB before)

## GPU Selection

| Quant | File | Min GPU | Cost/hr | Use case |
|---|---|---|---|---|
| Q3_K_M | 11 GB | RTX 4090 (24 GB) | ~$1.12 | Fastest, lowest quality |
| Q4_K_M | 14 GB | RTX 4090 (24 GB) | ~$1.12 | Recommended starting point |
| Q5_K_M | 16 GB | L40S (48 GB) | ~$0.90 | Better quality |
| Q8_0 | 23 GB | L40S (48 GB) | ~$0.90 | Near-lossless |

## Inference Settings (Distilled + Quantized)

- LoRA strength: 0.6–0.7
- Sampling steps: 6–8
- CFG: 1.5–2.0
- Resolution: 480p default

## What We Don't Need Anymore

- `loader.py` — the custom safetensors loader (ComfyUI handles loading)
- `handler.py` — the custom RunPod handler (ComfyUI base image provides its own)
- `Dockerfile` (old) — replaced by `Dockerfile.comfyui`
- `sulphur_distil_fp8mixed.safetensors` — monolithic checkpoint (replaced by GGUF + separate components)
- Network volume — replaced by HF repo + RunPod Model Cache

## Future Enhancements (Day 2+)

### Video MP4 Output (VHS + VideoOutputBridge)
- Add `ComfyUI-VideoHelperSuite` node → `VHS_VideoCombine` merges frames into MP4
- Add `VideoOutputBridge` node → repackages VHS video output into standard `images` payload format so RunPod's handler picks it up without modification
- Result: API returns MP4 video file (or S3 URL) instead of individual frame images

### Prompt Enhancer Integration
- Add `sulphur_prompt_enhancer_model-q8_0.gguf` (~9.5 GB) and `mmproj-BF16.gguf` (~2 GB) to HF repo
- Add ComfyUI LLM-loader node to run enhancer on CPU as a pre-processing step
- Alternative: run enhancer as a separate microservice before calling ComfyUI

### Higher Quality Quants
- Add Q5_K_M (16 GB) or Q8_0 (23 GB) GGUF files to HF repo for quality-sensitive jobs
- Requires GPU upgrade: L40S (48 GB) at ~$0.90/hr
- Simple filename swap in the workflow JSON — no code changes

### Audio Generation
- Add LTX-2.3 audio VAE loader and vocoder nodes (VAE file already in repo)
- Enable native sync audio with video output
- Requires KJNodes audio pipeline nodes (already installed)

## Build & Deploy Steps

1. Download model files to HF repo (or upload via huggingface-cli)
2. Build and push Docker image (RunPod native build pointing at Dockerfile.comfyui)
3. Create RunPod serverless endpoint (RTX 4090, Model Cache: Floppyshy/sulphur-2-runpod)
4. Write and test ComfyUI workflow JSON
5. Submit test job, verify video output
