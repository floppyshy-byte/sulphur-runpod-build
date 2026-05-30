# Sulphur-2 ComfyUI + GGUF — Deployment Plan

## Architecture

```
RunPod Serverless Container (stateless)
├── ComfyUI + ComfyUI-GGUF + ComfyUI-LTXVideo + KJNodes
├── start-wrapper.sh → symlinks models from network volume
└── handler.py → receives ComfyUI workflow JSON, returns video

RunPod Network Volume (persistent)
├── models/unet/          → GGUF quantized diffusion model
├── models/loras/         → distill LoRA
├── models/vae/           → LTX-2.3 video + audio VAE
├── models/text_encoders/ → Gemma-3-12B
├── models/prompt_enhancer/ → optional CPU-side prompt enhancer
└── workflows/            → ComfyUI workflow JSON files
```

## Docker Image

Base: `runpod/worker-comfyui:5.8.5-base`

Custom nodes installed at build time:
- `city96/ComfyUI-GGUF` — UnetLoaderGGUF, DualCLIPLoaderGGUF
- `Lightricks/ComfyUI-LTXVideo` — LTX pipeline nodes (scheduler, sampler)
- `Kijai/ComfyUI-KJNodes` — LTX-2.3 VAE loader, audio VAE support

`start-wrapper.sh` runs before `start.sh` to symlink models from `/runpod-volume` into ComfyUI directories.

## Network Volume Layout

```
/runpod-volume/
  models/
    unet/
      sulphur-2-base-Q4_K_M.gguf           ~14 GB
    loras/
      Sulphur-2-distill-lora.safetensors   ~10 GB
    vae/
      ltx_2.3_video_vae.safetensors        ~1.5 GB
      ltx_2.3_audio_vae.safetensors        ~500 MB
    text_encoders/
      gemma-3-12b-it/                       ~24 GB (safetensors, from our HF repo)
    prompt_enhancer/
      sulphur_prompt_enhancer_model-q8_0.gguf   ~9.5 GB
      mmproj-BF16.gguf                           ~2 GB
```

Total network volume: ~60 GB

## HF Repo Changes

`Floppyshy/sulphur-2-runpod`:
- REMOVE: `sulphur_distil_fp8mixed.safetensors` (29 GB, monolithic — not needed for ComfyUI)
- ADD: `sulphur-2-base-Q4_K_M.gguf` (~14 GB, from Abiray conversion)
- ADD: `Sulphur-2-distill-lora.safetensors` (~10 GB, if not already present)
- KEEP: `text_encoder/` (Gemma-3-12B safetensors)
- KEEP: `tokenizer/` (tokenizer configs)

## GPU Selection

| Quant | File | Min GPU | Cost/hr | Use case |
|---|---|---|---|---|
| Q3_K_M | 11 GB | RTX 4090 (24 GB) | ~$1.12 | Fastest, lowest quality |
| Q4_K_M | 14 GB | RTX 4090 (24 GB) | ~$1.12 | Recommended starting point |
| Q5_K_M | 16 GB | L40S (48 GB) | ~$0.90 | Better quality |
| Q8_0 | 23 GB | L40S (48 GB) | ~$0.90 | Near-lossless |

## API Payload (ComfyUI workflow JSON)

```json
{
  "input": {
    "workflow": {
      "10": {
        "inputs": {"unet_name": "sulphur-2-base-Q4_K_M.gguf"},
        "class_type": "UnetLoaderGGUF"
      },
      "11": {
        "inputs": {
          "lora_name": "Sulphur-2-distill-lora.safetensors",
          "strength_model": 0.65,
          "model": ["10", 0]
        },
        "class_type": "LoraLoader"
      },
      "12": {
        "inputs": {
          "steps": 8,
          "cfg": 1.7,
          "sampler_name": "euler",
          "scheduler": "normal",
          "model": ["11", 0]
        },
        "class_type": "KSampler"
      }
    }
  }
}
```

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

## Build & Deploy Steps

1. Update `Dockerfile.comfyui` with KJNodes
2. Build and push Docker image (GHCR or RunPod native build)
3. Populate network volume with all model files
4. Update HF repo (remove safetensors, add GGUF)
5. Create RunPod serverless endpoint (RTX 4090, attached network volume)
6. Write and test ComfyUI workflow JSON
7. Submit test job, verify video output
