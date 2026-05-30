# Sulphur-2 RunPod Serverless (ComfyUI + GGUF)

Text-to-video and image-to-video generation using
[Sulphur-2](https://huggingface.co/SulphurAI/Sulphur-2-base) on RunPod Serverless.

## Quick Links

- **[HF Model Repo](https://huggingface.co/Floppyshy/sulphur-2-runpod)** — all model files (GGUF, LoRA, VAE, text encoder)
- **[Workflow Diagrams](workflow-diagram.html)** — architecture, cold start, inference flow, node graph
- **[Deployment Plan](PLAN.md)** — full deployment plan and configuration

## Architecture

- **Docker image:** ComfyUI + ComfyUI-GGUF + ComfyUI-LTXVideo + KJNodes + VHS
- **Models:** Stored on HuggingFace, cached to workers via RunPod Model Cache
- **GPU:** RTX 4090 (24 GB) for Q4_K_M, or L40S (48 GB) for higher quants
- **Cold start:** ~5-10 seconds (models pre-cached on worker)

## Files

- `Dockerfile` — Docker image
- `start-wrapper.sh` — symlinks models from HF cache into ComfyUI at boot
- `workflow-sulphur2-t2v.json` — ComfyUI workflow for text-to-video
- `PLAN.md` — full deployment plan
- `download-models.sh` — script to populate the HF repo

## API Usage

POST to your RunPod endpoint:

```json
{
  "input": {
    "workflow": <workflow-sulphur2-t2v.json contents>
  }
}
```

Customize the prompt by replacing the text widget in node 4 (positive prompt) and node 5 (negative prompt).

## Inference Settings

- LoRA strength: 0.65
- Sampling steps: 6–8
- CFG: 1.5–2.0
- Default resolution: 480p (768×512)
- Default frames: 65

## HF Repo Contents

| File | Size | Purpose |
|------|------|---------|
| `sulphur-2-base-Q4_K_M.gguf` | 13 GB | Quantized diffusion model |
| `sulphur_lora_rank_768.safetensors` | 10 GB | Distill LoRA |
| `LTX23_video_vae_bf16.safetensors` | 1.4 GB | Video VAE |
| `LTX23_audio_vae_bf16.safetensors` | 348 MB | Audio VAE |
| `taeltx2_3.safetensors` | 22 MB | Tiny upscaler VAE |
| `text_encoder/` | 12 GB | Gemma-3-12B FP8 scaled |
| `tokenizer/` | — | Tokenizer configs |
| `prompt_enhancer/` | 6 GB | Prompt enhancer GGUF |
