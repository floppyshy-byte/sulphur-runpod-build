# Sulphur-2 RunPod Serverless (ComfyUI + FP8)

Text-to-video and image-to-video generation using
[Sulphur-2](https://huggingface.co/Civitai/Sulphur-2-distilled-fp8) on RunPod Serverless.

## Quick Links

- **[HF Model Repo (Primary)](https://huggingface.co/Civitai/Sulphur-2-distilled-fp8)** — FP8 checkpoint (transformer + VAE + distill, ~29 GB)
- **[HF Model Repo (Components)](https://huggingface.co/Floppyshy/sulphur-2-runpod)** — text encoder, connector, tokenizer
- **[Workflow Diagrams](workflow-diagram.html)** — architecture, cold start, inference flow, node graph
- **[Deployment Plan](PLAN.md)** — full deployment plan and configuration

## Architecture

- **Docker image:** ComfyUI + ComfyUI-LTXVideo + KJNodes + VHS
- **Models:** Stored on HuggingFace, cached to workers via RunPod Model Cache (multi-repo)
- **GPU:** L40S (48 GB) — RTX 4090 (24 GB) is insufficient for FP8
- **Cold start:** ~5–10 seconds (models pre-cached on worker)

## Files

- `Dockerfile` — Docker image
- `start-wrapper.sh` — symlinks models from one or more HF cache repos into ComfyUI dirs
- `workflow-sulphur2-t2v.json` — ComfyUI workflow for text-to-video
- `workflow-sulphur2-smoke.json` — lightweight smoke-test workflow
- `PLAN.md` — full deployment plan

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

- Sampling steps: 6–8
- CFG: 1.5–2.0
- Default resolution: 480p (768×512)
- Default frames: 65

## HF Repo Contents

### Primary: `Civitai/Sulphur-2-distilled-fp8`

| File | Size | Purpose |
|------|------|---------|
| `sulphur_distil_fp8mixed.safetensors` | 29 GB | FP8 checkpoint (transformer + VAE + distilled) |

### Components: `Floppyshy/sulphur-2-runpod`

| File | Size | Purpose |
|------|------|---------|
| `text_encoder/gemma_3_12B_it_fp8_scaled.safetensors` | 12 GB | Gemma 3 12B text encoder |
| `ltx-2.3-22b-distilled_embeddings_connectors.safetensors` | ~6 GB | Gemma → LTX projection connector |

Both repos should be configured in RunPod Model Cache. `start-wrapper.sh` will symlink files from all cached repos.
