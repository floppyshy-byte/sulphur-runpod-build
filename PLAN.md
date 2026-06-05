# Sulphur-2 ComfyUI + FP8 — Deployment Plan

## Architecture

```
RunPod Serverless Container (stateless)
├── ComfyUI + ComfyUI-LTXVideo + KJNodes
├── start-wrapper.sh → symlinks models from HF cache into ComfyUI dirs
└── handler.py → receives ComfyUI workflow JSON, returns video frames

HuggingFace Repos (cached via RunPod Model Cache)
├── Civitai/Sulphur-2-distilled-fp8
│   └── sulphur_distil_fp8mixed.safetensors           ~29 GB
└── Floppyshy/sulphur-2-runpod
    ├── text_encoder/gemma_3_12B_it_fp8_scaled.safetensors  ~12 GB
    ├── ltx-2.3-22b-distilled_embeddings_connectors.safetensors  ~6 GB
    └── tokenizer/                                      ~4 MB

RunPod Model Cache →
  /runpod-volume/huggingface-cache/hub/
    models--Civitai--Sulphur-2-distilled-fp8/snapshots/<hash>/
    models--Floppyshy--sulphur-2-runpod/snapshots/<hash>/
```

No network volume — everything is in HF repos, cached to workers via RunPod Model Cache.

## Docker Image

Base: `runpod/worker-comfyui:5.8.5-base`

Custom nodes installed at build time:
- `Lightricks/ComfyUI-LTXVideo` — LTX pipeline nodes (scheduler, sampler)
- `Kijai/ComfyUI-KJNodes` — LTX-2.3 VAE loader, audio VAE support
- `Kosinkadink/ComfyUI-VideoHelperSuite` — video I/O (VHS)

`start-wrapper.sh` runs before `start.sh` to symlink models from all HF cache repos into ComfyUI directories.

## RunPod Model Cache Configuration

Configure **both** repos in your RunPod endpoint Model Cache settings:

1. `Civitai/Sulphur-2-distilled-fp8` — primary checkpoint
2. `Floppyshy/sulphur-2-runpod` — text encoder + connector + tokenizer

`start-wrapper.sh` automatically discovers and symlinks from all cached repos.

## GPU Selection

| Setup | File | Min GPU | Cost/hr | Use case |
|---|---|---|---|---|
| FP8 distilled | 29 GB | L40S (48 GB) | ~$0.90 | Full quality, distill baked in |

**RTX 4090 (24 GB) is insufficient** for the 29 GB FP8 checkpoint. L40S (48 GB) or A100 (80 GB) required.

## Inference Settings (Distilled FP8)

- Sampling steps: 6–8
- CFG: 1.5–2.0
- Resolution: 480p default
- No separate LoRA — distill is baked into the checkpoint

## Workflow Nodes

| Node | File | class_type | Output Used |
|---|---|---|---|
| Checkpoint | `sulphur_distil_fp8mixed.safetensors` | `CheckpointLoaderSimple` | `[0]` MODEL, `[2]` VAE |
| Text Encoder | `gemma_3_12B_it_fp8_scaled.safetensors` | `DualCLIPLoaderGGUF` | `[0]` CLIP |
| Connector | `ltx-2.3-22b-distilled_embeddings_connectors.safetensors` | `DualCLIPLoaderGGUF` | `[0]` CLIP |

## Build & Deploy Steps

1. Build and push Docker image (RunPod native build pointing at Dockerfile)
2. Configure RunPod Model Cache with both HF repos
3. Create RunPod serverless endpoint (L40S 48 GB)
4. Write and test ComfyUI workflow JSON
5. Submit test job, verify video output
