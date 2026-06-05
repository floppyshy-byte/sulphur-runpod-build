# Sulphur-2 ComfyUI FP8 — RunPod Serverless

This runs Sulphur-2 video generation on RunPod Serverless using ComfyUI with
the FP8 distilled checkpoint. The monolithic safetensors file includes the
transformer, VAE, and distilled LoRA weights all in one.

## Architecture

```
RunPod Serverless (stateless container)
  → ComfyUI + ComfyUI-LTXVideo + ComfyUI-KJNodes
  → Network Volume (/runpod-volume) holds model weights
```

The Docker image is small (~6-8 GB) — no models baked in. Models live on a
persistent Network Volume, symlinked at boot.

## Quickstart

### 1. Build and push the image

```bash
docker build -f Dockerfile -t ghcr.io/floppyshy-byte/sulphur-runpod-fp8:latest .
docker push ghcr.io/floppyshy-byte/sulphur-runpod-fp8:latest
```

### 2. Populate the Network Volume

Spin up a temporary GPU Pod with a Network Volume attached, then:

```bash
# Download Sulphur-2 FP8 checkpoint
mkdir -p /workspace/models/checkpoints
huggingface-cli download Civitai/Sulphur-2-distilled-fp8 \
  --local-dir /workspace/models/checkpoints \
  --include "*.safetensors"

# Download Gemma-3-12B text encoder
mkdir -p /workspace/models/text_encoders
huggingface-cli download Floppyshy/sulphur-2-runpod \
  --local-dir /workspace/models/text_encoders \
  --include "text_encoder/*"

# Download LTX connector
huggingface-cli download Floppyshy/sulphur-2-runpod \
  --local-dir /workspace/models/clip \
  --include "*connector*.safetensors"
```

### 3. Create the RunPod Serverless Endpoint

- Template: the Docker image from step 1
- GPU: **L40S (48 GB)** — RTX 4090 (24 GB) will OOM
- Network Volume: attach the volume from step 2
- Container Disk: 20 GB
- Execution Timeout: 600 seconds

### 4. Call the API

```json
{
  "input": {
    "workflow": {
      "1": {
        "inputs": {
          "ckpt_name": "sulphur_distil_fp8mixed.safetensors"
        },
        "class_type": "CheckpointLoaderSimple"
      },
      "4": {
        "inputs": {
          "text": "A cat walking on a sunlit sidewalk",
          "clip": ["3", 0]
        },
        "class_type": "CLIPTextEncode"
      }
    }
  }
}
```

The handler returns base64-encoded video frames or S3 URLs (if BUCKET_ENDPOINT_URL is configured).

## GPU Requirements

| Setup | File Size | Min GPU | Cost/hr |
|---|---|---|---|
| FP8 distilled | 29 GB | L40S (48 GB) | ~$0.90 |

The FP8 mixed checkpoint requires ~29 GB of VRAM just for the model weights.
An RTX 4090 (24 GB) will OOM during inference.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `COMFY_LOG_LEVEL` | `DEBUG` | ComfyUI log verbosity |
| `REFRESH_WORKER` | `false` | Refresh worker between jobs |
| `BUCKET_ENDPOINT_URL` | — | S3 endpoint for storing generated videos |
| `COMFY_ORG_API_KEY` | — | Comfy.org API key for API nodes |

## Differences from the old GGUF approach

| | Old (GGUF) | New (FP8 checkpoint) |
|---|---|---|
| Docker image | ~6 GB | ~6 GB |
| Min GPU | RTX 4090 (24 GB) | L40S (48 GB) |
| Cost/hr | ~$1.12 | ~$0.90 |
| Model format | GGUF (Q4_K_M) + LoRA + VAE | Single FP8 safetensors |
| API payload | ComfyUI workflow JSON | ComfyUI workflow JSON |
| Custom code | Dockerfile + start wrapper | Dockerfile + start wrapper |
