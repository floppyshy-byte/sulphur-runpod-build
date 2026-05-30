# Sulphur-2 ComfyUI GGUF — RunPod Serverless

This runs Sulphur-2 video generation on RunPod Serverless using ComfyUI with
GGUF-quantized weights. The GGUF format lets us use cheaper GPUs (RTX 4090 24GB
or L40S 48GB) instead of an A100 80GB.

## Architecture

```
RunPod Serverless (stateless container)
  → ComfyUI + ComfyUI-GGUF + ComfyUI-LTXVideo
  → Network Volume (/runpod-volume) holds model weights
```

The Docker image is small (~6-8 GB) — no models baked in. Models live on a
persistent Network Volume, symlinked at boot.

## Quickstart

### 1. Build and push the image

```bash
docker build -f Dockerfile.comfyui -t ghcr.io/floppyshy-byte/sulphur-runpod-comfyui:latest .
docker push ghcr.io/floppyshy-byte/sulphur-runpod-comfyui:latest
```

### 2. Populate the Network Volume

Spin up a temporary GPU Pod with a Network Volume attached, then:

```bash
# Download Sulphur-2 GGUF (Q4_K_M is a good starting point)
mkdir -p /workspace/models/unet
wget -P /workspace/models/unet \
  https://huggingface.co/Abiray/Sulphur-2-base-GGUF/resolve/main/sulphur_dev-Q4_K_M.gguf

# Download Gemma-3-12B text encoder
mkdir -p /workspace/models/text_encoders/gemma-3-12b-it
huggingface-cli download unsloth/gemma-3-12b-it \
  --local-dir /workspace/models/text_encoders/gemma-3-12b-it

# Download LTX VAE (from official LTX-Video repo)
# Place in /workspace/models/vae/
```

### 3. Create the RunPod Serverless Endpoint

- Template: the Docker image from step 1
- GPU: RTX 4090 (24GB) for Q3/Q4 quants, L40S (48GB) for Q5+
- Network Volume: attach the volume from step 2
- Container Disk: 20 GB
- Execution Timeout: 600 seconds (video generation takes time)

### 4. Call the API

```json
{
  "input": {
    "workflow": {
      "35": {
        "inputs": {
          "unet_name": "sulphur_dev-Q4_K_M.gguf"
        },
        "class_type": "UnetLoaderGGUF"
      },
      "6": {
        "inputs": {
          "text": "A cat walking on a sunlit sidewalk",
          "clip": ["35", 1]
        },
        "class_type": "CLIPTextEncode"
      }
    }
  }
}
```

The handler returns base64-encoded video frames or S3 URLs (if BUCKET_ENDPOINT_URL is configured).

## GPU / Quant Selection

| Quant | File Size | Min GPU | Cost/hr |
|---|---|---|---|
| Q3_K_M | 11.1 GB | RTX 4090 (24 GB) | ~$1.12 |
| Q4_K_M | 14.3 GB | RTX 4090 (24 GB) | ~$1.12 |
| Q5_K_M | 16.1 GB | L40S (48 GB) | ~$0.90 |
| Q6_K | 17.8 GB | L40S (48 GB) | ~$0.90 |
| Q8_0 | 22.8 GB | L40S (48 GB) | ~$0.90 |

Higher quants = better quality, higher quants also need more VRAM for activations.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `COMFY_LOG_LEVEL` | `DEBUG` | ComfyUI log verbosity |
| `REFRESH_WORKER` | `false` | Refresh worker between jobs |
| `BUCKET_ENDPOINT_URL` | — | S3 endpoint for storing generated videos |
| `COMFY_ORG_API_KEY` | — | Comfy.org API key for API nodes |

## Differences from the diffusers approach

| | diffusers (old) | ComfyUI + GGUF (new) |
|---|---|---|
| Docker image | ~12 GB | ~6 GB |
| Min GPU | A100 80GB | RTX 4090 24GB |
| Cost/hr | ~$2.74 | ~$1.12 |
| Model format | Single safetensors (FP8mixed) | GGUF (quants Q3-Q8) |
| API payload | Simple {prompt, width, frames} | ComfyUI workflow JSON |
| Custom code | handler.py + loader.py | Just Dockerfile + start wrapper |
