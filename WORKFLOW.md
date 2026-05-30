# Sulphur-2 ComfyUI + GGUF — Architecture & Workflow

## System Architecture

```mermaid
flowchart TB
    subgraph Client["🖥️ Client"]
        A[API POST<br/>workflow JSON + prompt]
    end

    subgraph RunPod["☁️ RunPod Serverless"]
        subgraph Container["📦 Stateless Container"]
            B[handler.py<br/>RunPod worker]
            C[start-wrapper.sh<br/>symlink models]
            D[ComfyUI<br/>localhost:8188]
            E[custom_nodes/<br/>GGUF, LTXVideo,<br/>KJNodes, VHS]
        end

        subgraph Cache["💾 RunPod Model Cache"]
            F[(HuggingFace Cache<br/>Floppyshy/<br/>sulphur-2-runpod)]
        end
    end

    subgraph HF["🤗 HuggingFace"]
        G[Floppyshy/<br/>sulphur-2-runpod<br/>~43 GB]
    end

    A -->|"POST /run"| B
    C -->|symlinks| F
    B -->|"queues workflow"| D
    D --> E
    D -->|reads models from| F
    G -->|"cached at deploy"| F
    B -->|"returns frames/video"| A
```

## Cold Start Sequence

```mermaid
sequenceDiagram
    participant R as RunPod
    participant W as Worker Container
    participant S as start-wrapper.sh
    participant C as ComfyUI
    participant H as HF Cache

    R->>W: Start container
    W->>S: CMD /start-wrapper.sh
    S->>H: Find snapshot in<br/>models--Floppyshy--sulphur-2-runpod
    S->>S: Symlink GGUF → models/unet/
    S->>S: Symlink LoRA → models/loras/
    S->>S: Symlink VAEs → models/vae/
    S->>S: Symlink text_encoder → models/text_encoders/
    S->>S: Symlink prompt_enhancer → models/prompt_enhancer/
    S->>W: exec /start.sh
    W->>C: Launch ComfyUI (python main.py)
    C->>C: Load custom nodes
    C-->>R: Health check OK
    R->>W: Job ready
```

## Inference Flow

```mermaid
sequenceDiagram
    participant Client
    participant Handler as handler.py
    participant Comfy as ComfyUI :8188
    participant GPU

    Client->>Handler: POST {input: {workflow: {...}}}
    Handler->>Handler: Validate workflow JSON
    Handler->>Comfy: Check health (GET /)
    Comfy-->>Handler: OK
    Handler->>Comfy: Upload images (if i2v)
    Handler->>Comfy: POST /prompt {workflow}
    Comfy->>GPU: Load GGUF (UnetLoaderGGUF)
    Comfy->>GPU: Load LoRA (LoraLoader, 0.65)
    Comfy->>GPU: Load text encoder (Gemma FP8)
    Comfy->>GPU: Load VAE (LTX23_video_vae_bf16)
    Comfy->>GPU: Encode prompt → condition
    Comfy->>GPU: Sample latents (8 steps, CFG 1.7)
    Comfy->>GPU: Decode latents → frames (VAE Decode)
    Comfy-->>Handler: WebSocket: executing... done
    Handler->>Comfy: GET /history/{prompt_id}
    Comfy-->>Handler: Output images
    Handler-->>Client: {images: [{filename, data: base64}]}
```

## ComfyUI Node Graph

```mermaid
flowchart LR
    subgraph Input["📥 Input"]
        P[Prompt<br/>'a cat walking']
    end

    subgraph Loaders["📦 Model Loaders"]
        GGUF[UnetLoaderGGUF<br/>Q4_K_M.gguf]
        LORA[LoraLoader<br/>distill loRA<br/>strength: 0.65]
        TE[Gemma Text Encoder<br/>FP8 scaled]
        VAE[VAE Loader<br/>LTX23 video VAE]
    end

    subgraph Pipeline["⚙️ LTX Pipeline"]
        ENCODE[CLIPTextEncode]
        SAMPLE[KSampler / LTX Sampler<br/>8 steps, CFG 1.7<br/>euler + normal]
        DECODE[VAE Decode]
    end

    subgraph Output["📤 Output"]
        FRAMES[Video Frames<br/>base64 or MP4]
    end

    P --> ENCODE
    TE --> ENCODE
    GGUF --> LORA
    LORA --> SAMPLE
    ENCODE -->|conditioning| SAMPLE
    SAMPLE -->|latents| DECODE
    VAE --> DECODE
    DECODE --> FRAMES
```

## File Layout

```mermaid
flowchart TB
    subgraph HFRepo["🤗 Floppyshy/sulphur-2-runpod"]
        direction LR
        U[sulphur-2-base-Q4_K_M.gguf<br/>13 GB]
        L[sulphur_lora_rank_768.safetensors<br/>10 GB]
        VV[LTX23_video_vae_bf16.safetensors<br/>1.4 GB]
        AV[LTX23_audio_vae_bf16.safetensors<br/>348 MB]
        TV[taeltx2_3.safetensors<br/>22 MB]
        TE[text_encoder/<br/>config + FP8 scaled<br/>12 GB]
        TK[tokenizer/<br/>10 files]
        PE[prompt_enhancer/<br/>Q4_K_M + mmproj<br/>6 GB]
    end

    subgraph ComfyDir["📁 /comfyui/models/"]
        direction LR
        MU[unet/ → GGUF]
        ML[loras/ → safetensors]
        MV[vae/ → safetensors]
        MT[text_encoders/ → Gemma]
        MP[prompt_enhancer/ → GGUF]
    end

    HFRepo -->|"start-wrapper.sh<br/>symlinks"| ComfyDir
```
