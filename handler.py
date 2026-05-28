"""
Sulphur-2 RunPod serverless handler.
Supports text-to-video (t2v) and image-to-video (i2v) via Diffusers pipeline.
Model: Civitai/Sulphur-2-distilled-fp8
"""
from __future__ import annotations

import base64
import gc
import os
import tempfile
import time
import traceback
from pathlib import Path

import runpod
import torch
from diffusers import DiffusionPipeline

# ---------------------------------------------------------------------------
# Model loading (lazy, cached globally)
# ---------------------------------------------------------------------------

MODEL_REPO = os.environ.get("SULPHUR_MODEL_REPO", "Civitai/Sulphur-2-distilled-fp8")
_pipe = None


def _recursive_list_dir(path, prefix=""):
    """Return a formatted recursive listing of a directory."""
    lines = []
    try:
        entries = sorted(os.listdir(path))
    except Exception as exc:
        return f"{prefix}[error: {exc}]"
    for e in entries:
        p = os.path.join(path, e)
        if os.path.isdir(p):
            lines.append(f"{prefix}{e}/")
            sub = _recursive_list_dir(p, prefix + "  ")
            if sub:
                lines.append(sub)
        else:
            try:
                size = os.path.getsize(p)
                lines.append(f"{prefix}{e}  ({size:,} bytes)")
            except Exception:
                lines.append(f"{prefix}{e}")
    return "\n".join(lines)


def _find_safetensors(cache_dir: str) -> str | None:
    """Find the first .safetensors file under cache_dir."""
    for root, _dirs, files in os.walk(cache_dir):
        for f in files:
            if f.endswith(".safetensors"):
                return os.path.join(root, f)
    return None


def _find_text_encoder_cache(cache_dir: str) -> str | None:
    """Find a cached Gemma text encoder under cache_dir."""
    for root, dirs, _files in os.walk(cache_dir):
        for d in dirs:
            if "gemma" in d.lower():
                return os.path.join(root, d)
    return None


def get_pipeline():
    global _pipe
    if _pipe is not None:
        return _pipe

    cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface/hub"))
    print(f"[debug] HF_HOME={cache_dir}", flush=True)

    # Deep recursive listing
    listing = _recursive_list_dir(cache_dir)
    print(f"[debug] Cache listing:\n{listing}", flush=True)

    # Find checkpoint
    checkpoint_path = _find_safetensors(cache_dir)
    if not checkpoint_path:
        raise FileNotFoundError(f"No .safetensors found under {cache_dir}")

    print(f"[startup] Found checkpoint: {checkpoint_path}", flush=True)

    # Check for text encoder cache
    te_cache = _find_text_encoder_cache(cache_dir)
    if te_cache:
        print(f"[startup] Found text encoder cache: {te_cache}", flush=True)
    else:
        print("[startup] No text encoder cache found", flush=True)

    t0 = time.perf_counter()

    # Try LTX2Pipeline.from_single_file first
    try:
        from diffusers import LTX2Pipeline

        print("[startup] Loading with LTX2Pipeline.from_single_file ...", flush=True)
        _pipe = LTX2Pipeline.from_single_file(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        _pipe = _pipe.to("cuda")
        print("[startup] LTX2Pipeline.from_single_file succeeded", flush=True)
    except Exception as e1:
        print(f"[startup] LTX2Pipeline.from_single_file failed: {e1}", flush=True)
        # Fallback: DiffusionPipeline.from_single_file
        try:
            print("[startup] Loading with DiffusionPipeline.from_single_file ...", flush=True)
            _pipe = DiffusionPipeline.from_single_file(
                checkpoint_path,
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            )
            _pipe = _pipe.to("cuda")
            print("[startup] DiffusionPipeline.from_single_file succeeded", flush=True)
        except Exception as e2:
            print(f"[startup] DiffusionPipeline.from_single_file failed: {e2}", flush=True)
            raise RuntimeError(
                f"Failed to load checkpoint from {checkpoint_path}. "
                f"LTX2Pipeline error: {e1}; DiffusionPipeline error: {e2}"
            )

    _pipe.set_progress_bar_config(disable=True)
    elapsed = time.perf_counter() - t0
    print(f"[startup] Pipeline loaded in {elapsed:.1f}s", flush=True)
    return _pipe


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_input(b64_data: str, suffix: str) -> str:
    """Write base64-encoded file to temp dir, return path."""
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(base64.b64decode(b64_data))
    tmp.close()
    return tmp.name


def _encode_video_b64(path: str) -> str:
    """Read a file and return base64."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _save_frames_to_mp4(frames, path: str, fps: int = 24) -> None:
    """Save a list of PIL/numpy frames to an MP4 file."""
    import numpy as np

    try:
        import imageio

        frames_np = []
        for f in frames:
            if hasattr(f, "cpu"):
                f = f.cpu().numpy()
            f = np.asarray(f)
            if f.dtype != np.uint8:
                f = (f * 255).astype(np.uint8)
            frames_np.append(f)

        with imageio.get_writer(path, fps=fps, format="FFMPEG", mode="I") as w:
            for frame in frames_np:
                w.append_data(frame)
    except Exception:
        # Fallback: imageio-ffmpeg plugin
        import imageio

        frames_np = []
        for f in frames:
            if hasattr(f, "cpu"):
                f = f.cpu().numpy()
            f = np.asarray(f)
            if f.dtype != np.uint8:
                f = (f * 255).astype(np.uint8)
            frames_np.append(f)

        with imageio.get_writer(path, fps=fps) as w:
            for frame in frames_np:
                w.append_data(frame)


# ---------------------------------------------------------------------------
# RunPod Handler
# ---------------------------------------------------------------------------

def handler(event):
    try:
        input_data = event.get("input", {})

        task = (input_data.get("task", "t2v") or "t2v").strip().lower()
        prompt = input_data.get("prompt", "")
        negative_prompt = input_data.get("negative_prompt", "")
        width = int(input_data.get("width", 768))
        height = int(input_data.get("height", 512))
        num_frames = int(input_data.get("num_frames", 65))
        num_inference_steps = int(input_data.get("num_inference_steps", 30))
        guidance_scale = float(input_data.get("guidance_scale", 3.5))
        seed = int(input_data.get("seed", 42))

        if not prompt:
            return {"error": "prompt is required"}

        if num_frames % 8 != 1:
            num_frames = ((num_frames // 8) * 8) + 1
            if num_frames < 9:
                num_frames = 9

        print(
            f"[inference] task={task} prompt={prompt[:80]!r} "
            f"size={width}x{height} frames={num_frames} "
            f"steps={num_inference_steps} cfg={guidance_scale} seed={seed}",
            flush=True,
        )

        pipe = get_pipeline()
        generator = torch.Generator(device="cuda").manual_seed(seed)

        t0 = time.perf_counter()

        if task in ("i2v", "image_to_video", "image to video"):
            b64_image = input_data.get("input_image_base64", "")
            if not b64_image:
                return {"error": "input_image_base64 required for i2v"}
            from PIL import Image

            image_path = _decode_input(b64_image, ".png")
            image = Image.open(image_path).convert("RGB")
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                width=width,
                height=height,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
            os.unlink(image_path)
        else:
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )

        frames = result.frames[0]
        elapsed = time.perf_counter() - t0

        # Save to temp MP4
        output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        output_path.close()
        _save_frames_to_mp4(frames, output_path.name, fps=24)

        output_b64 = _encode_video_b64(output_path.name)
        os.unlink(output_path.name)

        gc.collect()
        torch.cuda.empty_cache()

        return {
            "output_type": "video",
            "media_type": "video/mp4",
            "output_base64": output_b64,
            "output_filename": "output.mp4",
            "elapsed_seconds": round(elapsed, 3),
            "num_frames": num_frames,
            "seed": seed,
        }

    except Exception as exc:
        traceback.print_exc()
        cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface/hub"))
        debug = {
            "hf_home": cache_dir,
            "cache_exists": os.path.isdir(cache_dir),
            "cache_listing": _recursive_list_dir(cache_dir) if os.path.isdir(cache_dir) else "N/A",
        }
        return {"error": str(exc), "traceback": traceback.format_exc(), "debug": debug}


if __name__ == "__main__":
    print("[Handler] Starting Sulphur-2 RunPod serverless worker...")
    runpod.serverless.start({"handler": handler})
