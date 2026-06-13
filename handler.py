#!/usr/bin/env python3
"""
Custom RunPod handler for Sulphur-2 ComfyUI workflows.

All models are pre-cached via RunPod model caching from HuggingFace repos
and symlinked into place by start-wrapper.sh before ComfyUI starts.

Encryption: AES-256-GCM via COMFY_ENCRYPTION_KEY env var (64 hex chars).
Supports encrypted_prompt injection into CLIPTextEncode nodes and encrypted
output images.

Input format:
{
  "workflow": { ...ComfyUI API node graph... },
  "images": [{"name": "input.png", "image": "<base64>"}],
  "encryption": true,                                  // optional
  "encrypted_prompt": "<base64>"                       // optional
}
"""

import base64
import json
import os
import secrets
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

import tempfile

import runpod
from runpod.serverless.utils import rp_upload

COMFYUI_URL = "http://127.0.0.1:8188"

# AES-256-GCM encryption — enabled when COMFY_ENCRYPTION_KEY is set (64 hex chars)
_ENCRYPTION_KEY: bytes | None = None
_RAW_KEY = os.getenv("COMFY_ENCRYPTION_KEY", "")
if _RAW_KEY:
    _key_bytes = bytes.fromhex(_RAW_KEY)
    if len(_key_bytes) != 32:
        raise RuntimeError(
            "COMFY_ENCRYPTION_KEY must be 64 hex characters (32 bytes)"
        )
    _ENCRYPTION_KEY = _key_bytes


def _aes_decrypt(encoded: str) -> bytes:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    data = base64.b64decode(encoded)
    nonce, ciphertext = data[:12], data[12:]
    return AESGCM(_ENCRYPTION_KEY).decrypt(nonce, ciphertext, None)


def _aes_encrypt_bytes(plaintext: bytes) -> bytes:
    """Encrypt plaintext bytes and return raw nonce + ciphertext + tag."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    nonce = secrets.token_bytes(12)
    ciphertext = AESGCM(_ENCRYPTION_KEY).encrypt(nonce, plaintext, None)
    return nonce + ciphertext


def _aes_encrypt(plaintext: bytes) -> str:
    """Encrypt plaintext bytes and return base64(nonce + ciphertext + tag)."""
    return base64.b64encode(_aes_encrypt_bytes(plaintext)).decode()


def _s3_configured() -> bool:
    """Return True if the base image's S3 uploader is configured."""
    return bool(os.environ.get("BUCKET_ENDPOINT_URL"))


def _upload_to_s3(data: bytes, filename: str, job_id: str) -> str:
    """Upload bytes via the base image's S3 helper and return the URL."""
    suffix = os.path.splitext(filename)[1] or ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        return rp_upload.upload_image(job_id, tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def _wait_for_comfyui(timeout: int = 120) -> None:
    """Block until ComfyUI's HTTP API responds or timeout expires."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{COMFYUI_URL}/system_stats", timeout=3)
            return
        except Exception:
            time.sleep(2)
    print(
        "[handler] FATAL: ComfyUI did not start within %ds, killing container" % timeout
    )
    os._exit(1)


def _upload_image(name: str, image_b64: str) -> None:
    """Upload a base64-encoded image to ComfyUI's input directory."""
    img_bytes = base64.b64decode(image_b64)
    boundary = "runpod-upload-boundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{name}"\r\n'
        f"Content-Type: image/png\r\n\r\n"
    ).encode() + img_bytes + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        f"{COMFYUI_URL}/upload/image",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        resp.read()


def _queue_prompt(workflow: dict) -> str:
    """Submit a ComfyUI workflow and return the prompt_id."""
    data = json.dumps({"prompt": workflow}).encode()
    req = urllib.request.Request(
        f"{COMFYUI_URL}/prompt",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())["prompt_id"]
    except Exception as exc:
        exc_type = type(exc).__name__
        if hasattr(exc, "code") and hasattr(exc, "read"):
            body = exc.read().decode(errors="replace")
            raise RuntimeError(
                f"ComfyUI /prompt returned {exc.code}: {body}"
            ) from exc
        raise RuntimeError(f"ComfyUI /prompt error [{exc_type}]: {exc}") from exc


def _poll_history(prompt_id: str, timeout: int = 600) -> dict:
    """Poll ComfyUI history until prompt completes; return the history entry."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                f"{COMFYUI_URL}/history/{prompt_id}", timeout=10
            ) as resp:
                history = json.loads(resp.read())
            if prompt_id in history:
                return history[prompt_id]
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"Prompt {prompt_id} did not complete within {timeout}s")


def _fetch_output_info(result: dict) -> list:
    """Extract output image info (filename, subfolder, type) from ComfyUI result."""
    output_images = []
    for node_output in result.get("outputs", {}).values():
        for img_info in node_output.get("images", []):
            output_images.append(img_info)
    return output_images


def _fetch_image_bytes(filename: str, subfolder: str, folder_type: str) -> bytes:
    """Fetch an output image/video from ComfyUI and return raw bytes."""
    params = urllib.parse.urlencode(
        {"filename": filename, "subfolder": subfolder, "type": folder_type}
    )
    with urllib.request.urlopen(f"{COMFYUI_URL}/view?{params}", timeout=30) as resp:
        return resp.read()


# ---------------------------------------------------------------------------
# RunPod handler
# ---------------------------------------------------------------------------


def handler(job: dict) -> dict:
    job_input = job.get("input", {})

    # Encryption mode: triggered by "encryption": true or legacy "encrypted" field
    encryption_enabled = (
        job_input.get("encryption") is True or "encrypted" in job_input
    )
    was_fully_encrypted = "encrypted" in job_input

    if encryption_enabled and not _ENCRYPTION_KEY:
        return {
            "error": "Encryption is enabled but COMFY_ENCRYPTION_KEY is not set"
        }

    # Decrypt full payload if it arrived in the legacy "encrypted" field
    if was_fully_encrypted:
        try:
            job_input = json.loads(_aes_decrypt(job_input["encrypted"]))
        except Exception as exc:
            return {"error": f"Failed to decrypt job input: {exc}"}

    workflow = job_input.get("workflow")
    images = job_input.get("images") or []

    if not workflow:
        return {"error": "No workflow provided"}

    # Decrypt encrypted_prompt and inject into CLIPTextEncode nodes
    encrypted_prompt = job_input.get("encrypted_prompt")
    if encryption_enabled and encrypted_prompt and _ENCRYPTION_KEY:
        try:
            decrypted_prompt = _aes_decrypt(encrypted_prompt).decode("utf-8")
            injected = False
            for node in workflow.values():
                if node.get("class_type") == "CLIPTextEncode":
                    node["inputs"]["text"] = decrypted_prompt
                    injected = True
            if not injected:
                return {
                    "error": "encrypted_prompt provided but no CLIPTextEncode node found in workflow"
                }
        except Exception as exc:
            return {"error": f"Failed to decrypt prompt: {exc}"}

    # Upload input images
    for img in images:
        try:
            _upload_image(img["name"], img["image"])
        except Exception as exc:
            return {"error": f"Failed to upload image {img['name']}: {exc}"}

    # Queue the workflow prompt
    try:
        prompt_id = _queue_prompt(workflow)
    except Exception as exc:
        return {"error": f"Failed to queue prompt: {exc}"}

    # Wait for completion
    try:
        result = _poll_history(prompt_id)
    except TimeoutError as exc:
        return {"error": str(exc)}

    # Collect output images, encrypting if enabled and uploading to S3/R2
    job_id = job.get("id") or str(uuid.uuid4())
    output_images = []
    for img_info in _fetch_output_info(result):
        filename = img_info["filename"]
        try:
            file_bytes = _fetch_image_bytes(
                filename,
                img_info.get("subfolder", ""),
                img_info.get("type", "output"),
            )
        except Exception as exc:
            return {"error": f"Failed to fetch output {filename}: {exc}"}

        out_item = {"filename": filename}
        if encryption_enabled:
            file_bytes = _aes_encrypt_bytes(file_bytes)
            out_item["encrypted"] = True

        if _s3_configured():
            try:
                out_item["data"] = _upload_to_s3(file_bytes, filename, job_id)
                out_item["type"] = "s3_url"
            except Exception as exc:
                return {"error": f"Failed to upload {filename} to S3: {exc}"}
        else:
            # Inline data URI fallback when S3 is not configured
            mime = "video/mp4" if filename.endswith(".mp4") else "image/png"
            out_item["data"] = f"data:{mime};base64," + base64.b64encode(file_bytes).decode()
            out_item["type"] = "base64"

        output_images.append(out_item)

    if not output_images:
        return {"error": "ComfyUI returned no output images"}

    return {"images": output_images}


if __name__ == "__main__":
    print("[handler] Waiting for ComfyUI...")
    _wait_for_comfyui()
    print("[handler] ComfyUI ready — starting RunPod serverless handler")
    runpod.serverless.start({"handler": handler})
