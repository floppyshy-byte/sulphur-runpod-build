"""
RunPod Serverless handler wrapper for Sulphur-2.

Decrypts an encrypted input payload (shape: {input: {encrypted: ...}}) before
delegrating to the upstream runpod/worker-comfyui handler. The upstream handler
is expected to be available as handler_base (the original /handler.py renamed
in the Docker image).

The COMFY_ENCRYPTION_KEY env var must match the key used by the Flask backend.
"""

import base64
import json
import os
import sys

import runpod

# The upstream worker-comfyui handler lives at /handler_base.py after the
# Dockerfile renames the original /handler.py.
sys.path.insert(0, "/")
from handler_base import handler as _base_handler


# ---------------------------------------------------------------------------
# AES-256-GCM decryption — matches vid-web-base crypto.py wire format:
# base64( nonce(12 bytes) + ciphertext + GCM tag(16 bytes) )
# ---------------------------------------------------------------------------
_RAW_KEY = os.environ.get("COMFY_ENCRYPTION_KEY", "")
_ENCRYPTION_KEY: bytes | None = None
if _RAW_KEY:
    _key_bytes = bytes.fromhex(_RAW_KEY)
    if len(_key_bytes) != 32:
        raise RuntimeError(
            "COMFY_ENCRYPTION_KEY must be 64 hex characters (32 bytes)"
        )
    _ENCRYPTION_KEY = _key_bytes


def _decrypt_string(encoded: str) -> str:
    """Decrypt a base64(nonce+ciphertext+tag) string, returning plaintext."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    if _ENCRYPTION_KEY is None:
        raise RuntimeError("COMFY_ENCRYPTION_KEY is not configured")

    data = base64.b64decode(encoded)
    nonce, ciphertext = data[:12], data[12:]
    plaintext = AESGCM(_ENCRYPTION_KEY).decrypt(nonce, ciphertext, None)
    return plaintext.decode("utf-8")


def handler(job):
    """Wrap the upstream handler with input decryption."""
    job_input = job.get("input", {})

    if "encrypted" in job_input:
        try:
            decrypted_json = _decrypt_string(job_input["encrypted"])
            job["input"] = json.loads(decrypted_json)
        except Exception as exc:
            return {
                "error": f"Failed to decrypt RunPod input: {exc}",
            }

    return _base_handler(job)


if __name__ == "__main__":
    print("sulphur-handler - Starting encrypted input wrapper...")
    runpod.serverless.start({"handler": handler})
