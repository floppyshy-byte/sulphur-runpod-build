"""
Model Cache diagnostic handler.
Tests whether Model Cache downloads HF repos and where files land.
All results returned in API response — no need to check logs.
"""
import os
import json
import sys

CACHE_ROOT = "/runpod-volume/huggingface-cache/hub"


def _walk(path, max_depth=5, depth=0):
    """Recursive directory listing up to max_depth."""
    if depth > max_depth:
        return [f"{'  '*depth}..."]
    lines = []
    try:
        entries = sorted(os.listdir(path))
    except Exception as e:
        return [f"{'  '*depth}[error: {e}]"]
    for entry in entries:
        full = os.path.join(path, entry)
        if os.path.isdir(full):
            lines.append(f"{'  '*depth}{entry}/")
            lines.extend(_walk(full, max_depth, depth + 1))
        elif os.path.islink(full):
            target = os.readlink(full)
            lines.append(f"{'  '*depth}{entry} -> {target}")
        else:
            size = os.path.getsize(full)
            lines.append(f"{'  '*depth}{entry}  ({size:,} bytes)")
    return lines


def handler(event):
    result = {}

    # 1. Does cache root exist?
    result["cache_root"] = CACHE_ROOT
    result["cache_root_exists"] = os.path.isdir(CACHE_ROOT)

    # 2. Full directory tree
    if result["cache_root_exists"]:
        result["cache_tree"] = "\n".join(_walk(CACHE_ROOT))
    else:
        result["cache_tree"] = "(cache root does not exist)"

    # 3. /runpod-volume top-level listing
    rpv = "/runpod-volume"
    result["runpod_volume_top"] = sorted(os.listdir(rpv)) if os.path.isdir(rpv) else "MISSING"

    # 4. Any huggingface-cache at all?
    hfc = "/runpod-volume/huggingface-cache"
    result["hfc_exists"] = os.path.isdir(hfc)
    if result["hfc_exists"]:
        result["hfc_top"] = sorted(os.listdir(hfc))

    # 5. Relevant env vars
    result["env"] = {
        k: v for k, v in sorted(os.environ.items())
        if any(t in k.lower() for t in ["hf", "cache", "model", "transformers", "offline", "home", "runpod"])
    }

    # 6. Disk usage
    try:
        import shutil
        du = shutil.disk_usage("/runpod-volume")
        result["disk_runpod_volume"] = {
            "total_gb": round(du.total / 1e9, 1),
            "used_gb": round(du.used / 1e9, 1),
            "free_gb": round(du.free / 1e9, 1),
        }
    except Exception:
        pass

    return result


# Dump summary at startup for logs too
print(f"[ModelCacheTest] Cache root ({CACHE_ROOT}) exists: {os.path.isdir(CACHE_ROOT)}")
if os.path.isdir(CACHE_ROOT):
    top = sorted(os.listdir(CACHE_ROOT))
    print(f"[ModelCacheTest] Cache top-level: {top}")

# This handler doesn't need runpod.serverless.start — it's a raw handler
# that we'll use with a minimal wrapper. But we need runpod to receive jobs.
if __name__ == "__main__":
    import runpod
    print("[ModelCacheTest] Starting RunPod handler...")
    runpod.serverless.start({"handler": handler})
