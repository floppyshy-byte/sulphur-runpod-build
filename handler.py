"""
Placeholder handler for RunPod's repo scanner.
The actual handler is inside the ComfyUI base image — our Dockerfile
overrides CMD with start-wrapper.sh, which chains to the base start.sh.
This file exists only to satisfy RunPod's requirement that the repo
contains runpod.serverless.start().
"""
import runpod

# This is never called — start-wrapper.sh replaces the CMD
runpod.serverless.start({"handler": lambda job: {}})
