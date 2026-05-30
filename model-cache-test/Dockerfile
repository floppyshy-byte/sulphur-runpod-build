FROM runpod/worker-comfyui:5.8.5-base

# Override the base CMD with our diagnostic handler
RUN pip install --no-cache-dir runpod 2>/dev/null || true

COPY handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]
