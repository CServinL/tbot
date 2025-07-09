from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from conductor.model_loader import ModelLoader
import threading
import time

model_loader = ModelLoader(
    cache_dir=..., device=..., torch_dtype=..., is_sdxl=..., logger=...
)
model_lock = threading.Lock()
device_busy = {}  # device string -> busy flag

app = FastAPI()

@app.post("/generate")
async def generate(request: Request):
    data = await request.json()
    prompt = data.get("prompt")
    # ...other params...
    model_id = "Lykon/dreamshaper-8"  # Or load from SETTINGS.md

    # Determine device key (e.g., "cuda", "cuda:0", "cpu")
    device_key = str(getattr(model_loader, "device", "cpu"))

    wait_start = time.time()
    max_wait = 900  # 15 minutes in seconds
    while True:
        acquired = model_lock.acquire(timeout=600)
        if not acquired:
            return JSONResponse({"error": "Server busy. Try again later."}, status_code=503)
        try:
            if device_busy.get(device_key, False):
                model_lock.release()
                if time.time() - wait_start > max_wait:
                    return JSONResponse({"error": "Device busy timeout. Try again later."}, status_code=504)
                time.sleep(0.2)
                continue
            # Set busy flag for this device
            device_busy[device_key] = True
        finally:
            if model_lock.locked():
                model_lock.release()
        break

    try:
        pipeline = None
        try:
            # Wait for model to be ready, up to 5 minutes
            model_wait_start = time.time()
            model_max_wait = 300  # 5 minutes
            while True:
                pipeline = model_loader.ensure_model_loaded(model_id)
                if pipeline is not None:
                    break
                if time.time() - model_wait_start > model_max_wait:
                    device_busy[device_key] = False
                    return JSONResponse({"error": "Model loading timed out. Try again later."}, status_code=504)
                time.sleep(0.2)
        except Exception as e:
            device_busy[device_key] = False
            return JSONResponse({"error": f"Failed to load model: {str(e)}"}, status_code=500)
        # If pipeline is None, treat as a loading failure
        if pipeline is None:
            device_busy[device_key] = False
            return JSONResponse({"error": "Failed to load model: Model loader returned None."}, status_code=500)

        # ...run generation...
        # result = pipeline(prompt, ...)
        # images = ...
        pass  # ...existing code...
    finally:
        model_loader.offload_model()
        device_busy[device_key] = False

    # ...return result...
    # return JSONResponse({"images": images})