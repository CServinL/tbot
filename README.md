# TBot
## A multimodal AI bot for your terminal.

Powered by local HTTP and MCP daemons for text and image generation.


## Diffusion HTTP/MCP Server

### Popular realistic models that work well with diffusers:
- poetry run python diffusiond/main.py --model "runwayml/stable-diffusion-v1-5"          # Standard SD 1.5
- poetry run python diffusiond/main.py --model "stabilityai/stable-diffusion-2-1"        # SD 2.1  
- poetry run python diffusiond/main.py --model "dreamlike-art/dreamlike-photoreal-2.0"   # Photorealistic
- poetry run python diffusiond/main.py --model "Lykon/DreamShaper"                       # Popular community model
- poetry run python diffusiond/main.py --model "SG161222/RealVisXL_V4.0" --precision fp32 --attention-precision fp32
- poetry run python diffusiond/main.py --model "Lykon/dreamshaper-8" --precision fp16 --device cuda


### Switching Models

#### Start with SDXL
#### Your current model: SG161222/RealVisXL_V4.0

#### Switch to SD 1.5 for faster generation
curl -X POST http://127.0.0.1:8000/switch-model \
  -H "Content-Type: application/json" \
  -d '{"model_id": "runwayml/stable-diffusion-v1-5", "precision": "fp16"}'

#### Generate with SD 1.5 (512x512 default)
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a beautiful woman", "width": 512, "height": 512}'

#### Switch back to SDXL for higher quality
curl -X POST http://127.0.0.1:8000/switch-model \
  -H "Content-Type: application/json" \
  -d '{"model_id": "SG161222/RealVisXL_V4.0"}'

curl -X POST http://127.0.0.1:8000/switch-model \
  -H "Content-Type: application/json" \
  -d '{"model_id": "Lykon/DreamShaper", "precision": "fp32"}'
