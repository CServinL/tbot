# TBot
## A multimodal AI bot for your terminal.

TBot is a comprehensive multimodal AI solution that provides both text generation (via Conductor) and image generation (via Diffusiond) capabilities. The system is designed with modularity in mind, offering flexible deployment options through HTTP APIs, MCP (Model Context Protocol) servers, and command-line interfaces.

## Architecture Overview

- **conductor/**: Intelligent LLM routing system with specialized engines for different tasks
- **diffusiond/**: Stable Diffusion image generation server with model switching capabilities  
- **tbot/**: Main command-line tool (work in progress) that will integrate multiple AI sources

---

## Conductor - LLM Text Generation System

Conductor is an intelligent LLM routing system that automatically classifies prompts and routes them to specialized engines optimized for different tasks like code generation, creative writing, mathematical reasoning, and more.

### Features
- **Intelligent Prompt Classification**: Automatically routes prompts to the most appropriate specialized engine
- **Multiple Specialized Engines**: Code generation, completion, review, creative writing, translation, Q&A, math, and more
- **Model Context Protocol (MCP)**: Full MCP server implementation for integration with MCP clients
- **Shared Model Loading**: Efficient memory usage by sharing loaded models across engines
- **Configurable**: Easy configuration through SETTINGS.md files

### Usage

#### Command Line Interface
```bash
# Start interactive CLI with default settings
python conductor/conductor.py

# Use custom config file
python conductor/conductor.py --config conductor/SETTINGS_8gb.md

# Skip model loading for faster startup (classification only)
python conductor/conductor.py --skip-model-loading

# Set logging level
python conductor/conductor.py --log-level DEBUG
```

#### Command Line Options
```
--config PATH              Configuration file path (default: conductor/SETTINGS.md)
--skip-model-loading       Skip loading models at startup
--log-level LEVEL          Set logging level: DEBUG, INFO, WARNING, ERROR (default: INFO)
```

#### MCP Server
Conductor provides a full MCP server implementation that can be used with MCP-compatible clients:

```python
from conductor.mcp_server import MCPServer
from conductor.conductor import Conductor

# Initialize conductor and MCP server
conductor = Conductor()
await conductor.initialize()

# MCP server handles protocol communication automatically
# Supports: completion, generation, streaming, model management, tools, resources
```

#### Single Request Usage
```python
from conductor.conductor import Conductor

conductor = Conductor()
await conductor.initialize()

# Auto-classify and generate
category, response = await conductor.generate("Write a Python function to sort a list")

# Specify category directly
category, response = await conductor.generate("def sort_list(items):", category="code_completion")
```

### Supported Engine Categories
- **code_generation**: Writing new code, functions, classes
- **code_completion**: Completing existing code snippets  
- **code_review**: Reviewing code for bugs, improvements
- **creative_writing**: Stories, poems, creative content
- **translation**: Language translation tasks
- **question_answering**: Factual questions and explanations
- **summarization**: Text summarization and TLDR
- **mathematical_reasoning**: Math problems and calculations
- **conversational_chat**: General conversation and chat
- **general_reasoning**: Catch-all for complex reasoning tasks
- **image_generation**: Routes to diffusiond for image creation

---

## Diffusiond - Image Generation System

Diffusiond is a high-performance Stable Diffusion server with dynamic model switching, optimized for both speed and quality.

### Features
- **Multiple Model Support**: Stable Diffusion 1.5, 2.1, SDXL, and community models
- **Dynamic Model Switching**: Change models without restarting the server
- **Precision Control**: fp16, fp32, bf16 support for speed/quality tradeoffs
- **Memory Optimization**: Lazy loading and efficient model management
- **HTTP API**: RESTful API for easy integration
- **MCP Server**: Model Context Protocol support

### Usage

#### Command Line Interface
```bash
# Start with Realistic Vision model (high quality portraits)
python diffusiond/main.py --model "SG161222/Realistic_Vision_V2.0"

# Start with SDXL for highest quality
python diffusiond/main.py --model "SG161222/RealVisXL_V4.0" --precision fp32

# Start with DreamShaper (versatile community model)
python diffusiond/main.py --model "Lykon/DreamShaper" --precision fp16

# Start with standard SD 1.5 for fastest generation
python diffusiond/main.py --model "runwayml/stable-diffusion-v1-5" --precision fp16

# Custom configuration
python diffusiond/main.py \
  --model "stabilityai/stable-diffusion-2-1" \
  --device cuda \
  --precision fp16 \
  --output-dir ./my_images \
  --http-port 8080 \
  --load-on-startup
```

#### Command Line Options
```
--model MODEL_ID           HuggingFace model ID (default: SG161222/Realistic_Vision_V2.0)
--device DEVICE            Device: auto, cuda, cpu, mps (default: auto)
--output-dir PATH          Directory for generated images (default: ./generated_images)
--precision PRECISION      Model precision: fp16, fp32, bf16 (default: fp16)
--attention-precision PREC  Attention precision: fp16, fp32, bf16 (default: fp16)
--models-dir PATH          Model cache directory (default: ./models)
--http-host HOST           HTTP server host (default: 127.0.0.1)
--http-port PORT           HTTP server port (default: 8000)
--load-on-startup          Load model at startup instead of lazy loading
```

#### HTTP API Examples

##### Generate Images
```bash
# Simple generation
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a beautiful sunset over mountains"}'

# Advanced generation with parameters
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a beautiful woman, highly detailed, 8k",
    "negative_prompt": "blurry, low quality",
    "width": 768,
    "height": 768,
    "steps": 30,
    "guidance_scale": 7.5,
    "seed": 42
  }'
```

##### Dynamic Model Switching
```bash
# Switch to SD 1.5 for faster generation
curl -X POST http://127.0.0.1:8000/switch-model \
  -H "Content-Type: application/json" \
  -d '{"model_id": "runwayml/stable-diffusion-v1-5", "precision": "fp16"}'

# Switch to SDXL for highest quality
curl -X POST http://127.0.0.1:8000/switch-model \
  -H "Content-Type: application/json" \
  -d '{"model_id": "SG161222/RealVisXL_V4.0", "precision": "fp32"}'

# Switch to DreamShaper community model
curl -X POST http://127.0.0.1:8000/switch-model \
  -H "Content-Type: application/json" \
  -d '{"model_id": "Lykon/DreamShaper", "precision": "fp16"}'
```

##### Server Status
```bash
# Check current model and status
curl http://127.0.0.1:8000/status

# List available endpoints
curl http://127.0.0.1:8000/docs
```

### Recommended Models

#### High Quality (SDXL)
- `SG161222/RealVisXL_V4.0` - Photorealistic, best quality
- `stabilityai/stable-diffusion-xl-base-1.0` - Official SDXL

#### Balanced (SD 1.5/2.1)  
- `SG161222/Realistic_Vision_V2.0` - Excellent for portraits
- `Lykon/DreamShaper` - Versatile community favorite
- `dreamlike-art/dreamlike-photoreal-2.0` - Photorealistic style

#### Fast (SD 1.5)
- `runwayml/stable-diffusion-v1-5` - Standard, fastest
- `stabilityai/stable-diffusion-2-1` - Good balance

---

## TBot - Main CLI Tool (Work in Progress)

TBot will serve as the main command-line interface that integrates multiple AI sources and provides a unified experience for both text and image generation. 

**Current Status**: In development. The modular architecture with Conductor and Diffusiond is complete and ready for integration.

**Planned Features**:
- Unified CLI for text and image generation
- Integration with multiple upstream model sources
- Conversation history and context management
- Plugin system for extending capabilities
- Configuration profiles for different use cases

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd tbot
   ```

2. **Install dependencies**:
   ```bash
   # Using poetry (recommended)
   poetry install
   
   # Or using pip
   pip install -r requirements.txt
   ```

3. **Download models** (optional - models download automatically on first use):
   ```bash
   # Conductor will download LLM models to ./models/
   # Diffusiond will download Stable Diffusion models to ./models/
   ```

## Configuration

- **Conductor**: Edit `conductor/SETTINGS.md` or use pre-configured profiles in `conductor/SETTINGS_*.md`
- **Diffusiond**: Configuration via command line arguments or `diffusiond/SETTINGS.md`

## Examples

### Integrated Workflow
```bash
# Terminal 1: Start diffusiond for image generation
python diffusiond/main.py --model "SG161222/Realistic_Vision_V2.0"

# Terminal 2: Start conductor for text generation  
python conductor/conductor.py

# In conductor CLI:
> generate an image of a sunset over mountains
# Automatically routes to diffusiond and returns image path

> write a python function to resize images
# Routes to code_generation engine and returns Python code

> translate "hello world" to spanish
# Routes to translation engine: "hola mundo"
```

## License

[Your license here]
