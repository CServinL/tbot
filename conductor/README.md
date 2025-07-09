# Conductor LLM Server

A modular, high-performance LLM server system that intelligently routes different types of tasks to specialized models and engines based on configuration.

## Features

- **Modular Architecture**: Separate engines for different task types (chat, code, math, translation, etc.)
- **Intelligent Model Management**: Automatic loading/unloading with memory optimization
- **Multiple Interfaces**: HTTP API and Model Context Protocol (MCP) support
- **Configuration-Driven**: Easy setup via `SETTINGS.md` file
- **Production Ready**: Comprehensive logging, error handling, and monitoring

## Quick Start

### Ensure you have the required models and dependencies
```bash

python conductor.py --skip-model-loading  # First run to check setup
```

### Configuration

Edit `conductor/SETTINGS.md` to configure your models and engines:

```markdown
# LLM Server Settings

## Conversational Persona
[Your conversational persona here]

| Category/Area | Best Open Source Model | VRAM/RAM Requirements | Technical Model Name | Stay Loaded | Precision |
|---------------|------------------------|----------------------|---------------------|-------------|-----------|
| **Conversational Chat** | Llama 3.1 8B | ~8GB (FP16) | `meta-llama/Llama-3.1-8B-Instruct` | true | FP16 |
| **Code Completion** | CodeLlama 7B | ~4GB (4-bit) | `codellama/CodeLlama-7b-hf` | true | 4-bit |
| ... | ... | ... | ... | ... | ... |
```

### 3. Run the Server

```bash
# Start HTTP server (default: localhost:8000)
poetry run python conductor/conductor.py

# Or with custom settings
poetry run python conductor/conductor.py --host 0.0.0.0 --port 8080 --log-level DEBUG

# MCP server only
poetry run python conductor/conductor.py --mcp-only
```

## Architecture

```
conductor/
├── conductor.py              # Main entry point
├── SETTINGS.md              # Configuration file
├── base_llm_engine.py       # Base engine class
├── model_loader.py          # Model loading/management
├── dependencies_loader.py   # Dependency management
├── http_server.py           # HTTP API server
├── mcp_server.py           # MCP protocol server
├── engines/                 # Specialized engines
│   ├── conversational_chat.py
│   ├── code_completion.py
│   ├── code_generation.py
│   ├── mathematical_reasoning.py
│   ├── translation.py
│   └── ...
└── utils/                   # Utility modules
    ├── config_parser.py
    ├── model_registry.py
    └── persona_loader.py
```

## Core Modules

### Base Engine (`base_llm_engine.py`)
Abstract base class that all specialized engines inherit from. Provides:
- Model loading/unloading interface
- Generation and streaming methods
- Memory management
- Health checking

### Model Loader (`model_loader.py`)
Handles all model loading operations:
- Supports multiple precisions (FP16, FP32, 4-bit, 8-bit)
- Memory monitoring and management
- Quantization support via BitsAndBytesConfig
- Device placement optimization

### Dependencies Loader (`dependencies_loader.py`)
Ensures all required packages are installed:
- Model-specific dependency detection
- Automatic installation of missing packages
- System requirements checking
- Version compatibility verification

### Configuration Parser (`utils/config_parser.py`)
Parses the `SETTINGS.md` file:
- Extracts model configurations
- Validates settings
- Supports hot-reloading
- Manages conversational personas

### Model Registry (`utils/model_registry.py`)
Tracks model states and metadata:
- Model status tracking (loading, loaded, error, etc.)
- Memory usage monitoring
- Generation count statistics
- Health checking

## Specialized Engines

Each engine is optimized for specific task types:

### Conversational Chat (`engines/conversational_chat.py`)
- Session management with conversation history
- Configurable personas
- Context-aware responses

### Code Completion (`engines/code_completion.py`)
- Language detection
- Fast, low-latency completion
- Caching for performance
- Multiple programming languages

### Code Generation (`engines/code_generation.py`)
- Full code solution generation
- Template-based generation
- Language-specific optimizations
- Documentation generation

### Mathematical Reasoning (`engines/mathematical_reasoning.py`)
- Step-by-step problem solving
- LaTeX formatting support
- Multiple math contexts (algebra, calculus, etc.)
- Verification and checking

### Translation (`engines/translation.py`)
- 200+ language support via NLLB
- Formal/informal register control
- Language detection
- Batch translation

### Scientific Research (`engines/scientific_research.py`)
- Literature review generation
- Methodology analysis
- Peer review standards
- Citation management

### Legal Analysis (`engines/legal_analysis.py`)
- Contract review
- Risk assessment
- Compliance checking
- Multiple legal areas

### And many more...

## HTTP API

### Generate Response
```http
POST /generate
Content-Type: application/json

{
  "prompt": "Explain quantum computing",
  "category": "general_reasoning",
  "max_tokens": 500,
  "temperature": 0.7
}
```

### Stream Response
```http
POST /generate/stream
Content-Type: application/json

{
  "prompt": "Write a story about...",
  "category": "creative_writing",
  "stream": true
}
```

### List Models
```http
GET /models
```

### Health Check
```http
GET /health
```

### System Statistics
```http
GET /stats
```

## MCP Protocol

The server supports Model Context Protocol for integration with compatible clients:

```python
import json
from conductor import ConductorServer

conductor = ConductorServer()
await conductor.initialize()

# MCP request
request = {
    "id": "1",
    "method": "tools/call",
    "params": {
        "name": "generate_text",
        "arguments": {
            "prompt": "Hello world",
            "category": "conversational_chat"
        }
    }
}

response = await conductor.handle_mcp_request(json.dumps(request))
```

## Configuration Options

### Model Settings
- `technical_model_name`: HuggingFace model identifier
- `stay_loaded`: Whether to keep in memory
- `precision`: FP16, FP32, 4-bit, 8-bit
- `vram_requirement`: Memory requirement estimate

### Engine Parameters
- `temperature`: Generation creativity (0.1-1.5)
- `max_tokens`: Maximum tokens to generate
- `top_p`: Nucleus sampling parameter
- `repetition_penalty`: Repetition penalty factor

### Server Options
- `host`: Bind address (default: 0.0.0.0)
- `port`: HTTP port (default: 8000)
- `log_level`: Logging verbosity
- `skip_model_loading`: Skip model loading for testing

## Memory Management

The system automatically manages memory usage:

1. **Stay-Loaded Models**: Keep frequently used models in memory
2. **On-Demand Loading**: Load specialized models when needed
3. **Automatic Unloading**: Free memory when models aren't needed
4. **Memory Monitoring**: Track usage and prevent OOM errors

### Memory Optimization Tips

- Use 4-bit quantization for speed-critical tasks
- Use FP16 for quality-important tasks
- Configure stay-loaded models based on usage patterns
- Monitor memory usage via `/stats` endpoint

## Development

### Adding New Engines

1. Create new engine file in `engines/` directory
2. Inherit from `BaseLLMEngine`
3. Implement required methods:
   ```python
   async def load_model(self) -> bool
   async def unload_model(self) -> bool
   async def generate(self, prompt: str, **kwargs) -> str
   async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]
   def get_system_prompt(self) -> Optional[str]
   ```
4. Add to engine registry in `conductor.py`
5. Update `SETTINGS.md` with new category

### Testing

```bash
# Run with model loading disabled
python conductor.py --skip-model-loading

# Test specific engine
python -m pytest tests/test_engines.py::test_code_completion

# Integration tests
python -m pytest tests/test_integration.py
```

## Deployment

### Docker
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt

EXPOSE 8000
CMD ["python", "conductor.py"]
```

### Systemd Service
```ini
[Unit]
Description=Conductor LLM Server
After=network.target

[Service]
Type=simple
User=conductor
WorkingDirectory=/opt/conductor
ExecStart=/opt/conductor/venv/bin/python conductor.py
Restart=always

[Install]
WantedBy=multi-user.target
```

## Monitoring

### Health Endpoints
- `/health` - Basic health status
- `/stats` - Detailed statistics
- `/models` - Model status information

### Logging
- Structured JSON logs
- Configurable log levels
- Request/response tracking
- Performance metrics

### Metrics
- Generation count per engine
- Memory usage per model
- Request latency
- Error rates

## Troubleshooting

### Common Issues

**Models not loading:**
- Check CUDA availability: `torch.cuda.is_available()`
- Verify memory requirements
- Check HuggingFace model names
- Review dependency installation

**Out of memory errors:**
- Reduce precision (use 4-bit quantization)
- Decrease models staying loaded
- Monitor with `/stats` endpoint
- Consider model sharding

**Performance issues:**
- Use appropriate precision for task
- Enable caching where applicable
- Monitor GPU utilization
- Check for memory leaks

### Debug Mode
```bash
python conductor.py --log-level DEBUG --skip-model-loading
```

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run tests: `python -m pytest`
5. Submit pull request

## License

[Your chosen license]

## Support

- Issues: [GitHub Issues](link)
- Discussions: [GitHub Discussions](link)
- Documentation: [Wiki](link)