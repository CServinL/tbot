# TBot Client-Server Architecture

TBot now follows a client-server architecture with clear separation of concerns:

## Architecture Overview

```
┌─────────┐    HTTP/MCP     ┌─────────────┐    HTTP/MCP     ┌─────────────┐
│  TBot   │ ◄──────────────► │ Conductor   │ ◄──────────────► │ Diffusiond  │
│ (Client)│                 │ (LLM Server)│                 │(Image Server)│
└─────────┘                 └─────────────┘                 └─────────────┘
```

## Components

### 1. TBot (`tbot/main.py`)
- **Pure command-line client**
- Communicates only with Conductor via HTTP (MCP support planned)
- No direct model loading or heavy dependencies
- Conductor handles routing to Diffusiond internally
- Features:
  - Interactive chat mode
  - Single prompt mode
  - Streaming responses
  - Health monitoring
  - Conversation history

### 2. Conductor (`conductor/`)
- **LLM server** with HTTP and MCP endpoints
- Dynamic model switching system
- Intelligent prompt classification and routing
- Features:
  - `/generate` - General text generation
  - `/chat/completions` - OpenAI-compatible endpoint
  - `/code/completion` - Code completion
  - `/code/generation` - Code generation
  - `/translate` - Translation
  - `/math/solve` - Mathematical reasoning
  - And more specialized endpoints

### 3. Diffusiond (`diffusiond/`)
- **Image generation server** with HTTP and MCP endpoints
- Stable Diffusion and ComfyUI compatibility
- Features:
  - `/generate` - Image generation
  - `/models` - Model management
  - ComfyUI workflow support

## Usage

### Start the Servers

1. **Start Conductor (LLM server):**
   ```bash
   poetry run conductor
   # or with custom config
   poetry run conductor --config path/to/SETTINGS.md
   ```

2. **Start Diffusiond (Image server):**
   ```bash
   poetry run diffusiond
   # or with custom settings
   poetry run diffusiond --model "SG161222/Realistic_Vision_V2.0"
   ```

### Use TBot Client

1. **Health check:**
   ```bash
   poetry run tbot --health
   ```

2. **Interactive mode:**
   ```bash
   poetry run tbot --interactive
   ```

3. **Single prompt:**
   ```bash
   poetry run tbot "What is the capital of France?"
   ```

4. **Image generation:**
   ```bash
   poetry run tbot "Generate an image of a sunset over mountains"
   ```

5. **Streaming mode:**
   ```bash
   poetry run tbot --interactive --stream
   ```

6. **Custom server URLs:**
   ```bash
   poetry run tbot --conductor-url http://192.168.1.100:8001 --diffusiond-url http://192.168.1.101:8000 --interactive
   ```

## Configuration

### TBot Client
- `--conductor-url`: Conductor server URL (default: http://localhost:8001)
- `--diffusiond-url`: Diffusiond server URL (default: http://localhost:8000)
- `--timeout`: Request timeout in seconds (default: 60)
- `--max-tokens`: Maximum tokens to generate
- `--temperature`: Generation temperature
- `--category`: Force specific category

### Server Configuration
- Conductor: Configure via `conductor/SETTINGS.md`
- Diffusiond: Configure via command-line arguments or `diffusiond/SETTINGS.md`

## Benefits of This Architecture

1. **Separation of Concerns**: Each component has a single responsibility
2. **Scalability**: Servers can run on different machines
3. **Resource Efficiency**: Only servers need heavy dependencies
4. **Flexibility**: Can mix and match different server implementations
5. **Development**: Easier to develop and test individual components
6. **Deployment**: Can deploy servers independently

## API Compatibility

### HTTP Endpoints
Both servers provide REST APIs for integration with other tools.

### MCP (Model Context Protocol)
Both servers support MCP for integration with Claude Desktop and other MCP clients.

## Development

### Running Tests
```bash
# Test the client-server communication
python test_tbot_client.py

# Test conductor dynamic switching
python conductor/tests/test_dynamic_switching.py

# Test diffusiond
cd tests_diffusiond && bash run_tests.sh
```

### Adding New Features
- **New text capabilities**: Add to Conductor engines
- **New image capabilities**: Add to Diffusiond
- **Client features**: Modify TBot client

This architecture provides a clean separation that makes the system more maintainable, scalable, and flexible.
