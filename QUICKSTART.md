# TBot Quick Start Guide

## Overview: AI-Powered CLI Tool

TBot is an AI-powered command-line tool that leverages LLM models for various tasks:
- **Summarization** - Summarize files and documents
- **Analysis** - Analyze text for insights, sentiment, security issues
- **Conversion** - Convert data to different formats (tables, JSON, etc.)
- **Explanation** - Explain complex content in simpler terms
- **Transformation** - Transform text between formats

## Architecture

TBot uses a client-server architecture:
- **TBot CLI** = Lightweight command-line tool
- **Conductor** = LLM text generation server  
- **Diffusiond** = Image generation server (used internally by Conductor)

## Quick Start

### 1. Install Dependencies
```bash
poetry install
```

### 2. Start Servers
```bash
# Start both servers with one command
poetry run tbot-servers

# Or start individually
poetry run tbot-servers --conductor-only
poetry run tbot-servers --diffusiond-only
```

### 3. Use TBot CLI Tool
```bash
# Health check
tbot --health

# Summarize a file
tbot summarize README.md

# Analyze a file for security issues
tbot analyze --type security config.yaml

# Convert log data to table
cat errors.log | tbot convert-to-table

# Explain complex code
tbot explain --level simple algorithm.py

# Interactive mode
poetry run tbot --interactive

# Single prompt
poetry run tbot "What is Python?"

# Image generation
poetry run tbot "Generate an image of a cat"
```

## Manual Server Management

### Start Conductor (LLM Server)
```bash
poetry run conductor
# Runs on http://localhost:8001
```

### Start Diffusiond (Image Server)
```bash
poetry run diffusiond
# Runs on http://localhost:8000
```

## Example Usage

### Text Generation
```bash
# Ask questions
poetry run tbot "Explain machine learning"

# Code generation  
poetry run tbot "Write a Python function to sort a list"

# Math problems
poetry run tbot "Solve: 2x + 3 = 11"
```

### Image Generation
```bash
# Generate images
poetry run tbot "Generate an image of a sunset"
poetry run tbot "Create a picture of a robot"
poetry run tbot "Draw a landscape with mountains"
```

### Interactive Mode
```bash
poetry run tbot --interactive

# In interactive mode:
💬 You: What is the capital of France?
📝 Response: The capital of France is Paris.

💬 You: Generate an image of that city
🎨 Generated image: generated_images/paris_123456.png

💬 You: /help
# Shows available commands

💬 You: /exit
# Exits the program
```

## Configuration

### Custom Server URLs
```bash
# Use remote servers
poetry run tbot --conductor-url http://192.168.1.100:8001 \
                --diffusiond-url http://192.168.1.101:8000 \
                --interactive
```

### Generation Parameters
```bash
# Control text generation
poetry run tbot "Tell me a story" \
                --max-tokens 500 \
                --temperature 0.8
```

## Available Commands (Interactive Mode)

- `/help` - Show help
- `/history` - Show conversation history  
- `/clear` - Clear conversation history
- `/health` - Check server health
- `/exit` or `/quit` - Exit

## Troubleshooting

### Servers Not Starting
1. Check if ports are available: `netstat -an | grep 800[01]`
2. Check dependencies: `poetry install`
3. Check CUDA availability for GPU: `python -c "import torch; print(torch.cuda.is_available())"`

### Connection Errors
1. Verify servers are running: `poetry run tbot --health`
2. Check firewall settings
3. Verify URLs are correct

### Out of Memory
1. Reduce model size in conductor/SETTINGS.md
2. Use CPU mode: `poetry run tbot-servers --device cpu`
3. Close other applications

## Advanced Usage

### Server Management
```bash
# Start with custom models
poetry run tbot-servers --diffusiond-model "runwayml/stable-diffusion-v1-5"

# Custom ports
poetry run tbot-servers --conductor-port 9001 --diffusiond-port 9000

# Debug mode
poetry run tbot-servers --log-level DEBUG
```

### Development
```bash
# Test client-server communication
python tbot/tests/test_tbot_client.py

# Test conductor dynamic switching
python conductor/tests/test_dynamic_switching.py
```

This new architecture provides better separation of concerns, scalability, and resource efficiency!
