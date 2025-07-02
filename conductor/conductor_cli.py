# ...existing code from conductor.py related to CLI interface...

import asyncio
import logging
import sys
import argparse
import json
import time
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, TYPE_CHECKING, AsyncGenerator
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
import importlib.util
import urllib.request

# Only import what we absolutely need at startup
if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

# Import dependencies loader with fallback
try:
    from dependencies_loader import DependenciesLoader
except ImportError:
    try:
        from conductor.dependencies_loader import DependenciesLoader
    except ImportError:
        class DependenciesLoader:
            def __init__(self): pass

            async def ensure_dependencies(self, model_name: str, precision: str = "FP16") -> bool: return True

            def check_system_requirements(self) -> Dict[str, Any]: return {'platform': 'unknown', 'memory_gb': 16.0}

            def get_missing_packages(self, packages: List[str]) -> List[str]: return []

            def get_installation_instructions(self, packages: List[str]) -> Dict[str, str]: return {}

            def validate_model_requirements(self, model_name: str, precision: str) -> Dict[str, Any]:
                return {'can_load': True, 'warnings': [], 'recommendations': []}

            def get_dependency_report(self) -> Dict[str, Any]: return {'system': {}, 'packages': {},
                                                                       'recommendations': []}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

from conductor.utils.config_parser import ConfigParser
from conductor.conductor import Conductor

def get_status(conductor) -> dict:
    """
    Return a status dictionary for the CLI /status command.
    """
    return {
        "engine_count": len(conductor.engines),
        "configured_engines": list(conductor.engine_configs.keys()),
        "loaded_engines": list(conductor.engines.keys()),
        "model_sharing": _get_model_sharing_info(conductor),
        "memory_status": conductor.model_loader.get_memory_status(),
    }

def _get_model_sharing_info(conductor) -> dict:
    """
    Return a mapping of model_name -> list of engine categories sharing that model.
    """
    sharing = {}
    for category, engine in conductor.engines.items():
        model_name = getattr(engine, "technical_model_name", None)
        if model_name:
            sharing.setdefault(model_name, []).append(category)
    return sharing

# ...existing code for CLI interface and main_sync...

async def cli_interface(conductor: "Conductor"):
    """Simple command-line interface"""
    print("\n=== Conductor CLI with MCP ===")
    print("Commands: /status, /engines, /mcp, /deps, /models, /vram, /test, /force-stop, /quit")
    print("Or just type any prompt to generate a response.\n")

    while True:
        try:
            user_input = input("> ").strip()

            if user_input.lower() in ['/quit', '/exit', 'quit', 'exit']:
                break
            elif user_input == '/status':
                status = get_status(conductor)
                print(f"Status: {json.dumps(status, indent=2)}")

                # Pretty print model sharing
                if status.get('model_sharing'):
                    print("\n=== Model Sharing ===")
                    for model_name, engines in status['model_sharing'].items():
                        print(f"{model_name}: {len(engines)} engines ({', '.join(engines)})")
            elif user_input == '/vram':
                # Show detailed VRAM usage
                print("\n=== VRAM Usage Details ===")
                memory_status = conductor.model_loader.get_memory_status()

                if memory_status['device'] == 'cuda':
                    print(f"GPU Device: {memory_status['device']}")
                    print(f"Total VRAM: {memory_status['total_memory_gb']:.1f} GB")
                    print(f"Used VRAM: {memory_status['used_memory_gb']:.1f} GB")
                    if 'reserved_memory_gb' in memory_status:
                        print(f"Reserved VRAM: {memory_status['reserved_memory_gb']:.1f} GB")
                        print(f"Free VRAM: {memory_status['free_memory_gb']:.1f} GB")
                    print(f"Available: {memory_status['available_memory_gb']:.1f} GB")
                    print(f"Usage: {(memory_status['used_memory_gb'] / memory_status['total_memory_gb']) * 100:.1f}%")
                else:
                    print(f"Device: {memory_status['device']}")
                    print(f"System RAM: {memory_status['total_memory_gb']:.1f} GB")
                    print(f"Used by models: {memory_status['used_memory_gb']:.1f} GB")

                print(f"\nLoaded Models: {memory_status['loaded_model_count']}")

                # Show which models are loaded and their estimated sizes
                loaded_models = conductor.model_loader.loaded_models
                if loaded_models:
                    print("\nModel Details:")
                    for model_name in loaded_models:
                        # Find engines using this model to get precision info
                        for category, engine in conductor.engines.items():
                            if engine.technical_model_name == model_name:
                                precision = engine.config.get('precision', 'FP16')
                                # Estimate model size
                                if '7b' in model_name.lower():
                                    base_size = 7
                                elif '8b' in model_name.lower():
                                    base_size = 8
                                elif '13b' in model_name.lower():
                                    base_size = 13
                                else:
                                    base_size = 7

                                if precision == '4-bit':
                                    estimated_size = base_size * 0.5
                                elif precision == 'FP16':
                                    estimated_size = base_size * 2.0
                                else:
                                    estimated_size = base_size * 2.0

                                print(f"  {model_name}")
                                print(f"    Precision: {precision}")
                                print(f"    Estimated size: ~{estimated_size:.1f} GB")
                                break
                else:
                    print("No models currently loaded")
            elif user_input == '/engines':
                print(f"Available engines: {list(conductor.engines.keys())}")
                print(f"Configured engines: {list(conductor.engine_configs.keys())}")

                # Show model sharing info
                print("\n=== Model Sharing Status ===")
                loaded_models = conductor.model_loader.get_loaded_model_info()
                for model_name in loaded_models:
                    # Find which engines use this model
                    using_engines = []
                    for category, engine in conductor.engines.items():
                        if engine.technical_model_name == model_name:
                            using_engines.append(category)
                    print(f"Model: {model_name}")
                    print(f"  Shared by {len(using_engines)} engines: {using_engines}")

                memory_status = conductor.model_loader.get_memory_status()
                print(f"\nMemory: {memory_status['used_memory_gb']:.1f}GB / {memory_status['total_memory_gb']:.1f}GB")
                print(f"Loaded models: {memory_status['loaded_model_count']}")
                print(f"Device: {memory_status['device']}")
            elif user_input == '/deps':
                print("\n=== Dependencies Report ===")
                report = conductor.dependencies_loader.get_dependency_report()
                print("System Info:")
                for key, value in report['system'].items():
                    if key != 'gpu_memory':
                        print(f"  {key}: {value}")
                print("\nPackage Status:")
                for pkg, status in report['packages'].items():
                    symbol = "✓" if status['available'] else "✗"
                    req_type = "required" if status['required'] else "optional"
                    print(f"  {symbol} {pkg} ({req_type})")
                if report['recommendations']:
                    print("\nRecommendations:")
                    for rec in report['recommendations']:
                        print(f"  • {rec}")
                print(f"\nModels Directory: {conductor.model_loader.models_dir.absolute()}")
            elif user_input == '/models':
                models_dir = conductor.model_loader.models_dir
                print(f"\n=== Models Directory: {models_dir.absolute()} ===")
                if models_dir.exists():
                    subdirs = [d for d in models_dir.iterdir() if d.is_dir()]
                    if subdirs:
                        print("Local models found:")
                        for subdir in subdirs:
                            model_name = subdir.name.replace('--', '/')
                            # Fix: use relative_to only if subdir is under cwd, else just show absolute path
                            try:
                                rel_path = subdir.relative_to(Path.cwd())
                            except ValueError:
                                rel_path = subdir
                            print(f"  {model_name} -> {rel_path}")
                    else:
                        print("No local models found")
                        print("Models will be downloaded here on first use")
                else:
                    print("Models directory does not exist yet")
                    print("Will be created automatically on first model download")
            elif user_input == '/mcp':
                test_request = {
                    "id": "test-1",
                    "method": "tools/call",
                    "params": {
                        "name": "generate_text",
                        "arguments": {"prompt": "Hello from MCP test", "category": "conversational_chat"}
                    }
                }
                response = await conductor.handle_mcp_request(json.dumps(test_request))
                print(f"MCP Test Response: {response}")
            elif user_input == '/test':
                # Simple test generation
                print("Testing generation with simple prompt...")
                try:
                    engine = await conductor.get_engine('conversational_chat')
                    if engine:
                        print(f"Engine loaded: {engine.is_loaded()}")
                        print(f"Model available: {engine.model is not None}")
                        print(f"Tokenizer available: {engine.tokenizer is not None}")

                        if engine.model and engine.tokenizer:
                            # Test with very simple generation
                            print("Attempting simple generation...")
                            # Use a dedicated test session to avoid history contamination
                            session_id = "__test__"
                            if hasattr(engine, "clear_conversation_history"):
                                engine.clear_conversation_history(session_id)
                            if hasattr(engine, 'persona'):
                                engine.persona = "You are a helpful assistant."

                            # Force test formatting
                            prompt = "Who is the father of computer science?"
                            prepared_prompt = engine._prepare_prompt(prompt)
                            print(f"\n[Test Prepared Prompt]:\n{prepared_prompt}\n")

                            response = await asyncio.wait_for(
                                engine.generate(prompt, max_tokens=50, session_id=session_id),
                                timeout=10.0
                            )
                            print(f"Test response: {response}")
                        else:
                            print("Model or tokenizer not available")
                    else:
                        print("No conversational_chat engine available")
                except Exception as e:
                    print(f"Test failed: {e}")
                    import traceback
                    print(f"Traceback: {traceback.format_exc()}")
            elif user_input == '/force-stop':
                print("Attempting to stop any hanging generation...")
                # Force cleanup
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print("Cleared CUDA cache")
            elif user_input == '/diffusion-model':
                info = conductor.get_diffusion_model_info()
                print(json.dumps(info, indent=2))
            elif user_input:
                try:
                    print(f"Generating response for: '{user_input}'...")

                    # Add timeout to prevent hanging
                    category, response = await asyncio.wait_for(
                        conductor.generate(user_input, max_tokens=1024),
                        timeout=30.0  # 30 second timeout
                    )
                    print(f"\n[{category}]: {response}\n")
                except asyncio.TimeoutError:
                    print("Error: Generation timed out after 30 seconds")
                    print("Try /force-stop to clear CUDA cache")
                except Exception as e:
                    print(f"Error: {e}")
                    # Show more detailed error for debugging
                    import traceback
                    print(f"Detailed error: {traceback.format_exc()}")

        except KeyboardInterrupt:
            break
        except EOFError:
            break

    print("\nGoodbye!")

