#!/usr/bin/env python3
"""
Conductor LLM Server - Main Entry Point

A modular LLM server system that routes different types of tasks to specialized models
and engines based on configuration defined in SETTINGS.md.
"""

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
from dataclasses import dataclass, asdict

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


class ConfigParser:
    """Parse SETTINGS.md configuration file"""

    def __init__(self, config_path: str = "conductor/SETTINGS.md"):
        self.config_path = Path(config_path)
        self._settings = {}
        self._persona = ""

    def parse_settings(self) -> Dict[str, Dict[str, Any]]:
        """Parse SETTINGS.md and extract model configurations"""
        if not self.config_path.exists():
            logger.error(f"Configuration file not found: {self.config_path}")
            return {}

        try:
            content = self.config_path.read_text(encoding='utf-8')
            return self._parse_markdown_table(content)
        except Exception as e:
            logger.error(f"Failed to parse configuration: {e}")
            return {}

    def _parse_markdown_table(self, content: str) -> Dict[str, Dict[str, Any]]:
        """Extract configuration from markdown table"""
        lines = content.split('\n')
        configs = {}

        # Extract persona
        persona_start = False
        persona_lines = []
        for line in lines:
            if '## Conversational Persona' in line:
                persona_start = True
                continue
            elif persona_start and line.startswith('##'):
                break
            elif persona_start and line.strip():
                persona_lines.append(line)
        self._persona = '\n'.join(persona_lines).strip()

        # Extract table data
        in_table = False
        for line in lines:
            line = line.strip()
            if not line:
                continue

            if '| Category/Area |' in line:
                in_table = True
                continue
            elif line.startswith('|---'):
                continue
            elif in_table and line.startswith('|') and '**' in line:
                parts = [p.strip() for p in line.split('|')[1:-1]]
                if len(parts) >= 5:
                    category = parts[0].replace('**', '').strip()
                    model_name = parts[1].strip()
                    vram_req = parts[2].strip()
                    technical_name = parts[3].strip().replace('`', '')
                    stay_loaded = parts[4].strip().lower() == 'true'
                    precision = parts[5].strip() if len(parts) > 5 else 'FP16'

                    category_key = category.lower().replace(' ', '_').replace('/', '_')
                    configs[category_key] = {
                        'category': category_key,
                        'model_name': model_name,
                        'technical_model_name': technical_name,
                        'vram_requirement': vram_req,
                        'stay_loaded': stay_loaded,
                        'precision': precision
                    }

        logger.info(f"Parsed {len(configs)} model configurations")
        return configs

    def get_conversational_persona(self) -> str:
        return self._persona

    def get_configuration_summary(self) -> Dict[str, Any]:
        return {
            'config_path': str(self.config_path),
            'persona_length': len(self._persona),
            'engine_count': len(self._settings),
            'engines': list(self._settings.keys()) if self._settings else []
        }


class ModelLoader:
    """Enhanced model loader with better error handling and model sharing"""

    def __init__(self, models_dir: str = "./models"):
        self.loaded_models: Dict[str, tuple] = {}
        self.device = None
        self._torch = None
        self._transformers = None
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        logger.info(f"Model cache directory: {self.models_dir.absolute()}")

    def _lazy_import_torch(self):
        """Lazy import torch only when needed"""
        if self._torch is None:
            try:
                import torch
                self._torch = torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"Initialized torch with device: {self.device}")
            except ImportError:
                logger.error("torch not available - install with: pip install torch")
                raise
        return self._torch

    def _lazy_import_transformers(self):
        """Lazy import transformers only when needed"""
        if self._transformers is None:
            try:
                import transformers
                from transformers import AutoTokenizer, AutoModelForCausalLM
                self._transformers = transformers
                self._transformers.AutoTokenizer = AutoTokenizer
                self._transformers.AutoModelForCausalLM = AutoModelForCausalLM
                logger.info("Loaded transformers library")
            except ImportError:
                logger.error("transformers not available - install with: pip install transformers")
                raise
        return self._transformers

    def _get_local_model_path(self, model_name: str) -> Optional[Path]:
        """Check if model exists locally"""
        safe_name = model_name.replace('/', '--')
        local_path = self.models_dir / safe_name

        if local_path.exists():
            model_files = ['config.json', 'pytorch_model.bin', 'model.safetensors']
            if any((local_path / f).exists() for f in model_files):
                logger.info(f"Found local model at: {local_path}")
                return local_path

        logger.info(f"Model not found locally, will download to: {local_path}")
        return None

    def _get_hf_token(self) -> Optional[str]:
        """Get HuggingFace token from various sources"""
        # Check environment variables
        token = os.getenv('HUGGINGFACE_HUB_TOKEN') or os.getenv('HF_TOKEN')
        if token:
            return token

        # Try HuggingFace CLI cache
        try:
            from huggingface_hub import HfFolder
            return HfFolder.get_token()
        except:
            pass

        try:
            from huggingface_hub.utils import get_token
            return get_token()
        except:
            pass

        # Try token file
        try:
            token_path = os.path.expanduser("~/.cache/huggingface/token")
            if os.path.exists(token_path):
                with open(token_path, 'r') as f:
                    return f.read().strip()
        except:
            pass

        return None

    def _load_tokenizer(self, model_name: str) -> Optional['PreTrainedTokenizer']:
        """Load tokenizer with enhanced error handling and fallback"""
        transformers = self._lazy_import_transformers()

        try:
            hf_token = self._get_hf_token()
            kwargs = {"trust_remote_code": True}

            if hf_token:
                kwargs["token"] = hf_token

            # Try fast tokenizer first
            try:
                kwargs["use_fast"] = True
                tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, **kwargs)
                logger.info(f"✓ Loaded fast tokenizer for {model_name}")
            except Exception as fast_error:
                if "PyPreTokenizerTypeWrapper" in str(fast_error):
                    logger.warning(f"Fast tokenizer failed for {model_name}, trying slow tokenizer")
                    kwargs["use_fast"] = False
                    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, **kwargs)
                    logger.info(f"✓ Loaded slow tokenizer for {model_name}")
                else:
                    raise fast_error

            # Ensure pad token is set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            return tokenizer

        except Exception as e:
            self._handle_tokenizer_error(e, model_name)
            return None

    def _handle_tokenizer_error(self, error: Exception, model_name: str):
        """Handle tokenizer loading errors with helpful messages"""
        error_msg = str(error)
        logger.error(f"Failed to load tokenizer for {model_name}: {error_msg}")

        if "PyPreTokenizerTypeWrapper" in error_msg:
            logger.error("=" * 60)
            logger.error("🔧 TOKENIZER COMPATIBILITY ISSUE")
            logger.error("=" * 60)
            logger.error("Quick fixes:")
            logger.error("1. pip install --upgrade tokenizers transformers")
            logger.error("2. rm -rf ~/.cache/huggingface/hub/models--mistralai*")
            logger.error("3. Try alternative model: HuggingFaceH4/zephyr-7b-beta")
            logger.error("=" * 60)
        elif "gated repo" in error_msg.lower():
            logger.error("=" * 60)
            logger.error("🔒 AUTHENTICATION REQUIRED")
            logger.error("=" * 60)
            logger.error(f"Model '{model_name}' requires authentication.")
            logger.error("Fix: huggingface-cli login")
            logger.error("=" * 60)

    async def load_model(self, model_name: str, precision: str = "FP16") -> tuple:
        """Load model with enhanced error handling"""
        if model_name in self.loaded_models:
            logger.info(f"Model {model_name} already loaded, returning cached version")
            return self.loaded_models[model_name]

        logger.info(f"Loading model {model_name} with precision {precision}")

        # Check current memory status
        memory_status = self.get_memory_status()
        logger.info(
            f"Memory status before loading: {memory_status['used_memory_gb']:.1f}GB used of {memory_status['total_memory_gb']:.1f}GB total")

        # Get paths and imports
        local_path = self._get_local_model_path(model_name)
        model_path = str(local_path) if local_path else model_name
        torch = self._lazy_import_torch()
        transformers = self._lazy_import_transformers()

        try:
            # Load tokenizer
            logger.info(f"Loading tokenizer for {model_name}")
            tokenizer = self._load_tokenizer(model_path)
            if tokenizer is None:
                logger.error(f"Cannot proceed without tokenizer for {model_name}")
                return None, None

            # Setup model loading kwargs
            cache_dir = None
            if not local_path:
                cache_dir = self.models_dir / model_name.replace('/', '--')
                cache_dir.mkdir(exist_ok=True)

            hf_token = self._get_hf_token()
            model_kwargs = {
                "trust_remote_code": True,
                "device_map": "auto" if self.device == "cuda" else None,
                "low_cpu_mem_usage": True
            }

            if cache_dir:
                model_kwargs["cache_dir"] = str(cache_dir)
            if hf_token:
                model_kwargs["token"] = hf_token

            # Configure precision with better memory management
            if precision == "4-bit" and self.device == "cuda":
                try:
                    from transformers import BitsAndBytesConfig

                    # Check available memory to decide on CPU offloading
                    memory_status = self.get_memory_status()
                    available_memory = memory_status.get('available_memory_gb', 0)

                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )

                    # If low on VRAM, enable CPU offloading for quantized models
                    if available_memory < 6.0:  # Less than 6GB available
                        logger.warning(
                            f"Limited VRAM ({available_memory:.1f}GB), enabling CPU offloading for 4-bit model")
                        quantization_config.llm_int8_enable_fp32_cpu_offload = True
                        # Use a more conservative device_map
                        model_kwargs["device_map"] = "balanced_low_0"

                    model_kwargs["quantization_config"] = quantization_config

                except ImportError:
                    logger.warning("BitsAndBytesConfig not available, using FP16")
                    model_kwargs["torch_dtype"] = torch.float16
            elif precision == "FP16":
                model_kwargs["torch_dtype"] = torch.float16

            # Load model
            logger.info(f"Loading model {model_name}")
            try:
                model = transformers.AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
                logger.info(f"✓ Successfully loaded model {model_name}")
            except Exception as model_error:
                error_msg = str(model_error)
                logger.error(f"Model loading failed for {model_name}:")
                logger.error(f"  Error type: {type(model_error).__name__}")
                logger.error(f"  Error message: {error_msg}")
                logger.error(f"  Model path: {model_path}")
                logger.error(f"  Model kwargs: {list(model_kwargs.keys())}")
                self._handle_model_error(model_error, model_name)
                return None, None

            # Only move model manually if not using device_map auto-placement
            if model_kwargs.get("device_map") is None and (self.device == "cpu" or precision not in ["4-bit"]):
                model = model.to(self.device)
            elif model_kwargs.get("device_map") == "auto":
                logger.info(f"Model {model_name} using automatic device placement, skipping manual .to() call")

            self.loaded_models[model_name] = (model, tokenizer)
            return model, tokenizer

        except Exception as e:
            error_msg = str(e)

            # Handle device placement conflicts
            if "offloaded to cpu or disk" in error_msg:
                logger.error("=" * 60)
                logger.error("🔧 DEVICE PLACEMENT CONFLICT")
                logger.error("=" * 60)
                logger.error("Model loaded successfully but device placement failed.")
                logger.error("This happens when using device_map='auto'.")
                logger.error("The model is actually working - this is just a placement issue.")
                logger.error("=" * 60)
                # The model might actually be loaded and working despite this error
                # Let's try to return it anyway if it exists
                if 'model' in locals() and model is not None:
                    logger.warning("Returning model despite placement warning - it should work")
                    self.loaded_models[model_name] = (model, tokenizer)
                    return model, tokenizer

            logger.error(f"Failed to load {model_name}: {error_msg}")

            # Log the full error details for debugging
            logger.error(f"Full error details for {model_name}:")
            logger.error(f"  Error type: {type(e).__name__}")
            logger.error(f"  Error message: {error_msg}")

            self._handle_model_error(e, model_name)
            return None, None

    def _handle_model_error(self, error: Exception, model_name: str):
        """Handle model loading errors with helpful messages"""
        error_msg = str(error)

        if "dispatched on the CPU or the disk" in error_msg and "quantized model" in error_msg:
            logger.error("=" * 60)
            logger.error("💾 INSUFFICIENT VRAM FOR QUANTIZED MODEL")
            logger.error("=" * 60)
            logger.error("Not enough contiguous VRAM for the quantized model.")
            logger.error("Solutions:")
            logger.error("1. Load smaller models first (reorder in SETTINGS.md)")
            logger.error("2. Use CPU offloading: llm_int8_enable_fp32_cpu_offload=True")
            logger.error("3. Reduce other models' VRAM usage")
            logger.error("4. Use even more aggressive quantization")
            logger.error("=" * 60)
        elif "offloaded to cpu or disk" in error_msg:
            logger.error("=" * 60)
            logger.error("🔧 DEVICE PLACEMENT CONFLICT")
            logger.error("=" * 60)
            logger.error("Model loaded successfully but has device placement conflict.")
            logger.error("This is usually harmless - the model should still work.")
            logger.error("Caused by device_map='auto' + manual .to(device) call.")
            logger.error("=" * 60)
        elif "rope_scaling" in error_msg.lower():
            logger.error("=" * 60)
            logger.error("🔧 TRANSFORMERS VERSION COMPATIBILITY ISSUE")
            logger.error("=" * 60)
            logger.error("The model uses a newer rope_scaling format.")
            logger.error("Quick fixes:")
            logger.error("1. pip install --upgrade transformers")
            logger.error("2. pip install transformers>=4.38.0")
            logger.error("3. Alternative: Try mistralai/Mistral-7B-Instruct-v0.3")
            logger.error("4. Alternative: Try HuggingFaceH4/zephyr-7b-beta")
            logger.error("=" * 60)
        elif "gated repo" in error_msg.lower():
            logger.error("=" * 60)
            logger.error("🔒 MODEL ACCESS REQUIRED")
            logger.error("=" * 60)
            logger.error(f"Model '{model_name}' requires access approval.")
            logger.error(f"1. Visit: https://huggingface.co/{model_name}")
            logger.error("2. Request access and wait for approval")
            logger.error("3. Run: huggingface-cli login")
            logger.error("4. Alternative: Use 'HuggingFaceH4/zephyr-7b-beta'")
            logger.error("=" * 60)
        elif "not found" in error_msg.lower():
            logger.error("=" * 60)
            logger.error("🔍 MODEL NOT FOUND")
            logger.error("=" * 60)
            logger.error(f"Model '{model_name}' not found.")
            logger.error("Try: 'HuggingFaceH4/zephyr-7b-beta' (no auth required)")
            logger.error("=" * 60)
        elif "OutOfMemoryError" in error_msg or "CUDA out of memory" in error_msg:
            logger.error("=" * 60)
            logger.error("💾 OUT OF MEMORY")
            logger.error("=" * 60)
            logger.error("Solutions:")
            logger.error("• Use 4-bit precision instead of FP16")
            logger.error("• Close other applications")
            logger.error("• Try smaller models")
            logger.error("=" * 60)

    def unload_model(self, model_name: str) -> bool:
        """Unload model from memory"""
        if model_name not in self.loaded_models:
            return False

        try:
            del self.loaded_models[model_name]
            if self._torch and self.device == "cuda":
                self._torch.cuda.empty_cache()
            logger.info(f"✓ Unloaded {model_name}")
            return True
        except Exception as e:
            logger.error(f"Error unloading {model_name}: {e}")
            return False

    def is_model_loaded(self, model_name: str) -> bool:
        """Check if specific model is loaded"""
        return model_name in self.loaded_models

    def get_loaded_model_info(self) -> Dict[str, Dict[str, Any]]:
        """Get info about loaded models"""
        info = {}
        for model_name in self.loaded_models:
            info[model_name] = {
                'loaded': True,
                'shared_by_engines': []  # Could track which engines use this model
            }
        return info

    def get_memory_status(self) -> Dict[str, Any]:
        """Get current memory status with detailed VRAM info"""
        if self._torch and self.device == "cuda":
            # Get GPU memory info
            total_memory = self._torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            used_memory = self._torch.cuda.memory_allocated() / (1024 ** 3)
            reserved_memory = self._torch.cuda.memory_reserved() / (1024 ** 3)
            free_memory = total_memory - reserved_memory

            return {
                'total_memory_gb': total_memory,
                'used_memory_gb': used_memory,
                'reserved_memory_gb': reserved_memory,
                'free_memory_gb': free_memory,
                'available_memory_gb': total_memory - used_memory,
                'loaded_model_count': len(self.loaded_models),
                'device': self.device,
                'memory_details': {
                    'allocated_gb': used_memory,
                    'cached_gb': reserved_memory - used_memory,
                    'free_gb': free_memory
                }
            }
        else:
            # Fallback estimates for CPU
            total_memory = 16.0  # Assume 16GB
            used_memory = len(self.loaded_models) * 8.0  # Rough estimate

            return {
                'total_memory_gb': total_memory,
                'used_memory_gb': used_memory,
                'available_memory_gb': total_memory - used_memory,
                'loaded_model_count': len(self.loaded_models),
                'device': self.device or 'cpu'
            }


class BaseEngine(ABC):
    """Base engine with essential functionality - uses shared models"""

    def __init__(self, config: Dict[str, Any], model_loader: ModelLoader, persona: str = ""):
        self.config = config
        self.category = config.get('category', 'unknown')
        self.technical_model_name = config.get('technical_model_name', '')
        self.model_loader = model_loader
        self.persona = persona
        # Don't store model/tokenizer directly - get from shared loader
        self.generation_count = 0

    async def load_model(self) -> bool:
        """Load the model for this engine (shared via model loader)"""
        try:
            logger.debug(f"Requesting shared model for {self.category} using {self.technical_model_name}")
            model, tokenizer = await self.model_loader.load_model(
                self.technical_model_name,
                self.config.get('precision', 'FP16')
            )

            if model and tokenizer:
                logger.debug(f"Successfully got shared model for {self.category}")
                return True
            else:
                logger.error(f"Model loader returned None for {self.category}: model={model}, tokenizer={tokenizer}")
                return False
        except Exception as e:
            logger.error(f"Exception in load_model for {self.category}: {type(e).__name__}: {e}")
            return False

    async def unload_model(self) -> bool:
        """Unload is handled by the shared model loader"""
        # Don't actually unload since other engines might be using it
        # Let the model loader handle lifecycle
        return True

    def is_loaded(self) -> bool:
        """Check if the shared model is loaded"""
        return self.model_loader.is_model_loaded(self.technical_model_name)

    @property
    def model(self):
        """Get the shared model instance"""
        if self.technical_model_name in self.model_loader.loaded_models:
            return self.model_loader.loaded_models[self.technical_model_name][0]
        return None

    @property
    def tokenizer(self):
        """Get the shared tokenizer instance"""
        if self.technical_model_name in self.model_loader.loaded_models:
            return self.model_loader.loaded_models[self.technical_model_name][1]
        return None

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using shared model"""
        if not self.model or not self.tokenizer:
            raise RuntimeError(f"Model not loaded for {self.category}")

        try:
            logger.info(f"[{self.category}] Starting generation for: '{prompt[:50]}...'")

            full_prompt = self._prepare_prompt(prompt)
            logger.info(f"[{self.category}] Prepared prompt")

            # Simple tokenization
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024  # Shorter for faster processing
            )
            logger.info(f"[{self.category}] Tokenized input - shape: {inputs['input_ids'].shape}")

            # Check device placement before generation
            model_device = next(self.model.parameters()).device
            logger.info(f"[{self.category}] Model device: {model_device}")

            # Move inputs to same device as model
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
            logger.info(f"[{self.category}] Moved inputs to device: {model_device}")

            # Very conservative generation parameters
            max_new_tokens = min(kwargs.get('max_tokens', 50), 50)  # Very short for testing
            temperature = 0.7

            logger.info(f"[{self.category}] About to call model.generate() with max_tokens={max_new_tokens}")

            # Import torch for no_grad
            torch = self.model_loader._lazy_import_torch()

            start_time = time.time()
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs.get('attention_mask'),
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    early_stopping=True,
                    num_beams=1,
                    repetition_penalty=1.1,
                    use_cache=True
                )

            generation_time = time.time() - start_time
            logger.info(f"[{self.category}] Generation completed in {generation_time:.2f} seconds")

            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            logger.info(f"[{self.category}] Decoded response: '{response[:100]}...'")

            self.generation_count += 1
            return response.strip()

        except Exception as e:
            logger.error(f"[{self.category}] Generation error: {e}")
            import traceback
            logger.error(f"[{self.category}] Traceback: {traceback.format_exc()}")
            return f"Error in {self.category}: {str(e)}"

    def _prepare_prompt(self, user_prompt: str) -> str:
        model_name = self.technical_model_name.lower()

        if "mistral" in model_name:
            return f"[INST] {user_prompt.strip()} [/INST]"

        system_prompt = self.get_system_prompt()
        if system_prompt:
            return f"{system_prompt}\n\n{user_prompt.strip()}"
        return user_prompt.strip()

    @abstractmethod
    def get_system_prompt(self) -> Optional[str]:
        """Get system prompt for this engine type"""
        pass

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage for this engine"""
        if not self.is_loaded():
            return {'memory_gb': 0, 'status': 'not_loaded'}

        precision = self.config.get('precision', 'FP16')
        if '4-bit' in precision:
            estimated_memory = 4.0
        elif 'FP16' in precision:
            estimated_memory = 8.0
        else:
            estimated_memory = 16.0

        return {'memory_gb': estimated_memory, 'status': 'loaded', 'precision': precision}

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        return {
            'engine': self.category,
            'model': self.technical_model_name,
            'loaded': self.is_loaded(),
            'memory_usage': self.get_memory_usage(),
            'generation_count': self.generation_count,
            'status': 'healthy' if self.is_loaded() else 'unloaded'
        }


# Specialized Engine Classes
class ConversationalChatEngine(BaseEngine):
    def get_system_prompt(self) -> Optional[str]:
        if self.persona:
            return f"System: {self.persona}\n\nYou are a helpful assistant."
        return "You are a helpful assistant."


class CodeCompletionEngine(BaseEngine):
    def get_system_prompt(self) -> Optional[str]:
        return "Complete the following code. Provide only the completion, no explanations."


class CodeGenerationEngine(BaseEngine):
    def get_system_prompt(self) -> Optional[str]:
        return "You are an expert programmer. Provide clear, well-commented code solutions."


class MathematicalReasoningEngine(BaseEngine):
    def get_system_prompt(self) -> Optional[str]:
        return "You are a mathematics expert. Show your work step-by-step and explain your reasoning clearly."


class TranslationEngine(BaseEngine):
    def get_system_prompt(self) -> Optional[str]:
        return "You are a professional translator. Provide accurate translations while preserving meaning and context."


class GeneralReasoningEngine(BaseEngine):
    def get_system_prompt(self) -> Optional[str]:
        if self.persona:
            return f"System: {self.persona}\n\nAnalyze and reason through problems systematically."
        return "Analyze and reason through problems systematically and logically."


class CreativeWritingEngine(BaseEngine):
    def get_system_prompt(self) -> Optional[str]:
        return "You are a creative writer. Craft engaging, original content with vivid descriptions and compelling narratives."


class InstructionFollowingEngine(BaseEngine):
    def get_system_prompt(self) -> Optional[str]:
        return "Follow instructions precisely and completely. Break down complex tasks into clear steps."


class SummarizationEngine(BaseEngine):
    def get_system_prompt(self) -> Optional[str]:
        return "Provide concise, accurate summaries that capture the key points and essential information."


class QuestionAnsweringEngine(BaseEngine):
    def get_system_prompt(self) -> Optional[str]:
        if self.persona:
            return f"System: {self.persona}\n\nProvide accurate, informative answers based on knowledge."
        return "Provide accurate, informative answers. Cite sources when possible and acknowledge limitations."


class ScientificResearchEngine(BaseEngine):
    def get_system_prompt(self) -> Optional[str]:
        return "You are a scientific researcher. Provide evidence-based analysis, cite relevant studies, and follow scientific methodology."


class LegalAnalysisEngine(BaseEngine):
    def get_system_prompt(self) -> Optional[str]:
        return "Analyze legal documents and situations. Provide thorough analysis while noting this is not legal advice."


class CodeReviewEngine(BaseEngine):
    def get_system_prompt(self) -> Optional[str]:
        return "Review code for quality, security, performance, and best practices. Provide constructive feedback and suggestions."


class LongContextEngine(BaseEngine):
    def get_system_prompt(self) -> Optional[str]:
        return "Process and analyze long documents or complex information. Maintain context throughout your analysis."


# MCP Protocol Support
@dataclass
class MCPRequest:
    id: str
    method: str
    params: Dict[str, Any]
    jsonrpc: str = "2.0"


@dataclass
class MCPResponse:
    id: str
    result: Any = None
    error: Optional[Dict[str, Any]] = None
    jsonrpc: str = "2.0"


class MCPServer:
    """Model Context Protocol server implementation"""

    def __init__(self, conductor):
        self.conductor = conductor
        self.client_capabilities = {}
        self.method_handlers = {
            'initialize': self._handle_initialize,
            'initialized': self._handle_initialized,
            'tools/list': self._handle_list_tools,
            'tools/call': self._handle_call_tool,
            'resources/list': self._handle_list_resources,
            'resources/read': self._handle_read_resource,
            'generation/generate': self._handle_generation,
            'models/list': self._handle_list_models,
            'system/health': self._handle_health_check,
            'system/stats': self._handle_system_stats,
            'system/dependencies': self._handle_dependencies_check,
        }

    async def handle_request(self, request_data: str) -> str:
        """Handle incoming MCP request"""
        try:
            request_dict = json.loads(request_data)
            if 'id' not in request_dict:
                return ""  # Notification

            request = MCPRequest(
                id=request_dict['id'],
                method=request_dict['method'],
                params=request_dict.get('params', {}),
                jsonrpc=request_dict.get('jsonrpc', '2.0')
            )

            response = await self._handle_request(request)
            return json.dumps(asdict(response))

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in request: {e}")
            return json.dumps(asdict(MCPResponse(id="unknown", error={"code": -32700, "message": "Parse error"})))
        except Exception as e:
            logger.error(f"Error handling request: {e}")
            return json.dumps(asdict(MCPResponse(
                id=request_dict.get('id', 'unknown'),
                error={"code": -32603, "message": "Internal error", "data": str(e)}
            )))

    async def _handle_request(self, request: MCPRequest) -> MCPResponse:
        """Handle an MCP request and return response"""
        handler = self.method_handlers.get(request.method)
        if not handler:
            return MCPResponse(id=request.id, error={"code": -32601, "message": f"Method not found: {request.method}"})

        try:
            result = await handler(request.params)
            return MCPResponse(id=request.id, result=result)
        except Exception as e:
            logger.error(f"Error in handler {request.method}: {e}")
            return MCPResponse(id=request.id, error={"code": -32603, "message": "Internal error", "data": str(e)})

    # MCP Method Handlers
    async def _handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self.client_capabilities = params.get('capabilities', {})
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "resources": {"subscribe": True, "listChanged": True},
                "tools": {"listChanged": True},
                "generation": {"streaming": False, "categories": list(self.conductor.engine_configs.keys())}
            },
            "serverInfo": {"name": "Conductor LLM Server", "version": "1.0.0", "description": "Multi-model LLM server"}
        }

    async def _handle_initialized(self, params: Dict[str, Any]) -> None:
        logger.info("MCP client initialization complete")
        return None

    async def _handle_list_tools(self, params: Dict[str, Any]) -> Dict[str, Any]:
        tools = [
            {
                "name": "generate_text",
                "description": "Generate text using specified category engine",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string", "description": "Input prompt"},
                        "category": {"type": "string", "description": "Task category"},
                        "max_tokens": {"type": "integer", "description": "Maximum tokens"},
                        "temperature": {"type": "number", "description": "Generation temperature"}
                    },
                    "required": ["prompt"]
                }
            },
            {
                "name": "classify_prompt",
                "description": "Classify a prompt to determine appropriate engine category",
                "inputSchema": {
                    "type": "object",
                    "properties": {"prompt": {"type": "string", "description": "Prompt to classify"}},
                    "required": ["prompt"]
                }
            }
        ]
        return {"tools": tools}

    async def _handle_call_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        tool_name = params.get('name')
        arguments = params.get('arguments', {})

        if tool_name == "generate_text":
            prompt = arguments.get('prompt', '')
            category = arguments.get('category')
            category_used, response = await self.conductor.generate(prompt, category, **arguments)
            return {"content": [{"type": "text", "text": f"[{category_used}]: {response}"}]}
        elif tool_name == "classify_prompt":
            prompt = arguments.get('prompt', '')
            category = await self.conductor.classify_prompt(prompt)
            return {"content": [{"type": "text", "text": f"Classified as: {category}"}]}
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

    async def _handle_list_resources(self, params: Dict[str, Any]) -> Dict[str, Any]:
        resources = [
            {"uri": "conductor://models", "name": "Model Registry",
             "description": "Information about all registered models", "mimeType": "application/json"},
            {"uri": "conductor://config", "name": "Configuration", "description": "Current server configuration",
             "mimeType": "application/json"},
            {"uri": "conductor://engines", "name": "Engine Status", "description": "Status of all LLM engines",
             "mimeType": "application/json"},
            {"uri": "conductor://dependencies", "name": "Dependencies Status",
             "description": "System requirements and dependency status", "mimeType": "application/json"}
        ]
        return {"resources": resources}

    async def _handle_read_resource(self, params: Dict[str, Any]) -> Dict[str, Any]:
        uri = params.get('uri')

        if uri == "conductor://models":
            models = {name: str(type(engine).__name__) for name, engine in self.conductor.engines.items()}
            return {"contents": [{"uri": uri, "mimeType": "application/json", "text": json.dumps(models, indent=2)}]}
        elif uri == "conductor://config":
            config_summary = self.conductor.config_parser.get_configuration_summary()
            return {"contents": [
                {"uri": uri, "mimeType": "application/json", "text": json.dumps(config_summary, indent=2)}]}
        elif uri == "conductor://engines":
            engine_status = {category: await engine.health_check() for category, engine in
                             self.conductor.engines.items()}
            return {
                "contents": [{"uri": uri, "mimeType": "application/json", "text": json.dumps(engine_status, indent=2)}]}
        elif uri == "conductor://dependencies":
            dependency_report = self.conductor.dependencies_loader.get_dependency_report()
            return {"contents": [
                {"uri": uri, "mimeType": "application/json", "text": json.dumps(dependency_report, indent=2)}]}
        else:
            raise ValueError(f"Unknown resource URI: {uri}")

    async def _handle_generation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        prompt = params.get('prompt', '')
        category = params.get('category')
        category_used, response = await self.conductor.generate(prompt, category, **params)
        return {"response": response, "category": category_used,
                "metadata": {"engine_loaded": category_used in self.conductor.engines}}

    async def _handle_list_models(self, params: Dict[str, Any]) -> Dict[str, Any]:
        models = {}
        for category, config in self.conductor.engine_configs.items():
            models[category] = {
                "technical_name": config['technical_model_name'],
                "stay_loaded": config['stay_loaded'],
                "precision": config['precision'],
                "loaded": category in self.conductor.engines
            }
        return {"models": models}

    async def _handle_health_check(self, params: Dict[str, Any]) -> Dict[str, Any]:
        status = self.conductor.get_status()
        memory_status = self.conductor.model_loader.get_memory_status()
        return {"status": "healthy", "engines": status, "memory": memory_status, "timestamp": time.time()}

    async def _handle_system_stats(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "engines": self.conductor.get_status(),
            "memory": self.conductor.model_loader.get_memory_status(),
            "loaded_models": list(self.conductor.model_loader.loaded_models.keys())
        }

    async def _handle_dependencies_check(self, params: Dict[str, Any]) -> Dict[str, Any]:
        model_name = params.get('model_name')
        precision = params.get('precision', 'FP16')
        dependency_report = self.conductor.dependencies_loader.get_dependency_report()

        result = {
            "system_info": dependency_report['system'],
            "package_status": dependency_report['packages'],
            "recommendations": dependency_report['recommendations']
        }

        if model_name:
            validation = self.conductor.dependencies_loader.validate_model_requirements(model_name, precision)
            missing_packages = self.conductor.dependencies_loader.get_missing_packages(['torch', 'transformers'])
            result['model_validation'] = {
                'model_name': model_name,
                'precision': precision,
                'can_load': validation['can_load'],
                'warnings': validation['warnings'],
                'recommendations': validation['recommendations'],
                'missing_packages': missing_packages
            }
            if missing_packages:
                result['installation_instructions'] = self.conductor.dependencies_loader.get_installation_instructions(
                    missing_packages)

        return result


class Conductor:
    """Main conductor with MCP support and efficient model sharing"""

    def __init__(self, config_path: str = "conductor/SETTINGS.md", models_dir: str = "./models"):
        self.config_parser = ConfigParser(config_path)
        self.model_loader = ModelLoader(models_dir)
        self.dependencies_loader = DependenciesLoader()
        self.engines: Dict[str, BaseEngine] = {}
        self.engine_configs: Dict[str, Dict[str, Any]] = {}
        self.persona = ""
        self.mcp_server: Optional[MCPServer] = None

        # Engine class mapping
        self.engine_classes = {
            'conversational_chat': ConversationalChatEngine,
            'code_completion': CodeCompletionEngine,
            'code_generation': CodeGenerationEngine,
            'mathematical_reasoning': MathematicalReasoningEngine,
            'translation': TranslationEngine,
            'general_reasoning': GeneralReasoningEngine,
            'creative_writing': CreativeWritingEngine,
            'instruction_following': InstructionFollowingEngine,
            'summarization': SummarizationEngine,
            'question_answering': QuestionAnsweringEngine,
            'scientific_research': ScientificResearchEngine,
            'legal_analysis': LegalAnalysisEngine,
            'code_review': CodeReviewEngine,
            'long_context': LongContextEngine
        }

    async def initialize(self, skip_model_loading: bool = False) -> bool:
        """Initialize the conductor"""
        try:
            logger.info("=== Initializing Conductor with MCP ===")

            # Check system requirements
            system_info = self.dependencies_loader.check_system_requirements()
            logger.info("=== System Requirements ===")
            logger.info(f"Platform: {system_info['platform']}")
            logger.info(f"Python: {system_info['python_version']}")
            logger.info(f"Memory: {system_info['memory_gb']:.1f} GB")
            logger.info(f"CUDA available: {system_info['cuda_available']}")
            if system_info['cuda_available']:
                logger.info(f"GPU count: {system_info['gpu_count']}")

            # Parse configuration
            self.engine_configs = self.config_parser.parse_settings()
            if not self.engine_configs:
                logger.error("Failed to parse configuration")
                return False

            self.persona = self.config_parser.get_conversational_persona()
            logger.info(f"Loaded {len(self.engine_configs)} engine configurations")

            # Initialize MCP server
            self.mcp_server = MCPServer(self)
            logger.info("✓ MCP server initialized")

            # Load models if not skipping
            if not skip_model_loading:
                await self._ensure_dependencies()
                await self._load_stay_loaded_models()

            logger.info("=== Conductor with MCP initialized ===")
            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False

    async def _ensure_dependencies(self):
        """Ensure all required dependencies are available"""
        logger.info("Checking dependencies for stay-loaded models...")
        stay_loaded_configs = [(category, config) for category, config in self.engine_configs.items() if
                               config.get('stay_loaded', False)]

        # Group by technical model name to avoid duplicate dependency checks
        unique_models = {}
        for category, config in stay_loaded_configs:
            technical_name = config['technical_model_name']
            precision = config.get('precision', 'FP16')
            if technical_name not in unique_models:
                unique_models[technical_name] = precision

        logger.info(f"Checking dependencies for {len(unique_models)} unique models...")

        for model_name, precision in unique_models.items():
            logger.info(f"Checking dependencies for {model_name} ({precision})")
            deps_ok = await self.dependencies_loader.ensure_dependencies(model_name, precision)
            if not deps_ok:
                logger.warning(f"Some dependencies missing for {model_name}")
                missing_packages = self.dependencies_loader.get_missing_packages(['torch', 'transformers'])
                if 'bitsandbytes' not in missing_packages and precision in ['4-bit', '8-bit']:
                    missing_packages.append('bitsandbytes')

                if missing_packages:
                    instructions = self.dependencies_loader.get_installation_instructions(missing_packages)
                    logger.info("Install missing packages:")
                    for pkg, instruction in instructions.items():
                        logger.info(f"  {instruction}")

            # Validate model requirements
            validation = self.dependencies_loader.validate_model_requirements(model_name, precision)
            if not validation['can_load']:
                logger.error(f"Cannot load {model_name}: insufficient resources")
                for warning in validation['warnings']:
                    logger.warning(f"  {warning}")
                for rec in validation['recommendations']:
                    logger.info(f"  Recommendation: {rec}")
            else:
                if validation['warnings']:
                    for warning in validation['warnings']:
                        logger.warning(f"  {warning}")

    async def _load_stay_loaded_models(self):
        """Load models that should stay loaded (shared across engines) with smart memory management"""
        stay_loaded = [(category, config) for category, config in self.engine_configs.items() if
                       config.get('stay_loaded', False)]
        logger.info(f"Loading {len(stay_loaded)} persistent models...")

        # Group engines by technical_model_name to avoid duplicate loading
        models_to_load = {}
        engines_by_model = {}

        for category, config in stay_loaded:
            technical_name = config['technical_model_name']
            if technical_name not in models_to_load:
                models_to_load[technical_name] = config
                engines_by_model[technical_name] = []
            engines_by_model[technical_name].append((category, config))

        logger.info(f"Will load {len(models_to_load)} unique models shared by {len(stay_loaded)} engines:")
        for technical_name, engine_list in engines_by_model.items():
            categories = [cat for cat, cfg in engine_list]
            logger.info(f"  {technical_name} → {categories}")

        # Sort models by memory efficiency (4-bit models first, then smaller models)
        def get_model_priority(item):
            technical_name, config = item
            precision = config.get('precision', 'FP16')
            # 4-bit models get highest priority (lowest number)
            if precision == '4-bit':
                return 1
            # Then by estimated size (smaller first)
            if '7b' in technical_name.lower():
                return 2
            elif '8b' in technical_name.lower():
                return 3
            else:
                return 4

        sorted_models = sorted(models_to_load.items(), key=get_model_priority)
        logger.info("Loading models in memory-efficient order:")
        for technical_name, config in sorted_models:
            precision = config.get('precision', 'FP16')
            logger.info(f"  {technical_name} ({precision})")

        # Load each unique model once in memory-efficient order
        for technical_name, config in sorted_models:
            logger.info(f"Loading shared model: {technical_name}")

            # Check available memory before loading
            memory_status = self.model_loader.get_memory_status()
            available_memory = memory_status.get('available_memory_gb', 0)
            logger.info(f"Available VRAM before loading: {available_memory:.1f}GB")

            # Load the model once
            model, tokenizer = await self.model_loader.load_model(
                technical_name,
                config.get('precision', 'FP16')
            )

            if model and tokenizer:
                # Create engines for all categories that use this model
                for category, engine_config in engines_by_model[technical_name]:
                    try:
                        await self._create_engine_for_category(category, engine_config)
                    except Exception as e:
                        logger.error(f"Failed to create engine {category}: {e}")
                logger.info(
                    f"✓ Model {technical_name} loaded and shared by {len(engines_by_model[technical_name])} engines")

                # Show updated memory status
                memory_status = self.model_loader.get_memory_status()
                logger.info(
                    f"VRAM after loading: {memory_status.get('used_memory_gb', 0):.1f}GB used, {memory_status.get('available_memory_gb', 0):.1f}GB available")
            else:
                logger.error(f"✗ Failed to load shared model: {technical_name}")
                # Skip creating engines for this model
                for category, _ in engines_by_model[technical_name]:
                    logger.error(f"✗ Skipping engine {category} (model failed to load)")

                # If we have limited memory, suggest alternatives
                memory_status = self.model_loader.get_memory_status()
                if memory_status.get('available_memory_gb', 0) < 6:
                    logger.error(f"Low VRAM available ({memory_status.get('available_memory_gb', 0):.1f}GB). Consider:")
                    logger.error("1. Using smaller models")
                    logger.error("2. Using more aggressive quantization")
                    logger.error("3. Loading fewer models simultaneously")

    async def _create_engine_for_category(self, category: str, config: Dict[str, Any]):
        """Create an engine for a category (assumes model is already loaded)"""
        engine_class = self.engine_classes.get(category, ConversationalChatEngine)

        # Create engine with special handling for conversational engines
        if 'conversational' in category or 'general_reasoning' in category:
            engine = engine_class(config, self.model_loader, self.persona)
        else:
            engine = engine_class(config, self.model_loader)

        # Engine doesn't need to load model - it's already loaded and shared
        # Just verify the model is available
        if engine.is_loaded():
            self.engines[category] = engine
            logger.info(f"✓ Created engine: {category}")
        else:
            logger.error(f"✗ Cannot create engine {category}: shared model not available")

    async def _create_and_load_engine(self, category: str, config: Dict[str, Any]):
        """Create and load an engine (for on-demand loading)"""
        technical_name = config['technical_model_name']

        # Check if model is already loaded
        if not self.model_loader.is_model_loaded(technical_name):
            logger.info(f"Loading model on-demand: {technical_name}")
            model, tokenizer = await self.model_loader.load_model(
                technical_name,
                config.get('precision', 'FP16')
            )
            if not (model and tokenizer):
                logger.error(f"Failed to load model {technical_name} for on-demand engine {category}")
                return

        # Create the engine (model is now loaded)
        await self._create_engine_for_category(category, config)

    async def get_engine(self, category: str) -> Optional[BaseEngine]:
        """Get engine for category, loading on-demand if needed"""
        if category in self.engines:
            return self.engines[category]

        # Try to load on-demand
        if category in self.engine_configs:
            config = self.engine_configs[category]
            try:
                await self._create_and_load_engine(category, config)
                return self.engines.get(category)
            except Exception as e:
                logger.error(f"Failed to load on-demand engine {category}: {e}")

        return None

    async def classify_prompt(self, prompt: str) -> str:
        """Intelligent prompt classification using general reasoning engine if available"""
        # Try to use general_reasoning engine for classification
        reasoning_engine = await self.get_engine('general_reasoning')

        if reasoning_engine:
            try:
                classification_prompt = f"""Analyze this user prompt and determine which specialized AI engine would be most appropriate to handle it.

Available engine categories:
- conversational_chat: General conversation, greetings, casual questions
- code_completion: Code completion, filling in partial code snippets
- code_generation: Creating new code, algorithms, full programs
- mathematical_reasoning: Math problems, equations, calculations, proofs
- translation: Translating between languages
- creative_writing: Stories, poems, creative content, fiction
- instruction_following: Step-by-step tasks, following complex instructions
- summarization: Summarizing documents, articles, long texts
- question_answering: Factual questions, explanations, informational queries
- scientific_research: Scientific analysis, research papers, academic content
- legal_analysis: Legal documents, contracts, legal advice
- code_review: Reviewing code quality, security, best practices
- long_context: Processing very long documents, complex analysis

User prompt: "{prompt}"

Respond with ONLY the category name (e.g., "code_generation" or "mathematical_reasoning"). Choose the most specific and appropriate category."""

                classification_result = await reasoning_engine.generate(classification_prompt, max_tokens=20,
                                                                        temperature=0.3)
                predicted_category = classification_result.strip().lower()

                # Validate against available categories
                if predicted_category in self.engine_configs:
                    logger.info(f"Classified prompt as: {predicted_category}")
                    return predicted_category

                # Try partial matches
                for category in self.engine_configs.keys():
                    if category in predicted_category or predicted_category in category:
                        logger.info(f"Classified prompt as: {category} (partial match)")
                        return category

            except Exception as e:
                logger.error(f"Error in AI classification: {e}")

        # Fallback to simple keyword-based classification
        return self._fallback_classify(prompt)

    def _fallback_classify(self, prompt: str) -> str:
        """Fallback keyword-based classification"""
        prompt_lower = prompt.lower()

        if any(word in prompt_lower for word in
               ['code', 'function', 'class', 'import', 'def ', 'var ', 'python', 'javascript']):
            return 'code_generation'
        elif any(word in prompt_lower for word in ['complete', 'finish', '...']):
            return 'code_completion'
        elif any(word in prompt_lower for word in
                 ['math', 'calculate', 'solve', 'equation', '+', '-', '*', '/', 'integral', 'derivative']):
            return 'mathematical_reasoning'
        elif any(word in prompt_lower for word in
                 ['translate', 'translation', 'spanish', 'french', 'german', 'chinese']):
            return 'translation'
        elif any(word in prompt_lower for word in ['story', 'poem', 'creative', 'write', 'fiction', 'novel']):
            return 'creative_writing'
        elif any(word in prompt_lower for word in ['summarize', 'summary', 'tldr', 'brief']):
            return 'summarization'
        elif any(word in prompt_lower for word in ['review', 'check', 'analyze code', 'bug']):
            return 'code_review'
        elif any(word in prompt_lower for word in ['research', 'study', 'scientific', 'academic', 'paper']):
            return 'scientific_research'
        elif any(word in prompt_lower for word in ['legal', 'contract', 'law', 'court']):
            return 'legal_analysis'
        elif len(prompt.split()) > 500:  # Very long prompts
            return 'long_context'
        else:
            return 'conversational_chat'

    async def generate(self, prompt: str, category: Optional[str] = None, **kwargs) -> tuple[str, str]:
        """Generate response, with automatic category detection if not specified"""
        if not category:
            category = await self.classify_prompt(prompt)

        engine = await self.get_engine(category)
        if not engine:
            # Fallback to conversational
            engine = await self.get_engine('conversational_chat')
            if not engine:
                raise RuntimeError("No engines available")

        response = await engine.generate(prompt, **kwargs)
        return category, response

    async def handle_mcp_request(self, request_data: str) -> str:
        """Handle MCP request"""
        if not self.mcp_server:
            raise RuntimeError("MCP server not initialized")
        return await self.mcp_server.handle_request(request_data)

    def get_status(self) -> Dict[str, Any]:
        """Get conductor status with model sharing info"""
        # Calculate model sharing statistics
        model_usage = {}
        for category, engine in self.engines.items():
            model_name = engine.technical_model_name
            if model_name not in model_usage:
                model_usage[model_name] = []
            model_usage[model_name].append(category)

        return {
            'engines_loaded': len(self.engines),
            'categories_configured': len(self.engine_configs),
            'unique_models_loaded': len(self.model_loader.loaded_models),
            'model_sharing': model_usage,
            'device': self.model_loader.device,
            'memory_status': self.model_loader.get_memory_status(),
            'mcp_enabled': self.mcp_server is not None
        }


# CLI interface
async def cli_interface(conductor: Conductor):
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
                status = conductor.get_status()
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
                            print(f"  {model_name} -> {subdir.relative_to(Path.cwd())}")
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
                            prompt = "Hi"
                            prepared_prompt = engine._prepare_prompt(prompt)
                            print(f"\n[Test Prepared Prompt]:\n{prepared_prompt}\n")

                            response = await asyncio.wait_for(
                                engine.generate(prompt, max_tokens=10, session_id=session_id),
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
            elif user_input:
                try:
                    print(f"Generating response for: '{user_input}'...")

                    # Add timeout to prevent hanging
                    category, response = await asyncio.wait_for(
                        conductor.generate(user_input),
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


# HTTP server support
async def start_http_server(conductor: Conductor, host: str = "localhost", port: int = 8000):
    """Start HTTP server with lazy FastAPI import"""
    try:
        from fastapi import FastAPI, HTTPException
        import uvicorn
    except ImportError:
        logger.error("FastAPI not available. Install with: pip install fastapi uvicorn")
        return

    app = FastAPI(title="Conductor with MCP", version="1.0.0")

    @app.post("/generate")
    async def generate(request: dict):
        prompt = request.get('prompt', '')
        category = request.get('category')
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt required")
        try:
            category_used, response = await conductor.generate(prompt, category, **request)
            return {"response": response, "category": category_used, "status": "success"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/mcp")
    async def mcp_handler(request: dict):
        try:
            request_json = json.dumps(request)
            response_json = await conductor.handle_mcp_request(request_json)
            if response_json:
                return json.loads(response_json)
            else:
                return {"status": "notification_processed"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/status")
    async def status():
        return conductor.get_status()

    @app.get("/health")
    async def health():
        return {"status": "healthy", "mcp_enabled": True}

    @app.get("/engines")
    async def list_engines():
        return {"loaded": list(conductor.engines.keys()), "configured": list(conductor.engine_configs.keys())}

    @app.get("/dependencies")
    async def check_dependencies():
        return conductor.dependencies_loader.get_dependency_report()

    logger.info(f"Starting HTTP server with MCP support on {host}:{port}")
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Conductor LLM Server with MCP")
    parser.add_argument("--config", default="conductor/SETTINGS.md", help="Config file path")
    parser.add_argument("--skip-model-loading", action="store_true", help="Skip loading models")
    parser.add_argument("--http", action="store_true", help="Start HTTP server")
    parser.add_argument("--mcp-only", action="store_true", help="Run as MCP server only")
    parser.add_argument("--host", default="localhost", help="HTTP host")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port")
    parser.add_argument("--log-level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO',
                        help="Logging level")

    args = parser.parse_args()

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Create conductor
    conductor = Conductor(args.config)

    # Initialize
    success = await conductor.initialize(skip_model_loading=args.skip_model_loading)
    if not success:
        logger.error("Failed to initialize conductor")
        sys.exit(1)

    # Start appropriate interface
    try:
        if args.mcp_only:
            logger.info("Running in MCP-only mode")
            logger.info("MCP server ready for requests")
            while True:
                await asyncio.sleep(1)
        elif args.http:
            await start_http_server(conductor, args.host, args.port)
        else:
            await cli_interface(conductor)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Runtime error: {e}")


if __name__ == "__main__":
    asyncio.run(main())