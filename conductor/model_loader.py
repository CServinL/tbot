import torch
import asyncio
import logging
import time
import gc
import os
import importlib.util
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer
)
import psutil
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class ModelInfo:
    technical_name: str
    max_context_window: int = 2048
    max_new_tokens: int = 1024
    description: str = ""
    special_flags: dict = field(default_factory=dict)
    stop_patterns: list = field(default_factory=list)

class ModelLoader:
    """Enhanced model loader with better error handling and model sharing"""

    def __init__(self, models_dir: str = "./models"):
        self.loaded_models: Dict[str, Tuple[PreTrainedModel, PreTrainedTokenizer]] = {}
        self.device = None
        self._torch = None
        self._transformers = None
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.model_infos: Dict[str, ModelInfo] = {}
        self.max_memory_gb = self._detect_max_memory_gb()
        self.total_memory_used = 0.0
        self.model_metadata = {}
        logger.info(f"Model cache directory: {self.models_dir.absolute()}")

    def _detect_max_memory_gb(self):
        """
        Detect the maximum available GPU memory (in GB), or fallback to system RAM if CUDA is not available.
        """
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        except Exception:
            pass
        # Fallback to system RAM
        import psutil
        return psutil.virtual_memory().total / 1024 ** 3

    def load_model_info_from_file(self, technical_name: str) -> ModelInfo:
        """(Deprecated) Load model information from a JSON file. Not used in .py ModelInfo workflow."""
        logger.warning("load_model_info_from_file is deprecated and not used.")
        return ModelInfo(technical_name=technical_name)

    def register_model_info(self, model_info: ModelInfo):
        """Register model information.

        Args:
            model_info (ModelInfo): The model information to register.
        """
        self.model_infos[model_info.technical_name] = model_info
        logger.info(f"Registered model info: {model_info.technical_name}")

    def get_model_info(self, technical_name: str):
        """
        Load ModelInfo from a Python file using the double-dash naming convention.
        E.g., google/flan-t5-large -> google--flan-t5-large.py
        """
        model_info_dir = getattr(self, "model_info_dir", None)
        if not model_info_dir:
            return None
        base = technical_name.replace("/", "--")
        path = os.path.join(model_info_dir, f"{base}.py")
        if not os.path.isfile(path):
            return None
        # Dynamically import the ModelInfo from the .py file
        spec = importlib.util.spec_from_file_location(f"model_info_{base}", path)
        if not spec or not spec.loader:
            return None
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception:
            return None
        # The ModelInfo instance must be named 'model_info' in the .py file
        return getattr(module, "model_info", None)

    def lazy_import_torch(self):
        if self._torch is None:
            try:
                import torch
                self._torch = torch
                logger.info("Torch module imported")
            except ImportError:
                logger.error("Torch module not found")
                self._torch = None
        return self._torch

    def _lazy_import_transformers(self):
        """Lazy import of transformers module."""
        if self._transformers is None:
            try:
                from transformers import (
                    AutoTokenizer,
                    AutoModelForCausalLM,
                    BitsAndBytesConfig,
                    PreTrainedModel,
                    PreTrainedTokenizer
                )
                self._transformers = (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, PreTrainedModel, PreTrainedTokenizer)
                logger.info("Transformers module imported")
            except ImportError:
                logger.error("Transformers module not found")
                raise

    def _get_local_model_path(self, model_name: str) -> Optional[Path]:
        """Get the local file path of a model.

        Args:
            model_name (str): The name of the model.

        Returns:
            Optional[Path]: The file path of the model, or None if not found.
        """
        model_path = self.models_dir / model_name
        if model_path.exists():
            return model_path
        else:
            return None

    def _get_hf_token(self) -> Optional[str]:
        """Get Hugging Face token from environment or cache.

        Returns:
            Optional[str]: The Hugging Face token, or None if not available.
        """
        return os.getenv("HF_TOKEN")

    def _load_tokenizer(self, model_name: str) -> Optional['PreTrainedTokenizer']:
        """Load tokenizer for a model.

        Args:
            model_name (str): The name of the model.

        Returns:
            Optional[PreTrainedTokenizer]: The loaded tokenizer, or None if failed.
        """
        self._lazy_import_transformers()
        from transformers import AutoTokenizer

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_fast=True
            )

            # Ensure pad token is set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            return tokenizer
        except Exception as e:
            logger.error(f"Error loading tokenizer for {model_name}: {e}")
            return None

    def _handle_tokenizer_error(self, error: Exception, model_name: str):
        """Handle errors that occur during tokenizer loading.

        Args:
            error (Exception): The exception that was raised.
            model_name (str): The name of the model.
        """
        logger.error(f"Error loading tokenizer for {model_name}: {error}")

    async def load_model(self, model_name: str, precision: str = "FP16") -> Tuple[Optional[PreTrainedModel], Optional[PreTrainedTokenizer]]:
        """Load model with specified precision.

        Args:
            model_name: HuggingFace model identifier
            precision: Model precision (FP16, FP32, 4-bit)

        Returns:
            Tuple of (model, tokenizer) or (None, None) if failed
        """
        if model_name in self.loaded_models:
            logger.info(f"Model {model_name} already loaded, returning cached version")
            return self.loaded_models[model_name]

        logger.info(f"Loading model {model_name} with precision {precision}")
        start_time = time.time()

        try:
            # Check if we have enough memory
            estimated_memory = self._estimate_model_memory(model_name, precision)
            if not self._check_memory_availability(estimated_memory):
                logger.error(f"Insufficient memory to load {model_name} (estimated {estimated_memory:.1f}GB)")
                return None, None

            # Configure precision settings
            torch_dtype = self._get_torch_dtype(precision)
            quantization_config = self._get_quantization_config(precision)

            # Load tokenizer
            logger.info(f"Loading tokenizer for {model_name}")
            tokenizer = await asyncio.get_event_loop().run_in_executor(
                None, self._load_tokenizer, model_name
            )

            if tokenizer is None:
                logger.error(f"Failed to load tokenizer for {model_name}")
                return None, None

            # Load model
            logger.info(f"Loading model {model_name}")
            model = await asyncio.get_event_loop().run_in_executor(
                None, self._load_model_sync, model_name, torch_dtype, quantization_config
            )

            if model is None:
                logger.error(f"Failed to load model {model_name}")
                return None, None

            # Store loaded model
            self.loaded_models[model_name] = (model, tokenizer)

            # Update metadata
            load_time = time.time() - start_time
            actual_memory = self._get_model_memory_usage(model)
            self.model_metadata[model_name] = {
                'precision': precision,
                'load_time': load_time,
                'memory_gb': actual_memory,
                'torch_dtype': str(torch_dtype),
                'quantized': precision == '4-bit',
                'device': str(model.device) if hasattr(model, 'device') else 'unknown'
            }

            self.total_memory_used += actual_memory

            logger.info(f"Successfully loaded {model_name} in {load_time:.2f}s, using {actual_memory:.1f}GB")
            return model, tokenizer

        except Exception as e:
            logger.error(f"Failed to load {model_name}: {str(e)}")
            return None, None

    def _load_model_sync(self,
                         model_name: str,
                         torch_dtype: torch.dtype,
                         quantization_config: Optional[BitsAndBytesConfig]) -> Optional[PreTrainedModel]:
        """Synchronously load model."""
        try:
            model = AutoModelForCausalLM.from_pretrained(  # type: ignore[no-untyped-call]
                model_name,
                torch_dtype=torch_dtype,
                quantization_config=quantization_config,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                use_cache=True
            )

            # Move to device if not using device_map
            if not torch.cuda.is_available() or quantization_config is None:
                model = model.to(self.device)  # type: ignore[attr-defined]

            return model  # type: ignore[no-any-return]
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return None

    def unload_model(self, model_name: str) -> bool:
        """Unload model from memory.

        Args:
            model_name: HuggingFace model identifier

        Returns:
            bool: True if successfully unloaded
        """
        if model_name not in self.loaded_models:
            logger.warning(f"Model {model_name} not loaded, cannot unload")
            return False

        logger.info(f"Unloading model {model_name}")

        try:
            model, tokenizer = self.loaded_models[model_name]

            # Get memory usage before deletion
            memory_used = self.model_metadata.get(model_name, {}).get('memory_gb', 0)

            # Delete model and tokenizer
            del model
            del tokenizer

            # Remove from tracking
            del self.loaded_models[model_name]
            if model_name in self.model_metadata:
                del self.model_metadata[model_name]

            # Update total memory
            self.total_memory_used = max(0, self.total_memory_used - memory_used)

            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info(f"Successfully unloaded {model_name}, freed {memory_used:.1f}GB")
            return True

        except Exception as e:
            logger.error(f"Error unloading model {model_name}: {e}")
            return False

    def is_model_loaded(self, model_name: str) -> bool:
        """Check if specific model is loaded."""
        return model_name in self.loaded_models

    def get_loaded_model_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all loaded models."""
        return {
            model_name: {
                **metadata,
                'is_loaded': True
            }
            for model_name, metadata in self.model_metadata.items()
        }

    def get_memory_status(self) -> Dict[str, Any]:
        """Get current memory status."""
        current_usage = self._get_current_memory_usage()
        return {
            'total_memory_gb': self.max_memory_gb,
            'used_memory_gb': current_usage,
            'available_memory_gb': self.max_memory_gb - current_usage,
            'tracked_model_memory_gb': self.total_memory_used,
            'loaded_model_count': len(self.loaded_models),
            'device': self.device
        }

    async def cleanup_unused_models(self, keep_models: list = None) -> int:
        """Cleanup models not in keep list.

        Args:
            keep_models: List of model names to keep loaded

        Returns:
            int: Number of models unloaded
        """
        if keep_models is None:
            keep_models = []

        models_to_unload = [
            name for name in self.loaded_models.keys()
            if name not in keep_models
        ]

        unloaded_count = 0
        for model_name in models_to_unload:
            if await self.unload_model(model_name):
                unloaded_count += 1

        logger.info(f"Cleanup complete: unloaded {unloaded_count} models")
        return unloaded_count

    def _get_torch_dtype(self, precision: str) -> torch.dtype:
        """Get torch dtype for precision."""
        precision_map = {
            "FP16": torch.float16,
            "FP32": torch.float32,
            "4-bit": torch.float16,  # 4-bit uses FP16 as base
            "8-bit": torch.float16
        }
        return precision_map.get(precision, torch.float16)

    def _get_quantization_config(self, precision: str) -> Optional[BitsAndBytesConfig]:
        """Get quantization config for precision."""
        if precision == "4-bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif precision == "8-bit":
            return BitsAndBytesConfig(
                load_in_8bit=True
            )
        return None

    def _estimate_model_memory(self, model_name: str, precision: str) -> float:
        """Estimate memory requirements for model."""
        # Simple estimation based on model name patterns
        # In production, you'd want more sophisticated estimation

        size_estimates = {
            '7b': 7, '8b': 8, '13b': 13, '34b': 34, '70b': 70
        }

        model_lower = model_name.lower()
        base_size = 7  # Default assumption

        for size_key, size_gb in size_estimates.items():
            if size_key in model_lower:
                base_size = size_gb
                break

        # Adjust for precision
        if precision == "FP32":
            return base_size * 4  # 4 bytes per parameter
        elif precision == "FP16":
            return base_size * 2  # 2 bytes per parameter
        elif precision == "4-bit":
            return base_size * 0.5  # 0.5 bytes per parameter
        elif precision == "8-bit":
            return base_size * 1  # 1 byte per parameter

        return base_size * 2  # Default to FP16

    def _check_memory_availability(self, required_memory: float) -> bool:
        """Check if enough memory is available."""
        current_usage = self._get_current_memory_usage()
        available = self.max_memory_gb - current_usage
        buffer = 2.0  # 2GB buffer

        return available >= (required_memory + buffer)

    def _get_model_memory_usage(self, model: PreTrainedModel) -> float:
        """Get actual memory usage of a loaded model."""
        try:
            if hasattr(model, 'get_memory_footprint'):
                return model.get_memory_footprint() / (1024 ** 3)

            # Fallback: estimate based on parameters
            param_count = sum(p.numel() for p in model.parameters())
            bytes_per_param = 2  # Assume FP16
            return (param_count * bytes_per_param) / (1024 ** 3)
        except:
            return 0.0

    def get_loaded_models(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all loaded models."""
        return {
            model_name: {
                **metadata,
                'is_loaded': True
            }
            for model_name, metadata in self.model_metadata.items()
        }

    def get_memory_status(self) -> Dict[str, Any]:
        """Get current memory status."""
        current_usage = self._get_current_memory_usage()
        return {
            'total_memory_gb': self.max_memory_gb,
            'used_memory_gb': current_usage,
            'available_memory_gb': self.max_memory_gb - current_usage,
            'tracked_model_memory_gb': self.total_memory_used,
            'loaded_model_count': len(self.loaded_models),
            'device': self.device
        }

    async def cleanup_unused_models(self, keep_models: list = None) -> int:
        """Cleanup models not in keep list.

        Args:
            keep_models: List of model names to keep loaded

        Returns:
            int: Number of models unloaded
        """
        if keep_models is None:
            keep_models = []

        models_to_unload = [
            name for name in self.loaded_models.keys()
            if name not in keep_models
        ]

        unloaded_count = 0
        for model_name in models_to_unload:
            if await self.unload_model(model_name):
                unloaded_count += 1

        logger.info(f"Cleanup complete: unloaded {unloaded_count} models")
        return unloaded_count

    def is_model_loaded(self, model_name: str) -> bool:
        """Check if specific model is loaded."""
        return model_name in self.loaded_models

    def _get_current_memory_usage(self):
        """
        Return the current GPU memory usage in GB.
        If CUDA is not available, returns 0.0.
        """
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1024 ** 3  # in GB
        except Exception:
            pass
        return 0.0

    # Optionally, keep the public alias for compatibility:
    get_current_memory_usage = _get_current_memory_usage