from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
import logging
import time

from conductor.model_loader import ModelLoader

# Type hints for models - using Union to allow different model types
# but ensuring they have at least the base PreTrainedModel interface
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from torch import Tensor
ModelType = Union[PreTrainedModel, None]
TokenizerType = Union[PreTrainedTokenizer, None]
TensorDict = Dict[str, Tensor]

logger = logging.getLogger(__name__)

class BaseEngine(ABC):
    """Base engine with essential functionality - supports both shared and individual model management"""

    def __init__(self, config: Dict[str, Any], model_loader: ModelLoader, persona: str = ""):
        self.config = config
        self.category = config.get('category', 'unknown')
        self.technical_model_name = config.get('technical_model_name', '')
        self.model_loader = model_loader
        self.persona = persona
        self.generation_count = 0
        
        # Individual model management attributes
        self.model_name = config.get('model_name', '')
        self.precision = config.get('precision', 'FP16')
        self.stay_loaded = config.get('stay_loaded', False)
        self.vram_requirement = config.get('vram_requirement', '')
        self._model = None
        self._tokenizer = None
        self.is_model_loaded = False
        self.load_time = None

    async def load_model(self) -> bool:
        """Load the model for this engine - can be overridden for individual model management"""
        try:
            logger.debug(f"Loading model for {self.category} using {self.technical_model_name}")
            # First try shared model approach
            result = await self.model_loader.load_model(
                self.technical_model_name,
                self.config.get('precision', 'FP16')
            )
            model = result[0] if result else None
            tokenizer = result[1] if result else None

            if model and tokenizer:
                # For shared models, don't store locally
                logger.debug(f"Successfully got shared model for {self.category}")
                self.is_model_loaded = True
                if self.load_time is None:
                    import time
                    self.load_time = time.time()
                return True
            else:
                logger.error(f"Model loader returned None for {self.category}: model={model}, tokenizer={tokenizer}")
                self.is_model_loaded = False
                return False
        except Exception as e:
            logger.error(f"Exception in load_model for {self.category}: {type(e).__name__}: {e}")
            self.is_model_loaded = False
            return False

    async def load_individual_model(self) -> bool:
        """Load model individually for this engine (not shared)"""
        try:
            logger.info(f"Loading individual model for {self.category}: {self.technical_model_name}")
            result = await self.model_loader.load_model(
                self.technical_model_name,
                self.precision
            )
            
            if result:
                self._model, self._tokenizer = result
                self.is_model_loaded = (self._model is not None and self._tokenizer is not None)
                
                if self.is_model_loaded:
                    import time
                    self.load_time = time.time()
                    logger.info(f"Successfully loaded individual model for {self.category}")
                    return True
                    
            logger.error(f"Failed to load individual model for {self.category}")
            self.is_model_loaded = False
            return False
            
        except Exception as e:
            logger.error(f"Error loading individual model for {self.category}: {e}")
            self.is_model_loaded = False
            self._model = None
            self._tokenizer = None
            return False

    async def unload_model(self) -> bool:
        """Unload model - handles both shared and individual models"""
        try:
            if self._model is not None or self._tokenizer is not None:
                # Individual model - actually unload it
                success = self.model_loader.unload_model(self.technical_model_name)
                if success:
                    self._model = None
                    self._tokenizer = None
                    self.is_model_loaded = False
                    logger.info(f"Individual model unloaded for {self.category}")
                return success
            else:
                # Shared model - just mark as unloaded for this engine
                self.is_model_loaded = False
                logger.debug(f"Marked shared model as unloaded for {self.category}")
                return True
        except Exception as e:
            logger.error(f"Error unloading model for {self.category}: {e}")
            return False

    def is_loaded(self) -> bool:
        """Check if model is loaded (either shared or individual)"""
        # Check individual model first
        if self._model is not None and self._tokenizer is not None:
            return True
        # Check shared model
        if self.technical_model_name in self.model_loader.loaded_models:
            return True
        return self.is_model_loaded

    @property
    def model(self) -> Optional['PreTrainedModel']:
        """Get model (individual first, then shared)"""
        # Return individual model if available
        if self._model is not None:
            return self._model
        # Fall back to shared model
        if self.technical_model_name in self.model_loader.loaded_models:
            return self.model_loader.loaded_models[self.technical_model_name][0]
        return None

    @property
    def tokenizer(self) -> Optional['PreTrainedTokenizer']:
        """Get tokenizer (individual first, then shared)"""
        # Return individual tokenizer if available
        if self._tokenizer is not None:
            return self._tokenizer
        # Fall back to shared tokenizer
        if self.technical_model_name in self.model_loader.loaded_models:
            return self.model_loader.loaded_models[self.technical_model_name][1]
        return None


    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate response using shared model"""
        if not self.model or not self.tokenizer:
            raise RuntimeError(f"Model not loaded for {self.category}")

        try:
            logger.info(f"[{self.category}] Starting generation for: '{prompt[:50]}...'")

            full_prompt = self._prepare_prompt(prompt)
            logger.info(f"[{self.category}] Prepared prompt")

            # Simple tokenization
            inputs: Any = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024  # Shorter for faster processing
            )
            # Skip shape logging to avoid type checker issues
            logger.info(f"[{self.category}] Tokenized input")

            # Check device placement before generation
            model_device = next(self.model.parameters()).device
            logger.info(f"[{self.category}] Model device: {model_device}")

            # Move inputs to same device as model - typed as Any for complex tensor operations
            inputs: Any = {k: v.to(model_device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            logger.info(f"[{self.category}] Moved inputs to device: {model_device}")

            # Use model info for max tokens/window if available
            model_info = self.model_loader.get_model_info(self.technical_model_name)
            model_max_tokens = model_info.max_new_tokens if model_info else 1024
            #model_max_window = model_info.max_context_window if model_info else 2048

            # Allow long outputs, but cap for model and hardware safety
            user_max_tokens = int(kwargs.get('max_tokens', model_max_tokens))
            max_new_tokens = min(user_max_tokens, model_max_tokens, 2048)
            temperature = kwargs.get('temperature', 0.7)

            logger.info(f"[{self.category}] About to call model.generate() with max_tokens={max_new_tokens}")

            # Import torch - typed as Any for dynamic import
            torch: Any = self.model_loader.lazy_import_torch()

            start_time = time.time()
            with torch.no_grad():
                # Model generation - typed as Any for complex PyTorch operations
                outputs: Any = self.model.generate(  # type: ignore
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs.get('attention_mask'),
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,  # type: ignore
                    eos_token_id=self.tokenizer.eos_token_id,  # type: ignore
                    early_stopping=True,
                    num_beams=1,
                    repetition_penalty=1.1,
                    use_cache=True
                )

            generation_time = time.time() - start_time
            logger.info(f"[{self.category}] Generation completed in {generation_time:.2f} seconds")

            # Decode response - typed as Any for complex tokenizer operations
            response: Any = self.tokenizer.decode(  # type: ignore
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            # Use stop patterns from ModelInfo if available, else default
            stop_patterns = []
            if model_info and getattr(model_info, "stop_patterns", None):
                stop_patterns = model_info.stop_patterns
            if not stop_patterns:
                stop_patterns = ["```", "\n>", "\n# ", "\nSources:", "\nLimitations:"]

            # Post-process: truncate at first code block or unwanted pattern
            for stop_pattern in stop_patterns:
                idx = response.find(stop_pattern)
                if idx != -1:
                    response = response[:idx].strip()
                    break

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

    def should_stay_loaded(self) -> bool:
        """Check if model should remain in memory."""
        return self.stay_loaded

    def get_config(self) -> Dict[str, Any]:
        """Get engine configuration."""
        return {
            'category': self.category,
            'model_name': self.model_name,
            'technical_model_name': self.technical_model_name,
            'precision': self.precision,
            'stay_loaded': self.stay_loaded,
            'vram_requirement': self.vram_requirement,
            'is_loaded': self.is_loaded(),
            'generation_count': self.generation_count,
            'load_time': self.load_time
        }

    def _parse_memory_requirement(self) -> float:
        """Parse memory requirement string to float GB."""
        import re
        match = re.search(r'~?(\d+(?:\.\d+)?)GB', self.vram_requirement)
        if match:
            return float(match.group(1))
        return 0.0

    def _get_default_generation_params(self) -> Dict[str, Any]:
        """Get default generation parameters for this engine type."""
        return {
            'max_new_tokens': 512,
            'temperature': 0.7,
            'do_sample': True,
            'top_p': 0.9,
            'repetition_penalty': 1.1,
            'pad_token_id': None  # Will be set when tokenizer is available
        }

    def _validate_generation_params(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and merge generation parameters."""
        defaults = self._get_default_generation_params()
        validated = defaults.copy()

        # Update with provided kwargs
        for key, value in kwargs.items():
            if key in ['max_new_tokens', 'max_tokens']:
                validated['max_new_tokens'] = min(value, 2048)  # Safety limit
            elif key in ['temperature']:
                validated['temperature'] = max(0.1, min(value, 2.0))  # Clamp temperature
            elif key in validated:
                validated[key] = value

        # Set pad_token_id if tokenizer available
        if self.tokenizer and validated['pad_token_id'] is None:
            validated['pad_token_id'] = self.tokenizer.eos_token_id  # type: ignore

        return validated

    def _extract_response(self, full_output: str, original_prompt: str) -> str:
        """Extract response from full model output."""
        response = full_output.replace(original_prompt, "").strip()
        # Remove common artifacts
        if response.startswith("Assistant:"):
            response = response[10:].strip()
        return response

    def increment_generation_count(self) -> None:
        """Increment the generation counter."""
        self.generation_count += 1

    async def warmup(self) -> bool:
        """Warm up the model with a simple generation."""
        if not self.is_loaded():
            return False

        try:
            await self.generate("Hello", max_new_tokens=5)
            logger.info(f"Warmup completed for {self.category}")
            return True
        except Exception as e:
            logger.error(f"Warmup failed for {self.category}: {e}")
            return False
