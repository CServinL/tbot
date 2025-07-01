from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, AsyncGenerator
import logging

logger = logging.getLogger(__name__)


class BaseLLMEngine(ABC):
    """Abstract base class for all LLM engines in the conductor system."""

    def __init__(self, config: Dict[str, Any]):
        self.category = config.get('category', 'unknown')
        self.model_name = config.get('model_name', '')
        self.technical_model_name = config.get('technical_model_name', '')
        self.precision = config.get('precision', 'FP16')
        self.stay_loaded = config.get('stay_loaded', False)
        self.vram_requirement = config.get('vram_requirement', '')
        self.persona = config.get('persona', None)
        self.model = None
        self.tokenizer = None
        self.is_model_loaded = False
        self.load_time = None
        self.generation_count = 0

    @abstractmethod
    async def load_model(self) -> bool:
        """Load the model with specified precision.

        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        pass

    @abstractmethod
    async def unload_model(self) -> bool:
        """Unload model from memory.

        Returns:
            bool: True if model unloaded successfully, False otherwise
        """
        pass

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate response with task-specific handling.

        Args:
            prompt: Input prompt for generation
            **kwargs: Additional generation parameters

        Returns:
            str: Generated response
        """
        pass

    @abstractmethod
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Generate streaming response with task-specific handling.

        Args:
            prompt: Input prompt for generation
            **kwargs: Additional generation parameters

        Yields:
            str: Partial response chunks
        """
        pass

    @abstractmethod
    def get_system_prompt(self) -> Optional[str]:
        """Return task-specific system prompt/persona.

        Returns:
            Optional[str]: System prompt or None if no persona needed
        """
        pass

    def is_loaded(self) -> bool:
        """Check if model is currently loaded in memory."""
        return self.is_model_loaded

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

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        if not self.is_loaded():
            return {'memory_gb': 0, 'status': 'not_loaded'}

        # This would be implemented with actual memory monitoring
        # For now, return estimated values from config
        estimated_memory = self._parse_memory_requirement()
        return {
            'memory_gb': estimated_memory,
            'status': 'loaded',
            'precision': self.precision
        }

    def _parse_memory_requirement(self) -> float:
        """Parse memory requirement string to float GB."""
        import re
        match = re.search(r'~?(\d+(?:\.\d+)?)GB', self.vram_requirement)
        if match:
            return float(match.group(1))
        return 0.0

    def _prepare_prompt(self, user_prompt: str) -> str:
        """Prepare prompt with system prompt if applicable."""
        system_prompt = self.get_system_prompt()
        if system_prompt:
            return f"{system_prompt}\n\nUser: {user_prompt}\nAssistant:"
        return user_prompt

    def _extract_response(self, full_output: str, original_prompt: str) -> str:
        """Extract response from full model output."""
        response = full_output.replace(original_prompt, "").strip()
        # Remove common artifacts
        if response.startswith("Assistant:"):
            response = response[10:].strip()
        return response

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
            validated['pad_token_id'] = self.tokenizer.eos_token_id

        return validated

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the engine."""
        return {
            'engine': self.category,
            'model': self.technical_model_name,
            'loaded': self.is_loaded(),
            'memory_usage': self.get_memory_usage(),
            'generation_count': self.generation_count,
            'status': 'healthy' if self.is_loaded() else 'unloaded'
        }

    def increment_generation_count(self):
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