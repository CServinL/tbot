import asyncio
import logging
import re
from typing import Dict, Any, Optional, AsyncGenerator, List
from base_llm_engine import BaseLLMEngine
from model_loader import ModelLoader

logger = logging.getLogger(__name__)


class CodeCompletionEngine(BaseLLMEngine):
    """Engine optimized for fast code completion."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_loader = ModelLoader()
        self.completion_cache: Dict[str, str] = {}
        self.max_cache_size = 1000
        self.language_patterns = {
            'python': ['.py', 'python', 'def ', 'import ', 'class '],
            'javascript': ['.js', '.ts', 'javascript', 'typescript', 'function ', 'const ', 'let '],
            'java': ['.java', 'java', 'public class', 'private ', 'public '],
            'cpp': ['.cpp', '.hpp', '.cc', 'c++', '#include', 'std::'],
            'rust': ['.rs', 'rust', 'fn ', 'let mut', 'impl '],
            'go': ['.go', 'golang', 'func ', 'package ', 'import '],
            'html': ['.html', '.htm', 'html', '<html', '<div', '<script'],
            'css': ['.css', 'css', '{', '}', 'color:', 'background:'],
            'sql': ['.sql', 'sql', 'SELECT', 'FROM', 'WHERE', 'INSERT'],
        }

    async def load_model(self) -> bool:
        """Load the code completion model."""
        try:
            logger.info(f"Loading code completion model: {self.technical_model_name}")
            self.model, self.tokenizer = await self.model_loader.load_model(
                self.technical_model_name,
                self.precision
            )

            if self.model is not None and self.tokenizer is not None:
                self.is_model_loaded = True
                self.load_time = asyncio.get_event_loop().time()
                logger.info(f"Successfully loaded code completion model")

                # Perform warmup with code snippet
                await self.warmup()
                return True
            else:
                logger.error("Failed to load code completion model")
                return False

        except Exception as e:
            logger.error(f"Error loading code completion model: {e}")
            return False

    async def unload_model(self) -> bool:
        """Unload the code completion model."""
        try:
            if self.is_model_loaded:
                success = await self.model_loader.unload_model(self.technical_model_name)
                if success:
                    self.model = None
                    self.tokenizer = None
                    self.is_model_loaded = False
                    self.completion_cache.clear()
                    logger.info("Code completion model unloaded")
                return success
            return True
        except Exception as e:
            logger.error(f"Error unloading code completion model: {e}")
            return False

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate code completion.

        Args:
            prompt: Code context for completion
            **kwargs: Additional parameters
                - language: Programming language hint
                - max_tokens: Maximum tokens to generate (default: 64)
                - stop_sequences: List of stop sequences

        Returns:
            str: Code completion
        """
        if not self.is_model_loaded:
            raise RuntimeError("Code completion model not loaded")

        try:
            # Check cache first
            cache_key = self._get_cache_key(prompt, kwargs)
            if cache_key in self.completion_cache:
                logger.debug("Cache hit for code completion")
                return self.completion_cache[cache_key]

            # Detect programming language
            language = kwargs.get('language') or self._detect_language(prompt)

            # Prepare prompt with context
            completion_prompt = self._prepare_code_prompt(prompt, language)

            # Get generation parameters optimized for code
            gen_params = self._get_code_generation_params(kwargs, language)

            # Tokenize input
            inputs = self.tokenizer(
                completion_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048  # Shorter context for faster completion
            )

            # Move to device
            if hasattr(self.model, 'device'):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Generate completion
            import torch
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=gen_params['max_new_tokens'],
                    temperature=gen_params['temperature'],
                    do_sample=gen_params['do_sample'],
                    top_p=gen_params['top_p'],
                    pad_token_id=gen_params['pad_token_id'],
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1,
                    use_cache=True
                )

            # Decode and process completion
            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            completion = self._extract_code_completion(full_output, completion_prompt, language)

            # Post-process completion
            completion = self._post_process_completion(completion, language, kwargs.get('stop_sequences', []))

            # Cache result
            self._cache_completion(cache_key, completion)

            self.increment_generation_count()

            logger.debug(f"Generated code completion: {len(completion)} chars")
            return completion

        except Exception as e:
            logger.error(f"Error generating code completion: {e}")
            raise

    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Generate streaming code completion.

        Args:
            prompt: Code context
            **kwargs: Additional parameters

        Yields:
            str: Completion chunks
        """
        # For code completion, streaming might not be as useful since completions are typically short
        # But we'll implement it for consistency

        if not self.is_model_loaded:
            raise RuntimeError("Code completion model not loaded")

        try:
            from transformers import TextIteratorStreamer
            import torch
            from threading import Thread

            language = kwargs.get('language') or self._detect_language(prompt)
            completion_prompt = self._prepare_code_prompt(prompt, language)
            gen_params = self._get_code_generation_params(kwargs, language)

            inputs = self.tokenizer(
                completion_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            )

            if hasattr(self.model, 'device'):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )

            generation_kwargs = {
                **inputs,
                'max_new_tokens': gen_params['max_new_tokens'],
                'temperature': gen_params['temperature'],
                'do_sample': gen_params['do_sample'],
                'top_p': gen_params['top_p'],
                'pad_token_id': gen_params['pad_token_id'],
                'eos_token_id': self.tokenizer.eos_token_id,
                'streamer': streamer
            }

            generation_thread = Thread(
                target=self.model.generate,
                kwargs=generation_kwargs
            )
            generation_thread.start()

            full_completion = ""
            stop_sequences = kwargs.get('stop_sequences', [])

            for chunk in streamer:
                full_completion += chunk

                # Check for stop sequences
                should_stop = False
                for stop_seq in stop_sequences:
                    if stop_seq in full_completion:
                        chunk = chunk[:chunk.find(stop_seq)]
                        should_stop = True
                        break

                if chunk:
                    yield chunk

                if should_stop:
                    break

            generation_thread.join()
            self.increment_generation_count()

        except Exception as e:
            logger.error(f"Error in streaming code completion: {e}")
            yield f"// Error: {str(e)}"

    def get_system_prompt(self) -> Optional[str]:
        """Code completion doesn't use system prompts."""
        return None

    def _detect_language(self, code: str) -> Optional[str]:
        """Detect programming language from code context.

        Args:
            code: Code snippet

        Returns:
            Optional[str]: Detected language or None
        """
        code_lower = code.lower()

        for language, patterns in self.language_patterns.items():
            for pattern in patterns:
                if pattern.lower() in code_lower:
                    return language

        return None

    def _prepare_code_prompt(self, prompt: str, language: Optional[str] = None) -> str:
        """Prepare prompt for code completion.

        Args:
            prompt: Original prompt
            language: Detected or specified language

        Returns:
            str: Prepared prompt
        """
        # For code completion, we typically don't add much context
        # The model should complete based on the existing code structure
        return prompt

    def _get_code_generation_params(self, kwargs: Dict[str, Any], language: Optional[str] = None) -> Dict[str, Any]:
        """Get generation parameters optimized for code completion.

        Args:
            kwargs: User-provided parameters
            language: Programming language

        Returns:
            Dict[str, Any]: Generation parameters
        """
        # Base parameters for code completion (deterministic, focused)
        params = {
            'max_new_tokens': kwargs.get('max_tokens', 64),  # Short completions
            'temperature': 0.2,  # Low temperature for deterministic code
            'do_sample': False,  # Deterministic sampling
            'top_p': 1.0,
            'repetition_penalty': 1.0,
            'pad_token_id': self.tokenizer.eos_token_id if self.tokenizer else None
        }

        # Language-specific adjustments
        if language == 'python':
            params['max_new_tokens'] = min(params['max_new_tokens'], 128)
        elif language in ['javascript', 'typescript']:
            params['max_new_tokens'] = min(params['max_new_tokens'], 96)
        elif language == 'html':
            params['max_new_tokens'] = min(params['max_new_tokens'], 256)

        # Override with user parameters
        if 'temperature' in kwargs:
            params['temperature'] = max(0.1, min(kwargs['temperature'], 0.5))  # Clamp for code

        return params

    def _extract_code_completion(self, full_output: str, original_prompt: str, language: Optional[str] = None) -> str:
        """Extract the completion part from full output.

        Args:
            full_output: Full model output
            original_prompt: Original prompt
            language: Programming language

        Returns:
            str: Extracted completion
        """
        # Remove the original prompt
        completion = full_output[len(original_prompt):].strip()

        return completion

    def _post_process_completion(self, completion: str, language: Optional[str] = None,
                                 stop_sequences: List[str] = None) -> str:
        """Post-process the completion for better code quality.

        Args:
            completion: Raw completion
            language: Programming language
            stop_sequences: List of stop sequences

        Returns:
            str: Post-processed completion
        """
        if not completion:
            return completion

        # Apply stop sequences
        if stop_sequences:
            for stop_seq in stop_sequences:
                if stop_seq in completion:
                    completion = completion[:completion.index(stop_seq)]

        # Language-specific post-processing
        if language == 'python':
            # Stop at next function/class definition
            for pattern in ['\ndef ', '\nclass ']:
                if pattern in completion:
                    completion = completion[:completion.index(pattern)]

        elif language in ['javascript', 'typescript']:
            # Stop at next function definition
            for pattern in ['\nfunction ', '\nconst ', '\nlet ']:
                if pattern in completion:
                    completion = completion[:completion.index(pattern)]

        elif language == 'java':
            # Stop at next method/class
            for pattern in ['\npublic ', '\nprivate ', '\nprotected ']:
                if pattern in completion:
                    completion = completion[:completion.index(pattern)]

        # General cleanup
        completion = completion.rstrip()

        # Remove incomplete lines for some languages
        if language in ['python', 'javascript', 'java', 'cpp']:
            lines = completion.split('\n')
            if len(lines) > 1 and not lines[-1].strip().endswith((':', ';', '}', ')', ']')):
                # Last line might be incomplete, remove it
                completion = '\n'.join(lines[:-1])

        return completion

    def _get_cache_key(self, prompt: str, kwargs: Dict[str, Any]) -> str:
        """Generate cache key for completion.

        Args:
            prompt: Code prompt
            kwargs: Parameters

        Returns:
            str: Cache key
        """
        import hashlib

        # Include relevant parameters in cache key
        cache_data = {
            'prompt': prompt[-200:],  # Only last 200 chars to avoid huge keys
            'language': kwargs.get('language'),
            'max_tokens': kwargs.get('max_tokens', 64),
            'temperature': kwargs.get('temperature', 0.2)
        }

        cache_str = str(sorted(cache_data.items()))
        return hashlib.md5(cache_str.encode()).hexdigest()

    def _cache_completion(self, cache_key: str, completion: str):
        """Cache a completion result.

        Args:
            cache_key: Cache key
            completion: Completion result
        """
        if len(self.completion_cache) >= self.max_cache_size:
            # Remove oldest entries (simple FIFO)
            oldest_keys = list(self.completion_cache.keys())[:100]
            for key in oldest_keys:
                del self.completion_cache[key]

        self.completion_cache[cache_key] = completion

    def clear_cache(self):
        """Clear the completion cache."""
        self.completion_cache.clear()
        logger.info("Cleared code completion cache")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict containing cache stats
        """
        return {
            'cache_size': len(self.completion_cache),
            'max_cache_size': self.max_cache_size,
            'cache_hit_ratio': getattr(self, '_cache_hits', 0) / max(self.generation_count, 1)
        }

    async def warmup(self) -> bool:
        """Warm up with a code completion example."""
        if not self.is_loaded():
            return False

        try:
            warmup_code = "def fibonacci(n):\n    if n <= 1:\n        return n\n    return"
            await self.generate(warmup_code, max_tokens=10)
            logger.info(f"Warmup completed for {self.category}")
            return True
        except Exception as e:
            logger.error(f"Warmup failed for {self.category}: {e}")
            return False