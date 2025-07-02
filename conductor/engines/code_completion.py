import logging
from typing import Dict, Any, Optional, List
from conductor.engines.base_engine import BaseEngine
from conductor.model_loader import ModelLoader

logger = logging.getLogger(__name__)


class CodeCompletionEngine(BaseEngine):
    """Code completion engine for intelligent code suggestions."""
    
    def __init__(self, config: Dict[str, Any], model_loader: ModelLoader, persona: str = ""):
        super().__init__(config, model_loader, persona)
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

    def get_system_prompt(self) -> Optional[str]:
        """Code completion doesn't use system prompts."""
        return None

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate code completion with enhanced context."""
        # Parse code completion parameters
        language = kwargs.get('language', self._detect_language(prompt))
        max_completion_length = kwargs.get('max_completion_length', 100)
        
        # Check cache first
        cache_key = self._get_cache_key(prompt, kwargs)
        if cache_key in self.completion_cache:
            logger.debug("Cache hit for code completion")
            return self.completion_cache[cache_key]
        
        # Build completion-specific prompt
        completion_prompt = self._build_completion_prompt(prompt, language)
        
        # Use shorter max_tokens for completions and lower temperature
        kwargs['max_tokens'] = max_completion_length
        kwargs['temperature'] = kwargs.get('temperature', 0.2)  # Lower temperature for code
        
        # Use parent's generate method
        completion = await super().generate(completion_prompt, **kwargs)
        
        # Post-process completion
        processed_completion = self._post_process_completion(completion, prompt, language)
        
        # Cache result
        self._cache_completion(cache_key, processed_completion)
        
        return processed_completion

    def _detect_language(self, code: str) -> Optional[str]:
        """Detect programming language from code context."""
        code_lower = code.lower()
        for language, patterns in self.language_patterns.items():
            for pattern in patterns:
                if pattern.lower() in code_lower:
                    return language
        return None

    def _build_completion_prompt(self, prompt: str, language: Optional[str] = None) -> str:
        """Build prompt for code completion."""
        if language:
            return f"Complete this {language} code:\n{prompt}"
        return f"Complete this code:\n{prompt}"

    def _post_process_completion(self, completion: str, original_prompt: str, language: Optional[str] = None) -> str:
        """Post-process completion for better code quality."""
        if not completion:
            return completion

        # Remove the original prompt if it appears in completion
        if original_prompt in completion:
            completion = completion.replace(original_prompt, "").strip()

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

        return completion.strip()

    def _get_cache_key(self, prompt: str, kwargs: Dict[str, Any]) -> str:
        """Generate cache key for completion."""
        import hashlib
        cache_data = {
            'prompt': prompt[-200:],  # Only last 200 chars
            'language': kwargs.get('language'),
            'max_tokens': kwargs.get('max_completion_length', 100),
        }
        cache_str = str(sorted(cache_data.items()))
        return hashlib.md5(cache_str.encode()).hexdigest()

    def _cache_completion(self, cache_key: str, completion: str) -> None:
        """Cache a completion result."""
        if len(self.completion_cache) >= self.max_cache_size:
            # Remove oldest entries (simple FIFO)
            oldest_keys = list(self.completion_cache.keys())[:100]
            for key in oldest_keys:
                del self.completion_cache[key]
        self.completion_cache[cache_key] = completion

    def clear_cache(self) -> None:
        """Clear the completion cache."""
        self.completion_cache.clear()
        logger.info("Cleared code completion cache")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'cache_size': len(self.completion_cache),
            'max_cache_size': self.max_cache_size,
        }

    async def complete_function(self, function_start: str, language: str = 'python') -> str:
        """Complete a function definition."""
        return await self.generate(function_start, language=language, max_completion_length=200)

    async def complete_class(self, class_start: str, language: str = 'python') -> str:
        """Complete a class definition."""
        return await self.generate(class_start, language=language, max_completion_length=300)