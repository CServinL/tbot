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

    def _get_default_generation_params(self) -> Dict[str, Any]:
        """Get default generation parameters optimized for code completion."""
        params = super()._get_default_generation_params()
        # Override defaults for code completion
        params.update({
            'max_new_tokens': 100,  # Shorter for completions
            'temperature': 0.2,     # Lower temperature for deterministic code
            'top_p': 0.95,         # Slightly higher for code diversity
            'repetition_penalty': 1.05  # Lower penalty for code patterns
        })
        return params

    def get_system_prompt(self) -> Optional[str]:
        """Get system prompt for code completion."""
        if self.persona:
            return self.persona
        return "You are a code completion assistant. Provide accurate, contextually appropriate code completions based on the given code context."

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
        
        # Set completion-specific parameters and validate them
        completion_kwargs = kwargs.copy()
        completion_kwargs['max_completion_length'] = max_completion_length
        if 'max_tokens' not in completion_kwargs:
            completion_kwargs['max_tokens'] = max_completion_length
        
        # Use base class parameter validation
        validated_params = self._validate_generation_params(completion_kwargs)
        
        # Use parent's generate method with validated parameters
        completion = await super().generate(completion_prompt, **validated_params)
        
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

        # Use base class response extraction first
        completion = self._extract_response(completion, original_prompt)

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
        cache_data: Dict[str, Any] = {
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

    def _should_use_cache(self, prompt: str, kwargs: Dict[str, Any]) -> bool:
        """Determine if caching should be used for this completion request."""
        # Don't cache very short prompts or requests with high temperature
        if len(prompt.strip()) < 10:
            return False
        if kwargs.get('temperature', 0.2) > 0.5:
            return False
        return True
    
    def _estimate_completion_quality(self, completion: str, language: Optional[str]) -> Dict[str, Any]:
        """Estimate the quality of a code completion."""
        quality_score = 0.0
        issues: List[str] = []
        
        if not completion.strip():
            return {'score': 0.0, 'issues': ['Empty completion']}
        
        # Basic syntax checks
        if language == 'python':
            # Check for basic Python syntax patterns
            if completion.count('(') != completion.count(')'):
                issues.append('Unmatched parentheses')
            if completion.count('[') != completion.count(']'):
                issues.append('Unmatched brackets')
            if completion.count('{') != completion.count('}'):
                issues.append('Unmatched braces')
        
        # Length appropriateness
        if len(completion) > 500:
            issues.append('Completion too long')
            quality_score -= 0.2
        elif len(completion) < 5:
            issues.append('Completion too short')
            quality_score -= 0.3
        else:
            quality_score += 0.3
        
        # Check for common artifacts
        if any(artifact in completion.lower() for artifact in ['assistant:', 'human:', '```']):
            issues.append('Contains conversation artifacts')
            quality_score -= 0.4
        
        # Base score
        quality_score += 0.7
        quality_score = max(0.0, min(1.0, quality_score))
        
        return {
            'score': quality_score,
            'issues': issues,
            'length': len(completion),
            'language': language
        }

    async def generate_with_quality_check(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        """Generate completion with quality assessment."""
        completion = await self.generate(prompt, **kwargs)
        language = kwargs.get('language', self._detect_language(prompt))
        quality = self._estimate_completion_quality(completion, language)
        
        return {
            'completion': completion,
            'quality': quality,
            'cached': self._get_cache_key(prompt, kwargs) in self.completion_cache
        }