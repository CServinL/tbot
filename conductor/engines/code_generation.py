import asyncio
import logging
import re
from typing import Dict, Any, Optional, AsyncGenerator, List, Tuple
from base_llm_engine import BaseLLMEngine
from model_loader import ModelLoader
from utils.persona_loader import PersonaLoader

logger = logging.getLogger(__name__)


class CodeGenerationEngine(BaseLLMEngine):
    """Engine for generating complete code solutions and implementations."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_loader = ModelLoader()
        self.persona_loader = PersonaLoader()
        self.code_templates = {
            'python': {
                'function': 'def {name}({params}):\n    """{docstring}"""\n    {body}',
                'class': 'class {name}:\n    """{docstring}"""\n    \n    def __init__(self{init_params}):\n        {init_body}\n    \n    {methods}',
                'script': '#!/usr/bin/env python3\n"""{docstring}"""\n\n{imports}\n\n{main_code}'
            },
            'javascript': {
                'function': 'function {name}({params}) {{\n    // {docstring}\n    {body}\n}}',
                'class': 'class {name} {{\n    // {docstring}\n    constructor({init_params}) {{\n        {init_body}\n    }}\n    \n    {methods}\n}}',
                'async_function': 'async function {name}({params}) {{\n    // {docstring}\n    {body}\n}}'
            },
            'java': {
                'class': 'public class {name} {{\n    // {docstring}\n    \n    public {name}({init_params}) {{\n        {init_body}\n    }}\n    \n    {methods}\n}}',
                'method': 'public {return_type} {name}({params}) {{\n    // {docstring}\n    {body}\n}}'
            }
        }

    async def load_model(self) -> bool:
        """Load the code generation model."""
        try:
            logger.info(f"Loading code generation model: {self.technical_model_name}")
            self.model, self.tokenizer = await self.model_loader.load_model(
                self.technical_model_name,
                self.precision
            )

            if self.model is not None and self.tokenizer is not None:
                self.is_model_loaded = True
                self.load_time = asyncio.get_event_loop().time()
                logger.info(f"Successfully loaded code generation model")

                # Perform warmup
                await self.warmup()
                return True
            else:
                logger.error("Failed to load code generation model")
                return False

        except Exception as e:
            logger.error(f"Error loading code generation model: {e}")
            return False

    async def unload_model(self) -> bool:
        """Unload the code generation model."""
        try:
            if self.is_model_loaded:
                success = await self.model_loader.unload_model(self.technical_model_name)
                if success:
                    self.model = None
                    self.tokenizer = None
                    self.is_model_loaded = False
                    logger.info("Code generation model unloaded")
                return success
            return True
        except Exception as e:
            logger.error(f"Error unloading code generation model: {e}")
            return False

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate code based on requirements.

        Args:
            prompt: Description of what code to generate
            **kwargs: Additional parameters
                - language: Target programming language
                - style: Code style (function, class, script, etc.)
                - include_tests: Whether to include unit tests
                - include_docs: Whether to include documentation
                - max_tokens: Maximum tokens to generate

        Returns:
            str: Generated code
        """
        if not self.is_model_loaded:
            raise RuntimeError("Code generation model not loaded")

        try:
            # Parse generation requirements
            language = kwargs.get('language', 'python')
            style = kwargs.get('style', 'auto')
            include_tests = kwargs.get('include_tests', False)
            include_docs = kwargs.get('include_docs', True)

            # Build structured prompt
            generation_prompt = self._build_code_generation_prompt(
                prompt, language, style, include_tests, include_docs
            )

            # Get generation parameters
            gen_params = self._get_code_generation_params(kwargs, language)

            # Tokenize input
            inputs = self.tokenizer(
                generation_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=3072  # Leave room for substantial code generation
            )

            # Move to device
            if hasattr(self.model, 'device'):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Generate code
            import torch
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=gen_params['max_new_tokens'],
                    temperature=gen_params['temperature'],
                    do_sample=gen_params['do_sample'],
                    top_p=gen_params['top_p'],
                    repetition_penalty=gen_params['repetition_penalty'],
                    pad_token_id=gen_params['pad_token_id'],
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1
                )

            # Decode and extract code
            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_code = self._extract_response(full_output, generation_prompt)

            # Post-process the generated code
            processed_code = self._post_process_code(generated_code, language, kwargs)

            self.increment_generation_count()

            logger.debug(f"Generated {language} code: {len(processed_code)} chars")
            return processed_code

        except Exception as e:
            logger.error(f"Error generating code: {e}")
            raise

    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Generate streaming code response.

        Args:
            prompt: Code generation request
            **kwargs: Additional parameters

        Yields:
            str: Code chunks
        """
        if not self.is_model_loaded:
            raise RuntimeError("Code generation model not loaded")

        try:
            from transformers import TextIteratorStreamer
            import torch
            from threading import Thread

            language = kwargs.get('language', 'python')
            generation_prompt = self._build_code_generation_prompt(
                prompt, language,
                kwargs.get('style', 'auto'),
                kwargs.get('include_tests', False),
                kwargs.get('include_docs', True)
            )

            gen_params = self._get_code_generation_params(kwargs, language)

            inputs = self.tokenizer(
                generation_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=3072
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
                'repetition_penalty': gen_params['repetition_penalty'],
                'pad_token_id': gen_params['pad_token_id'],
                'eos_token_id': self.tokenizer.eos_token_id,
                'streamer': streamer
            }

            generation_thread = Thread(
                target=self.model.generate,
                kwargs=generation_kwargs
            )
            generation_thread.start()

            # Stream the response
            for chunk in streamer:
                yield chunk

            generation_thread.join()
            self.increment_generation_count()

        except Exception as e:
            logger.error(f"Error in streaming code generation: {e}")
            yield f"# Error: {str(e)}"

    def get_system_prompt(self) -> Optional[str]:
        """Get system prompt for code generation."""
        return self.persona_loader.get_persona_for_category('code_generation')

    def _build_code_generation_prompt(self,
                                      user_request: str,
                                      language: str,
                                      style: str,
                                      include_tests: bool,
                                      include_docs: bool) -> str:
        """Build structured prompt for code generation.

        Args:
            user_request: User's code request
            language: Target programming language
            style: Code style/structure
            include_tests: Whether to include tests
            include_docs: Whether to include documentation

        Returns:
            str: Structured prompt
        """
        system_prompt = self.get_system_prompt()

        prompt_parts = []

        if system_prompt:
            prompt_parts.append(system_prompt)

        # Add specific instructions for code generation
        instructions = [
            f"Generate {language} code for the following request:",
            f"Language: {language}",
            f"Style: {style}",
        ]

        if include_docs:
            instructions.append("Include comprehensive documentation and comments")

        if include_tests:
            instructions.append("Include unit tests for the generated code")

        instructions.extend([
            "Follow best practices and coding standards",
            "Make the code production-ready and well-structured",
            "Include error handling where appropriate"
        ])

        prompt_parts.append("\n".join(instructions))
        prompt_parts.append(f"\nRequest: {user_request}")
        prompt_parts.append(f"\n{language} code:")

        return "\n\n".join(prompt_parts)

    def _get_code_generation_params(self, kwargs: Dict[str, Any], language: str) -> Dict[str, Any]:
        """Get generation parameters for code generation.

        Args:
            kwargs: User parameters
            language: Programming language

        Returns:
            Dict[str, Any]: Generation parameters
        """
        # Base parameters for code generation
        params = {
            'max_new_tokens': kwargs.get('max_tokens', 1024),
            'temperature': 0.3,  # Lower temperature for more deterministic code
            'do_sample': True,
            'top_p': 0.85,
            'repetition_penalty': 1.1,
            'pad_token_id': self.tokenizer.eos_token_id if self.tokenizer else None
        }

        # Language-specific adjustments
        if language in ['python', 'javascript']:
            params['temperature'] = 0.4  # Slightly higher for more creative solutions
        elif language in ['java', 'cpp', 'c']:
            params['temperature'] = 0.2  # Lower for more structured languages
        elif language in ['html', 'css']:
            params['temperature'] = 0.5  # Higher for creative styling

        # Override with user parameters
        if 'temperature' in kwargs:
            params['temperature'] = max(0.1, min(kwargs['temperature'], 1.0))

        return params

    def _post_process_code(self, code: str, language: str, kwargs: Dict[str, Any]) -> str:
        """Post-process generated code for quality and formatting.

        Args:
            code: Generated code
            language: Programming language
            kwargs: Generation parameters

        Returns:
            str: Post-processed code
        """
        if not code.strip():
            return code

        # Remove common artifacts
        code = self._remove_code_artifacts(code)

        # Language-specific formatting
        if language == 'python':
            code = self._format_python_code(code)
        elif language in ['javascript', 'typescript']:
            code = self._format_javascript_code(code)
        elif language == 'java':
            code = self._format_java_code(code)

        # Add header comment if requested
        if kwargs.get('include_header', True):
            code = self._add_code_header(code, language, kwargs.get('description', ''))

        return code.strip()

    def _remove_code_artifacts(self, code: str) -> str:
        """Remove common generation artifacts from code.

        Args:
            code: Raw generated code

        Returns:
            str: Cleaned code
        """
        # Remove common prefixes that models sometimes add
        prefixes_to_remove = [
            "Here's the code:",
            "Here is the code:",
            "```python\n",
            "```javascript\n",
            "```java\n",
            "```cpp\n",
            "```\n"
        ]

        for prefix in prefixes_to_remove:
            if code.startswith(prefix):
                code = code[len(prefix):]

        # Remove closing code fences
        if code.endswith("```"):
            code = code[:-3]

        # Remove explanatory text that sometimes appears at the end
        lines = code.split('\n')
        clean_lines = []

        for line in lines:
            # Stop at common explanatory phrases
            if any(phrase in line.lower() for phrase in [
                "this code", "the above", "explanation:", "note that", "here's how"
            ]) and not line.strip().startswith(('#', '//', '/*')):
                break
            clean_lines.append(line)

        return '\n'.join(clean_lines)

    def _format_python_code(self, code: str) -> str:
        """Format Python-specific code improvements.

        Args:
            code: Python code

        Returns:
            str: Formatted Python code
        """
        lines = code.split('\n')
        formatted_lines = []

        for line in lines:
            # Ensure proper spacing after commas
            line = re.sub(r',(?!\s)', ', ', line)

            # Ensure proper spacing around operators
            line = re.sub(r'(\w)=(\w)', r'\1 = \2', line)

            formatted_lines.append(line)

        return '\n'.join(formatted_lines)

    def _format_javascript_code(self, code: str) -> str:
        """Format JavaScript-specific code improvements.

        Args:
            code: JavaScript code

        Returns:
            str: Formatted JavaScript code
        """
        lines = code.split('\n')
        formatted_lines = []

        for line in lines:
            # Ensure semicolons at end of statements (basic check)
            stripped = line.strip()
            if (stripped and
                    not stripped.endswith((';', '{', '}', ')', ',')) and
                    not stripped.startswith(('if', 'for', 'while', 'function', 'class', '//'))):
                if not any(stripped.endswith(ending) for ending in ['{', '}', ':', '(', ')']):
                    line = line.rstrip() + ';'

            formatted_lines.append(line)

        return '\n'.join(formatted_lines)

    def _format_java_code(self, code: str) -> str:
        """Format Java-specific code improvements.

        Args:
            code: Java code

        Returns:
            str: Formatted Java code
        """
        lines = code.split('\n')
        formatted_lines = []

        for line in lines:
            # Ensure proper Java naming conventions (basic)
            # This is a simplified version - in practice you'd want more sophisticated formatting
            formatted_lines.append(line)

        return '\n'.join(formatted_lines)

    def _add_code_header(self, code: str, language: str, description: str) -> str:
        """Add a header comment to the generated code.

        Args:
            code: Generated code
            language: Programming language
            description: Code description

        Returns:
            str: Code with header
        """
        comment_styles = {
            'python': '#',
            'javascript': '//',
            'typescript': '//',
            'java': '//',
            'cpp': '//',
            'c': '//',
            'go': '//',
            'rust': '//',
            'html': '<!--',
            'css': '/*'
        }

        comment_char = comment_styles.get(language, '#')

        if language in ['html']:
            header = f"<!-- Generated code: {description} -->\n"
        elif language in ['css']:
            header = f"/* Generated code: {description} */\n"
        else:
            header = f"{comment_char} Generated code: {description}\n{comment_char} Created by Conductor LLM Server\n\n"

        return header + code

    async def warmup(self) -> bool:
        """Warm up with a code generation example."""
        if not self.is_loaded():
            return False

        try:
            warmup_prompt = "Create a simple function that adds two numbers"
            await self.generate(warmup_prompt, language="python", max_tokens=100)
            logger.info(f"Warmup completed for {self.category}")
            return True
        except Exception as e:
            logger.error(f"Warmup failed for {self.category}: {e}")
            return False

    def get_supported_languages(self) -> List[str]:
        """Get list of supported programming languages.

        Returns:
            List[str]: Supported languages
        """
        return list(self.code_templates.keys()) + [
            'typescript', 'cpp', 'c', 'go', 'rust', 'php', 'ruby', 'swift', 'kotlin'
        ]

    def get_language_info(self, language: str) -> Dict[str, Any]:
        """Get information about a specific language's capabilities.

        Args:
            language: Programming language

        Returns:
            Dict containing language info
        """
        templates = self.code_templates.get(language, {})

        return {
            'language': language,
            'has_templates': bool(templates),
            'available_styles': list(templates.keys()) if templates else ['auto'],
            'recommended_max_tokens': {
                'python': 1024,
                'javascript': 1024,
                'java': 1536,
                'cpp': 1536,
                'html': 512,
                'css': 256
            }.get(language, 1024)
        }