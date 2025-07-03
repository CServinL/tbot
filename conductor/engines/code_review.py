import logging
import re
from typing import Dict, Any, Optional, AsyncGenerator, List
from conductor.engines.base_engine import BaseEngine
from conductor.model_loader import ModelLoader

logger = logging.getLogger(__name__)


class CodeReviewEngine(BaseEngine):
    def __init__(self, config: Dict[str, Any], model_loader: ModelLoader, persona: str = ""):
        super().__init__(config, model_loader, persona)

        # Review types
        self.review_types: Dict[str, str] = {
            'security_review': 'Focus on security vulnerabilities and best practices',
            'performance_review': 'Analyze performance bottlenecks and optimizations',
            'style_review': 'Check coding style and conventions',
            'architecture_review': 'Evaluate code architecture and design patterns',
            'bug_detection': 'Identify potential bugs and logical errors',
            'maintainability': 'Assess code maintainability and readability',
            'testing_review': 'Review test coverage and testing strategies',
            'documentation_review': 'Evaluate code documentation and comments',
            'dependency_review': 'Analyze dependencies and external libraries',
            'comprehensive': 'Full code review covering all aspects'
        }

        # Programming languages and their specific considerations
        self.language_specifics: Dict[str, Dict[str, Any]] = {
            'python': {
                'style_guide': 'PEP 8',
                'common_issues': ['indentation', 'naming_conventions', 'imports'],
                'security_concerns': ['injection', 'deserialization', 'path_traversal'],
                'performance_tips': ['list_comprehensions', 'generators', 'profiling']
            },
            'javascript': {
                'style_guide': 'ESLint/Prettier',
                'common_issues': ['hoisting', 'closures', 'async_handling'],
                'security_concerns': ['xss', 'injection', 'prototype_pollution'],
                'performance_tips': ['dom_manipulation', 'memory_leaks', 'bundling']
            },
            'java': {
                'style_guide': 'Google Java Style',
                'common_issues': ['memory_management', 'exception_handling', 'concurrency'],
                'security_concerns': ['deserialization', 'injection', 'access_control'],
                'performance_tips': ['gc_optimization', 'collections', 'profiling']
            },
            'cpp': {
                'style_guide': 'Google C++ Style',
                'common_issues': ['memory_leaks', 'undefined_behavior', 'resource_management'],
                'security_concerns': ['buffer_overflow', 'integer_overflow', 'use_after_free'],
                'performance_tips': ['smart_pointers', 'move_semantics', 'optimization']
            },
            'go': {
                'style_guide': 'gofmt',
                'common_issues': ['error_handling', 'goroutine_leaks', 'interface_usage'],
                'security_concerns': ['injection', 'path_traversal', 'race_conditions'],
                'performance_tips': ['profiling', 'memory_allocation', 'concurrency']
            },
            'rust': {
                'style_guide': 'rustfmt',
                'common_issues': ['borrowing', 'lifetime_management', 'error_handling'],
                'security_concerns': ['unsafe_blocks', 'dependency_vulnerabilities'],
                'performance_tips': ['zero_cost_abstractions', 'memory_efficiency']
            }
        }

        # Severity levels for issues
        self.severity_levels: Dict[str, str] = {
            'critical': 'Critical issues that must be fixed immediately',
            'high': 'High priority issues that should be addressed soon',
            'medium': 'Medium priority issues for consideration',
            'low': 'Low priority suggestions and improvements',
            'info': 'Informational notes and best practices'
        }

    def _get_default_generation_params(self) -> Dict[str, Any]:
        """Get default generation parameters optimized for code review."""
        params = super()._get_default_generation_params()
        # Override defaults for code review
        params.update({
            'max_new_tokens': 1024,     # Enough for detailed reviews
            'temperature': 0.4,         # Lower temperature for precise analysis
            'top_p': 0.85,             # Focused but not too narrow
            'repetition_penalty': 1.1   # Prevent repetitive analysis
        })
        return params

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Perform code review and analysis.

        Args:
            prompt: Code to review (can include context/description)
            **kwargs: Additional parameters
                - code_to_review: Explicit code to review (if different from prompt)
                - language: Programming language
                - review_type: Type of review to perform
                - severity_filter: Minimum severity level to report
                - include_suggestions: Whether to include fix suggestions
                - context: Additional context about the code
                - focus_areas: Specific areas to focus on

        Returns:
            str: Code review with findings and recommendations
        """
        try:
            # Parse review parameters
            code_to_review = kwargs.get('code_to_review', prompt)
            language = kwargs.get('language', self._detect_language(code_to_review))
            review_type = kwargs.get('review_type', 'comprehensive')
            severity_filter = kwargs.get('severity_filter', 'low')
            include_suggestions = kwargs.get('include_suggestions', True)
            context = kwargs.get('context')
            focus_areas = kwargs.get('focus_areas', [])

            # Analyze the code
            code_analysis = self._analyze_code_structure(code_to_review, language)

            # Build code review prompt
            review_prompt = self._build_review_prompt(
                code_to_review, language, review_type, severity_filter,
                include_suggestions, context, focus_areas, code_analysis
            )

            # Get generation parameters optimized for code review
            gen_params = self._get_review_params(kwargs, review_type, language)

            # Use parent's generate method with the built prompt and parameters
            response = await super().generate(review_prompt, **gen_params)

            # Post-process the review
            processed_review = self._post_process_code_review(
                response, review_type, language, code_analysis, kwargs
            )

            logger.debug(f"Generated {review_type} code review for {language}: {len(processed_review)} chars")
            return processed_review

        except Exception as e:
            logger.error(f"Error generating code review: {e}")
            raise

    async def generate_stream(self, prompt: str, **kwargs: Any) -> AsyncGenerator[str, None]:
        """Generate streaming code review.

        Args:
            prompt: Code to review
            **kwargs: Additional parameters

        Yields:
            str: Code review chunks
        """
        # Parse review parameters
        code_to_review = kwargs.get('code_to_review', prompt)
        language = kwargs.get('language', self._detect_language(code_to_review))
        review_type = kwargs.get('review_type', 'comprehensive')
        severity_filter = kwargs.get('severity_filter', 'low')
        include_suggestions = kwargs.get('include_suggestions', True)
        context = kwargs.get('context')
        focus_areas = kwargs.get('focus_areas', [])

        # Analyze the code
        code_analysis = self._analyze_code_structure(code_to_review, language)

        # Build code review prompt
        review_prompt = self._build_review_prompt(
            code_to_review, language, review_type, severity_filter,
            include_suggestions, context, focus_areas, code_analysis
        )

        # Get generation parameters optimized for code review
        gen_params = self._get_review_params(kwargs, review_type, language)

        # Since base class doesn't have generate_stream, use regular generate
        try:
            result = await super().generate(review_prompt, **gen_params)
            processed_result = self._post_process_code_review(
                result, review_type, language, code_analysis, kwargs
            )
            yield processed_result
        except Exception as e:
            logger.error(f"Error in streaming code review: {e}")
            yield f"// Error: {str(e)}"

    def get_system_prompt(self) -> Optional[str]:
        """Get system prompt for code review."""
        if self.persona:
            return self.persona
        
        # Default code review system prompt
        return ("You are an expert code reviewer. Analyze the provided code for bugs, "
                "security issues, performance problems, style violations, and best practices. "
                "Provide specific, actionable feedback with clear explanations and suggestions for improvement.")

    def _detect_language(self, code: str) -> str:
        """Detect programming language from code."""
        # Simple language detection based on patterns
        code_lower = code.lower()

        if 'def ' in code_lower or 'import ' in code_lower or code_lower.count('    ') > code_lower.count('\t'):
            return 'python'
        elif 'function ' in code_lower or 'const ' in code_lower or 'let ' in code_lower:
            return 'javascript'
        elif 'public class' in code_lower or 'import java' in code_lower:
            return 'java'
        elif '#include' in code_lower or 'std::' in code_lower:
            return 'cpp'
        elif 'func ' in code_lower or 'package ' in code_lower:
            return 'go'
        elif 'fn ' in code_lower or 'use std::' in code_lower:
            return 'rust'
        else:
            return 'unknown'

    def _analyze_code_structure(self, code: str, language: str) -> Dict[str, Any]:
        """Analyze code structure for review context.

        Args:
            code: Code to analyze
            language: Programming language

        Returns:
            Dict containing code analysis
        """
        analysis: Dict[str, Any] = {
            'line_count': len(code.split('\n')),
            'complexity_estimate': 'medium',
            'has_functions': False,
            'has_classes': False,
            'has_comments': False,
            'has_imports': False,
            'indentation_style': 'spaces'
        }

        # Detect functions/methods
        function_patterns = {
            'python': r'def\s+\w+',
            'javascript': r'function\s+\w+',
            'java': r'(public|private|protected).*\s+\w+\s*\(',
            'cpp': r'\w+::\w+|\w+\s+\w+\s*\(',
            'go': r'func\s+\w+',
            'rust': r'fn\s+\w+'
        }

        if language in function_patterns:
            pattern = function_patterns[language]
            if re.search(pattern, code):
                analysis['has_functions'] = True

        # Detect classes
        class_patterns = {
            'python': r'class\s+\w+',
            'javascript': r'class\s+\w+',
            'java': r'(public|private)?\s*class\s+\w+',
            'cpp': r'class\s+\w+',
            'rust': r'struct\s+\w+|impl\s+\w+'
        }

        if language in class_patterns:
            pattern = class_patterns[language]
            if re.search(pattern, code):
                analysis['has_classes'] = True

        # Check for comments
        if '//' in code or '/*' in code or '#' in code:
            analysis['has_comments'] = True

        # Check for imports
        import_patterns = ['import ', 'from ', '#include', 'use ', 'require(']
        if any(pattern in code for pattern in import_patterns):
            analysis['has_imports'] = True

        # Estimate complexity
        complexity_indicators = len(re.findall(r'\b(if|for|while|switch|case)\b', code))
        if complexity_indicators > 10:
            analysis['complexity_estimate'] = 'high'
        elif complexity_indicators < 3:
            analysis['complexity_estimate'] = 'low'

        # Check indentation
        if '\t' in code and code.count('\t') > code.count('    '):
            analysis['indentation_style'] = 'tabs'

        return analysis

    def _build_review_prompt(self,
                             code: str,
                             language: str,
                             review_type: str,
                             severity_filter: str,
                             include_suggestions: bool,
                             context: Optional[str],
                             focus_areas: List[str],
                             analysis: Dict[str, Any]) -> str:
        """Build prompt for code review.

        Args:
            code: Code to review
            language: Programming language
            review_type: Type of review
            severity_filter: Minimum severity level
            include_suggestions: Whether to include suggestions
            context: Additional context
            focus_areas: Areas to focus on
            analysis: Code analysis

        Returns:
            str: Code review prompt
        """
        system_prompt = self.get_system_prompt()

        prompt_parts: List[str] = []

        if system_prompt:
            prompt_parts.append(system_prompt)

        # Add context if provided
        if context:
            prompt_parts.append(f"Context: {context}")

        # Add review instructions
        instructions: List[str] = []

        # Review type specific instructions
        if review_type in self.review_types:
            review_description = self.review_types[review_type]
            instructions.append(f"Review focus: {review_description}")

        # Language-specific instructions
        if language in self.language_specifics:
            lang_info = self.language_specifics[language]
            instructions.append(f"Apply {language} best practices and {lang_info['style_guide']} guidelines")

            if review_type == 'security_review':
                instructions.append(
                    f"Pay attention to {language}-specific security concerns: {', '.join(lang_info['security_concerns'])}")
            elif review_type == 'performance_review':
                instructions.append(
                    f"Consider {language} performance optimizations: {', '.join(lang_info['performance_tips'])}")

        # Focus areas
        if focus_areas:
            instructions.append(f"Pay special attention to: {', '.join(focus_areas)}")

        # Severity and suggestions
        instructions.append(f"Report issues of {severity_filter} severity and above")

        if include_suggestions:
            instructions.append("Provide specific suggestions and improved code examples where applicable")

        # General review guidelines
        instructions.extend([
            "Categorize issues by severity (critical, high, medium, low, info)",
            "Provide clear explanations for each issue identified",
            "Focus on correctness, security, performance, and maintainability",
            "Consider code readability and best practices"
        ])

        # Analysis-based instructions
        if analysis['complexity_estimate'] == 'high':
            instructions.append("Pay extra attention to code complexity and potential simplification")

        if not analysis['has_comments']:
            instructions.append("Consider commenting and documentation needs")

        if instructions:
            prompt_parts.append("Review instructions:\n" + "\n".join([f"- {inst}" for inst in instructions]))

        prompt_parts.append(f"Code to review ({language}):\n```{language}\n{code}\n```")
        prompt_parts.append("Code review:")

        return "\n\n".join(prompt_parts)

    def _get_review_params(self, kwargs: Dict[str, Any], review_type: str, language: str) -> Dict[str, Any]:
        """Get generation parameters for code review.

        Args:
            kwargs: User parameters
            review_type: Type of review
            language: Programming language

        Returns:
            Dict[str, Any]: Generation parameters
        """
        # Use base class parameter validation with review-specific overrides
        review_kwargs = kwargs.copy()
        
        # Adjust based on review type
        if review_type == 'comprehensive':
            review_kwargs['max_new_tokens'] = review_kwargs.get('max_new_tokens', 2048)
        elif review_type == 'security_review':
            review_kwargs['temperature'] = 0.3  # Very precise for security
            review_kwargs['max_new_tokens'] = review_kwargs.get('max_new_tokens', 1536)
        elif review_type == 'performance_review':
            review_kwargs['temperature'] = 0.5
        elif review_type == 'style_review':
            review_kwargs['temperature'] = 0.3
        elif review_type == 'architecture_review':
            review_kwargs['max_new_tokens'] = review_kwargs.get('max_new_tokens', 1536)

        # Override with user temperature if provided, but clamp for code review
        if 'temperature' in kwargs:
            review_kwargs['temperature'] = max(0.1, min(kwargs['temperature'], 0.8))

        # Use base class validation which handles clamping and safety limits
        return self._validate_generation_params(review_kwargs)

    def _post_process_code_review(self,
                                  review: str,
                                  review_type: str,
                                  language: str,
                                  analysis: Dict[str, Any],
                                  kwargs: Dict[str, Any]) -> str:
        """Post-process code review for formatting and clarity.

        Args:
            review: Raw code review
            review_type: Type of review
            language: Programming language
            analysis: Code analysis
            kwargs: Generation parameters

        Returns:
            str: Post-processed code review
        """
        if not review.strip():
            return review

        # Clean up common artifacts
        review = self._clean_review_artifacts(review)

        # Format by review type
        if review_type == 'comprehensive':
            review = self._format_comprehensive_review(review)
        elif review_type == 'security_review':
            review = self._format_security_review(review)
        elif review_type == 'performance_review':
            review = self._format_performance_review(review)

        # Add summary if requested
        if kwargs.get('include_summary', True):
            review = self._add_review_summary(review, analysis)

        return review.strip()

    def _clean_review_artifacts(self, review: str) -> str:
        """Remove common code review artifacts.

        Args:
            review: Raw review

        Returns:
            str: Cleaned review
        """
        # Remove common prefixes
        prefixes_to_remove = [
            "Code review:",
            "Here's the code review:",
            "Code analysis:",
            "Review findings:"
        ]

        for prefix in prefixes_to_remove:
            if review.startswith(prefix):
                review = review[len(prefix):].strip()

        return review

    def _format_comprehensive_review(self, review: str) -> str:
        """Format as comprehensive review with sections.

        Args:
            review: Review content

        Returns:
            str: Formatted comprehensive review
        """
        if not any(header in review for header in ['Security', 'Performance', 'Style', 'Maintainability']):
            review = f"**Comprehensive Code Review**\n\n{review}"

        return review

    def _format_security_review(self, review: str) -> str:
        """Format as security-focused review.

        Args:
            review: Review content

        Returns:
            str: Security-formatted review
        """
        if 'security' not in review.lower():
            review = f"**Security Review**\n\n{review}"

        return review

    def _format_performance_review(self, review: str) -> str:
        """Format as performance-focused review.

        Args:
            review: Review content

        Returns:
            str: Performance-formatted review
        """
        if 'performance' not in review.lower():
            review = f"**Performance Review**\n\n{review}"

        return review

    def _add_review_summary(self, review: str, analysis: Dict[str, Any]) -> str:
        """Add summary to code review.

        Args:
            review: Code review
            analysis: Code analysis

        Returns:
            str: Review with summary
        """
        summary = f"\n\n**Code Overview:**\n"
        summary += f"- Lines of code: {analysis['line_count']}\n"
        summary += f"- Estimated complexity: {analysis['complexity_estimate']}\n"
        summary += f"- Contains functions: {'Yes' if analysis['has_functions'] else 'No'}\n"
        summary += f"- Contains classes: {'Yes' if analysis['has_classes'] else 'No'}\n"
        summary += f"- Has documentation: {'Yes' if analysis['has_comments'] else 'No'}"

        return review + summary

    async def find_bugs(self,
                        code: str,
                        language: Optional[str] = None) -> str:
        """Specialized bug detection in code.

        Args:
            code: Code to analyze for bugs
            language: Programming language

        Returns:
            str: Bug analysis report
        """
        return await self.generate(
            code,
            language=language,
            review_type='bug_detection',
            severity_filter='medium',
            include_suggestions=True,
            focus_areas=['logical_errors', 'runtime_errors', 'edge_cases']
        )

    async def security_audit(self,
                             code: str,
                             language: Optional[str] = None) -> str:
        """Perform security audit of code.

        Args:
            code: Code to audit
            language: Programming language

        Returns:
            str: Security audit report
        """
        return await self.generate(
            code,
            language=language,
            review_type='security_review',
            severity_filter='low',
            include_suggestions=True,
            focus_areas=['vulnerabilities', 'input_validation', 'authentication', 'authorization']
        )

    async def warmup(self) -> bool:
        """Warm up with a code review example."""
        if not self.is_loaded():
            return False

        try:
            warmup_code = """
def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)
"""
            await self.generate(warmup_code, language='python', review_type='style_review', max_tokens=200)
            logger.info(f"Warmup completed for {self.category}")
            return True
        except Exception as e:
            logger.error(f"Warmup failed for {self.category}: {e}")
            return False

    def get_review_types(self) -> List[str]:
        """Get available review types.

        Returns:
            List[str]: Available review types
        """
        return list(self.review_types.keys())

    def get_supported_languages(self) -> List[str]:
        """Get supported programming languages.

        Returns:
            List[str]: Supported languages
        """
        return list(self.language_specifics.keys())

    def get_language_info(self, language: str) -> Dict[str, Any]:
        """Get information about a programming language.

        Args:
            language: Programming language

        Returns:
            Dict containing language info
        """
        if language in self.language_specifics:
            return self.language_specifics[language]
        else:
            return {
                'style_guide': 'General best practices',
                'common_issues': ['readability', 'maintainability'],
                'security_concerns': ['input_validation', 'error_handling'],
                'performance_tips': ['algorithm_efficiency', 'resource_management']
            }