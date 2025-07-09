import logging
import re
from typing import Dict, Any, Optional, List
from conductor.engines.base_engine import BaseEngine
from conductor.model_loader import ModelLoader

logger = logging.getLogger(__name__)


class MathematicalReasoningEngine(BaseEngine):
    def __init__(self, config: Dict[str, Any], model_loader: ModelLoader, persona: str = ""):
        super().__init__(config, model_loader, persona)
        self.math_contexts: Dict[str, str] = {
            'algebra': 'algebraic equations and expressions',
            'calculus': 'differentiation, integration, and limits',
            'geometry': 'geometric shapes, proofs, and spatial reasoning',
            'statistics': 'statistical analysis and probability',
            'linear_algebra': 'matrices, vectors, and linear transformations',
            'discrete_math': 'combinatorics, graph theory, and discrete structures',
            'number_theory': 'prime numbers, modular arithmetic, and number properties',
            'optimization': 'optimization problems and mathematical programming'
        }

    def _get_default_generation_params(self) -> Dict[str, Any]:
        """Get default generation parameters optimized for mathematical reasoning."""
        params = super()._get_default_generation_params()
        # Override defaults for mathematical reasoning
        params.update({
            'max_new_tokens': 1024,     # Increased for detailed mathematical explanations
            'temperature': 0.4,         # Lower temperature for more precise reasoning
            'top_p': 0.85,             # Focused vocabulary for mathematical terms
            'repetition_penalty': 1.1   # Prevent repetition in step-by-step solutions
        })
        return params

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate mathematical solution and reasoning.

        Args:
            prompt: Mathematical problem or question
            **kwargs: Additional parameters
                - math_context: Type of math problem (algebra, calculus, etc.)
                - show_work: Whether to show step-by-step work
                - verify_answer: Whether to verify the final answer
                - use_latex: Whether to format math expressions in LaTeX

        Returns:
            str: Mathematical solution with reasoning
        """
        if not self.is_loaded():
            raise RuntimeError("Mathematical reasoning model not loaded")

        try:
            # Parse mathematical context
            math_context = kwargs.get('math_context', 'general')
            show_work = kwargs.get('show_work', True)
            verify_answer = kwargs.get('verify_answer', True)
            use_latex = kwargs.get('use_latex', False)

            # Build mathematical prompt
            math_prompt = self._build_math_prompt(
                prompt, math_context, show_work, verify_answer, use_latex
            )

            # Get generation parameters
            gen_params = self._get_math_generation_params(kwargs)

            # Use parent's generate method with the built prompt and parameters
            response = await super().generate(math_prompt, **gen_params)

            # Post-process mathematical content
            processed_solution = self._post_process_math_solution(
                response, use_latex, kwargs
            )

            self.increment_generation_count()

            logger.debug(f"Generated mathematical solution: {len(processed_solution)} chars")
            return processed_solution

        except Exception as e:
            logger.error(f"Error generating mathematical solution: {e}")
            raise

    def get_system_prompt(self) -> Optional[str]:
        """Get system prompt for mathematical reasoning."""
        if self.persona:
            return self.persona
        
        # Default system prompt for mathematical reasoning
        return """You are an expert mathematician and problem solver. You excel at:
- Solving complex mathematical problems step-by-step
- Providing clear explanations for mathematical concepts
- Verifying solutions using multiple approaches
- Formatting mathematical expressions clearly
- Breaking down complex problems into manageable steps
- Applying various mathematical techniques and theorems

Always show your work, explain your reasoning, and verify your final answer when possible."""

    def _build_math_prompt(self,
                           problem: str,
                           math_context: str,
                           show_work: bool,
                           verify_answer: bool,
                           use_latex: bool) -> str:
        """Build structured prompt for mathematical reasoning.

        Args:
            problem: Mathematical problem statement
            math_context: Type of mathematics
            show_work: Whether to show step-by-step work
            verify_answer: Whether to verify the answer
            use_latex: Whether to use LaTeX formatting

        Returns:
            str: Structured mathematical prompt
        """
        system_prompt = self.get_system_prompt()

        prompt_parts: List[str] = []

        if system_prompt:
            prompt_parts.append(system_prompt)

        # Add mathematical context
        if math_context in self.math_contexts:
            context_description = self.math_contexts[math_context]
            prompt_parts.append(f"This is a {context_description} problem.")

        # Add instructions
        instructions: List[str] = []

        if show_work:
            instructions.append("Show your work step-by-step")
            instructions.append("Explain your reasoning clearly")

        if verify_answer:
            instructions.append("Verify your final answer")
            instructions.append("Check your solution by substitution or alternative method when possible")

        if use_latex:
            instructions.append("Format mathematical expressions using LaTeX notation")
            instructions.append("Use \\( \\) for inline math and \\[ \\] for display math")

        instructions.append("State your final answer clearly")

        if instructions:
            prompt_parts.append("Instructions: " + ". ".join(instructions))

        prompt_parts.append(f"Problem: {problem}")
        prompt_parts.append("Solution:")

        return "\n\n".join(prompt_parts)

    def _get_math_generation_params(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Get generation parameters for mathematical reasoning.

        Args:
            kwargs: User parameters

        Returns:
            Dict[str, Any]: Generation parameters
        """
        # Get base parameters and then customize for mathematical reasoning
        params = self._get_default_generation_params()
        
        # Override with user parameters
        if 'max_tokens' in kwargs:
            params['max_new_tokens'] = kwargs['max_tokens']
        if 'temperature' in kwargs:
            params['temperature'] = max(0.1, min(kwargs['temperature'], 1.0))
        if 'top_p' in kwargs:
            params['top_p'] = kwargs['top_p']
        if 'repetition_penalty' in kwargs:
            params['repetition_penalty'] = kwargs['repetition_penalty']

        # Adjust based on problem complexity
        if 'complex' in kwargs.get('math_context', '').lower():
            params['max_new_tokens'] = min(2048, params['max_new_tokens'] * 2)

        return params

    def _post_process_math_solution(self, solution: str, use_latex: bool, kwargs: Dict[str, Any]) -> str:
        """Post-process mathematical solution for clarity and formatting.

        Args:
            solution: Raw mathematical solution
            use_latex: Whether LaTeX formatting was requested
            kwargs: Additional parameters

        Returns:
            str: Post-processed solution
        """
        if not solution.strip():
            return solution

        # Clean up common artifacts
        solution = self._clean_math_artifacts(solution)

        # Format mathematical expressions
        if use_latex:
            solution = self._format_latex_expressions(solution)
        else:
            solution = self._format_plain_math(solution)

        # Ensure proper structure
        solution = self._structure_math_solution(solution)

        return solution.strip()

    def _clean_math_artifacts(self, solution: str) -> str:
        """Remove common mathematical generation artifacts.

        Args:
            solution: Raw solution

        Returns:
            str: Cleaned solution
        """
        # Remove common prefixes
        prefixes_to_remove = [
            "Here's the solution:",
            "Let me solve this step by step:",
            "Solution:",
            "Step-by-step solution:"
        ]

        for prefix in prefixes_to_remove:
            if solution.startswith(prefix):
                solution = solution[len(prefix):].strip()

        # Remove excessive spacing
        solution = re.sub(r'\n\s*\n\s*\n', '\n\n', solution)

        return solution

    def _format_latex_expressions(self, solution: str) -> str:
        """Format mathematical expressions with LaTeX.

        Args:
            solution: Solution text

        Returns:
            str: LaTeX-formatted solution
        """
        # This is a basic implementation - in practice you'd want more sophisticated LaTeX formatting

        # Convert common mathematical notation
        replacements = [
            (r'\b(\d+)/(\d+)\b', r'\\frac{\1}{\2}'),  # Fractions
            (r'\^(\d+)', r'^{\1}'),  # Exponents
            (r'sqrt\(([^)]+)\)', r'\\sqrt{\1}'),  # Square roots
            (r'integral', r'\\int'),  # Integrals
            (r'sum', r'\\sum'),  # Summations
        ]

        for pattern, replacement in replacements:
            solution = re.sub(pattern, replacement, solution)

        return solution

    def _format_plain_math(self, solution: str) -> str:
        """Format mathematical expressions in plain text.

        Args:
            solution: Solution text

        Returns:
            str: Plain text formatted solution
        """
        # Ensure proper spacing around operators
        solution = re.sub(r'(\w)\+(\w)', r'\1 + \2', solution)
        solution = re.sub(r'(\w)-(\w)', r'\1 - \2', solution)
        solution = re.sub(r'(\w)\*(\w)', r'\1 × \2', solution)
        solution = re.sub(r'(\w)/(\w)', r'\1 ÷ \2', solution)
        solution = re.sub(r'(\w)=(\w)', r'\1 = \2', solution)

        return solution

    def _structure_math_solution(self, solution: str) -> str:
        """Ensure proper structure in mathematical solution.

        Args:
            solution: Solution text

        Returns:
            str: Well-structured solution
        """
        lines = solution.split('\n')
        structured_lines: List[str] = []

        step_counter = 1
        in_steps = False

        for line in lines:
            stripped = line.strip()

            # Identify step-by-step sections
            if any(keyword in stripped.lower() for keyword in ['step', 'first', 'next', 'then', 'finally']):
                if not in_steps:
                    in_steps = True
                    structured_lines.append("\n**Step-by-step solution:**")

                if not stripped.startswith('Step'):
                    line = f"Step {step_counter}: {stripped}"
                    step_counter += 1

            # Identify final answer
            elif any(keyword in stripped.lower() for keyword in ['final answer', 'answer:', 'therefore', 'conclusion']):
                if in_steps:
                    structured_lines.append("")  # Add spacing
                structured_lines.append(f"**Final Answer:** {stripped}")
                continue

            structured_lines.append(line)

        return '\n'.join(structured_lines)

    def solve_equation(self, equation: str, variable: str = 'x') -> Dict[str, Any]:
        """Specialized method for solving equations.

        Args:
            equation: Mathematical equation to solve
            variable: Variable to solve for

        Returns:
            Dict containing solution details
        """
        # This would integrate with symbolic math libraries in a full implementation
        # For now, it's a placeholder that uses the general generation method

        return {
            'equation': equation,
            'variable': variable,
            'method': 'llm_generation',
            'note': 'This is a placeholder - would integrate with SymPy or similar for exact solutions'
        }

    async def warmup(self) -> bool:
        """Warm up with a mathematical problem."""
        if not self.is_loaded():
            return False

        try:
            warmup_problem = "Solve for x: 2x + 5 = 13"
            await self.generate(warmup_problem, max_tokens=100)
            logger.info(f"Warmup completed for {self.category}")
            return True
        except Exception as e:
            logger.error(f"Warmup failed for {self.category}: {e}")
            return False

    def get_math_contexts(self) -> List[str]:
        """Get available mathematical contexts.

        Returns:
            List[str]: Available math contexts
        """
        return list(self.math_contexts.keys())

    def get_context_info(self, context: str) -> Dict[str, Any]:
        """Get information about a mathematical context.

        Args:
            context: Math context name

        Returns:
            Dict containing context info
        """
        return {
            'context': context,
            'description': self.math_contexts.get(context, 'General mathematics'),
            'recommended_params': {
                'show_work': True,
                'verify_answer': True,
                'max_tokens': 512
            }
        }