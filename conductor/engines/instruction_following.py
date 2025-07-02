import logging
import re
from typing import Dict, Any, Optional, AsyncGenerator, List, Union
from conductor.engines.base_engine import BaseEngine
from conductor.model_loader import ModelLoader

logger = logging.getLogger(__name__)


class InstructionFollowingEngine(BaseEngine):
    def __init__(self, config: Dict[str, Any], model_loader: ModelLoader, persona: str = ""):
        super().__init__(config, model_loader, persona)

        # Instruction types and patterns
        self.instruction_types = {
            'task_completion': 'Complete a specific task or assignment',
            'format_conversion': 'Convert content from one format to another',
            'analysis_request': 'Analyze and provide insights on given content',
            'transformation': 'Transform or modify existing content',
            'extraction': 'Extract specific information from content',
            'classification': 'Classify or categorize content',
            'comparison': 'Compare multiple items or concepts',
            'synthesis': 'Combine multiple sources into cohesive output'
        }

        # Common instruction patterns
        self.instruction_patterns = {
            'step_by_step': r'step.?by.?step|one.?by.?one|sequentially',
            'list_format': r'list|bullet.?points|numbered|enumerate',
            'table_format': r'table|tabular|rows?.?and.?columns?',
            'summary_request': r'summarize|sum.?up|briefly|concise',
            'detailed_request': r'detailed?|comprehensive|thorough|in.?depth',
            'example_request': r'example|instance|case|illustration',
            'comparison_request': r'compare|contrast|difference|similarity',
            'explanation_request': r'explain|clarify|describe|elaborate'
        }

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Follow instructions and generate appropriate response.

        Args:
            prompt: Instructions to follow
            **kwargs: Additional parameters
                - instruction_type: Type of instruction being given
                - context: Additional context for the instruction
                - format_requirements: Specific format requirements
                - examples: Example inputs/outputs
                - constraints: Constraints or limitations

        Returns:
            str: Response following the instructions
        """
        # Parse instruction parameters
        instruction_type = kwargs.get('instruction_type', 'task_completion')
        context = kwargs.get('context')
        format_requirements = kwargs.get('format_requirements')
        examples = kwargs.get('examples', [])
        constraints = kwargs.get('constraints', [])

        # Analyze the instruction
        instruction_analysis = self._analyze_instruction(prompt)

        # Build instruction-following prompt
        instruction_prompt = self._build_instruction_prompt(
            prompt, instruction_type, context, format_requirements,
            examples, constraints, instruction_analysis
        )

        # Get generation parameters
        gen_params = self._get_instruction_params(kwargs, instruction_analysis)
        
        # Use parent's generate method with the built prompt and parameters
        response = await super().generate(instruction_prompt, **gen_params)

        # Post-process based on instruction requirements
        processed_response = self._post_process_instruction_response(
            response, instruction_analysis, kwargs
        )

        logger.debug(f"Followed instruction: {len(processed_response)} chars")
        return processed_response

    async def generate_stream(self, prompt: str, **kwargs: Any) -> AsyncGenerator[str, None]:
        """Generate streaming instruction response.

        Args:
            prompt: Instructions to follow
            **kwargs: Additional parameters

        Yields:
            str: Response chunks
        """
        # Parse instruction parameters
        instruction_type = kwargs.get('instruction_type', 'task_completion')
        context = kwargs.get('context')
        format_requirements = kwargs.get('format_requirements')
        examples = kwargs.get('examples', [])
        constraints = kwargs.get('constraints', [])

        # Analyze the instruction
        instruction_analysis = self._analyze_instruction(prompt)

        # Build instruction-following prompt
        instruction_prompt = self._build_instruction_prompt(
            prompt, instruction_type, context, format_requirements,
            examples, constraints, instruction_analysis
        )

        # Get generation parameters
        gen_params = self._get_instruction_params(kwargs, instruction_analysis)
        
        # Use parent's generate_stream method with the built prompt and parameters
        async for chunk in super().generate_stream(instruction_prompt, **gen_params):
            yield chunk

    def get_system_prompt(self) -> Optional[str]:
        """Get system prompt for instruction following."""
        if hasattr(self, 'persona_loader'):
            return self.persona_loader.get_persona_for_category('instruction_following')
        return None

    def _analyze_instruction(self, instruction: str) -> Dict[str, Any]:
        """Analyze instruction to understand requirements.

        Args:
            instruction: Instruction text to analyze

        Returns:
            Dict containing instruction analysis
        """
        analysis = {
            'detected_patterns': [],
            'complexity': 'medium',
            'requires_formatting': False,
            'requires_examples': False,
            'requires_step_by_step': False,
            'output_type': 'text'
        }

        instruction_lower = instruction.lower()

        # Detect instruction patterns
        for pattern_name, pattern_regex in self.instruction_patterns.items():
            if re.search(pattern_regex, instruction_lower):
                analysis['detected_patterns'].append(pattern_name)

        # Determine complexity
        complexity_indicators = len(re.findall(r'[.!?]', instruction))
        if complexity_indicators > 5 or len(instruction) > 500:
            analysis['complexity'] = 'high'
        elif complexity_indicators < 2 or len(instruction) < 100:
            analysis['complexity'] = 'low'

        # Check for formatting requirements
        if any(pattern in analysis['detected_patterns'] for pattern in ['list_format', 'table_format']):
            analysis['requires_formatting'] = True

        # Check for step-by-step requirement
        if 'step_by_step' in analysis['detected_patterns']:
            analysis['requires_step_by_step'] = True

        # Check for example requirement
        if 'example_request' in analysis['detected_patterns']:
            analysis['requires_examples'] = True

        # Determine output type
        if 'table_format' in analysis['detected_patterns']:
            analysis['output_type'] = 'table'
        elif 'list_format' in analysis['detected_patterns']:
            analysis['output_type'] = 'list'
        elif 'step_by_step' in analysis['detected_patterns']:
            analysis['output_type'] = 'steps'

        return analysis

    def _build_instruction_prompt(self,
                                  instruction: str,
                                  instruction_type: str,
                                  context: Optional[str],
                                  format_requirements: Optional[str],
                                  examples: List[Dict[str, str]],
                                  constraints: List[str],
                                  analysis: Dict[str, Any]) -> str:
        """Build prompt for instruction following.

        Args:
            instruction: Original instruction
            instruction_type: Type of instruction
            context: Additional context
            format_requirements: Format requirements
            examples: Example inputs/outputs
            constraints: Constraints
            analysis: Instruction analysis

        Returns:
            str: Structured instruction prompt
        """
        system_prompt = self.get_system_prompt()

        prompt_parts = []

        if system_prompt:
            prompt_parts.append(system_prompt)

        # Add context if provided
        if context:
            prompt_parts.append(f"Context: {context}")

        # Add examples if provided
        if examples:
            examples_text = "Examples:"
            for i, example in enumerate(examples, 1):
                if 'input' in example and 'output' in example:
                    examples_text += f"\nExample {i}:\nInput: {example['input']}\nOutput: {example['output']}"
            prompt_parts.append(examples_text)

        # Add format requirements
        format_instructions = []

        if format_requirements:
            format_instructions.append(f"Format requirement: {format_requirements}")

        # Add format instructions based on analysis
        if analysis['requires_formatting']:
            if analysis['output_type'] == 'list':
                format_instructions.append("Present your response as a clear, organized list")
            elif analysis['output_type'] == 'table':
                format_instructions.append("Present your response in a well-structured table format")
            elif analysis['output_type'] == 'steps':
                format_instructions.append("Present your response as numbered steps")

        if analysis['requires_examples'] and not examples:
            format_instructions.append("Include relevant examples in your response")

        # Add constraints
        if constraints:
            constraint_text = "Constraints:\n" + "\n".join([f"- {constraint}" for constraint in constraints])
            format_instructions.append(constraint_text)

        # General instruction following guidelines
        format_instructions.extend([
            "Follow the instruction precisely and completely",
            "Maintain accuracy and relevance",
            "Be thorough but concise"
        ])

        if format_instructions:
            prompt_parts.append("Instructions: " + "\n".join(format_instructions))

        prompt_parts.append(f"Task: {instruction}")
        prompt_parts.append("Response:")

        return "\n\n".join(prompt_parts)

    def _get_instruction_params(self, kwargs: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Get generation parameters for instruction following.

        Args:
            kwargs: User parameters
            analysis: Instruction analysis

        Returns:
            Dict[str, Any]: Generation parameters
        """
        # Base parameters for instruction following
        params = {
            'max_new_tokens': kwargs.get('max_tokens', 1024),
            'temperature': 0.5,  # Moderate temperature for balanced creativity/accuracy
            'do_sample': True,
            'top_p': 0.9,
            'repetition_penalty': 1.1,
            'pad_token_id': self.tokenizer.eos_token_id if self.tokenizer else None
        }

        # Adjust based on instruction complexity
        if analysis['complexity'] == 'high':
            params['max_new_tokens'] = min(2048, params['max_new_tokens'] * 2)
            params['temperature'] = 0.6  # Slightly higher for complex tasks
        elif analysis['complexity'] == 'low':
            params['max_new_tokens'] = min(512, params['max_new_tokens'])
            params['temperature'] = 0.4  # Lower for simple, precise tasks

        # Adjust based on output type
        if analysis['output_type'] in ['list', 'table', 'steps']:
            params['temperature'] = 0.4  # Lower for structured output
            params['repetition_penalty'] = 1.05  # Allow some repetition in structured formats

        # Adjust for formatting requirements
        if analysis['requires_formatting']:
            params['temperature'] = 0.4

        # Override with user parameters
        if 'temperature' in kwargs:
            params['temperature'] = max(0.1, min(kwargs['temperature'], 1.0))

        return params

    def _post_process_instruction_response(self,
                                           response: str,
                                           analysis: Dict[str, Any],
                                           kwargs: Dict[str, Any]) -> str:
        """Post-process response based on instruction requirements.

        Args:
            response: Raw response
            analysis: Instruction analysis
            kwargs: Generation parameters

        Returns:
            str: Post-processed response
        """
        if not response.strip():
            return response

        # Clean up common artifacts
        response = self._clean_instruction_artifacts(response)

        # Apply formatting based on analysis
        if analysis['requires_formatting']:
            if analysis['output_type'] == 'list':
                response = self._format_as_list(response)
            elif analysis['output_type'] == 'table':
                response = self._format_as_table(response)
            elif analysis['output_type'] == 'steps':
                response = self._format_as_steps(response)

        # Validate completeness
        if kwargs.get('validate_completeness', True):
            response = self._validate_instruction_completeness(response, analysis)

        return response.strip()

    def _clean_instruction_artifacts(self, response: str) -> str:
        """Remove common instruction following artifacts.

        Args:
            response: Raw response

        Returns:
            str: Cleaned response
        """
        # Remove common prefixes
        prefixes_to_remove = [
            "Here's the response:",
            "Following the instruction:",
            "As requested:",
            "Based on the instruction:"
        ]

        for prefix in prefixes_to_remove:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()

        return response

    def _format_as_list(self, text: str) -> str:
        """Format text as a structured list.

        Args:
            text: Text to format

        Returns:
            str: List-formatted text
        """
        lines = text.split('\n')
        formatted_lines = []

        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith(('-', '•', '*', '1.', '2.', '3.')):
                # Convert to list item if it looks like content
                if len(stripped) > 10 and '.' in stripped:
                    formatted_lines.append(f"- {stripped}")
                else:
                    formatted_lines.append(stripped)
            else:
                formatted_lines.append(stripped)

        return '\n'.join(formatted_lines)

    def _format_as_table(self, text: str) -> str:
        """Format text as a table structure.

        Args:
            text: Text to format

        Returns:
            str: Table-formatted text
        """
        # Basic table formatting - in practice you'd want more sophisticated table detection
        lines = text.split('\n')

        # Look for tabular data
        table_lines = []
        for line in lines:
            if '|' in line or '\t' in line:
                table_lines.append(line)
            elif line.strip() and len(line.split()) > 2:
                # Convert to table format
                cells = line.split()
                if len(cells) <= 5:  # Reasonable number of columns
                    table_lines.append(' | '.join(cells))
                else:
                    table_lines.append(line)
            else:
                table_lines.append(line)

        return '\n'.join(table_lines)

    def _format_as_steps(self, text: str) -> str:
        """Format text as numbered steps.

        Args:
            text: Text to format

        Returns:
            str: Step-formatted text
        """
        lines = text.split('\n')
        formatted_lines = []
        step_counter = 1

        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith(('Step', '1.', '2.', '3.')):
                # Convert to step if it looks like an action
                if len(stripped) > 15:
                    formatted_lines.append(f"Step {step_counter}: {stripped}")
                    step_counter += 1
                else:
                    formatted_lines.append(stripped)
            else:
                formatted_lines.append(stripped)

        return '\n'.join(formatted_lines)

    def _validate_instruction_completeness(self,
                                           response: str,
                                           analysis: Dict[str, Any]) -> str:
        """Validate that response addresses instruction requirements.

        Args:
            response: Generated response
            analysis: Instruction analysis

        Returns:
            str: Validated response (may include completeness notes)
        """
        issues = []

        # Check for required examples
        if analysis['requires_examples'] and 'example' not in response.lower():
            issues.append("Note: Response may benefit from including examples")

        # Check for step-by-step format
        if analysis['requires_step_by_step'] and 'step' not in response.lower():
            issues.append("Note: Response may benefit from step-by-step format")

        # Check minimum length for complex instructions
        if analysis['complexity'] == 'high' and len(response) < 200:
            issues.append("Note: Response may be too brief for the complexity of the instruction")

        if issues:
            validation_note = "\n\n" + "\n".join(issues)
            return response + validation_note

        return response

    async def execute_multi_step_instruction(self,
                                             steps: List[str],
                                             context: Optional[str] = None) -> List[str]:
        """Execute a multi-step instruction sequence.

        Args:
            steps: List of instruction steps
            context: Shared context for all steps

        Returns:
            List[str]: Results for each step
        """
        results = []
        accumulated_context = context or ""

        for i, step in enumerate(steps, 1):
            step_context = accumulated_context
            if results:
                step_context += f"\n\nPrevious step results:\n{chr(10).join(results)}"

            result = await self.generate(
                step,
                context=step_context,
                instruction_type='task_completion'
            )

            results.append(result)
            accumulated_context += f"\nStep {i} completed: {step}"

        return results

    async def warmup(self) -> bool:
        """Warm up with an instruction following example."""
        if not self.is_loaded():
            return False

        try:
            warmup_instruction = "List three benefits of regular exercise"
            await self.generate(warmup_instruction, max_tokens=150)
            logger.info(f"Warmup completed for {self.category}")
            return True
        except Exception as e:
            logger.error(f"Warmup failed for {self.category}: {e}")
            return False

    def get_instruction_types(self) -> List[str]:
        """Get available instruction types.

        Returns:
            List[str]: Available instruction types
        """
        return list(self.instruction_types.keys())

    def get_instruction_type_info(self, instruction_type: str) -> Dict[str, Any]:
        """Get information about a specific instruction type.

        Args:
            instruction_type: Type of instruction

        Returns:
            Dict containing instruction type info
        """
        return {
            'type': instruction_type,
            'description': self.instruction_types.get(instruction_type, 'Custom instruction type'),
            'recommended_params': {
                'temperature': 0.5,
                'validate_completeness': True
            }
        }