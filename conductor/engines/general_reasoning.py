import asyncio
import logging
from typing import Dict, Any, Optional, AsyncGenerator, List
from base_llm_engine import BaseLLMEngine
from model_loader import ModelLoader
from utils.persona_loader import PersonaLoader

logger = logging.getLogger(__name__)


class GeneralReasoningEngine(BaseLLMEngine):
    """Engine for general reasoning tasks and analytical thinking."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_loader = ModelLoader()
        self.persona_loader = PersonaLoader()
        self.reasoning_contexts = {
            'logical': 'logical deduction and inference',
            'analytical': 'analytical thinking and problem breakdown',
            'causal': 'cause and effect relationships',
            'comparative': 'comparison and contrast analysis',
            'critical': 'critical thinking and evaluation',
            'strategic': 'strategic planning and decision making',
            'ethical': 'ethical considerations and moral reasoning',
            'systematic': 'systematic approach to complex problems'
        }

    async def load_model(self) -> bool:
        """Load the general reasoning model."""
        try:
            logger.info(f"Loading general reasoning model: {self.technical_model_name}")
            self.model, self.tokenizer = await self.model_loader.load_model(
                self.technical_model_name,
                self.precision
            )

            if self.model is not None and self.tokenizer is not None:
                self.is_model_loaded = True
                self.load_time = asyncio.get_event_loop().time()
                logger.info(f"Successfully loaded general reasoning model")

                # Perform warmup
                await self.warmup()
                return True
            else:
                logger.error("Failed to load general reasoning model")
                return False

        except Exception as e:
            logger.error(f"Error loading general reasoning model: {e}")
            return False

    async def unload_model(self) -> bool:
        """Unload the general reasoning model."""
        try:
            if self.is_model_loaded:
                success = await self.model_loader.unload_model(self.technical_model_name)
                if success:
                    self.model = None
                    self.tokenizer = None
                    self.is_model_loaded = False
                    logger.info("General reasoning model unloaded")
                return success
            return True
        except Exception as e:
            logger.error(f"Error unloading general reasoning model: {e}")
            return False

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate reasoning-based response.

        Args:
            prompt: Question or problem requiring reasoning
            **kwargs: Additional parameters
                - reasoning_type: Type of reasoning needed
                - step_by_step: Whether to show reasoning steps
                - consider_alternatives: Whether to explore multiple perspectives
                - depth: Level of analysis depth (shallow, medium, deep)

        Returns:
            str: Reasoned response with analysis
        """
        if not self.is_model_loaded:
            raise RuntimeError("General reasoning model not loaded")

        try:
            # Parse reasoning parameters
            reasoning_type = kwargs.get('reasoning_type', 'analytical')
            step_by_step = kwargs.get('step_by_step', True)
            consider_alternatives = kwargs.get('consider_alternatives', True)
            depth = kwargs.get('depth', 'medium')

            # Build reasoning prompt
            reasoning_prompt = self._build_reasoning_prompt(
                prompt, reasoning_type, step_by_step, consider_alternatives, depth
            )

            # Get generation parameters
            gen_params = self._get_reasoning_params(kwargs, reasoning_type)

            # Tokenize input
            inputs = self.tokenizer(
                reasoning_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            )

            # Move to device
            if hasattr(self.model, 'device'):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Generate reasoning
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
                    eos_token_id=self.tokenizer.eos_token_id
                )

            # Decode and process response
            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            reasoning_response = self._extract_response(full_output, reasoning_prompt)

            # Post-process reasoning
            processed_response = self._post_process_reasoning(
                reasoning_response, reasoning_type, kwargs
            )

            self.increment_generation_count()

            logger.debug(f"Generated reasoning response: {len(processed_response)} chars")
            return processed_response

        except Exception as e:
            logger.error(f"Error generating reasoning response: {e}")
            raise

    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Generate streaming reasoning response.

        Args:
            prompt: Question or problem
            **kwargs: Additional parameters

        Yields:
            str: Response chunks
        """
        if not self.is_model_loaded:
            raise RuntimeError("General reasoning model not loaded")

        try:
            from transformers import TextIteratorStreamer
            import torch
            from threading import Thread

            reasoning_type = kwargs.get('reasoning_type', 'analytical')
            reasoning_prompt = self._build_reasoning_prompt(
                prompt, reasoning_type,
                kwargs.get('step_by_step', True),
                kwargs.get('consider_alternatives', True),
                kwargs.get('depth', 'medium')
            )

            gen_params = self._get_reasoning_params(kwargs, reasoning_type)

            inputs = self.tokenizer(
                reasoning_prompt,
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

            for chunk in streamer:
                yield chunk

            generation_thread.join()
            self.increment_generation_count()

        except Exception as e:
            logger.error(f"Error in streaming reasoning: {e}")
            yield f"Error: {str(e)}"

    def get_system_prompt(self) -> Optional[str]:
        """Get system prompt for general reasoning."""
        return self.persona_loader.get_persona_for_category('general_reasoning')

    def _build_reasoning_prompt(self,
                                question: str,
                                reasoning_type: str,
                                step_by_step: bool,
                                consider_alternatives: bool,
                                depth: str) -> str:
        """Build structured prompt for reasoning tasks.

        Args:
            question: Question or problem to reason about
            reasoning_type: Type of reasoning approach
            step_by_step: Whether to show reasoning steps
            consider_alternatives: Whether to explore alternatives
            depth: Level of analysis depth

        Returns:
            str: Structured reasoning prompt
        """
        system_prompt = self.get_system_prompt()

        prompt_parts = []

        if system_prompt:
            prompt_parts.append(system_prompt)

        # Add reasoning context
        if reasoning_type in self.reasoning_contexts:
            context_description = self.reasoning_contexts[reasoning_type]
            prompt_parts.append(f"Apply {context_description} to analyze this question.")

        # Add instructions based on parameters
        instructions = []

        if step_by_step:
            instructions.append("Break down your reasoning into clear, logical steps")
            instructions.append("Show your thought process explicitly")

        if consider_alternatives:
            instructions.append("Consider multiple perspectives and alternative viewpoints")
            instructions.append("Acknowledge potential counterarguments or limitations")

        # Depth-specific instructions
        if depth == 'shallow':
            instructions.append("Provide a concise analysis focusing on key points")
        elif depth == 'medium':
            instructions.append("Provide a balanced analysis with supporting details")
        elif depth == 'deep':
            instructions.append("Provide comprehensive analysis with extensive reasoning")
            instructions.append("Explore implications and connections to broader concepts")

        instructions.extend([
            "Support your reasoning with logic and evidence where applicable",
            "Be explicit about assumptions you're making",
            "Conclude with a clear, well-reasoned answer"
        ])

        if instructions:
            prompt_parts.append("Instructions: " + ". ".join(instructions))

        prompt_parts.append(f"Question: {question}")
        prompt_parts.append("Analysis:")

        return "\n\n".join(prompt_parts)

    def _get_reasoning_params(self, kwargs: Dict[str, Any], reasoning_type: str) -> Dict[str, Any]:
        """Get generation parameters for reasoning tasks.

        Args:
            kwargs: User parameters
            reasoning_type: Type of reasoning

        Returns:
            Dict[str, Any]: Generation parameters
        """
        # Base parameters for reasoning
        params = {
            'max_new_tokens': kwargs.get('max_tokens', 768),
            'temperature': 0.6,  # Moderate temperature for thoughtful responses
            'do_sample': True,
            'top_p': 0.9,
            'repetition_penalty': 1.1,
            'pad_token_id': self.tokenizer.eos_token_id if self.tokenizer else None
        }

        # Adjust based on reasoning type
        if reasoning_type in ['logical', 'analytical']:
            params['temperature'] = 0.5  # Lower for more structured reasoning
        elif reasoning_type in ['creative', 'strategic']:
            params['temperature'] = 0.7  # Higher for more creative approaches
        elif reasoning_type == 'critical':
            params['temperature'] = 0.6
            params['top_p'] = 0.85  # Slightly more focused

        # Adjust based on depth
        depth = kwargs.get('depth', 'medium')
        if depth == 'deep':
            params['max_new_tokens'] = min(1024, params['max_new_tokens'] * 1.5)
        elif depth == 'shallow':
            params['max_new_tokens'] = min(384, params['max_new_tokens'] * 0.7)

        # Override with user parameters
        if 'temperature' in kwargs:
            params['temperature'] = max(0.1, min(kwargs['temperature'], 1.0))

        return params

    def _post_process_reasoning(self,
                                reasoning: str,
                                reasoning_type: str,
                                kwargs: Dict[str, Any]) -> str:
        """Post-process reasoning response for clarity and structure.

        Args:
            reasoning: Raw reasoning response
            reasoning_type: Type of reasoning applied
            kwargs: Generation parameters

        Returns:
            str: Post-processed reasoning
        """
        if not reasoning.strip():
            return reasoning

        # Clean up common artifacts
        reasoning = self._clean_reasoning_artifacts(reasoning)

        # Structure the reasoning if step-by-step was requested
        if kwargs.get('step_by_step', True):
            reasoning = self._structure_reasoning_steps(reasoning)

        # Add reasoning type indicator if helpful
        if kwargs.get('show_reasoning_type', False):
            reasoning = f"**{reasoning_type.title()} Reasoning:**\n\n{reasoning}"

        return reasoning.strip()

    def _clean_reasoning_artifacts(self, reasoning: str) -> str:
        """Remove common reasoning artifacts.

        Args:
            reasoning: Raw reasoning text

        Returns:
            str: Cleaned reasoning
        """
        # Remove common prefixes
        prefixes_to_remove = [
            "Here's my analysis:",
            "Let me analyze this:",
            "Analysis:",
            "My reasoning:"
        ]

        for prefix in prefixes_to_remove:
            if reasoning.startswith(prefix):
                reasoning = reasoning[len(prefix):].strip()

        return reasoning

    def _structure_reasoning_steps(self, reasoning: str) -> str:
        """Structure reasoning into clear steps.

        Args:
            reasoning: Reasoning text

        Returns:
            str: Structured reasoning
        """
        lines = reasoning.split('\n')
        structured_lines = []

        step_counter = 1
        in_steps = False

        for line in lines:
            stripped = line.strip()

            # Identify potential reasoning steps
            if any(keyword in stripped.lower() for keyword in [
                'first', 'second', 'third', 'next', 'then', 'finally',
                'step 1', 'step 2', 'initially', 'subsequently'
            ]):
                if not in_steps:
                    in_steps = True
                    if structured_lines and not structured_lines[-1].strip():
                        structured_lines.append("**Reasoning Steps:**")
                    else:
                        structured_lines.append("\n**Reasoning Steps:**")

                if not stripped.startswith(('Step', '1.', '2.', '3.', '-', '*')):
                    line = f"{step_counter}. {stripped}"
                    step_counter += 1

            # Identify conclusions
            elif any(keyword in stripped.lower() for keyword in [
                'therefore', 'in conclusion', 'conclusion', 'final answer',
                'summary', 'to summarize'
            ]):
                if in_steps and structured_lines:
                    structured_lines.append("")  # Add spacing
                structured_lines.append(f"**Conclusion:** {stripped}")
                continue

            structured_lines.append(line)

        return '\n'.join(structured_lines)

    async def analyze_problem(self,
                              problem: str,
                              context: Optional[str] = None,
                              frameworks: Optional[List[str]] = None) -> Dict[str, Any]:
        """Specialized method for problem analysis.

        Args:
            problem: Problem statement to analyze
            context: Additional context information
            frameworks: Analysis frameworks to apply

        Returns:
            Dict containing structured analysis
        """
        analysis_prompt = f"Analyze this problem systematically: {problem}"

        if context:
            analysis_prompt += f"\n\nContext: {context}"

        if frameworks:
            analysis_prompt += f"\n\nApply these frameworks: {', '.join(frameworks)}"

        response = await self.generate(
            analysis_prompt,
            reasoning_type='analytical',
            step_by_step=True,
            consider_alternatives=True,
            depth='deep'
        )

        return {
            'problem': problem,
            'analysis': response,
            'context': context,
            'frameworks_applied': frameworks or [],
            'method': 'systematic_analysis'
        }

    async def compare_options(self,
                              options: List[str],
                              criteria: Optional[List[str]] = None) -> str:
        """Compare multiple options systematically.

        Args:
            options: List of options to compare
            criteria: Evaluation criteria

        Returns:
            str: Comparative analysis
        """
        comparison_prompt = f"Compare and evaluate these options:\n"
        for i, option in enumerate(options, 1):
            comparison_prompt += f"{i}. {option}\n"

        if criteria:
            comparison_prompt += f"\nEvaluation criteria: {', '.join(criteria)}"

        comparison_prompt += "\n\nProvide a systematic comparison with pros and cons for each option."

        return await self.generate(
            comparison_prompt,
            reasoning_type='comparative',
            step_by_step=True,
            consider_alternatives=True
        )

    async def warmup(self) -> bool:
        """Warm up with a reasoning example."""
        if not self.is_loaded():
            return False

        try:
            warmup_question = "What factors should be considered when making an important decision?"
            await self.generate(warmup_question, max_tokens=4000, reasoning_type='analytical')
            logger.info(f"Warmup completed for {self.category}")
            return True
        except Exception as e:
            logger.error(f"Warmup failed for {self.category}: {e}")
            return False

    def get_reasoning_types(self) -> List[str]:
        """Get available reasoning types.

        Returns:
            List[str]: Available reasoning types
        """
        return list(self.reasoning_contexts.keys())

    def get_reasoning_info(self, reasoning_type: str) -> Dict[str, Any]:
        """Get information about a specific reasoning type.

        Args:
            reasoning_type: Type of reasoning

        Returns:
            Dict containing reasoning type info
        """
        return {
            'type': reasoning_type,
            'description': self.reasoning_contexts.get(reasoning_type, 'General reasoning'),
            'recommended_params': {
                'step_by_step': True,
                'consider_alternatives': True,
                'depth': 'medium'
            }
        }