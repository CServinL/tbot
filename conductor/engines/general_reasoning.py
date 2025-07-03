import logging
from typing import Dict, Any, Optional, List
from conductor.engines.base_engine import BaseEngine
from conductor.model_loader import ModelLoader

logger = logging.getLogger(__name__)


class GeneralReasoningEngine(BaseEngine):
    """General reasoning engine for analytical and logical problem solving."""
    
    def __init__(self, config: Dict[str, Any], model_loader: ModelLoader, persona: str = ""):
        super().__init__(config, model_loader, persona)
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

    def get_system_prompt(self) -> Optional[str]:
        """Get system prompt for general reasoning."""
        if self.persona:
            return self.persona
        return "You are an analytical AI assistant specialized in logical reasoning and problem solving. Break down complex problems systematically, provide clear reasoning steps, and offer well-structured solutions."

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate reasoning-based response with enhanced prompt building."""
        # Parse reasoning parameters
        reasoning_type = kwargs.get('reasoning_type', 'analytical')
        step_by_step = kwargs.get('step_by_step', True)
        consider_alternatives = kwargs.get('consider_alternatives', True)
        depth = kwargs.get('depth', 'medium')

        # Build reasoning prompt
        reasoning_prompt = self._build_reasoning_prompt(
            prompt, reasoning_type, step_by_step, consider_alternatives, depth
        )

        # Use parent's generate method with the enhanced prompt
        return await super().generate(reasoning_prompt, **kwargs)

    def _build_reasoning_prompt(self,
                                question: str,
                                reasoning_type: str,
                                step_by_step: bool,
                                consider_alternatives: bool,
                                depth: str) -> str:
        """Build structured prompt for reasoning tasks."""
        system_prompt = self.get_system_prompt()

        prompt_parts: List[str] = []

        if system_prompt:
            prompt_parts.append(system_prompt)

        # Add reasoning context
        if reasoning_type in self.reasoning_contexts:
            context_description = self.reasoning_contexts[reasoning_type]
            prompt_parts.append(f"Apply {context_description} to analyze this question.")

        # Add instructions based on parameters
        instructions: List[str] = []

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

    def get_reasoning_types(self) -> List[str]:
        """Get available reasoning types."""
        return list(self.reasoning_contexts.keys())

    def get_reasoning_info(self, reasoning_type: str) -> Dict[str, Any]:
        """Get information about a specific reasoning type."""
        return {
            'type': reasoning_type,
            'description': self.reasoning_contexts.get(reasoning_type, 'General reasoning'),
            'recommended_params': {
                'step_by_step': True,
                'consider_alternatives': True,
                'depth': 'medium'
            }
        }

    async def analyze_problem(self, problem: str, context: Optional[str] = None) -> str:
        """Specialized method for problem analysis."""
        analysis_prompt = f"Analyze this problem systematically: {problem}"
        if context:
            analysis_prompt += f"\n\nContext: {context}"

        return await self.generate(
            analysis_prompt,
            reasoning_type='analytical',
            step_by_step=True,
            consider_alternatives=True,
            depth='deep'
        )

    async def compare_options(self, options: List[str], criteria: Optional[List[str]] = None) -> str:
        """Compare multiple options systematically."""
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