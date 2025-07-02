import logging
from typing import Dict, Any, Optional, List
from conductor.engines.base_engine import BaseEngine
from conductor.model_loader import ModelLoader
from conductor.utils.persona_loader import PersonaLoader

logger = logging.getLogger(__name__)


class SummarizationEngine(BaseEngine):
    """Summarization engine for text condensation and key point extraction."""
    
    def __init__(self, config: Dict[str, Any], model_loader: ModelLoader, persona: str = ""):
        super().__init__(config, model_loader, persona)
        self.persona_loader = PersonaLoader()
        
        # Summary types and styles
        self.summary_types = {
            'extractive': 'Extract key sentences from the original text',
            'abstractive': 'Generate new sentences that capture the essence',
            'bullet_points': 'Present key points as bulleted list',
            'outline': 'Create hierarchical outline of main topics',
            'executive': 'High-level summary for decision makers',
            'technical': 'Focus on technical details and specifications',
            'narrative': 'Tell the story in a condensed narrative form'
        }
        
        # Summary lengths
        self.summary_lengths = {
            'brief': {'sentences': 3, 'words': 50, 'description': 'Very concise overview'},
            'short': {'sentences': 5, 'words': 100, 'description': 'Concise summary'},
            'medium': {'sentences': 8, 'words': 200, 'description': 'Balanced detail'},
            'long': {'sentences': 12, 'words': 400, 'description': 'Comprehensive summary'},
            'detailed': {'sentences': 20, 'words': 800, 'description': 'In-depth analysis'}
        }

    def get_system_prompt(self) -> Optional[str]:
        """Get system prompt for summarization."""
        return self.persona_loader.get_persona_for_category('summarization')

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate summary with specified parameters."""
        # Parse summarization parameters
        text_to_summarize = kwargs.get('text', prompt)
        summary_type = kwargs.get('summary_type', 'abstractive')
        summary_length = kwargs.get('summary_length', 'medium')
        focus_areas = kwargs.get('focus_areas', [])
        
        # Build summarization prompt
        summary_prompt = self._build_summary_prompt(
            text_to_summarize, summary_type, summary_length, focus_areas
        )
        
        # Adjust generation parameters for summaries
        if summary_length in self.summary_lengths:
            max_words = self.summary_lengths[summary_length]['words']
            kwargs['max_tokens'] = int(max_words * 1.3)  # Allow some buffer
        
        # Use parent's generate method
        summary = await super().generate(summary_prompt, **kwargs)
        
        # Post-process summary
        return self._post_process_summary(summary, summary_type, summary_length)

    def _build_summary_prompt(self, 
                             text: str, 
                             summary_type: str, 
                             summary_length: str,
                             focus_areas: List[str]) -> str:
        """Build structured prompt for summarization."""
        system_prompt = self.get_system_prompt()
        
        prompt_parts: List[str] = []
        
        if system_prompt:
            prompt_parts.append(system_prompt)
        
        # Add summarization instructions
        type_instruction = self.summary_types.get(summary_type, 'Create a clear summary')
        length_info = self.summary_lengths.get(summary_length, {})
        
        instructions: List[str] = [
            f"Task: {type_instruction}",
            f"Target length: {length_info.get('description', summary_length)} ({length_info.get('words', 200)} words)",
        ]
        
        if focus_areas:
            instructions.append(f"Focus on: {', '.join(focus_areas)}")
            
        if summary_type == 'bullet_points':
            instructions.append("Format the summary as clear bullet points")
        elif summary_type == 'outline':
            instructions.append("Format as a hierarchical outline with main topics and subtopics")
        elif summary_type == 'executive':
            instructions.append("Focus on key decisions, impacts, and actionable insights")
        
        prompt_parts.append("Instructions: " + ". ".join(instructions))
        prompt_parts.append(f"Text to summarize:\n{text}")
        prompt_parts.append("Summary:")
        
        return "\n\n".join(prompt_parts)

    def _post_process_summary(self, summary: str, summary_type: str, summary_length: str) -> str:
        """Post-process summary for quality and format."""
        if not summary.strip():
            return summary
        
        # Clean up common artifacts
        summary = summary.strip()
        
        # Remove redundant phrases
        redundant_phrases = [
            "Here's a summary:",
            "Summary:",
            "In summary,",
            "To summarize,",
            "The text discusses"
        ]
        
        for phrase in redundant_phrases:
            if summary.startswith(phrase):
                summary = summary[len(phrase):].strip()
        
        # Format specific post-processing
        if summary_type == 'bullet_points' and not summary.startswith(('•', '*', '-')):
            # Convert to bullet points if not already formatted
            sentences = summary.split('. ')
            summary = '\n'.join(f"• {sentence.strip('.')}" for sentence in sentences if sentence.strip())
        
        return summary

    async def summarize_text(self, 
                           text: str, 
                           summary_type: str = 'abstractive',
                           length: str = 'medium',
                           **kwargs: Any) -> str:
        """Convenience method for text summarization."""
        return await self.generate(
            text,
            text=text,
            summary_type=summary_type,
            summary_length=length,
            **kwargs
        )

    async def create_executive_summary(self, text: str, **kwargs: Any) -> str:
        """Create an executive summary for business documents."""
        return await self.summarize_text(
            text,
            summary_type='executive',
            length='medium',
            focus_areas=['key decisions', 'business impact', 'recommendations'],
            **kwargs
        )

    async def create_bullet_summary(self, text: str, **kwargs: Any) -> str:
        """Create a bullet-point summary."""
        return await self.summarize_text(
            text,
            summary_type='bullet_points',
            length='short',
            **kwargs
        )

    def get_available_summary_types(self) -> List[str]:
        """Get list of available summary types."""
        return list(self.summary_types.keys())

    def get_available_lengths(self) -> List[str]:
        """Get list of available summary lengths."""
        return list(self.summary_lengths.keys())

    def get_summary_type_info(self, summary_type: str) -> Dict[str, Any]:
        """Get information about a specific summary type."""
        return {
            'type': summary_type,
            'description': self.summary_types.get(summary_type, 'Standard summary'),
            'recommended_use': self._get_type_recommendations(summary_type)
        }

    def _get_type_recommendations(self, summary_type: str) -> str:
        """Get recommendations for when to use each summary type."""
        recommendations = {
            'extractive': 'Best for preserving original wording and key quotes',
            'abstractive': 'Best for general-purpose summaries and readability',
            'bullet_points': 'Best for quick reference and action items',
            'outline': 'Best for complex documents with multiple topics',
            'executive': 'Best for business documents and decision-making',
            'technical': 'Best for technical documentation and specifications',
            'narrative': 'Best for stories, reports, and chronological content'
        }
        return recommendations.get(summary_type, 'General purpose summarization')
