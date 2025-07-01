import os
import logging
from typing import Dict, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class PersonaLoader:
    """Loads and manages different personas/system prompts for various tasks."""

    def __init__(self, personas_dir: str = "conductor/personas"):
        self.personas_dir = Path(personas_dir)
        self.cached_personas: Dict[str, str] = {}
        self.default_conversational_persona = ""

        # Task-specific prompt templates
        self.task_prompts = {
            'code_generation': {
                'system_prompt': "You are an expert programmer. Generate clean, efficient, and well-commented code. Follow best practices and explain your approach when helpful.",
                'temperature': 0.3,
                'max_tokens': 1024
            },
            'code_completion': {
                'system_prompt': None,  # No system prompt for completion
                'temperature': 0.2,
                'max_tokens': 128
            },
            'mathematical_reasoning': {
                'system_prompt': "You are a mathematics expert. Solve problems step-by-step, showing your work clearly. Use proper mathematical notation and explain your reasoning.",
                'temperature': 0.4,
                'max_tokens': 512
            },
            'creative_writing': {
                'system_prompt': "You are a creative writer. Write engaging, imaginative content with rich descriptions and compelling narratives. Adapt your style to the requested genre or format.",
                'temperature': 0.8,
                'max_tokens': 1024
            },
            'translation': {
                'system_prompt': "You are a professional translator. Provide accurate, natural-sounding translations that preserve meaning, tone, and cultural context.",
                'temperature': 0.3,
                'max_tokens': 512
            },
            'summarization': {
                'system_prompt': "You are an expert at summarization. Create concise, accurate summaries that capture the key points and main ideas of the source material.",
                'temperature': 0.4,
                'max_tokens': 256
            },
            'scientific_research': {
                'system_prompt': "You are a research scientist. Provide accurate, evidence-based information. Cite sources when possible and clearly distinguish between established facts and current research or hypotheses.",
                'temperature': 0.5,
                'max_tokens': 1024
            },
            'legal_analysis': {
                'system_prompt': "You are a legal analyst. Provide thorough analysis based on legal principles and precedents. Always note that this is for informational purposes and recommend consulting qualified legal professionals for specific legal advice.",
                'temperature': 0.4,
                'max_tokens': 1024
            },
            'code_review': {
                'system_prompt': "You are a senior code reviewer. Analyze code for correctness, efficiency, security, and maintainability. Provide constructive feedback and suggest improvements.",
                'temperature': 0.4,
                'max_tokens': 512
            },
            'long_context': {
                'system_prompt': "You are analyzing a long document or conversation. Maintain awareness of the full context while providing relevant responses. Organize your thoughts clearly when dealing with complex, multi-part information.",
                'temperature': 0.6,
                'max_tokens': 1024
            }
        }

    def set_conversational_persona(self, persona: str):
        """Set the default conversational persona.

        Args:
            persona: Conversational persona text from settings
        """
        self.default_conversational_persona = persona
        logger.info("Set conversational persona")

    def load_conversational_persona(self) -> str:
        """Load the conversational persona.

        Returns:
            str: Conversational persona text
        """
        return self.default_conversational_persona

    def get_persona_for_category(self, category: str) -> Optional[str]:
        """Get appropriate persona/system prompt for a category.

        Args:
            category: Category name

        Returns:
            Optional[str]: System prompt or None
        """
        # Categories that use conversational persona
        conversational_categories = {
            'conversational_chat',
            'general_reasoning',
            'question_answering',
            'instruction_following'
        }

        if category in conversational_categories:
            return self.default_conversational_persona

        # Task-specific prompts
        task_config = self.task_prompts.get(category)
        if task_config:
            return task_config.get('system_prompt')

        return None

    def get_generation_params_for_category(self, category: str) -> Dict[str, Any]:
        """Get generation parameters for a category.

        Args:
            category: Category name

        Returns:
            Dict[str, Any]: Generation parameters
        """
        # Default parameters
        default_params = {
            'temperature': 0.7,
            'max_new_tokens': 512,
            'do_sample': True,
            'top_p': 0.9,
            'repetition_penalty': 1.1
        }

        # Task-specific overrides
        task_config = self.task_prompts.get(category, {})

        # Apply task-specific parameters
        if 'temperature' in task_config:
            default_params['temperature'] = task_config['temperature']
        if 'max_tokens' in task_config:
            default_params['max_new_tokens'] = task_config['max_tokens']

        # Category-specific adjustments
        if category == 'code_completion':
            default_params.update({
                'do_sample': False,  # Deterministic for code
                'top_p': 1.0,
                'repetition_penalty': 1.0
            })
        elif category == 'creative_writing':
            default_params.update({
                'top_p': 0.95,
                'repetition_penalty': 1.05
            })
        elif category in ['mathematical_reasoning', 'scientific_research', 'legal_analysis']:
            default_params.update({
                'top_p': 0.85,
                'repetition_penalty': 1.1
            })

        return default_params

    def create_full_prompt(self, category: str, user_prompt: str) -> str:
        """Create full prompt with system prompt for a category.

        Args:
            category: Category name
            user_prompt: User input prompt

        Returns:
            str: Full prompt including system prompt if applicable
        """
        system_prompt = self.get_persona_for_category(category)

        if system_prompt:
            # Format with system prompt
            if category == 'conversational_chat':
                return f"{system_prompt}\n\nUser: {user_prompt}\nAssistant:"
            else:
                return f"System: {system_prompt}\n\nUser: {user_prompt}\nAssistant:"
        else:
            # No system prompt needed
            return user_prompt

    def load_custom_persona(self, persona_name: str) -> Optional[str]:
        """Load a custom persona from file.

        Args:
            persona_name: Name of persona file (without extension)

        Returns:
            Optional[str]: Persona content or None if not found
        """
        if persona_name in self.cached_personas:
            return self.cached_personas[persona_name]

        persona_file = self.personas_dir / f"{persona_name}.txt"

        if not persona_file.exists():
            logger.warning(f"Persona file not found: {persona_file}")
            return None

        try:
            with open(persona_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()

            self.cached_personas[persona_name] = content
            logger.info(f"Loaded custom persona: {persona_name}")
            return content

        except Exception as e:
            logger.error(f"Error loading persona {persona_name}: {e}")
            return None

    def save_custom_persona(self, persona_name: str, content: str) -> bool:
        """Save a custom persona to file.

        Args:
            persona_name: Name of persona
            content: Persona content

        Returns:
            bool: True if saved successfully
        """
        try:
            # Create personas directory if it doesn't exist
            self.personas_dir.mkdir(parents=True, exist_ok=True)

            persona_file = self.personas_dir / f"{persona_name}.txt"

            with open(persona_file, 'w', encoding='utf-8') as f:
                f.write(content)

            # Update cache
            self.cached_personas[persona_name] = content

            logger.info(f"Saved custom persona: {persona_name}")
            return True

        except Exception as e:
            logger.error(f"Error saving persona {persona_name}: {e}")
            return False

    def list_available_personas(self) -> Dict[str, Any]:
        """List all available personas.

        Returns:
            Dict containing persona information
        """
        personas = {
            'built_in': list(self.task_prompts.keys()),
            'custom': [],
            'conversational': bool(self.default_conversational_persona)
        }

        # Check for custom persona files
        if self.personas_dir.exists():
            try:
                for persona_file in self.personas_dir.glob("*.txt"):
                    personas['custom'].append(persona_file.stem)
            except Exception as e:
                logger.error(f"Error listing custom personas: {e}")

        return personas

    def get_persona_info(self, category: str) -> Dict[str, Any]:
        """Get detailed information about a persona.

        Args:
            category: Category name

        Returns:
            Dict containing persona info
        """
        system_prompt = self.get_persona_for_category(category)
        generation_params = self.get_generation_params_for_category(category)

        return {
            'category': category,
            'has_system_prompt': system_prompt is not None,
            'system_prompt_length': len(system_prompt) if system_prompt else 0,
            'generation_params': generation_params,
            'is_conversational': category in ['conversational_chat', 'general_reasoning', 'question_answering',
                                              'instruction_following']
        }

    def validate_persona(self, persona_content: str) -> Dict[str, Any]:
        """Validate persona content.

        Args:
            persona_content: Persona text to validate

        Returns:
            Dict containing validation results
        """
        issues = []
        warnings = []

        # Check length
        if len(persona_content) > 2000:
            warnings.append("Persona is quite long, may affect context window")

        if len(persona_content) < 10:
            issues.append("Persona is too short to be effective")

        # Check for common issues
        if not persona_content.strip():
            issues.append("Persona is empty")

        if persona_content.count('\n') > 20:
            warnings.append("Persona has many line breaks, consider condensing")

        # Check for potentially problematic content
        problematic_terms = ['jailbreak', 'ignore instructions', 'forget previous', 'system override']
        for term in problematic_terms:
            if term.lower() in persona_content.lower():
                issues.append(f"Potentially problematic content detected: '{term}'")

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'length': len(persona_content),
            'word_count': len(persona_content.split())
        }

    def get_effective_prompt_length(self, category: str, user_prompt: str) -> int:
        """Get total prompt length including system prompt.

        Args:
            category: Category name
            user_prompt: User input

        Returns:
            int: Total prompt length in characters
        """
        full_prompt = self.create_full_prompt(category, user_prompt)
        return len(full_prompt)

    def clear_cache(self):
        """Clear the persona cache."""
        self.cached_personas.clear()
        logger.info("Cleared persona cache")

    def reload_personas(self):
        """Reload all personas from files."""
        self.clear_cache()
        logger.info("Reloaded personas")

    def get_persona_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded personas.

        Returns:
            Dict containing persona statistics
        """
        return {
            'cached_personas': len(self.cached_personas),
            'built_in_personas': len(self.task_prompts),
            'has_conversational_persona': bool(self.default_conversational_persona),
            'conversational_persona_length': len(self.default_conversational_persona),
            'personas_dir_exists': self.personas_dir.exists()
        }