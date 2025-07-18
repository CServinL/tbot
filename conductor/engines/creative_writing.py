import logging
from typing import Dict, Any, Optional, AsyncGenerator, List
from conductor.engines.base_engine import BaseEngine
from conductor.model_loader import ModelLoader

logger = logging.getLogger(__name__)


class CreativeWritingEngine(BaseEngine):
    def __init__(self, config: Dict[str, Any], model_loader: ModelLoader, persona: str = ""):
        super().__init__(config, model_loader, persona)

        # Writing genres and styles
        self.writing_genres: Dict[str, Dict[str, Any]] = {
            'fiction': {
                'description': 'Creative fictional narratives',
                'styles': ['literary', 'genre', 'experimental', 'commercial']
            },
            'poetry': {
                'description': 'Poetic compositions',
                'styles': ['free_verse', 'sonnet', 'haiku', 'limerick', 'narrative']
            },
            'screenplay': {
                'description': 'Scripts for film and television',
                'styles': ['feature', 'short', 'tv_episode', 'commercial']
            },
            'dialogue': {
                'description': 'Character conversations and exchanges',
                'styles': ['dramatic', 'comedic', 'realistic', 'stylized']
            },
            'prose': {
                'description': 'Creative non-fiction and essays',
                'styles': ['personal', 'observational', 'argumentative', 'descriptive']
            },
            'humor': {
                'description': 'Comedic writing and satire',
                'styles': ['satirical', 'observational', 'absurdist', 'parody']
            }
        }

        # Writing techniques and elements
        self.writing_techniques: Dict[str, str] = {
            'show_dont_tell': 'Use vivid scenes and actions instead of exposition',
            'dialogue_driven': 'Advance plot and reveal character through dialogue',
            'stream_of_consciousness': 'Present thoughts and feelings as they occur',
            'unreliable_narrator': 'Use a narrator whose credibility is questionable',
            'multiple_perspectives': 'Tell story from different viewpoints',
            'experimental_structure': 'Use non-traditional narrative structures',
            'rich_imagery': 'Employ vivid sensory descriptions',
            'symbolism': 'Use symbolic elements to convey deeper meaning'
        }

    def _get_default_generation_params(self) -> Dict[str, Any]:
        """Get default generation parameters optimized for creative writing."""
        params = super()._get_default_generation_params()
        # Override defaults for creative writing
        params.update({
            'max_new_tokens': 1024,     # Enough for creative pieces
            'temperature': 0.8,         # Higher temperature for creativity
            'top_p': 0.95,             # Higher top_p for diverse vocabulary
            'repetition_penalty': 1.05  # Light repetition penalty for creative flow
        })
        return params

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate creative writing.

        Args:
            prompt: Writing prompt or request
            **kwargs: Additional parameters
                - genre: Writing genre (fiction, poetry, etc.)
                - style: Writing style within genre
                - length: Target length (short, medium, long)
                - tone: Desired tone (dramatic, humorous, etc.)
                - techniques: Writing techniques to employ
                - characters: Character descriptions if applicable
                - setting: Setting description if applicable

        Returns:
            str: Creative writing piece
        """
        try:
            # Parse creative writing parameters
            genre = kwargs.get('genre', 'fiction')
            style = kwargs.get('style', 'literary')
            length = kwargs.get('length', 'medium')
            tone = kwargs.get('tone', 'engaging')
            techniques = kwargs.get('techniques', [])
            characters = kwargs.get('characters')
            setting = kwargs.get('setting')

            # Build creative writing prompt
            writing_prompt = self._build_creative_prompt(
                prompt, genre, style, length, tone, techniques, characters, setting
            )

            # Get generation parameters optimized for creativity
            gen_params = self._get_creative_params(kwargs, genre, style)

            # Use parent's generate method with the built prompt and parameters
            response = await super().generate(writing_prompt, **gen_params)

            # Post-process the creative writing
            processed_text = self._post_process_creative_writing(
                response, genre, style, kwargs
            )

            logger.debug(f"Generated {genre} writing: {len(processed_text)} chars")
            return processed_text

        except Exception as e:
            logger.error(f"Error generating creative writing: {e}")
            raise

    async def generate_stream(self, prompt: str, **kwargs: Any) -> AsyncGenerator[str, None]:
        """Generate streaming creative writing.

        Args:
            prompt: Writing prompt
            **kwargs: Additional parameters

        Yields:
            str: Writing chunks
        """
        # Parse creative writing parameters
        genre = kwargs.get('genre', 'fiction')
        style = kwargs.get('style', 'literary')
        length = kwargs.get('length', 'medium')
        tone = kwargs.get('tone', 'engaging')
        techniques = kwargs.get('techniques', [])
        characters = kwargs.get('characters')
        setting = kwargs.get('setting')

        # Build creative writing prompt
        writing_prompt = self._build_creative_prompt(
            prompt, genre, style, length, tone, techniques, characters, setting
        )

        # Get generation parameters optimized for creativity
        gen_params = self._get_creative_params(kwargs, genre, style)

        # Since base class doesn't have generate_stream, use regular generate
        try:
            result = await super().generate(writing_prompt, **gen_params)
            processed_result = self._post_process_creative_writing(
                result, genre, style, kwargs
            )
            yield processed_result
        except Exception as e:
            logger.error(f"Error in streaming creative writing: {e}")
            yield f"[Error: {str(e)}]"

    def get_system_prompt(self) -> Optional[str]:
        """Get system prompt for creative writing."""
        if self.persona:
            return self.persona
        
        # Default creative writing system prompt
        return ("You are a creative writing assistant. Help create engaging, well-written content "
                "in various genres and styles. Focus on compelling narratives, vivid descriptions, "
                "authentic characters, and engaging dialogue.")

    def _build_creative_prompt(self,
                               user_prompt: str,
                               genre: str,
                               style: str,
                               length: str,
                               tone: str,
                               techniques: List[str],
                               characters: Optional[str] = None,
                               setting: Optional[str] = None) -> str:
        """Build structured prompt for creative writing.

        Args:
            user_prompt: User's writing request
            genre: Writing genre
            style: Writing style
            length: Target length
            tone: Desired tone
            techniques: Writing techniques to use
            characters: Character descriptions
            setting: Setting description

        Returns:
            str: Structured creative writing prompt
        """
        system_prompt = self.get_system_prompt()

        prompt_parts: List[str] = []

        if system_prompt:
            prompt_parts.append(system_prompt)

        # Add creative writing instructions
        instructions: List[str] = [
            f"Write a {genre} piece in a {style} style",
            f"Target length: {length}",
            f"Tone: {tone}"
        ]

        # Add genre-specific guidance
        if genre in self.writing_genres:
            genre_info = self.writing_genres[genre]
            instructions.append(f"Focus on {genre_info['description'].lower()}")

        # Add technique instructions
        if techniques:
            technique_descriptions: List[str] = []
            for technique in techniques:
                if technique in self.writing_techniques:
                    technique_descriptions.append(self.writing_techniques[technique])

            if technique_descriptions:
                instructions.append("Writing techniques to employ:")
                instructions.extend([f"- {desc}" for desc in technique_descriptions])

        # Add character information
        if characters:
            instructions.append(f"Characters: {characters}")

        # Add setting information
        if setting:
            instructions.append(f"Setting: {setting}")

        # Length-specific instructions
        length_guidance: Dict[str, str] = {
            'short': 'Keep it concise and impactful (200-500 words)',
            'medium': 'Develop the piece with good detail (500-1000 words)',
            'long': 'Create a substantial piece with rich development (1000+ words)'
        }

        if length in length_guidance:
            instructions.append(length_guidance[length])

        # General creative writing guidance
        instructions.extend([
            "Use vivid, engaging language",
            "Create compelling characters and situations",
            "Show emotion and conflict",
            "Maintain consistent voice and style"
        ])

        prompt_parts.append("Instructions: " + "\n".join(instructions))
        prompt_parts.append(f"Writing prompt: {user_prompt}")
        prompt_parts.append("Creative piece:")

        return "\n\n".join(prompt_parts)

    def _get_creative_params(self, kwargs: Dict[str, Any], genre: str, style: str) -> Dict[str, Any]:
        """Get generation parameters optimized for creative writing.

        Args:
            kwargs: User parameters
            genre: Writing genre
            style: Writing style

        Returns:
            Dict[str, Any]: Generation parameters
        """
        # Use base class parameter validation with creative-specific overrides
        creative_kwargs = kwargs.copy()
        
        # Adjust based on genre
        if genre == 'poetry':
            creative_kwargs['temperature'] = creative_kwargs.get('temperature', 0.9)  # Even higher for poetry
            creative_kwargs['repetition_penalty'] = 1.0  # Allow repetition in poetry
        elif genre == 'screenplay':
            creative_kwargs['temperature'] = creative_kwargs.get('temperature', 0.7)  # More structured for scripts
        elif genre == 'dialogue':
            creative_kwargs['temperature'] = creative_kwargs.get('temperature', 0.75)  # Balanced for natural conversation
        elif genre == 'humor':
            creative_kwargs['temperature'] = creative_kwargs.get('temperature', 0.85)  # High for creative humor

        # Adjust based on style
        if style == 'experimental':
            creative_kwargs['temperature'] = creative_kwargs.get('temperature', 0.9)
            creative_kwargs['top_p'] = creative_kwargs.get('top_p', 0.98)
        elif style == 'commercial':
            creative_kwargs['temperature'] = creative_kwargs.get('temperature', 0.7)
        elif style == 'literary':
            creative_kwargs['temperature'] = creative_kwargs.get('temperature', 0.8)
            creative_kwargs['top_p'] = creative_kwargs.get('top_p', 0.9)

        # Adjust based on length
        length = kwargs.get('length', 'medium')
        if length == 'short':
            creative_kwargs['max_new_tokens'] = min(600, creative_kwargs.get('max_new_tokens', 600))
        elif length == 'long':
            creative_kwargs['max_new_tokens'] = min(2048, creative_kwargs.get('max_new_tokens', 2048))

        # Override with user temperature if provided, but clamp for creative writing
        if 'temperature' in kwargs:
            creative_kwargs['temperature'] = max(0.3, min(kwargs['temperature'], 1.5))

        # Use base class validation which handles clamping and safety limits
        return self._validate_generation_params(creative_kwargs)

    def _post_process_creative_writing(self,
                                       text: str,
                                       genre: str,
                                       style: str,
                                       kwargs: Dict[str, Any]) -> str:
        """Post-process creative writing for quality and formatting.

        Args:
            text: Raw creative text
            genre: Writing genre
            style: Writing style
            kwargs: Generation parameters

        Returns:
            str: Post-processed creative writing
        """
        if not text.strip():
            return text

        # Clean up common artifacts
        text = self._clean_creative_artifacts(text)

        # Genre-specific formatting
        if genre == 'poetry':
            text = self._format_poetry(text)
        elif genre == 'screenplay':
            text = self._format_screenplay(text)
        elif genre == 'dialogue':
            text = self._format_dialogue(text)

        # Add title if requested
        if kwargs.get('add_title', False):
            text = self._add_creative_title(text, genre, kwargs.get('title_style', 'simple'))

        return text.strip()

    def _clean_creative_artifacts(self, text: str) -> str:
        """Remove common creative writing artifacts.

        Args:
            text: Raw creative text

        Returns:
            str: Cleaned text
        """
        # Remove common prefixes
        prefixes_to_remove = [
            "Here's a", "Here is a", "This is a", "I'll write",
            "Let me write", "Creative piece:", "Story:"
        ]

        for prefix in prefixes_to_remove:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()
                # Remove common follow-ups
                if text.startswith((':',)):
                    text = text[1:].strip()

        return text

    def _format_poetry(self, poem: str) -> str:
        """Format poetry with proper line breaks and stanza structure.

        Args:
            poem: Raw poem text

        Returns:
            str: Formatted poem
        """
        lines = poem.split('\n')
        formatted_lines: List[str] = []

        for line in lines:
            # Clean up line
            line = line.strip()
            if line:
                formatted_lines.append(line)
            elif formatted_lines and formatted_lines[-1]:  # Empty line for stanza break
                formatted_lines.append('')

        return '\n'.join(formatted_lines)

    def _format_screenplay(self, script: str) -> str:
        """Format screenplay with proper script formatting.

        Args:
            script: Raw script text

        Returns:
            str: Formatted screenplay
        """
        lines = script.split('\n')
        formatted_lines: List[str] = []

        for line in lines:
            stripped = line.strip()

            # Character names (usually all caps)
            if stripped and stripped.isupper() and len(stripped.split()) <= 3:
                formatted_lines.append(f"\n{stripped}")
            # Dialogue
            elif stripped and not stripped.startswith('('):
                formatted_lines.append(f"    {stripped}")
            # Action/stage directions
            elif stripped.startswith('(') and stripped.endswith(')'):
                formatted_lines.append(f"        {stripped}")
            else:
                formatted_lines.append(stripped)

        return '\n'.join(formatted_lines)

    def _format_dialogue(self, dialogue: str) -> str:
        """Format dialogue with proper attribution and punctuation.

        Args:
            dialogue: Raw dialogue text

        Returns:
            str: Formatted dialogue
        """
        # Basic dialogue formatting - in practice you'd want more sophisticated parsing
        lines = dialogue.split('\n')
        formatted_lines: List[str] = []

        for line in lines:
            stripped = line.strip()
            if stripped and '"' in stripped:
                # Ensure proper dialogue formatting
                if not stripped.startswith('"'):
                    # Find the first quote and format accordingly
                    quote_pos = stripped.find('"')
                    if quote_pos > 0:
                        speaker = stripped[:quote_pos].strip()
                        quote = stripped[quote_pos:].strip()
                        formatted_lines.append(f'{speaker} {quote}')
                    else:
                        formatted_lines.append(stripped)
                else:
                    formatted_lines.append(stripped)
            else:
                formatted_lines.append(stripped)

        return '\n'.join(formatted_lines)

    def _add_creative_title(self, text: str, genre: str, title_style: str) -> str:
        """Add a title to the creative piece.

        Args:
            text: Creative text
            genre: Writing genre
            title_style: Title style preference

        Returns:
            str: Text with title
        """
        # Simple title generation based on content
        # In practice, you might use the model to generate a title

        if genre == 'poetry':
            title = "Untitled Poem"
        elif genre == 'fiction':
            title = "A Short Story"
        elif genre == 'screenplay':
            title = "Scene"
        else:
            title = f"Creative {genre.title()}"

        if title_style == 'elegant':
            return f"# {title}\n\n{text}"
        elif title_style == 'simple':
            return f"{title}\n{'=' * len(title)}\n\n{text}"
        else:
            return f"**{title}**\n\n{text}"

    async def generate_character(self,
                                 character_type: str,
                                 traits: Optional[List[str]] = None,
                                 background: Optional[str] = None) -> Dict[str, Any]:
        """Generate a character description.

        Args:
            character_type: Type of character (protagonist, antagonist, etc.)
            traits: Desired character traits
            background: Character background information

        Returns:
            Dict containing character information
        """
        prompt = f"Create a detailed {character_type} character"

        if traits:
            prompt += f" with these traits: {', '.join(traits)}"

        if background:
            prompt += f" with this background: {background}"

        prompt += ". Include physical description, personality, motivations, and backstory."

        character_description = await self.generate(
            prompt,
            genre='prose',
            style='descriptive',
            length='medium'
        )

        return {
            'character_type': character_type,
            'traits': traits or [],
            'background': background,
            'description': character_description,
            'method': 'ai_generated'
        }

    async def generate_plot_outline(self,
                                    premise: str,
                                    genre: str = 'fiction',
                                    length: str = 'short') -> str:
        """Generate a plot outline.

        Args:
            premise: Story premise
            genre: Story genre
            length: Story length

        Returns:
            str: Plot outline
        """
        outline_prompt = f"""Create a plot outline for a {length} {genre} story based on this premise: {premise}

Include:
- Setup/Introduction
- Inciting incident
- Rising action
- Climax
- Resolution

Structure the outline clearly with main plot points."""

        return await self.generate(
            outline_prompt,
            genre='prose',
            style='structured',
            length='medium'
        )

    async def warmup(self) -> bool:
        """Warm up with a creative writing example."""
        if not self.is_loaded():
            return False

        try:
            warmup_prompt = "Write a short scene about a character discovering something unexpected"
            await self.generate(
                warmup_prompt,
                genre='fiction',
                style='literary',
                length='short',
                max_tokens=200
            )
            logger.info(f"Warmup completed for {self.category}")
            return True
        except Exception as e:
            logger.error(f"Warmup failed for {self.category}: {e}")
            return False

    def get_genres(self) -> List[str]:
        """Get available writing genres.

        Returns:
            List[str]: Available genres
        """
        return list(self.writing_genres.keys())

    def get_genre_info(self, genre: str) -> Dict[str, Any]:
        """Get information about a specific genre.

        Args:
            genre: Writing genre

        Returns:
            Dict containing genre info
        """
        if genre in self.writing_genres:
            return self.writing_genres[genre]
        else:
            return {'description': 'Custom genre', 'styles': ['custom']}

    def get_writing_techniques(self) -> List[str]:
        """Get available writing techniques.

        Returns:
            List[str]: Available techniques
        """
        return list(self.writing_techniques.keys())