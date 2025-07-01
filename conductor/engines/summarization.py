import asyncio
import logging
import re
from typing import Dict, Any, Optional, AsyncGenerator, List
from base_llm_engine import BaseLLMEngine
from model_loader import ModelLoader
from utils.persona_loader import PersonaLoader

logger = logging.getLogger(__name__)


class SummarizationEngine(BaseLLMEngine):
    """Engine specialized for text summarization tasks."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_loader = ModelLoader()
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
            'short': {'sentences': 5, 'words': 100, 'description': 'Short summary of key points'},
            'medium': {'sentences': 10, 'words': 200, 'description': 'Balanced summary with details'},
            'long': {'sentences': 15, 'words': 400, 'description': 'Comprehensive summary'},
            'detailed': {'sentences': 20, 'words': 600, 'description': 'Thorough summary with context'}
        }

        # Content types that may require different approaches
        self.content_types = [
            'academic_paper', 'news_article', 'research_report', 'book_chapter',
            'meeting_transcript', 'legal_document', 'technical_manual', 'blog_post',
            'email_thread', 'conversation', 'financial_report', 'policy_document'
        ]

    async def load_model(self) -> bool:
        """Load the summarization model."""
        try:
            logger.info(f"Loading summarization model: {self.technical_model_name}")
            self.model, self.tokenizer = await self.model_loader.load_model(
                self.technical_model_name,
                self.precision
            )

            if self.model is not None and self.tokenizer is not None:
                self.is_model_loaded = True
                self.load_time = asyncio.get_event_loop().time()
                logger.info(f"Successfully loaded summarization model")

                # Perform warmup
                await self.warmup()
                return True
            else:
                logger.error("Failed to load summarization model")
                return False

        except Exception as e:
            logger.error(f"Error loading summarization model: {e}")
            return False

    async def unload_model(self) -> bool:
        """Unload the summarization model."""
        try:
            if self.is_model_loaded:
                success = await self.model_loader.unload_model(self.technical_model_name)
                if success:
                    self.model = None
                    self.tokenizer = None
                    self.is_model_loaded = False
                    logger.info("Summarization model unloaded")
                return success
            return True
        except Exception as e:
            logger.error(f"Error unloading summarization model: {e}")
            return False

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate summary of the provided text.

        Args:
            prompt: Text to summarize (can include instructions)
            **kwargs: Additional parameters
                - text_to_summarize: Explicit text to summarize (if different from prompt)
                - summary_type: Type of summary (extractive, abstractive, etc.)
                - length: Target summary length (brief, short, medium, long, detailed)
                - focus_areas: Specific areas to emphasize
                - content_type: Type of content being summarized
                - preserve_tone: Whether to preserve original tone
                - include_key_quotes: Whether to include important quotes

        Returns:
            str: Generated summary
        """
        if not self.is_model_loaded:
            raise RuntimeError("Summarization model not loaded")

        try:
            # Parse summarization parameters
            text_to_summarize = kwargs.get('text_to_summarize', prompt)
            summary_type = kwargs.get('summary_type', 'abstractive')
            length = kwargs.get('length', 'medium')
            focus_areas = kwargs.get('focus_areas', [])
            content_type = kwargs.get('content_type', 'general')
            preserve_tone = kwargs.get('preserve_tone', False)
            include_quotes = kwargs.get('include_key_quotes', False)

            # Analyze the input text
            text_analysis = self._analyze_text_for_summary(text_to_summarize)

            # Build summarization prompt
            summary_prompt = self._build_summary_prompt(
                text_to_summarize, summary_type, length, focus_areas,
                content_type, preserve_tone, include_quotes, text_analysis
            )

            # Get generation parameters
            gen_params = self._get_summary_params(kwargs, summary_type, length)

            # Tokenize input (handle long texts)
            inputs = self.tokenizer(
                summary_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=3584  # Leave room for summary generation
            )

            # Move to device
            if hasattr(self.model, 'device'):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Generate summary
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

            # Decode and process summary
            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            summary_text = self._extract_response(full_output, summary_prompt)

            # Post-process summary
            processed_summary = self._post_process_summary(
                summary_text, summary_type, length, text_analysis, kwargs
            )

            self.increment_generation_count()

            logger.debug(f"Generated {summary_type} summary ({length}): {len(processed_summary)} chars")
            return processed_summary

        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            raise

    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Generate streaming summary.

        Args:
            prompt: Text to summarize
            **kwargs: Additional parameters

        Yields:
            str: Summary chunks
        """
        if not self.is_model_loaded:
            raise RuntimeError("Summarization model not loaded")

        try:
            from transformers import TextIteratorStreamer
            import torch
            from threading import Thread

            text_to_summarize = kwargs.get('text_to_summarize', prompt)
            summary_type = kwargs.get('summary_type', 'abstractive')
            length = kwargs.get('length', 'medium')

            text_analysis = self._analyze_text_for_summary(text_to_summarize)
            summary_prompt = self._build_summary_prompt(
                text_to_summarize, summary_type, length,
                kwargs.get('focus_areas', []),
                kwargs.get('content_type', 'general'),
                kwargs.get('preserve_tone', False),
                kwargs.get('include_key_quotes', False),
                text_analysis
            )

            gen_params = self._get_summary_params(kwargs, summary_type, length)

            inputs = self.tokenizer(
                summary_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=3584
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
            logger.error(f"Error in streaming summarization: {e}")
            yield f"Error: {str(e)}"

    def get_system_prompt(self) -> Optional[str]:
        """Get system prompt for summarization."""
        return self.persona_loader.get_persona_for_category('summarization')

    def _analyze_text_for_summary(self, text: str) -> Dict[str, Any]:
        """Analyze text to inform summarization approach.

        Args:
            text: Text to analyze

        Returns:
            Dict containing text analysis
        """
        analysis = {
            'word_count': len(text.split()),
            'sentence_count': len(re.findall(r'[.!?]+', text)),
            'paragraph_count': len([p for p in text.split('\n\n') if p.strip()]),
            'complexity': 'medium',
            'has_headings': bool(re.search(r'^[A-Z][^.]*:?\s*$', text, re.MULTILINE)),
            'has_lists': bool(re.search(r'^\s*[-*•]\s+', text, re.MULTILINE)),
            'has_numbers': bool(re.search(r'\d+', text)),
            'estimated_reading_time': 0
        }

        # Estimate reading time (average 200 words per minute)
        analysis['estimated_reading_time'] = analysis['word_count'] / 200

        # Determine complexity
        avg_sentence_length = analysis['word_count'] / max(analysis['sentence_count'], 1)
        if avg_sentence_length > 25 or analysis['word_count'] > 2000:
            analysis['complexity'] = 'high'
        elif avg_sentence_length < 15 and analysis['word_count'] < 500:
            analysis['complexity'] = 'low'

        return analysis

    def _build_summary_prompt(self,
                              text: str,
                              summary_type: str,
                              length: str,
                              focus_areas: List[str],
                              content_type: str,
                              preserve_tone: bool,
                              include_quotes: bool,
                              analysis: Dict[str, Any]) -> str:
        """Build prompt for summarization.

        Args:
            text: Text to summarize
            summary_type: Type of summary
            length: Target length
            focus_areas: Areas to focus on
            content_type: Type of content
            preserve_tone: Whether to preserve tone
            include_quotes: Whether to include quotes
            analysis: Text analysis

        Returns:
            str: Summarization prompt
        """
        system_prompt = self.get_system_prompt()

        prompt_parts = []

        if system_prompt:
            prompt_parts.append(system_prompt)

        # Add summarization instructions
        instructions = []

        # Type-specific instructions
        if summary_type in self.summary_types:
            instructions.append(self.summary_types[summary_type])

        # Length-specific instructions
        if length in self.summary_lengths:
            length_info = self.summary_lengths[length]
            instructions.append(
                f"Target length: {length_info['description']} (approximately {length_info['words']} words)")

        # Content type considerations
        if content_type == 'academic_paper':
            instructions.append("Focus on methodology, findings, and conclusions")
        elif content_type == 'news_article':
            instructions.append("Emphasize who, what, when, where, why")
        elif content_type == 'meeting_transcript':
            instructions.append("Highlight decisions made and action items")
        elif content_type == 'technical_manual':
            instructions.append("Focus on key procedures and important warnings")

        # Focus areas
        if focus_areas:
            instructions.append(f"Pay special attention to: {', '.join(focus_areas)}")

        # Tone and style
        if preserve_tone:
            instructions.append("Preserve the tone and style of the original text")

        if include_quotes:
            instructions.append("Include key quotes that capture important points")

        # Format instructions based on summary type
        if summary_type == 'bullet_points':
            instructions.append("Present the summary as clear bullet points")
        elif summary_type == 'outline':
            instructions.append("Structure as a hierarchical outline with main points and sub-points")
        elif summary_type == 'executive':
            instructions.append("Write for a business audience focusing on key takeaways and implications")

        # General quality instructions
        instructions.extend([
            "Maintain accuracy and avoid adding information not in the original",
            "Ensure the summary is coherent and flows logically",
            "Focus on the most important and relevant information"
        ])

        if instructions:
            prompt_parts.append("Summarization instructions:\n" + "\n".join([f"- {inst}" for inst in instructions]))

        prompt_parts.append(f"Text to summarize:\n{text}")
        prompt_parts.append("Summary:")

        return "\n\n".join(prompt_parts)

    def _get_summary_params(self, kwargs: Dict[str, Any], summary_type: str, length: str) -> Dict[str, Any]:
        """Get generation parameters for summarization.

        Args:
            kwargs: User parameters
            summary_type: Type of summary
            length: Target length

        Returns:
            Dict[str, Any]: Generation parameters
        """
        # Base parameters for summarization
        params = {
            'max_new_tokens': 512,
            'temperature': 0.4,  # Lower temperature for more focused summaries
            'do_sample': True,
            'top_p': 0.85,
            'repetition_penalty': 1.1,
            'pad_token_id': self.tokenizer.eos_token_id if self.tokenizer else None
        }

        # Adjust based on target length
        if length in self.summary_lengths:
            target_words = self.summary_lengths[length]['words']
            # Rough conversion: 1 token ≈ 0.75 words
            params['max_new_tokens'] = min(1024, int(target_words / 0.75 * 1.5))  # Add buffer

        # Adjust based on summary type
        if summary_type == 'extractive':
            params['temperature'] = 0.3  # Lower for extractive summaries
        elif summary_type == 'abstractive':
            params['temperature'] = 0.5  # Higher for creative abstractive summaries
        elif summary_type in ['bullet_points', 'outline']:
            params['temperature'] = 0.4
            params['repetition_penalty'] = 1.05  # Allow some repetition in structured formats

        # Override with user parameters
        if 'temperature' in kwargs:
            params['temperature'] = max(0.1, min(kwargs['temperature'], 1.0))

        if 'max_tokens' in kwargs:
            params['max_new_tokens'] = min(1024, kwargs['max_tokens'])

        return params

    def _post_process_summary(self,
                              summary: str,
                              summary_type: str,
                              length: str,
                              analysis: Dict[str, Any],
                              kwargs: Dict[str, Any]) -> str:
        """Post-process summary for quality and formatting.

        Args:
            summary: Raw summary
            summary_type: Type of summary
            length: Target length
            analysis: Text analysis
            kwargs: Generation parameters

        Returns:
            str: Post-processed summary
        """
        if not summary.strip():
            return summary

        # Clean up common artifacts
        summary = self._clean_summary_artifacts(summary)

        # Apply format-specific processing
        if summary_type == 'bullet_points':
            summary = self._format_bullet_points(summary)
        elif summary_type == 'outline':
            summary = self._format_outline(summary)
        elif summary_type == 'executive':
            summary = self._format_executive_summary(summary)

        # Validate length constraints
        if kwargs.get('enforce_length', True):
            summary = self._enforce_length_constraints(summary, length)

        # Final quality checks
        summary = self._final_quality_check(summary, analysis)

        return summary.strip()

    def _clean_summary_artifacts(self, summary: str) -> str:
        """Remove common summarization artifacts.

        Args:
            summary: Raw summary

        Returns:
            str: Cleaned summary
        """
        # Remove common prefixes
        prefixes_to_remove = [
            "Here's a summary:",
            "Summary:",
            "Here is the summary:",
            "The text can be summarized as:",
            "In summary,"
        ]

        for prefix in prefixes_to_remove:
            if summary.lower().startswith(prefix.lower()):
                summary = summary[len(prefix):].strip()

        # Remove redundant phrases
        redundant_phrases = [
            "The text discusses",
            "The article talks about",
            "This document covers"
        ]

        for phrase in redundant_phrases:
            summary = summary.replace(phrase, "").strip()

        return summary

    def _format_bullet_points(self, summary: str) -> str:
        """Format summary as bullet points.

        Args:
            summary: Summary text

        Returns:
            str: Bullet-formatted summary
        """
        # Split into sentences and convert to bullets
        sentences = re.split(r'[.!?]+', summary)
        bullet_points = []

        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:
                if not sentence.startswith(('•', '-', '*')):
                    bullet_points.append(f"• {sentence}")
                else:
                    bullet_points.append(sentence)

        return '\n'.join(bullet_points)

    def _format_outline(self, summary: str) -> str:
        """Format summary as hierarchical outline.

        Args:
            summary: Summary text

        Returns:
            str: Outline-formatted summary
        """
        # Basic outline formatting - in practice you'd want more sophisticated structure detection
        paragraphs = summary.split('\n\n')
        outline_parts = []

        for i, paragraph in enumerate(paragraphs, 1):
            if paragraph.strip():
                outline_parts.append(f"{i}. {paragraph.strip()}")

        return '\n\n'.join(outline_parts)

    def _format_executive_summary(self, summary: str) -> str:
        """Format as executive summary with clear sections.

        Args:
            summary: Summary text

        Returns:
            str: Executive-formatted summary
        """
        # Add executive summary header if not present
        if not summary.lower().startswith('executive summary'):
            summary = f"**Executive Summary**\n\n{summary}"

        return summary

    def _enforce_length_constraints(self, summary: str, target_length: str) -> str:
        """Enforce length constraints on summary.

        Args:
            summary: Summary text
            target_length: Target length category

        Returns:
            str: Length-constrained summary
        """
        if target_length not in self.summary_lengths:
            return summary

        target_words = self.summary_lengths[target_length]['words']
        current_words = len(summary.split())

        # If too long, truncate intelligently
        if current_words > target_words * 1.2:  # 20% tolerance
            words = summary.split()
            truncated_words = words[:int(target_words * 1.1)]

            # Try to end at sentence boundary
            truncated_text = ' '.join(truncated_words)
            last_sentence_end = max(
                truncated_text.rfind('.'),
                truncated_text.rfind('!'),
                truncated_text.rfind('?')
            )

            if last_sentence_end > len(truncated_text) * 0.8:
                summary = truncated_text[:last_sentence_end + 1]

        return summary

    def _final_quality_check(self, summary: str, analysis: Dict[str, Any]) -> str:
        """Perform final quality checks on summary.

        Args:
            summary: Summary text
            analysis: Original text analysis

        Returns:
            str: Quality-checked summary
        """
        # Ensure summary is substantially shorter than original
        summary_words = len(summary.split())
        original_words = analysis['word_count']

        if summary_words > original_words * 0.7:  # Summary shouldn't be more than 70% of original
            logger.warning("Summary may be too long relative to original text")

        # Ensure summary has content
        if summary_words < 10:
            logger.warning("Summary may be too short")

        return summary

    async def summarize_multiple_texts(self,
                                       texts: List[str],
                                       combine_method: str = 'separate') -> Union[List[str], str]:
        """Summarize multiple texts.

        Args:
            texts: List of texts to summarize
            combine_method: How to handle multiple texts ('separate', 'combined')

        Returns:
            Union[List[str], str]: Individual summaries or combined summary
        """
        if combine_method == 'separate':
            summaries = []
            for text in texts:
                summary = await self.generate(text, summary_type='abstractive', length='short')
                summaries.append(summary)
            return summaries

        elif combine_method == 'combined':
            combined_text = '\n\n---\n\n'.join(texts)
            return await self.generate(
                combined_text,
                summary_type='abstractive',
                length='medium',
                content_type='multiple_documents'
            )

        else:
            raise ValueError(f"Unknown combine_method: {combine_method}")

    async def warmup(self) -> bool:
        """Warm up with a summarization example."""
        if not self.is_loaded():
            return False

        try:
            warmup_text = """
            Artificial intelligence has made significant progress in recent years. 
            Machine learning algorithms can now perform tasks that were once thought 
            to require human intelligence. These advances have applications in many 
            fields including healthcare, finance, and transportation.
            """
            await self.generate(warmup_text, summary_type='abstractive', length='brief', max_tokens=100)
            logger.info(f"Warmup completed for {self.category}")
            return True
        except Exception as e:
            logger.error(f"Warmup failed for {self.category}: {e}")
            return False

    def get_summary_types(self) -> List[str]:
        """Get available summary types.

        Returns:
            List[str]: Available summary types
        """
        return list(self.summary_types.keys())

    def get_summary_lengths(self) -> List[str]:
        """Get available summary lengths.

        Returns:
            List[str]: Available lengths
        """
        return list(self.summary_lengths.keys())

    def get_content_types(self) -> List[str]:
        """Get supported content types.

        Returns:
            List[str]: Supported content types
        """
        return self.content_types.copy()

    def estimate_summary_length(self, text: str, target_length: str) -> Dict[str, int]:
        """Estimate summary length for given text and target.

        Args:
            text: Input text
            target_length: Target length category

        Returns:
            Dict containing length estimates
        """
        analysis = self._analyze_text_for_summary(text)

        if target_length in self.summary_lengths:
            target_info = self.summary_lengths[target_length]
            compression_ratio = target_info['words'] / analysis['word_count']
        else:
            compression_ratio = 0.2  # Default 20% compression

        return {
            'original_words': analysis['word_count'],
            'estimated_summary_words': int(analysis['word_count'] * compression_ratio),
            'compression_ratio': compression_ratio,
            'target_length': target_length
        }