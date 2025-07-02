import logging
import re
from typing import Dict, Any, Optional, AsyncGenerator, List
from conductor.engines.base_engine import BaseEngine
from conductor.model_loader import ModelLoader

logger = logging.getLogger(__name__)


class QuestionAnsweringEngine(BaseEngine):
    def __init__(self, config: Dict[str, Any], model_loader: ModelLoader, persona: str = ""):
        super().__init__(config, model_loader, persona)

        # Question types and their characteristics
        self.question_types = {
            'factual': {
                'description': 'Questions seeking specific facts or information',
                'examples': ['What is the capital of France?', 'When was the Declaration of Independence signed?'],
                'approach': 'direct_answer'
            },
            'explanatory': {
                'description': 'Questions asking for explanations or how something works',
                'examples': ['How does photosynthesis work?', 'Why do seasons change?'],
                'approach': 'detailed_explanation'
            },
            'analytical': {
                'description': 'Questions requiring analysis and reasoning',
                'examples': ['What are the implications of climate change?',
                             'How might this policy affect the economy?'],
                'approach': 'analytical_reasoning'
            },
            'comparative': {
                'description': 'Questions comparing different things',
                'examples': ['What is the difference between X and Y?', 'How do these approaches compare?'],
                'approach': 'comparative_analysis'
            },
            'opinion': {
                'description': 'Questions seeking perspectives or opinions',
                'examples': ['What do you think about this proposal?', 'What are the pros and cons?'],
                'approach': 'balanced_perspective'
            },
            'procedural': {
                'description': 'Questions about how to do something',
                'examples': ['How do I change a tire?', 'What steps should I follow?'],
                'approach': 'step_by_step'
            },
            'contextual': {
                'description': 'Questions that require understanding context',
                'examples': ['Based on this document, what can we conclude?'],
                'approach': 'context_based'
            }
        }

        # Answer formats
        self.answer_formats = {
            'direct': 'Provide a clear, direct answer',
            'detailed': 'Provide comprehensive explanation with details',
            'structured': 'Organize answer with clear sections',
            'bullet_points': 'Present answer as bullet points',
            'pros_and_cons': 'List advantages and disadvantages',
            'step_by_step': 'Break down into sequential steps'
        }

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Answer questions with appropriate depth and format.

        Args:
            prompt: Question to answer
            **kwargs: Additional parameters
                - context: Additional context for answering
                - question_type: Type of question (factual, explanatory, etc.)
                - answer_format: Desired answer format
                - confidence_level: Show confidence in the answer
                - sources_requested: Whether to mention information sources
                - depth: Level of detail (shallow, medium, deep)

        Returns:
            str: Answer to the question
        """
        if not self.is_model_loaded:
            raise RuntimeError("Question answering model not loaded")

        try:
            # Parse question parameters
            context = kwargs.get('context')
            question_type = kwargs.get('question_type')
            answer_format = kwargs.get('answer_format', 'detailed')
            show_confidence = kwargs.get('confidence_level', False)
            mention_sources = kwargs.get('sources_requested', False)
            depth = kwargs.get('depth', 'medium')

            # Analyze the question
            question_analysis = self._analyze_question(prompt)

            # Use detected type if not specified
            if not question_type:
                question_type = question_analysis.get('detected_type', 'explanatory')

            # Build question answering prompt
            qa_prompt = self._build_qa_prompt(
                prompt, context, question_type, answer_format,
                show_confidence, mention_sources, depth, question_analysis
            )

            # Get generation parameters
            gen_params = self._get_qa_params(kwargs, question_type, answer_format)

            # Tokenize input
            inputs = self.tokenizer(
                qa_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            )

            # Move to device
            if hasattr(self.model, 'device'):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Generate answer
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

            # Decode and process answer
            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer_text = self._extract_response(full_output, qa_prompt)

            # Post-process answer
            processed_answer = self._post_process_answer(
                answer_text, question_type, answer_format, question_analysis, kwargs
            )

            self.increment_generation_count()

            logger.debug(f"Answered {question_type} question: {len(processed_answer)} chars")
            return processed_answer

        except Exception as e:
            logger.error(f"Error answering question: {e}")
            raise

    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Generate streaming question answer.

        Args:
            prompt: Question to answer
            **kwargs: Additional parameters

        Yields:
            str: Answer chunks
        """
        if not self.is_model_loaded:
            raise RuntimeError("Question answering model not loaded")

        try:
            from transformers import TextIteratorStreamer
            import torch
            from threading import Thread

            question_analysis = self._analyze_question(prompt)
            question_type = kwargs.get('question_type') or question_analysis.get('detected_type', 'explanatory')

            qa_prompt = self._build_qa_prompt(
                prompt,
                kwargs.get('context'),
                question_type,
                kwargs.get('answer_format', 'detailed'),
                kwargs.get('confidence_level', False),
                kwargs.get('sources_requested', False),
                kwargs.get('depth', 'medium'),
                question_analysis
            )

            gen_params = self._get_qa_params(kwargs, question_type, kwargs.get('answer_format', 'detailed'))

            inputs = self.tokenizer(
                qa_prompt,
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
            logger.error(f"Error in streaming question answering: {e}")
            yield f"Error: {str(e)}"

    def get_system_prompt(self) -> Optional[str]:
        """Get system prompt for question answering."""
        return self.persona_loader.get_persona_for_category('question_answering')

    def _analyze_question(self, question: str) -> Dict[str, Any]:
        """Analyze question to determine type and approach.

        Args:
            question: Question text to analyze

        Returns:
            Dict containing question analysis
        """
        analysis = {
            'detected_type': 'explanatory',
            'complexity': 'medium',
            'has_context': False,
            'requires_reasoning': False,
            'is_multi_part': False,
            'question_words': []
        }

        question_lower = question.lower()

        # Detect question words
        question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'can', 'do', 'does', 'is', 'are']
        analysis['question_words'] = [word for word in question_words if word in question_lower]

        # Detect question type based on patterns
        if any(word in question_lower for word in ['what is', 'when was', 'where is', 'who is']):
            analysis['detected_type'] = 'factual'
        elif any(word in question_lower for word in ['how does', 'why does', 'explain', 'describe']):
            analysis['detected_type'] = 'explanatory'
        elif any(word in question_lower for word in ['compare', 'difference', 'versus', 'vs']):
            analysis['detected_type'] = 'comparative'
        elif any(word in question_lower for word in ['what do you think', 'opinion', 'view', 'perspective']):
            analysis['detected_type'] = 'opinion'
        elif any(word in question_lower for word in ['how to', 'steps', 'procedure', 'process']):
            analysis['detected_type'] = 'procedural'
        elif any(word in question_lower for word in ['analyze', 'implications', 'impact', 'effect']):
            analysis['detected_type'] = 'analytical'

        # Check for complexity indicators
        complexity_indicators = [
            'implications', 'consequences', 'analyze', 'evaluate', 'assess',
            'complex', 'detailed', 'comprehensive', 'thorough'
        ]
        if any(indicator in question_lower for indicator in complexity_indicators):
            analysis['complexity'] = 'high'
        elif len(question.split()) < 8:
            analysis['complexity'] = 'low'

        # Check for multi-part questions
        if any(connector in question for connector in ['and', 'also', '?']):
            question_count = question.count('?')
            if question_count > 1 or ' and ' in question_lower:
                analysis['is_multi_part'] = True

        # Check if reasoning is required
        reasoning_indicators = ['why', 'how', 'analyze', 'explain', 'implications', 'because']
        if any(indicator in question_lower for indicator in reasoning_indicators):
            analysis['requires_reasoning'] = True

        return analysis

    def _build_qa_prompt(self,
                         question: str,
                         context: Optional[str],
                         question_type: str,
                         answer_format: str,
                         show_confidence: bool,
                         mention_sources: bool,
                         depth: str,
                         analysis: Dict[str, Any]) -> str:
        """Build prompt for question answering.

        Args:
            question: Question to answer
            context: Additional context
            question_type: Type of question
            answer_format: Desired format
            show_confidence: Whether to show confidence
            mention_sources: Whether to mention sources
            depth: Level of detail
            analysis: Question analysis

        Returns:
            str: Question answering prompt
        """
        system_prompt = self.get_system_prompt()

        prompt_parts = []

        if system_prompt:
            prompt_parts.append(system_prompt)

        # Add context if provided
        if context:
            prompt_parts.append(f"Context: {context}")

        # Add question type specific instructions
        instructions = []

        if question_type in self.question_types:
            type_info = self.question_types[question_type]
            approach = type_info['approach']

            if approach == 'direct_answer':
                instructions.append("Provide a clear, direct answer to the question")
            elif approach == 'detailed_explanation':
                instructions.append("Provide a thorough explanation with relevant details")
            elif approach == 'analytical_reasoning':
                instructions.append("Analyze the question and provide reasoned insights")
            elif approach == 'comparative_analysis':
                instructions.append("Compare and contrast the relevant elements")
            elif approach == 'balanced_perspective':
                instructions.append("Present multiple perspectives and viewpoints")
            elif approach == 'step_by_step':
                instructions.append("Break down the answer into clear steps")
            elif approach == 'context_based':
                instructions.append("Base your answer on the provided context")

        # Add format-specific instructions
        if answer_format in self.answer_formats:
            instructions.append(self.answer_formats[answer_format])

        # Add depth instructions
        if depth == 'shallow':
            instructions.append("Provide a concise answer focusing on key points")
        elif depth == 'medium':
            instructions.append("Provide a balanced answer with appropriate detail")
        elif depth == 'deep':
            instructions.append("Provide comprehensive coverage with extensive detail")

        # Handle multi-part questions
        if analysis['is_multi_part']:
            instructions.append("Address each part of the question systematically")

        # Confidence and sources
        if show_confidence:
            instructions.append("Indicate your confidence level in the answer")

        if mention_sources:
            instructions.append("Mention the types of sources your knowledge comes from")

        # General quality instructions
        instructions.extend([
            "Ensure accuracy and relevance in your response",
            "Be clear and well-organized",
            "Acknowledge limitations or uncertainties when appropriate"
        ])

        if instructions:
            prompt_parts.append("Instructions: " + "\n".join([f"- {inst}" for inst in instructions]))

        prompt_parts.append(f"Question: {question}")
        prompt_parts.append("Answer:")

        return "\n\n".join(prompt_parts)

    def _get_qa_params(self, kwargs: Dict[str, Any], question_type: str, answer_format: str) -> Dict[str, Any]:
        """Get generation parameters for question answering.

        Args:
            kwargs: User parameters
            question_type: Type of question
            answer_format: Answer format

        Returns:
            Dict[str, Any]: Generation parameters
        """
        # Base parameters for question answering
        params = {
            'max_new_tokens': kwargs.get('max_tokens', 512),
            'temperature': 0.6,  # Balanced for informative answers
            'do_sample': True,
            'top_p': 0.9,
            'repetition_penalty': 1.1,
            'pad_token_id': self.tokenizer.eos_token_id if self.tokenizer else None
        }

        # Adjust based on question type
        if question_type == 'factual':
            params['temperature'] = 0.4  # Lower for factual accuracy
        elif question_type == 'explanatory':
            params['temperature'] = 0.6
            params['max_new_tokens'] = min(768, params['max_new_tokens'] * 1.5)
        elif question_type == 'analytical':
            params['temperature'] = 0.7  # Higher for analytical thinking
            params['max_new_tokens'] = min(1024, params['max_new_tokens'] * 2)
        elif question_type == 'opinion':
            params['temperature'] = 0.7
        elif question_type == 'procedural':
            params['temperature'] = 0.5  # More structured for procedures

        # Adjust based on answer format
        if answer_format in ['bullet_points', 'structured', 'step_by_step']:
            params['temperature'] = 0.5  # More structured output
        elif answer_format == 'detailed':
            params['max_new_tokens'] = min(1024, params['max_new_tokens'] * 1.5)

        # Depth adjustments
        depth = kwargs.get('depth', 'medium')
        if depth == 'deep':
            params['max_new_tokens'] = min(1024, params['max_new_tokens'] * 2)
        elif depth == 'shallow':
            params['max_new_tokens'] = min(256, params['max_new_tokens'] * 0.6)

        # Override with user parameters
        if 'temperature' in kwargs:
            params['temperature'] = max(0.1, min(kwargs['temperature'], 1.0))

        return params

    def _post_process_answer(self,
                             answer: str,
                             question_type: str,
                             answer_format: str,
                             analysis: Dict[str, Any],
                             kwargs: Dict[str, Any]) -> str:
        """Post-process answer for quality and formatting.

        Args:
            answer: Raw answer
            question_type: Type of question
            answer_format: Answer format
            analysis: Question analysis
            kwargs: Generation parameters

        Returns:
            str: Post-processed answer
        """
        if not answer.strip():
            return answer

        # Clean up common artifacts
        answer = self._clean_qa_artifacts(answer)

        # Apply format-specific processing
        if answer_format == 'bullet_points':
            answer = self._format_as_bullets(answer)
        elif answer_format == 'structured':
            answer = self._format_as_structured(answer)
        elif answer_format == 'step_by_step':
            answer = self._format_as_steps(answer)
        elif answer_format == 'pros_and_cons':
            answer = self._format_as_pros_cons(answer)

        # Add confidence indicators if requested
        if kwargs.get('confidence_level', False):
            answer = self._add_confidence_indicators(answer, analysis)

        return answer.strip()

    def _clean_qa_artifacts(self, answer: str) -> str:
        """Remove common Q&A artifacts.

        Args:
            answer: Raw answer

        Returns:
            str: Cleaned answer
        """
        # Remove common prefixes
        prefixes_to_remove = [
            "The answer is:",
            "Answer:",
            "To answer your question:",
            "Based on the question:"
        ]

        for prefix in prefixes_to_remove:
            if answer.startswith(prefix):
                answer = answer[len(prefix):].strip()

        return answer

    def _format_as_bullets(self, answer: str) -> str:
        """Format answer as bullet points.

        Args:
            answer: Answer text

        Returns:
            str: Bullet-formatted answer
        """
        # Split into logical points and format as bullets
        sentences = re.split(r'[.!?]+', answer)
        bullets = []

        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 15:
                if not sentence.startswith(('•', '-', '*')):
                    bullets.append(f"• {sentence}")
                else:
                    bullets.append(sentence)

        return '\n'.join(bullets)

    def _format_as_structured(self, answer: str) -> str:
        """Format answer with clear structure.

        Args:
            answer: Answer text

        Returns:
            str: Structured answer
        """
        # Add basic structure headers
        paragraphs = answer.split('\n\n')
        structured_parts = []

        for i, paragraph in enumerate(paragraphs):
            if paragraph.strip():
                if i == 0:
                    structured_parts.append(f"**Overview:**\n{paragraph.strip()}")
                else:
                    structured_parts.append(f"**Point {i}:**\n{paragraph.strip()}")

        return '\n\n'.join(structured_parts)

    def _format_as_steps(self, answer: str) -> str:
        """Format answer as numbered steps.

        Args:
            answer: Answer text

        Returns:
            str: Step-formatted answer
        """
        # Convert to numbered steps
        sentences = answer.split('. ')
        steps = []

        for i, sentence in enumerate(sentences, 1):
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:
                if not sentence.startswith(('Step', f'{i}.')):
                    steps.append(f"{i}. {sentence}")
                else:
                    steps.append(sentence)

        return '\n'.join(steps)

    def _format_as_pros_cons(self, answer: str) -> str:
        """Format answer as pros and cons.

        Args:
            answer: Answer text

        Returns:
            str: Pros/cons formatted answer
        """
        # Basic pros/cons formatting
        # In practice, you'd want more sophisticated detection
        if 'advantage' in answer.lower() or 'benefit' in answer.lower():
            answer = answer.replace('Advantages:', '\n**Pros:**\n')
            answer = answer.replace('Disadvantages:', '\n**Cons:**\n')
            answer = answer.replace('Benefits:', '\n**Pros:**\n')
            answer = answer.replace('Drawbacks:', '\n**Cons:**\n')

        return answer

    def _add_confidence_indicators(self, answer: str, analysis: Dict[str, Any]) -> str:
        """Add confidence indicators to answer.

        Args:
            answer: Answer text
            analysis: Question analysis

        Returns:
            str: Answer with confidence indicators
        """
        confidence_level = "medium"

        # Determine confidence based on question type
        if analysis['detected_type'] == 'factual':
            confidence_level = "high"
        elif analysis['detected_type'] in ['opinion', 'analytical']:
            confidence_level = "medium"
        elif analysis['complexity'] == 'high':
            confidence_level = "low"

        confidence_note = f"\n\n*Confidence level: {confidence_level}*"
        return answer + confidence_note

    async def answer_batch_questions(self, questions: List[str], **kwargs) -> List[str]:
        """Answer multiple questions in batch.

        Args:
            questions: List of questions to answer
            **kwargs: Parameters applied to all questions

        Returns:
            List[str]: Answers to each question
        """
        answers = []
        for question in questions:
            answer = await self.generate(question, **kwargs)
            answers.append(answer)
        return answers

    async def warmup(self) -> bool:
        """Warm up with a question answering example."""
        if not self.is_loaded():
            return False

        try:
            warmup_question = "What is artificial intelligence?"
            await self.generate(warmup_question, question_type='explanatory', max_tokens=150)
            logger.info(f"Warmup completed for {self.category}")
            return True
        except Exception as e:
            logger.error(f"Warmup failed for {self.category}: {e}")
            return False

    def get_question_types(self) -> List[str]:
        """Get available question types.

        Returns:
            List[str]: Available question types
        """
        return list(self.question_types.keys())

    def get_answer_formats(self) -> List[str]:
        """Get available answer formats.

        Returns:
            List[str]: Available answer formats
        """
        return list(self.answer_formats.keys())

    def get_question_type_info(self, question_type: str) -> Dict[str, Any]:
        """Get information about a specific question type.

        Args:
            question_type: Type of question

        Returns:
            Dict containing question type info
        """
        if question_type in self.question_types:
            return self.question_types[question_type]
        else:
            return {
                'description': 'Custom question type',
                'examples': [],
                'approach': 'general'
            }