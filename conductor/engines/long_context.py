import logging
import re
from typing import Dict, Any, Optional, AsyncGenerator, List, Tuple
from conductor.engines.base_engine import BaseEngine
from conductor.model_loader import ModelLoader

logger = logging.getLogger(__name__)


class LongContextEngine(BaseEngine):
    def __init__(self, config: Dict[str, Any], model_loader: ModelLoader, persona: str = ""):
        super().__init__(config, model_loader, persona)

        # Context processing strategies
        self.context_strategies: Dict[str, str] = {
            'full_context': 'Process entire context as one unit',
            'sliding_window': 'Use sliding window approach for very long texts',
            'hierarchical': 'Break into sections and process hierarchically',
            'summarize_and_refine': 'Summarize sections then refine with details',
            'key_extraction': 'Extract key information and process selectively',
            'chunk_and_merge': 'Process in chunks and merge results'
        }

        # Document types that benefit from long context
        self.document_types: Dict[str, str] = {
            'research_paper': 'Academic papers and research documents',
            'legal_document': 'Contracts, legal briefs, and regulatory texts',
            'technical_manual': 'Technical documentation and manuals',
            'book_chapter': 'Book chapters and long-form content',
            'meeting_transcript': 'Long meeting recordings and transcripts',
            'email_thread': 'Extended email conversations',
            'code_repository': 'Large codebases and documentation',
            'financial_report': 'Annual reports and financial documents',
            'policy_document': 'Government policies and procedures',
            'conversation_log': 'Extended conversation histories'
        }

        # Task types for long context
        self.long_context_tasks: Dict[str, str] = {
            'comprehensive_summary': 'Create comprehensive summary of entire document',
            'key_insights': 'Extract key insights and main themes',
            'question_answering': 'Answer questions based on full context',
            'cross_reference': 'Find connections and cross-references',
            'timeline_extraction': 'Extract chronological information',
            'entity_analysis': 'Analyze entities and relationships',
            'sentiment_analysis': 'Analyze sentiment across the document',
            'topic_modeling': 'Identify main topics and themes',
            'contradiction_detection': 'Find contradictions or inconsistencies',
            'action_items': 'Extract action items and decisions'
        }

        # Memory management for long contexts
        self.context_limits: Dict[str, int] = {
            'max_input_tokens': 32768,  # Adjust based on model capabilities
            'chunk_size': 4096,
            'overlap_size': 512,
            'summary_compression_ratio': 3  # Changed from 0.3 to avoid float/int issues
        }

    def _get_default_generation_params(self) -> Dict[str, Any]:
        """Get default generation parameters optimized for long context processing."""
        params = super()._get_default_generation_params()
        # Override defaults for long context processing
        params.update({
            'max_new_tokens': 2048,     # Longer outputs for comprehensive analysis
            'temperature': 0.5,         # Balanced temperature for long context
            'top_p': 0.9,              # Diverse vocabulary for detailed analysis
            'repetition_penalty': 1.2   # Higher penalty to avoid repetition in long outputs
        })
        return params

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Process long context and generate response.

        Args:
            prompt: Long context or question about long context
            **kwargs: Additional parameters
                - context_text: Explicit long context text
                - strategy: Context processing strategy
                - document_type: Type of document being processed
                - task_type: Type of task to perform
                - preserve_structure: Whether to preserve document structure
                - focus_query: Specific query to focus on
                - chunk_overlap: Overlap between chunks for processing

        Returns:
            str: Response based on long context analysis
        """
        if not self.is_loaded():
            raise RuntimeError("Long context model not loaded")

        try:
            # Parse long context parameters
            context_text = kwargs.get('context_text', prompt)
            strategy = kwargs.get('strategy', 'full_context')
            task_type = kwargs.get('task_type', 'comprehensive_summary')
            focus_query = kwargs.get('focus_query')
            chunk_overlap = kwargs.get('chunk_overlap', self.context_limits['overlap_size'])

            # Analyze context length and complexity
            context_analysis = self._analyze_long_context(context_text)

            # Determine optimal processing strategy
            if not strategy or strategy == 'auto':
                strategy = self._determine_optimal_strategy(context_analysis, task_type)

            # Process long context based on strategy
            if context_analysis['requires_chunking']:
                response = await self._process_chunked_context(
                    context_text, strategy, task_type, focus_query, chunk_overlap, kwargs
                )
            else:
                response = await self._process_full_context(
                    context_text, task_type, focus_query, kwargs
                )

            # Post-process response
            processed_response = self._post_process_long_context_response(
                response, strategy, task_type, context_analysis, kwargs
            )

            self.increment_generation_count()

            logger.debug(f"Processed long context with {strategy} strategy: {len(processed_response)} chars")
            return processed_response

        except Exception as e:
            logger.error(f"Error processing long context: {e}")
            raise

    async def generate_stream(self, prompt: str, **kwargs: Any) -> AsyncGenerator[str, None]:
        """Generate streaming response for long context.

        Args:
            prompt: Long context or question
            **kwargs: Additional parameters

        Yields:
            str: Response chunks
        """
        if not self.is_loaded():
            raise RuntimeError("Long context model not loaded")

        try:
            # For long context, we'll process in segments and stream results
            context_text = kwargs.get('context_text', prompt)
            context_analysis = self._analyze_long_context(context_text)

            if context_analysis['requires_chunking']:
                # Stream chunked processing results
                async for chunk in self._stream_chunked_processing(context_text, kwargs):
                    yield chunk
            else:
                # Stream full context processing
                async for chunk in self._stream_full_context(context_text, kwargs):
                    yield chunk

            self.increment_generation_count()

        except Exception as e:
            logger.error(f"Error in streaming long context: {e}")
            yield f"Error: {str(e)}"

    def get_system_prompt(self) -> Optional[str]:
        """Get system prompt for long context processing."""
        if self.persona:
            return self.persona
        
        # Default system prompt for long context processing
        return """You are an expert at analyzing and processing long documents. You excel at:
- Extracting key information from extensive texts
- Summarizing complex documents while preserving important details
- Finding connections and relationships across large amounts of text
- Breaking down complex information into clear, actionable insights
- Maintaining context and coherence across multiple sections
- Identifying important patterns, themes, and contradictions

Always provide thorough, well-structured responses that demonstrate deep understanding of the full context."""

    def _analyze_long_context(self, text: str) -> Dict[str, Any]:
        """Analyze long context to determine processing approach.

        Args:
            text: Long context text

        Returns:
            Dict containing context analysis
        """
        analysis: Dict[str, Any] = {
            'char_count': len(text),
            'word_count': len(text.split()),
            'line_count': len(text.split('\n')),
            'paragraph_count': len([p for p in text.split('\n\n') if p.strip()]),
            'estimated_tokens': len(text) // 4,  # Rough estimation
            'requires_chunking': False,
            'complexity': 'medium',
            'has_structure': False,
            'has_sections': False,
            'has_metadata': False
        }

        # Determine if chunking is required
        if analysis['estimated_tokens'] > self.context_limits['max_input_tokens']:
            analysis['requires_chunking'] = True

        # Analyze structure
        if any(marker in text for marker in ['# ', '## ', '### ', '#### ']):
            analysis['has_structure'] = True
            analysis['has_sections'] = True
        elif text.count('\n\n') > 5:
            analysis['has_sections'] = True

        # Check for metadata patterns
        if any(pattern in text.lower() for pattern in ['title:', 'author:', 'date:', 'abstract:']):
            analysis['has_metadata'] = True

        # Estimate complexity
        if analysis['word_count'] > 10000 or analysis['paragraph_count'] > 50:
            analysis['complexity'] = 'high'
        elif analysis['word_count'] < 1000 or analysis['paragraph_count'] < 5:
            analysis['complexity'] = 'low'

        return analysis

    def _determine_optimal_strategy(self, analysis: Dict[str, Any], task_type: str) -> str:
        """Determine optimal processing strategy based on analysis.

        Args:
            analysis: Context analysis
            task_type: Type of task

        Returns:
            str: Recommended strategy
        """
        # Strategy selection logic
        if not analysis['requires_chunking']:
            return 'full_context'

        if task_type == 'comprehensive_summary':
            if analysis['has_structure']:
                return 'hierarchical'
            else:
                return 'summarize_and_refine'
        elif task_type == 'question_answering':
            return 'key_extraction'
        elif task_type in ['cross_reference', 'contradiction_detection']:
            return 'chunk_and_merge'
        elif task_type in ['timeline_extraction', 'entity_analysis']:
            return 'sliding_window'
        else:
            return 'hierarchical' if analysis['has_structure'] else 'chunk_and_merge'

    async def _process_full_context(self,
                                    context: str,
                                    task_type: str,
                                    focus_query: Optional[str],
                                    kwargs: Dict[str, Any]) -> str:
        """Process context as a single unit.

        Args:
            context: Full context text
            task_type: Task to perform
            focus_query: Optional focus query
            kwargs: Additional parameters

        Returns:
            str: Processed response
        """
        # Build prompt for full context processing
        task_prompt = self._build_long_context_prompt(
            context, task_type, focus_query, 'full_context'
        )

        # Get generation parameters
        gen_params = self._get_long_context_params(kwargs, task_type)

        # Process with model
        return await self._generate_with_context(task_prompt, gen_params)

    async def _process_chunked_context(self,
                                       context: str,
                                       strategy: str,
                                       task_type: str,
                                       focus_query: Optional[str],
                                       chunk_overlap: int,
                                       kwargs: Dict[str, Any]) -> str:
        """Process context in chunks using specified strategy.

        Args:
            context: Long context text
            strategy: Processing strategy
            task_type: Task to perform
            focus_query: Optional focus query
            chunk_overlap: Overlap between chunks
            kwargs: Additional parameters

        Returns:
            str: Processed response
        """
        if strategy == 'sliding_window':
            return await self._sliding_window_processing(context, task_type, focus_query, chunk_overlap, kwargs)
        elif strategy == 'hierarchical':
            return await self._hierarchical_processing(context, task_type, focus_query, kwargs)
        elif strategy == 'summarize_and_refine':
            return await self._summarize_and_refine(context, task_type, focus_query, kwargs)
        elif strategy == 'key_extraction':
            return await self._key_extraction_processing(context, task_type, focus_query, kwargs)
        elif strategy == 'chunk_and_merge':
            return await self._chunk_and_merge_processing(context, task_type, focus_query, chunk_overlap, kwargs)
        else:
            # Default to chunk and merge
            return await self._chunk_and_merge_processing(context, task_type, focus_query, chunk_overlap, kwargs)

    async def _sliding_window_processing(self,
                                         context: str,
                                         task_type: str,
                                         focus_query: Optional[str],
                                         overlap: int,
                                         kwargs: Dict[str, Any]) -> str:
        """Process using sliding window approach.

        Args:
            context: Context text
            task_type: Task type
            focus_query: Focus query
            overlap: Window overlap
            kwargs: Additional parameters

        Returns:
            str: Processed response
        """
        chunks = self._create_overlapping_chunks(context, int(self.context_limits['chunk_size']), overlap)
        chunk_results: List[str] = []

        for i, chunk in enumerate(chunks):
            chunk_prompt = self._build_long_context_prompt(
                chunk, task_type, focus_query, 'sliding_window', chunk_index=i
            )

            gen_params = self._get_long_context_params(kwargs, task_type)
            result = await self._generate_with_context(chunk_prompt, gen_params)
            chunk_results.append(result)

        # Merge results
        return self._merge_chunk_results(chunk_results, task_type)

    async def _hierarchical_processing(self,
                                       context: str,
                                       task_type: str,
                                       focus_query: Optional[str],
                                       kwargs: Dict[str, Any]) -> str:
        """Process using hierarchical approach (sections first, then details).

        Args:
            context: Context text
            task_type: Task type
            focus_query: Focus query
            kwargs: Additional parameters

        Returns:
            str: Processed response
        """
        # Identify sections
        sections = self._identify_sections(context)

        # Process each section
        section_results: List[Tuple[str, str]] = []
        for section_title, section_content in sections:
            section_prompt = self._build_long_context_prompt(
                section_content, task_type, focus_query, 'hierarchical', section_title=section_title
            )

            gen_params = self._get_long_context_params(kwargs, task_type)
            result = await self._generate_with_context(section_prompt, gen_params)
            section_results.append((section_title, result))

        # Synthesize section results
        return self._synthesize_section_results(section_results, task_type)

    async def _summarize_and_refine(self,
                                    context: str,
                                    task_type: str,
                                    focus_query: Optional[str],
                                    kwargs: Dict[str, Any]) -> str:
        """Process by summarizing chunks then refining.

        Args:
            context: Context text
            task_type: Task type
            focus_query: Focus query
            kwargs: Additional parameters

        Returns:
            str: Processed response
        """
        # First pass: summarize chunks
        chunks = self._create_chunks(context, int(self.context_limits['chunk_size']))
        summaries: List[str] = []

        for chunk in chunks:
            summary_prompt = f"Summarize this text focusing on key information:\n\n{chunk}"
            gen_params = self._get_long_context_params(kwargs, 'summary')
            summary = await self._generate_with_context(summary_prompt, gen_params)
            summaries.append(summary)

        # Second pass: process combined summaries
        combined_summaries = '\n\n'.join(summaries)
        final_prompt = self._build_long_context_prompt(
            combined_summaries, task_type, focus_query, 'summarize_and_refine'
        )

        gen_params = self._get_long_context_params(kwargs, task_type)
        return await self._generate_with_context(final_prompt, gen_params)

    async def _key_extraction_processing(self,
                                         context: str,
                                         task_type: str,
                                         focus_query: Optional[str],
                                         kwargs: Dict[str, Any]) -> str:
        """Process by extracting key information first.

        Args:
            context: Context text
            task_type: Task type
            focus_query: Focus query
            kwargs: Additional parameters

        Returns:
            str: Processed response
        """
        # Extract key information
        key_info = await self._extract_key_information(context, focus_query, kwargs)

        # Process key information
        final_prompt = self._build_long_context_prompt(
            key_info, task_type, focus_query, 'key_extraction'
        )

        gen_params = self._get_long_context_params(kwargs, task_type)
        return await self._generate_with_context(final_prompt, gen_params)

    async def _chunk_and_merge_processing(self,
                                          context: str,
                                          task_type: str,
                                          focus_query: Optional[str],
                                          overlap: int,
                                          kwargs: Dict[str, Any]) -> str:
        """Process chunks independently then merge results.

        Args:
            context: Context text
            task_type: Task type
            focus_query: Focus query
            overlap: Chunk overlap
            kwargs: Additional parameters

        Returns:
            str: Processed response
        """
        chunks = self._create_overlapping_chunks(context, int(self.context_limits['chunk_size']), overlap)
        chunk_results: List[str] = []

        for chunk in chunks:
            chunk_prompt = self._build_long_context_prompt(
                chunk, task_type, focus_query, 'chunk_and_merge'
            )

            gen_params = self._get_long_context_params(kwargs, task_type)
            result = await self._generate_with_context(chunk_prompt, gen_params)
            chunk_results.append(result)

        # Merge and deduplicate results
        return self._merge_and_deduplicate_results(chunk_results, task_type)

    def _create_chunks(self, text: str, chunk_size: int) -> List[str]:
        """Create non-overlapping chunks from text.

        Args:
            text: Text to chunk
            chunk_size: Size of each chunk in characters

        Returns:
            List[str]: Text chunks
        """
        chunks: List[str] = []
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i + chunk_size])
        return chunks

    def _create_overlapping_chunks(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Create overlapping chunks from text.

        Args:
            text: Text to chunk
            chunk_size: Size of each chunk
            overlap: Overlap between chunks

        Returns:
            List[str]: Overlapping text chunks
        """
        chunks: List[str] = []
        step = chunk_size - overlap

        for i in range(0, len(text), step):
            chunk = text[i:i + chunk_size]
            if chunk:
                chunks.append(chunk)

        return chunks

    def _identify_sections(self, text: str) -> List[Tuple[str, str]]:
        """Identify sections in structured text.

        Args:
            text: Text to analyze

        Returns:
            List[Tuple[str, str]]: List of (title, content) pairs
        """
        sections: List[Tuple[str, str]] = []

        # Look for markdown-style headers
        header_pattern = r'^(#{1,6})\s+(.+)$'
        lines = text.split('\n')

        current_section = None
        current_content: List[str] = []

        for line in lines:
            header_match = re.match(header_pattern, line)
            if header_match:
                # Save previous section
                if current_section:
                    sections.append((current_section, '\n'.join(current_content)))

                # Start new section
                current_section = header_match.group(2)
                current_content = []
            else:
                current_content.append(line)

        # Save last section
        if current_section:
            sections.append((current_section, '\n'.join(current_content)))

        # If no headers found, create sections based on paragraphs
        if not sections:
            paragraphs = text.split('\n\n')
            for i, paragraph in enumerate(paragraphs):
                if paragraph.strip():
                    sections.append((f"Section {i + 1}", paragraph))

        return sections

    async def _extract_key_information(self,
                                       context: str,
                                       focus_query: Optional[str],
                                       kwargs: Dict[str, Any]) -> str:
        """Extract key information from context.

        Args:
            context: Full context text
            focus_query: Optional focus query
            kwargs: Additional parameters

        Returns:
            str: Extracted key information
        """
        extraction_prompt = "Extract the most important information, key facts, and main points from this text:\n\n"

        if focus_query:
            extraction_prompt += f"Focus specifically on information related to: {focus_query}\n\n"

        extraction_prompt += context

        gen_params = self._get_long_context_params(kwargs, 'key_extraction')
        return await self._generate_with_context(extraction_prompt, gen_params)

    def _build_long_context_prompt(self,
                                   text: str,
                                   task_type: str,
                                   focus_query: Optional[str],
                                   strategy: str,
                                   **context_info: Any) -> str:
        """Build prompt for long context processing.

        Args:
            text: Text to process
            task_type: Task type
            focus_query: Optional focus query
            strategy: Processing strategy
            **context_info: Additional context information

        Returns:
            str: Long context prompt
        """
        system_prompt = self.get_system_prompt()

        prompt_parts: List[str] = []

        if system_prompt:
            prompt_parts.append(system_prompt)

        # Add task instructions
        instructions: List[str] = []

        if task_type in self.long_context_tasks:
            task_description = self.long_context_tasks[task_type]
            instructions.append(f"Task: {task_description}")

        if focus_query:
            instructions.append(f"Focus on: {focus_query}")

        # Strategy-specific instructions
        if strategy == 'hierarchical' and 'section_title' in context_info:
            instructions.append(f"Processing section: {context_info['section_title']}")
        elif strategy == 'sliding_window' and 'chunk_index' in context_info:
            instructions.append(f"Processing chunk {context_info['chunk_index'] + 1}")

        # General long context instructions
        instructions.extend([
            "Maintain awareness of the broader context",
            "Focus on the most relevant and important information",
            "Preserve key relationships and connections"
        ])

        if instructions:
            prompt_parts.append("Instructions: " + "\n".join([f"- {inst}" for inst in instructions]))

        prompt_parts.append(f"Text to process:\n{text}")
        prompt_parts.append("Analysis:")

        return "\n\n".join(prompt_parts)

    def _get_long_context_params(self, kwargs: Dict[str, Any], task_type: str) -> Dict[str, Any]:
        """Get generation parameters for long context processing.

        Args:
            kwargs: User parameters
            task_type: Task type

        Returns:
            Dict[str, Any]: Generation parameters
        """
        # Get base parameters and then customize for long context
        params = self._get_default_generation_params()
        
        # Override with user parameters
        if 'max_tokens' in kwargs:
            params['max_new_tokens'] = kwargs['max_tokens']
        if 'temperature' in kwargs:
            params['temperature'] = kwargs['temperature']
        if 'top_p' in kwargs:
            params['top_p'] = kwargs['top_p']
        if 'repetition_penalty' in kwargs:
            params['repetition_penalty'] = kwargs['repetition_penalty']

        # Adjust based on task type
        if task_type == 'comprehensive_summary':
            params['max_new_tokens'] = min(2048, params['max_new_tokens'] * 2)
        elif task_type in ['timeline_extraction', 'entity_analysis']:
            params['temperature'] = 0.4  # More structured output
        elif task_type == 'question_answering':
            params['temperature'] = 0.5

        return params

    async def _generate_with_context(self, prompt: str, gen_params: Dict[str, Any]) -> str:
        """Generate response with long context handling.

        Args:
            prompt: Generation prompt
            gen_params: Generation parameters

        Returns:
            str: Generated response
        """
        # Use parent's generate method with the built prompt and parameters
        return await super().generate(prompt, **gen_params)

    def _merge_chunk_results(self, results: List[str], task_type: str) -> str:
        """Merge results from multiple chunks.

        Args:
            results: List of chunk results
            task_type: Task type

        Returns:
            str: Merged results
        """
        if task_type == 'comprehensive_summary':
            return self._merge_summaries(results)
        elif task_type == 'timeline_extraction':
            return self._merge_timelines(results)
        elif task_type == 'entity_analysis':
            return self._merge_entities(results)
        else:
            return '\n\n'.join(results)

    def _merge_summaries(self, summaries: List[str]) -> str:
        """Merge multiple summaries into one coherent summary."""
        combined = "Combined Summary:\n\n"
        for i, summary in enumerate(summaries, 1):
            combined += f"Section {i}:\n{summary}\n\n"
        return combined

    def _merge_timelines(self, timelines: List[str]) -> str:
        """Merge timeline information from multiple chunks."""
        return "Timeline Information:\n\n" + "\n\n".join(timelines)

    def _merge_entities(self, entity_lists: List[str]) -> str:
        """Merge entity analysis from multiple chunks."""
        return "Entity Analysis:\n\n" + "\n\n".join(entity_lists)

    def _merge_and_deduplicate_results(self, results: List[str], task_type: str) -> str:
        """Merge results and remove duplicates."""
        # Simple deduplication - in practice you'd want more sophisticated merging
        unique_results: List[str] = []
        seen_content: set[str] = set()

        for result in results:
            result_key = result[:100]  # Use first 100 chars as key
            if result_key not in seen_content:
                seen_content.add(result_key)
                unique_results.append(result)

        return self._merge_chunk_results(unique_results, task_type)

    def _synthesize_section_results(self, section_results: List[Tuple[str, str]], task_type: str) -> str:
        """Synthesize results from hierarchical processing."""
        synthesis = f"Hierarchical Analysis ({task_type}):\n\n"

        for section_title, result in section_results:
            synthesis += f"## {section_title}\n{result}\n\n"

        return synthesis

    async def _stream_full_context(self, context: str, kwargs: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Stream processing of full context."""
        # For streaming full context, we'll process normally but yield chunks
        task_type = kwargs.get('task_type', 'comprehensive_summary')
        result = await self._process_full_context(context, task_type, kwargs.get('focus_query'), kwargs)

        # Yield result in chunks
        chunk_size = 100
        for i in range(0, len(result), chunk_size):
            yield result[i:i + chunk_size]

    async def _stream_chunked_processing(self, context: str, kwargs: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Stream processing of chunked context."""
        # Process chunks and yield results as they complete
        chunks = self._create_chunks(context, int(self.context_limits['chunk_size']))
        task_type = kwargs.get('task_type', 'comprehensive_summary')

        for i, chunk in enumerate(chunks):
            yield f"\n--- Processing chunk {i + 1}/{len(chunks)} ---\n"

            chunk_prompt = self._build_long_context_prompt(
                chunk, task_type, kwargs.get('focus_query'), 'streaming'
            )

            gen_params = self._get_long_context_params(kwargs, task_type)
            result = await self._generate_with_context(chunk_prompt, gen_params)
            yield result + "\n\n"

    def _post_process_long_context_response(self,
                                            response: str,
                                            strategy: str,
                                            task_type: str,
                                            analysis: Dict[str, Any],
                                            kwargs: Dict[str, Any]) -> str:
        """Post-process long context response."""
        if not response.strip():
            return response

        # Add context metadata if requested
        if kwargs.get('include_metadata', False):
            metadata = f"\n\n**Context Analysis:**\n"
            metadata += f"- Original length: {analysis['word_count']} words\n"
            metadata += f"- Processing strategy: {strategy}\n"
            metadata += f"- Task type: {task_type}\n"
            metadata += f"- Complexity: {analysis['complexity']}"

            response += metadata

        return response.strip()

    async def process_document(self,
                               document_text: str,
                               document_type: str = 'general',
                               tasks: Optional[List[str]] = None) -> Dict[str, Any]:
        """Process a document with multiple analysis tasks.

        Args:
            document_text: Full document text
            document_type: Type of document
            tasks: List of tasks to perform

        Returns:
            Dict containing results for each task
        """
        if tasks is None:
            tasks = ['comprehensive_summary', 'key_insights', 'entity_analysis']

        results = {}

        for task in tasks:
            result = await self.generate(
                document_text,
                task_type=task,
                document_type=document_type,
                strategy='auto'
            )
            results[task] = result

        return {
            'document_analysis': results,
            'document_type': document_type,
            'tasks_performed': tasks,
            'processing_method': 'long_context_analysis'
        }

    async def warmup(self) -> bool:
        """Warm up with a long context example."""
        if not self.is_loaded():
            return False

        try:
            warmup_text = """
            This is a sample long document for testing the long context processing capabilities.
            It contains multiple paragraphs and sections to simulate a real document.

            The first section discusses the importance of context in natural language processing.
            Context allows models to understand relationships between different parts of a text.

            The second section explores different strategies for handling long contexts.
            These include chunking, summarization, and hierarchical processing approaches.
            """

            await self.generate(
                warmup_text,
                task_type='comprehensive_summary',
                strategy='auto',
                max_tokens=200
            )
            logger.info(f"Warmup completed for {self.category}")
            return True
        except Exception as e:
            logger.error(f"Warmup failed for {self.category}: {e}")
            return False

    def get_context_strategies(self) -> List[str]:
        """Get available context processing strategies."""
        return list(self.context_strategies.keys())

    def get_task_types(self) -> List[str]:
        """Get available long context task types."""
        return list(self.long_context_tasks.keys())

    def get_document_types(self) -> List[str]:
        """Get supported document types."""
        return list(self.document_types.keys())

    def get_context_limits(self) -> Dict[str, Any]:
        """Get current context limits."""
        return self.context_limits.copy()