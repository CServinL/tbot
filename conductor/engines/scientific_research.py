import asyncio
import logging
import re
from typing import Dict, Any, Optional, AsyncGenerator, List, Tuple
from base_llm_engine import BaseLLMEngine
from model_loader import ModelLoader
from utils.persona_loader import PersonaLoader

logger = logging.getLogger(__name__)


class ScientificResearchEngine(BaseLLMEngine):
    """Engine specialized for scientific research analysis and literature review."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_loader = ModelLoader()
        self.persona_loader = PersonaLoader()

        # Scientific domains
        self.scientific_domains = {
            'biology': 'Biological sciences, life sciences, and related fields',
            'chemistry': 'Chemical sciences, materials science, and molecular studies',
            'physics': 'Physical sciences, theoretical and applied physics',
            'medicine': 'Medical research, clinical studies, and healthcare',
            'computer_science': 'Computer science, AI, and computational research',
            'engineering': 'Engineering disciplines and applied sciences',
            'environmental': 'Environmental science, climate research, ecology',
            'psychology': 'Psychological research and behavioral sciences',
            'neuroscience': 'Brain research, cognitive science, neurobiology',
            'mathematics': 'Mathematical research and theoretical mathematics',
            'astronomy': 'Astronomical research and space sciences',
            'geology': 'Earth sciences, geology, and related fields'
        }

        # Research task types
        self.research_tasks = {
            'literature_review': 'Comprehensive review of existing literature',
            'methodology_analysis': 'Analysis of research methodologies',
            'results_interpretation': 'Interpretation of research findings',
            'hypothesis_generation': 'Generation of research hypotheses',
            'experimental_design': 'Design of experiments and studies',
            'data_analysis': 'Statistical and scientific data analysis',
            'paper_summary': 'Summarization of scientific papers',
            'research_proposal': 'Development of research proposals',
            'peer_review': 'Critical evaluation of scientific work',
            'meta_analysis': 'Analysis across multiple studies'
        }

        # Citation and evidence standards
        self.evidence_levels = {
            'systematic_review': 'Highest level - systematic reviews and meta-analyses',
            'randomized_trial': 'High level - randomized controlled trials',
            'cohort_study': 'Medium-high level - prospective cohort studies',
            'case_control': 'Medium level - case-control studies',
            'cross_sectional': 'Lower level - cross-sectional studies',
            'case_series': 'Low level - case series and case reports',
            'expert_opinion': 'Lowest level - expert opinion and consensus'
        }

    async def load_model(self) -> bool:
        """Load the scientific research model (typically larger model for complex analysis)."""
        try:
            logger.info(f"Loading scientific research model: {self.technical_model_name}")

            # Ensure dependencies for scientific analysis
            from dependencies_loader import DependenciesLoader
            deps_loader = DependenciesLoader()
            await deps_loader.ensure_dependencies(
                self.technical_model_name,
                self.precision,
                features=['optimizations']  # May need optimizations for large model
            )

            self.model, self.tokenizer = await self.model_loader.load_model(
                self.technical_model_name,
                self.precision
            )

            if self.model is not None and self.tokenizer is not None:
                self.is_model_loaded = True
                self.load_time = asyncio.get_event_loop().time()
                logger.info(f"Successfully loaded scientific research model")

                # Perform warmup
                await self.warmup()
                return True
            else:
                logger.error("Failed to load scientific research model")
                return False

        except Exception as e:
            logger.error(f"Error loading scientific research model: {e}")
            return False

    async def unload_model(self) -> bool:
        """Unload the scientific research model."""
        try:
            if self.is_model_loaded:
                success = await self.model_loader.unload_model(self.technical_model_name)
                if success:
                    self.model = None
                    self.tokenizer = None
                    self.is_model_loaded = False
                    logger.info("Scientific research model unloaded")
                return success
            return True
        except Exception as e:
            logger.error(f"Error unloading scientific research model: {e}")
            return False

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate scientific research analysis.

        Args:
            prompt: Research question or content to analyze
            **kwargs: Additional parameters
                - domain: Scientific domain (biology, chemistry, etc.)
                - task_type: Type of research task
                - evidence_level: Required evidence level
                - include_citations: Whether to include citation guidance
                - peer_review_mode: Whether to apply peer review standards
                - methodology_focus: Focus on methodological aspects
                - statistical_analysis: Include statistical considerations

        Returns:
            str: Scientific analysis or research content
        """
        if not self.is_model_loaded:
            raise RuntimeError("Scientific research model not loaded")

        try:
            # Parse research parameters
            domain = kwargs.get('domain', 'general')
            task_type = kwargs.get('task_type', 'literature_review')
            evidence_level = kwargs.get('evidence_level', 'high')
            include_citations = kwargs.get('include_citations', True)
            peer_review_mode = kwargs.get('peer_review_mode', True)
            methodology_focus = kwargs.get('methodology_focus', False)
            statistical_analysis = kwargs.get('statistical_analysis', False)

            # Build scientific research prompt
            research_prompt = self._build_research_prompt(
                prompt, domain, task_type, evidence_level, include_citations,
                peer_review_mode, methodology_focus, statistical_analysis
            )

            # Get generation parameters for research
            gen_params = self._get_research_params(kwargs, task_type, domain)

            # Tokenize input (handle longer research texts)
            inputs = self.tokenizer(
                research_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096  # Longer context for research
            )

            # Move to device
            if hasattr(self.model, 'device'):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Generate research content
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

            # Decode and process research content
            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            research_content = self._extract_response(full_output, research_prompt)

            # Post-process research content
            processed_content = self._post_process_research_content(
                research_content, task_type, domain, kwargs
            )

            self.increment_generation_count()

            logger.debug(f"Generated {task_type} research content for {domain}: {len(processed_content)} chars")
            return processed_content

        except Exception as e:
            logger.error(f"Error generating scientific research content: {e}")
            raise

    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Generate streaming scientific research content.

        Args:
            prompt: Research question or content
            **kwargs: Additional parameters

        Yields:
            str: Research content chunks
        """
        if not self.is_model_loaded:
            raise RuntimeError("Scientific research model not loaded")

        try:
            from transformers import TextIteratorStreamer
            import torch
            from threading import Thread

            domain = kwargs.get('domain', 'general')
            task_type = kwargs.get('task_type', 'literature_review')

            research_prompt = self._build_research_prompt(
                prompt, domain, task_type,
                kwargs.get('evidence_level', 'high'),
                kwargs.get('include_citations', True),
                kwargs.get('peer_review_mode', True),
                kwargs.get('methodology_focus', False),
                kwargs.get('statistical_analysis', False)
            )

            gen_params = self._get_research_params(kwargs, task_type, domain)

            inputs = self.tokenizer(
                research_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096
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
            logger.error(f"Error in streaming scientific research: {e}")
            yield f"Error: {str(e)}"

    def get_system_prompt(self) -> Optional[str]:
        """Get system prompt for scientific research."""
        return self.persona_loader.get_persona_for_category('scientific_research')

    def _build_research_prompt(self,
                               research_query: str,
                               domain: str,
                               task_type: str,
                               evidence_level: str,
                               include_citations: bool,
                               peer_review_mode: bool,
                               methodology_focus: bool,
                               statistical_analysis: bool) -> str:
        """Build prompt for scientific research tasks.

        Args:
            research_query: Research question or content
            domain: Scientific domain
            task_type: Type of research task
            evidence_level: Required evidence level
            include_citations: Whether to include citations
            peer_review_mode: Whether to apply peer review standards
            methodology_focus: Focus on methodology
            statistical_analysis: Include statistical analysis

        Returns:
            str: Research prompt
        """
        system_prompt = self.get_system_prompt()

        prompt_parts = []

        if system_prompt:
            prompt_parts.append(system_prompt)

        # Add domain context
        if domain in self.scientific_domains:
            domain_description = self.scientific_domains[domain]
            prompt_parts.append(f"Scientific domain: {domain} - {domain_description}")

        # Add task-specific instructions
        instructions = []

        if task_type in self.research_tasks:
            task_description = self.research_tasks[task_type]
            instructions.append(f"Task: {task_description}")

        # Evidence level requirements
        if evidence_level in self.evidence_levels:
            evidence_description = self.evidence_levels[evidence_level]
            instructions.append(f"Evidence standard: {evidence_description}")

        # Research quality standards
        instructions.extend([
            "Apply rigorous scientific methodology",
            "Ensure accuracy and objectivity",
            "Distinguish between established facts and current research",
            "Acknowledge limitations and uncertainties"
        ])

        # Specific focus areas
        if methodology_focus:
            instructions.append("Pay special attention to research methodologies and experimental design")

        if statistical_analysis:
            instructions.append("Include statistical considerations and analysis approaches")

        if peer_review_mode:
            instructions.extend([
                "Apply peer review standards for critical evaluation",
                "Identify potential methodological issues or limitations",
                "Assess the strength of evidence presented"
            ])

        # Citation and reference guidelines
        if include_citations:
            instructions.extend([
                "Indicate where citations would be appropriate (use [Citation needed] format)",
                "Distinguish between different types of evidence",
                "Reference established scientific principles where applicable"
            ])

        # Domain-specific instructions
        if domain == 'medicine':
            instructions.append("Apply evidence-based medicine principles")
        elif domain == 'physics':
            instructions.append("Include relevant physical principles and mathematical relationships")
        elif domain == 'biology':
            instructions.append("Consider biological mechanisms and evolutionary context")
        elif domain == 'chemistry':
            instructions.append("Include molecular and chemical mechanisms where relevant")

        if instructions:
            prompt_parts.append("Research instructions:\n" + "\n".join([f"- {inst}" for inst in instructions]))

        prompt_parts.append(f"Research query: {research_query}")
        prompt_parts.append("Scientific analysis:")

        return "\n\n".join(prompt_parts)

    def _get_research_params(self, kwargs: Dict[str, Any], task_type: str, domain: str) -> Dict[str, Any]:
        """Get generation parameters for scientific research.

        Args:
            kwargs: User parameters
            task_type: Research task type
            domain: Scientific domain

        Returns:
            Dict[str, Any]: Generation parameters
        """
        # Base parameters for scientific research
        params = {
            'max_new_tokens': kwargs.get('max_tokens', 1024),
            'temperature': 0.5,  # Moderate temperature for balanced accuracy/insight
            'do_sample': True,
            'top_p': 0.85,
            'repetition_penalty': 1.1,
            'pad_token_id': self.tokenizer.eos_token_id if self.tokenizer else None
        }

        # Adjust based on task type
        if task_type == 'literature_review':
            params['max_new_tokens'] = min(2048, params['max_new_tokens'] * 2)
            params['temperature'] = 0.6  # Slightly higher for comprehensive analysis
        elif task_type == 'methodology_analysis':
            params['temperature'] = 0.4  # Lower for precise methodological analysis
        elif task_type == 'hypothesis_generation':
            params['temperature'] = 0.7  # Higher for creative hypothesis generation
        elif task_type == 'peer_review':
            params['temperature'] = 0.5
            params['top_p'] = 0.8  # More focused for critical evaluation
        elif task_type == 'data_analysis':
            params['temperature'] = 0.4  # Lower for analytical precision

        # Adjust based on domain complexity
        complex_domains = ['physics', 'mathematics', 'neuroscience']
        if domain in complex_domains:
            params['max_new_tokens'] = min(2048, params['max_new_tokens'] * 1.5)

        # Override with user parameters
        if 'temperature' in kwargs:
            params['temperature'] = max(0.1, min(kwargs['temperature'], 1.0))

        return params

    def _post_process_research_content(self,
                                       content: str,
                                       task_type: str,
                                       domain: str,
                                       kwargs: Dict[str, Any]) -> str:
        """Post-process scientific research content.

        Args:
            content: Raw research content
            task_type: Research task type
            domain: Scientific domain
            kwargs: Generation parameters

        Returns:
            str: Post-processed research content
        """
        if not content.strip():
            return content

        # Clean up common artifacts
        content = self._clean_research_artifacts(content)

        # Apply task-specific formatting
        if task_type == 'literature_review':
            content = self._format_literature_review(content)
        elif task_type == 'methodology_analysis':
            content = self._format_methodology_analysis(content)
        elif task_type == 'peer_review':
            content = self._format_peer_review(content)
        elif task_type == 'research_proposal':
            content = self._format_research_proposal(content)

        # Add scientific rigor indicators
        if kwargs.get('peer_review_mode', True):
            content = self._add_scientific_rigor_notes(content)

        return content.strip()

    def _clean_research_artifacts(self, content: str) -> str:
        """Remove common research generation artifacts.

        Args:
            content: Raw content

        Returns:
            str: Cleaned content
        """
        # Remove common prefixes
        prefixes_to_remove = [
            "Here's the research analysis:",
            "Scientific analysis:",
            "Research findings:",
            "Based on scientific literature:"
        ]

        for prefix in prefixes_to_remove:
            if content.startswith(prefix):
                content = content[len(prefix):].strip()

        return content

    def _format_literature_review(self, content: str) -> str:
        """Format content as literature review.

        Args:
            content: Content to format

        Returns:
            str: Formatted literature review
        """
        # Add literature review structure
        if not any(header in content for header in ['Introduction', 'Methods', 'Results', 'Discussion']):
            # Basic structure for literature review
            paragraphs = content.split('\n\n')
            if len(paragraphs) > 1:
                structured_content = f"**Overview:**\n{paragraphs[0]}\n\n"
                structured_content += f"**Current Research:**\n" + '\n\n'.join(paragraphs[1:])
                return structured_content

        return content

    def _format_methodology_analysis(self, content: str) -> str:
        """Format content as methodology analysis.

        Args:
            content: Content to format

        Returns:
            str: Formatted methodology analysis
        """
        # Add methodology-specific headers
        if 'methodology' not in content.lower():
            content = f"**Methodological Analysis:**\n\n{content}"

        return content

    def _format_peer_review(self, content: str) -> str:
        """Format content as peer review.

        Args:
            content: Content to format

        Returns:
            str: Formatted peer review
        """
        # Add peer review structure
        if not any(section in content for section in ['Strengths', 'Weaknesses', 'Recommendations']):
            # Basic peer review structure
            content = f"**Peer Review Assessment:**\n\n{content}"

        return content

    def _format_research_proposal(self, content: str) -> str:
        """Format content as research proposal.

        Args:
            content: Content to format

        Returns:
            str: Formatted research proposal
        """
        # Add research proposal structure
        if not any(section in content for section in ['Hypothesis', 'Methodology', 'Expected Outcomes']):
            content = f"**Research Proposal:**\n\n{content}"

        return content

    def _add_scientific_rigor_notes(self, content: str) -> str:
        """Add scientific rigor and limitation notes.

        Args:
            content: Research content

        Returns:
            str: Content with rigor notes
        """
        # Add disclaimer about the AI nature of the analysis
        rigor_note = ("\n\n**Note:** This analysis is generated by an AI system and should be "
                      "verified against current scientific literature. For research purposes, "
                      "consult peer-reviewed sources and domain experts.")

        return content + rigor_note

    async def analyze_research_paper(self,
                                     paper_text: str,
                                     focus_areas: Optional[List[str]] = None) -> Dict[str, Any]:
        """Analyze a research paper comprehensively.

        Args:
            paper_text: Text of the research paper
            focus_areas: Specific areas to focus analysis on

        Returns:
            Dict containing analysis results
        """
        analysis_tasks = [
            "methodology",
            "findings",
            "limitations",
            "significance"
        ]

        if focus_areas:
            analysis_tasks.extend(focus_areas)

        results = {}

        for task in analysis_tasks:
            prompt = f"Analyze the {task} of this research paper:\n\n{paper_text}"

            analysis = await self.generate(
                prompt,
                task_type='methodology_analysis' if task == 'methodology' else 'paper_summary',
                peer_review_mode=True,
                domain='general'
            )

            results[task] = analysis

        return {
            'paper_analysis': results,
            'analysis_method': 'comprehensive_ai_analysis',
            'focus_areas': analysis_tasks
        }

    async def generate_research_hypothesis(self,
                                           research_area: str,
                                           background: str,
                                           domain: str = 'general') -> str:
        """Generate research hypotheses for a given area.

        Args:
            research_area: Area of research interest
            background: Background information and context
            domain: Scientific domain

        Returns:
            str: Generated research hypotheses
        """
        hypothesis_prompt = f"""
        Research area: {research_area}

        Background: {background}

        Generate testable research hypotheses for this area, including:
        1. Primary hypothesis
        2. Alternative hypotheses
        3. Null hypothesis
        4. Rationale for each hypothesis
        5. Potential experimental approaches
        """

        return await self.generate(
            hypothesis_prompt,
            task_type='hypothesis_generation',
            domain=domain,
            methodology_focus=True
        )

    async def warmup(self) -> bool:
        """Warm up with a scientific research example."""
        if not self.is_loaded():
            return False

        try:
            warmup_query = "Analyze the current understanding of climate change impacts on marine ecosystems"
            await self.generate(
                warmup_query,
                domain='environmental',
                task_type='literature_review',
                max_tokens=200
            )
            logger.info(f"Warmup completed for {self.category}")
            return True
        except Exception as e:
            logger.error(f"Warmup failed for {self.category}: {e}")
            return False

    def get_scientific_domains(self) -> List[str]:
        """Get available scientific domains.

        Returns:
            List[str]: Available domains
        """
        return list(self.scientific_domains.keys())

    def get_research_tasks(self) -> List[str]:
        """Get available research task types.

        Returns:
            List[str]: Available task types
        """
        return list(self.research_tasks.keys())

    def get_evidence_levels(self) -> List[str]:
        """Get available evidence levels.

        Returns:
            List[str]: Available evidence levels
        """
        return list(self.evidence_levels.keys())

    def get_domain_info(self, domain: str) -> Dict[str, Any]:
        """Get information about a scientific domain.

        Args:
            domain: Scientific domain

        Returns:
            Dict containing domain info
        """
        return {
            'domain': domain,
            'description': self.scientific_domains.get(domain, 'General scientific research'),
            'recommended_tasks': ['literature_review', 'methodology_analysis', 'paper_summary'],
            'evidence_standards': 'peer_reviewed_sources'
        }