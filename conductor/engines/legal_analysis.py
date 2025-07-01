import asyncio
import logging
import re
from typing import Dict, Any, Optional, AsyncGenerator, List, Tuple
from base_llm_engine import BaseLLMEngine
from model_loader import ModelLoader
from utils.persona_loader import PersonaLoader

logger = logging.getLogger(__name__)


class LegalAnalysisEngine(BaseLLMEngine):
    """Engine specialized for legal document analysis and legal research."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_loader = ModelLoader()
        self.persona_loader = PersonaLoader()

        # Legal practice areas
        self.legal_areas = {
            'contract_law': 'Contract analysis, terms, and obligations',
            'corporate_law': 'Corporate governance, compliance, and business law',
            'employment_law': 'Employment agreements, workplace policies, labor law',
            'intellectual_property': 'Patents, trademarks, copyrights, trade secrets',
            'real_estate': 'Property law, transactions, leases, zoning',
            'litigation': 'Civil litigation, disputes, procedural matters',
            'regulatory': 'Government regulations, compliance, administrative law',
            'tax_law': 'Tax regulations, compliance, planning strategies',
            'criminal_law': 'Criminal statutes, procedures, defense strategies',
            'family_law': 'Divorce, custody, domestic relations',
            'immigration': 'Immigration law, visas, citizenship matters',
            'bankruptcy': 'Insolvency, debt restructuring, bankruptcy proceedings'
        }

        # Document types
        self.document_types = {
            'contract': 'Contracts, agreements, and binding documents',
            'statute': 'Laws, regulations, and statutory provisions',
            'case_law': 'Court decisions, precedents, and judicial opinions',
            'brief': 'Legal briefs, memoranda, and arguments',
            'policy': 'Corporate policies, procedures, and guidelines',
            'compliance': 'Compliance documents and regulatory filings',
            'patent': 'Patent applications and intellectual property documents',
            'pleading': 'Court pleadings, motions, and filings',
            'opinion': 'Legal opinions and advisory documents',
            'transaction': 'Transaction documents and due diligence materials'
        }

        # Analysis types
        self.analysis_types = {
            'risk_assessment': 'Identify and evaluate legal risks',
            'compliance_review': 'Check compliance with applicable laws',
            'contract_review': 'Analyze contract terms and obligations',
            'precedent_analysis': 'Research relevant case law and precedents',
            'regulatory_analysis': 'Analyze regulatory requirements',
            'due_diligence': 'Comprehensive legal due diligence review',
            'clause_analysis': 'Detailed analysis of specific clauses',
            'comparative_analysis': 'Compare multiple documents or provisions',
            'summary': 'Executive summary of legal documents',
            'redlining': 'Suggest revisions and improvements'
        }

    async def load_model(self) -> bool:
        """Load the legal analysis model (typically larger model for complex analysis)."""
        try:
            logger.info(f"Loading legal analysis model: {self.technical_model_name}")

            # Ensure dependencies for legal analysis
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
                logger.info(f"Successfully loaded legal analysis model")

                # Perform warmup
                await self.warmup()
                return True
            else:
                logger.error("Failed to load legal analysis model")
                return False

        except Exception as e:
            logger.error(f"Error loading legal analysis model: {e}")
            return False

    async def unload_model(self) -> bool:
        """Unload the legal analysis model."""
        try:
            if self.is_model_loaded:
                success = await self.model_loader.unload_model(self.technical_model_name)
                if success:
                    self.model = None
                    self.tokenizer = None
                    self.is_model_loaded = False
                    logger.info("Legal analysis model unloaded")
                return success
            return True
        except Exception as e:
            logger.error(f"Error unloading legal analysis model: {e}")
            return False

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate legal analysis.

        Args:
            prompt: Legal question or document to analyze
            **kwargs: Additional parameters
                - legal_area: Area of law (contract_law, corporate_law, etc.)
                - document_type: Type of document being analyzed
                - analysis_type: Type of analysis requested
                - jurisdiction: Relevant jurisdiction
                - risk_focus: Focus on risk assessment
                - compliance_standards: Specific compliance requirements
                - cite_precedents: Whether to reference legal precedents

        Returns:
            str: Legal analysis and recommendations
        """
        if not self.is_model_loaded:
            raise RuntimeError("Legal analysis model not loaded")

        try:
            # Parse legal parameters
            legal_area = kwargs.get('legal_area', 'general')
            document_type = kwargs.get('document_type', 'general')
            analysis_type = kwargs.get('analysis_type', 'summary')
            jurisdiction = kwargs.get('jurisdiction', 'general')
            risk_focus = kwargs.get('risk_focus', True)
            compliance_standards = kwargs.get('compliance_standards', [])
            cite_precedents = kwargs.get('cite_precedents', False)

            # Build legal analysis prompt
            legal_prompt = self._build_legal_prompt(
                prompt, legal_area, document_type, analysis_type,
                jurisdiction, risk_focus, compliance_standards, cite_precedents
            )

            # Get generation parameters for legal analysis
            gen_params = self._get_legal_params(kwargs, analysis_type, legal_area)

            # Tokenize input (handle longer legal documents)
            inputs = self.tokenizer(
                legal_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096  # Longer context for legal documents
            )

            # Move to device
            if hasattr(self.model, 'device'):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Generate legal analysis
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

            # Decode and process legal analysis
            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            legal_analysis = self._extract_response(full_output, legal_prompt)

            # Post-process legal analysis
            processed_analysis = self._post_process_legal_analysis(
                legal_analysis, analysis_type, legal_area, kwargs
            )

            self.increment_generation_count()

            logger.debug(f"Generated {analysis_type} legal analysis for {legal_area}: {len(processed_analysis)} chars")
            return processed_analysis

        except Exception as e:
            logger.error(f"Error generating legal analysis: {e}")
            raise

    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Generate streaming legal analysis.

        Args:
            prompt: Legal question or document
            **kwargs: Additional parameters

        Yields:
            str: Legal analysis chunks
        """
        if not self.is_model_loaded:
            raise RuntimeError("Legal analysis model not loaded")

        try:
            from transformers import TextIteratorStreamer
            import torch
            from threading import Thread

            legal_area = kwargs.get('legal_area', 'general')
            analysis_type = kwargs.get('analysis_type', 'summary')

            legal_prompt = self._build_legal_prompt(
                prompt, legal_area,
                kwargs.get('document_type', 'general'),
                analysis_type,
                kwargs.get('jurisdiction', 'general'),
                kwargs.get('risk_focus', True),
                kwargs.get('compliance_standards', []),
                kwargs.get('cite_precedents', False)
            )

            gen_params = self._get_legal_params(kwargs, analysis_type, legal_area)

            inputs = self.tokenizer(
                legal_prompt,
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
            logger.error(f"Error in streaming legal analysis: {e}")
            yield f"Error: {str(e)}"

    def get_system_prompt(self) -> Optional[str]:
        """Get system prompt for legal analysis."""
        return self.persona_loader.get_persona_for_category('legal_analysis')

    def _build_legal_prompt(self,
                            legal_query: str,
                            legal_area: str,
                            document_type: str,
                            analysis_type: str,
                            jurisdiction: str,
                            risk_focus: bool,
                            compliance_standards: List[str],
                            cite_precedents: bool) -> str:
        """Build prompt for legal analysis.

        Args:
            legal_query: Legal question or document to analyze
            legal_area: Area of law
            document_type: Type of document
            analysis_type: Type of analysis
            jurisdiction: Relevant jurisdiction
            risk_focus: Whether to focus on risks
            compliance_standards: Compliance requirements
            cite_precedents: Whether to cite precedents

        Returns:
            str: Legal analysis prompt
        """
        system_prompt = self.get_system_prompt()

        prompt_parts = []

        if system_prompt:
            prompt_parts.append(system_prompt)

        # Add legal context
        context_parts = []

        if legal_area in self.legal_areas:
            area_description = self.legal_areas[legal_area]
            context_parts.append(f"Legal area: {legal_area} - {area_description}")

        if document_type in self.document_types:
            doc_description = self.document_types[document_type]
            context_parts.append(f"Document type: {document_type} - {doc_description}")

        if jurisdiction != 'general':
            context_parts.append(f"Jurisdiction: {jurisdiction}")

        if context_parts:
            prompt_parts.append("Legal context:\n" + "\n".join(context_parts))

        # Add analysis instructions
        instructions = []

        if analysis_type in self.analysis_types:
            analysis_description = self.analysis_types[analysis_type]
            instructions.append(f"Analysis type: {analysis_description}")

        # Core legal analysis instructions
        instructions.extend([
            "Provide thorough legal analysis based on established legal principles",
            "Identify key legal issues and implications",
            "Consider applicable laws, regulations, and legal standards",
            "Maintain objectivity and professional legal perspective"
        ])

        # Risk-focused instructions
        if risk_focus:
            instructions.extend([
                "Identify potential legal risks and exposure",
                "Assess likelihood and impact of identified risks",
                "Suggest risk mitigation strategies where appropriate"
            ])

        # Compliance instructions
        if compliance_standards:
            instructions.append(f"Evaluate compliance with: {', '.join(compliance_standards)}")

        # Precedent instructions
        if cite_precedents:
            instructions.extend([
                "Reference relevant legal precedents and case law where applicable",
                "Note: Use [Case citation needed] format for specific cases",
                "Distinguish between binding and persuasive authority"
            ])

        # Professional disclaimers
        instructions.extend([
            "Acknowledge any assumptions or limitations in the analysis",
            "Note that this analysis is for informational purposes",
            "Recommend consultation with qualified legal counsel for specific matters"
        ])

        if instructions:
            prompt_parts.append("Analysis instructions:\n" + "\n".join([f"- {inst}" for inst in instructions]))

        prompt_parts.append(f"Legal matter: {legal_query}")
        prompt_parts.append("Legal analysis:")

        return "\n\n".join(prompt_parts)

    def _get_legal_params(self, kwargs: Dict[str, Any], analysis_type: str, legal_area: str) -> Dict[str, Any]:
        """Get generation parameters for legal analysis.

        Args:
            kwargs: User parameters
            analysis_type: Type of analysis
            legal_area: Legal area

        Returns:
            Dict[str, Any]: Generation parameters
        """
        # Base parameters for legal analysis
        params = {
            'max_new_tokens': kwargs.get('max_tokens', 1024),
            'temperature': 0.4,  # Lower temperature for legal precision
            'do_sample': True,
            'top_p': 0.85,
            'repetition_penalty': 1.1,
            'pad_token_id': self.tokenizer.eos_token_id if self.tokenizer else None
        }

        # Adjust based on analysis type
        if analysis_type == 'risk_assessment':
            params['max_new_tokens'] = min(1536, params['max_new_tokens'] * 1.5)
            params['temperature'] = 0.5  # Slightly higher for comprehensive risk analysis
        elif analysis_type == 'due_diligence':
            params['max_new_tokens'] = min(2048, params['max_new_tokens'] * 2)
            params['temperature'] = 0.4
        elif analysis_type == 'contract_review':
            params['max_new_tokens'] = min(1536, params['max_new_tokens'] * 1.5)
            params['temperature'] = 0.3  # Very precise for contract analysis
        elif analysis_type == 'precedent_analysis':
            params['temperature'] = 0.5
        elif analysis_type == 'summary':
            params['temperature'] = 0.4

        # Adjust based on legal area complexity
        complex_areas = ['intellectual_property', 'tax_law', 'regulatory']
        if legal_area in complex_areas:
            params['max_new_tokens'] = min(2048, params['max_new_tokens'] * 1.5)

        # Override with user parameters
        if 'temperature' in kwargs:
            params['temperature'] = max(0.1, min(kwargs['temperature'], 0.8))  # Cap at 0.8 for legal

        return params

    def _post_process_legal_analysis(self,
                                     analysis: str,
                                     analysis_type: str,
                                     legal_area: str,
                                     kwargs: Dict[str, Any]) -> str:
        """Post-process legal analysis for formatting and disclaimers.

        Args:
            analysis: Raw legal analysis
            analysis_type: Type of analysis
            legal_area: Legal area
            kwargs: Generation parameters

        Returns:
            str: Post-processed legal analysis
        """
        if not analysis.strip():
            return analysis

        # Clean up common artifacts
        analysis = self._clean_legal_artifacts(analysis)

        # Apply analysis-specific formatting
        if analysis_type == 'risk_assessment':
            analysis = self._format_risk_assessment(analysis)
        elif analysis_type == 'contract_review':
            analysis = self._format_contract_review(analysis)
        elif analysis_type == 'compliance_review':
            analysis = self._format_compliance_review(analysis)
        elif analysis_type == 'due_diligence':
            analysis = self._format_due_diligence(analysis)

        # Add professional disclaimers
        analysis = self._add_legal_disclaimers(analysis, kwargs)

        return analysis.strip()

    def _clean_legal_artifacts(self, analysis: str) -> str:
        """Remove common legal analysis artifacts.

        Args:
            analysis: Raw analysis

        Returns:
            str: Cleaned analysis
        """
        # Remove common prefixes
        prefixes_to_remove = [
            "Legal analysis:",
            "Here's the legal analysis:",
            "Based on legal review:",
            "Legal assessment:"
        ]

        for prefix in prefixes_to_remove:
            if analysis.startswith(prefix):
                analysis = analysis[len(prefix):].strip()

        return analysis

    def _format_risk_assessment(self, analysis: str) -> str:
        """Format as risk assessment.

        Args:
            analysis: Analysis content

        Returns:
            str: Risk-formatted analysis
        """
        if not any(header in analysis for header in ['Risk', 'Likelihood', 'Impact', 'Mitigation']):
            analysis = f"**Legal Risk Assessment:**\n\n{analysis}"

        return analysis

    def _format_contract_review(self, analysis: str) -> str:
        """Format as contract review.

        Args:
            analysis: Analysis content

        Returns:
            str: Contract review formatted analysis
        """
        if not any(header in analysis for header in ['Terms', 'Obligations', 'Rights', 'Limitations']):
            analysis = f"**Contract Review Analysis:**\n\n{analysis}"

        return analysis

    def _format_compliance_review(self, analysis: str) -> str:
        """Format as compliance review.

        Args:
            analysis: Analysis content

        Returns:
            str: Compliance-formatted analysis
        """
        if 'compliance' not in analysis.lower():
            analysis = f"**Compliance Review:**\n\n{analysis}"

        return analysis

    def _format_due_diligence(self, analysis: str) -> str:
        """Format as due diligence review.

        Args:
            analysis: Analysis content

        Returns:
            str: Due diligence formatted analysis
        """
        if 'due diligence' not in analysis.lower():
            analysis = f"**Legal Due Diligence Review:**\n\n{analysis}"

        return analysis

    def _add_legal_disclaimers(self, analysis: str, kwargs: Dict[str, Any]) -> str:
        """Add appropriate legal disclaimers.

        Args:
            analysis: Legal analysis
            kwargs: Generation parameters

        Returns:
            str: Analysis with disclaimers
        """
        # Standard legal AI disclaimer
        disclaimer = ("\n\n**IMPORTANT DISCLAIMER:** This analysis is generated by an AI system "
                      "and is provided for informational purposes only. It does not constitute "
                      "legal advice and should not be relied upon for making legal decisions. "
                      "Please consult with a qualified attorney for specific legal guidance.")

        # Add jurisdiction disclaimer if relevant
        jurisdiction = kwargs.get('jurisdiction', 'general')
        if jurisdiction != 'general':
            disclaimer += (f" This analysis considers {jurisdiction} law but may not reflect "
                           "the most current legal developments.")

        return analysis + disclaimer

    async def analyze_contract(self,
                               contract_text: str,
                               contract_type: str = 'general',
                               focus_areas: Optional[List[str]] = None) -> Dict[str, Any]:
        """Perform comprehensive contract analysis.

        Args:
            contract_text: Text of the contract
            contract_type: Type of contract
            focus_areas: Specific areas to focus on

        Returns:
            Dict containing contract analysis
        """
        analysis_areas = focus_areas or [
            'key_terms',
            'obligations',
            'risks',
            'termination_provisions',
            'dispute_resolution'
        ]

        results = {}

        for area in analysis_areas:
            prompt = f"Analyze the {area} in this {contract_type} contract:\n\n{contract_text}"

            analysis = await self.generate(
                prompt,
                legal_area='contract_law',
                document_type='contract',
                analysis_type='contract_review',
                risk_focus=True
            )

            results[area] = analysis

        return {
            'contract_analysis': results,
            'contract_type': contract_type,
            'analysis_areas': analysis_areas,
            'analysis_method': 'ai_contract_review'
        }

    async def assess_legal_risk(self,
                                situation: str,
                                legal_area: str = 'general',
                                jurisdiction: str = 'general') -> str:
        """Assess legal risks in a given situation.

        Args:
            situation: Description of the situation
            legal_area: Relevant area of law
            jurisdiction: Applicable jurisdiction

        Returns:
            str: Risk assessment analysis
        """
        risk_prompt = f"""
        Assess the legal risks in the following situation:

        {situation}

        Provide:
        1. Identification of potential legal issues
        2. Risk level assessment (high/medium/low)
        3. Potential consequences
        4. Recommended mitigation strategies
        5. Suggested next steps
        """

        return await self.generate(
            risk_prompt,
            legal_area=legal_area,
            analysis_type='risk_assessment',
            jurisdiction=jurisdiction,
            risk_focus=True
        )

    async def warmup(self) -> bool:
        """Warm up with a legal analysis example."""
        if not self.is_loaded():
            return False

        try:
            warmup_query = "Analyze the key considerations in a standard employment agreement"
            await self.generate(
                warmup_query,
                legal_area='employment_law',
                analysis_type='summary',
                max_tokens=200
            )
            logger.info(f"Warmup completed for {self.category}")
            return True
        except Exception as e:
            logger.error(f"Warmup failed for {self.category}: {e}")
            return False

    def get_legal_areas(self) -> List[str]:
        """Get available legal practice areas.

        Returns:
            List[str]: Available legal areas
        """
        return list(self.legal_areas.keys())

    def get_document_types(self) -> List[str]:
        """Get available document types.

        Returns:
            List[str]: Available document types
        """
        return list(self.document_types.keys())

    def get_analysis_types(self) -> List[str]:
        """Get available analysis types.

        Returns:
            List[str]: Available analysis types
        """
        return list(self.analysis_types.keys())

    def get_legal_area_info(self, legal_area: str) -> Dict[str, Any]:
        """Get information about a legal practice area.

        Args:
            legal_area: Legal practice area

        Returns:
            Dict containing area info
        """
        return {
            'area': legal_area,
            'description': self.legal_areas.get(legal_area, 'General legal practice'),
            'common_documents': ['contract', 'brief', 'opinion'],
            'typical_analyses': ['risk_assessment', 'compliance_review', 'summary']
        }