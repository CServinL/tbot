import logging
from typing import Dict, Any, Optional, AsyncGenerator, List
from conductor.engines.base_engine import BaseEngine
from conductor.model_loader import ModelLoader

logger = logging.getLogger(__name__)


class TranslationEngine(BaseEngine):
    def __init__(self, config: Dict[str, Any], model_loader: ModelLoader, persona: str = ""):
        super().__init__(config, model_loader, persona)

        # Common language codes for NLLB-200
        self.language_codes: Dict[str, str] = {
            'english': 'eng_Latn',
            'spanish': 'spa_Latn',
            'french': 'fra_Latn',
            'german': 'deu_Latn',
            'italian': 'ita_Latn',
            'portuguese': 'por_Latn',
            'russian': 'rus_Cyrl',
            'chinese': 'zho_Hans',
            'japanese': 'jpn_Jpan',
            'korean': 'kor_Hang',
            'arabic': 'arb_Arab',
            'hindi': 'hin_Deva',
            'turkish': 'tur_Latn',
            'dutch': 'nld_Latn',
            'swedish': 'swe_Latn',
            'polish': 'pol_Latn',
            'czech': 'ces_Latn',
            'hungarian': 'hun_Latn',
            'romanian': 'ron_Latn',
            'bulgarian': 'bul_Cyrl',
            'greek': 'ell_Grek',
            'hebrew': 'heb_Hebr',
            'thai': 'tha_Thai',
            'vietnamese': 'vie_Latn',
            'indonesian': 'ind_Latn',
            'malay': 'zsm_Latn',
            'tagalog': 'tgl_Latn',
            'swahili': 'swh_Latn',
            'yoruba': 'yor_Latn',
            'hausa': 'hau_Latn'
        }

        # Reverse mapping for code to language name
        self.code_to_language: Dict[str, str] = {v: k for k, v in self.language_codes.items()}

    def _get_default_generation_params(self) -> Dict[str, Any]:
        """Get default generation parameters optimized for translation."""
        params = super()._get_default_generation_params()
        # Override defaults for translation
        params.update({
            'max_new_tokens': 512,  # Reasonable for most translations
            'temperature': 0.3,     # Lower temperature for consistent translations
            'top_p': 0.9,          # Slightly higher for language diversity
            'repetition_penalty': 1.1  # Prevent repetitive translations
        })
        return params

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate translation.

        Args:
            prompt: Text to translate
            **kwargs: Additional parameters
                - source_lang: Source language (e.g., 'english', 'spa_Latn')
                - target_lang: Target language (e.g., 'spanish', 'fra_Latn')
                - preserve_formatting: Whether to preserve formatting
                - formal_register: Whether to use formal language register

        Returns:
            str: Translated text
        """
        try:
            # Parse translation parameters
            source_lang = kwargs.get('source_lang', 'english')
            target_lang = kwargs.get('target_lang', 'spanish')
            preserve_formatting = kwargs.get('preserve_formatting', True)
            formal_register = kwargs.get('formal_register', False)

            # Convert language names to codes if needed
            source_code = self._get_language_code(source_lang)
            target_code = self._get_language_code(target_lang)

            if not source_code or not target_code:
                raise ValueError(f"Unsupported language pair: {source_lang} -> {target_lang}")

            # Set language codes for NLLB models
            self._configure_translation_languages(source_code, target_code)

            # Build translation prompt
            translation_prompt = self._build_translation_prompt(
                prompt, source_lang, target_lang, formal_register
            )

            # Get generation parameters optimized for translation
            gen_params = self._get_translation_params(kwargs)

            # Use parent's generate method with the built prompt and parameters
            response = await super().generate(translation_prompt, **gen_params)

            # Post-process the translation
            processed_translation = self._post_process_translation(
                response, preserve_formatting, kwargs
            )

            logger.debug(f"Generated translation: {source_lang} -> {target_lang}")
            return processed_translation

        except Exception as e:
            logger.error(f"Error generating translation: {e}")
            raise

    async def generate_stream(self, prompt: str, **kwargs: Any) -> AsyncGenerator[str, None]:
        """Generate streaming translation.

        Args:
            prompt: Text to translate
            **kwargs: Additional parameters

        Yields:
            str: Translation chunks
        """
        # Parse translation parameters
        source_lang = kwargs.get('source_lang', 'english')
        target_lang = kwargs.get('target_lang', 'spanish')
        preserve_formatting = kwargs.get('preserve_formatting', True)
        formal_register = kwargs.get('formal_register', False)

        # Convert language names to codes if needed
        source_code = self._get_language_code(source_lang)
        target_code = self._get_language_code(target_lang)

        if not source_code or not target_code:
            yield f"Error: Unsupported language pair: {source_lang} -> {target_lang}"
            return

        # Set language codes for NLLB models
        self._configure_translation_languages(source_code, target_code)

        # Build translation prompt
        translation_prompt = self._build_translation_prompt(
            prompt, source_lang, target_lang, formal_register
        )

        # Get generation parameters optimized for translation
        gen_params = self._get_translation_params(kwargs)

        # Since base class doesn't have generate_stream, use regular generate
        try:
            result = await super().generate(translation_prompt, **gen_params)
            processed_result = self._post_process_translation(
                result, preserve_formatting, kwargs
            )
            yield processed_result
        except Exception as e:
            logger.error(f"Error in streaming translation: {e}")
            yield f"Error: {str(e)}"

    def get_system_prompt(self) -> Optional[str]:
        """Get system prompt for translation."""
        # Use the persona from BaseEngine if available
        if self.persona:
            return self.persona
        
        # Default translation system prompt
        return ("You are a professional translator. Translate the text accurately "
                "while preserving the meaning, tone, and style of the original. "
                "Provide only the translation without additional explanations.")

    def _configure_translation_languages(self, source_code: str, target_code: str) -> None:
        """Configure tokenizer language codes for NLLB models.
        
        Args:
            source_code: Source language code
            target_code: Target language code
        """
        if self.tokenizer is None:
            return
            
        # For NLLB models, we need to set the target language
        if hasattr(self.tokenizer, 'src_lang'):
            self.tokenizer.src_lang = source_code  # type: ignore
        if hasattr(self.tokenizer, 'tgt_lang'):
            self.tokenizer.tgt_lang = target_code  # type: ignore

    def _get_language_code(self, language: str) -> Optional[str]:
        """Get language code from language name or return if already a code.

        Args:
            language: Language name or code

        Returns:
            Optional[str]: Language code or None if not found
        """
        # If it's already a code format (contains underscore)
        if '_' in language and language in self.code_to_language:
            return language

        # Try to find by name
        language_lower = language.lower()
        return self.language_codes.get(language_lower)

    def _build_translation_prompt(self,
                                  text: str,
                                  source_lang: str,
                                  target_lang: str,
                                  formal_register: bool) -> str:
        """Build prompt for translation.

        Args:
            text: Text to translate
            source_lang: Source language
            target_lang: Target language
            formal_register: Whether to use formal register

        Returns:
            str: Translation prompt
        """
        system_prompt = self.get_system_prompt()

        # For NLLB models, often the input text is used directly
        # But for instruction-following models, we need a structured prompt
        if 'nllb' in self.technical_model_name.lower():
            return text
        else:
            # Build structured prompt for instruction-following models
            prompt_parts: List[str] = []

            if system_prompt:
                prompt_parts.append(system_prompt)

            register_instruction = "formal" if formal_register else "natural"

            instructions: List[str] = [
                f"Translate the following text from {source_lang} to {target_lang}.",
                f"Use {register_instruction} language register.",
                "Preserve the meaning and tone of the original text.",
                "Maintain any formatting or structure present in the original."
            ]

            prompt_parts.append("\n".join(instructions))
            prompt_parts.append(f"Text to translate: {text}")
            prompt_parts.append(f"Translation to {target_lang}:")

            return "\n\n".join(prompt_parts)

    def _get_translation_params(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Get generation parameters for translation.

        Args:
            kwargs: User parameters

        Returns:
            Dict[str, Any]: Generation parameters
        """
        # Use base class parameter validation with translation-specific overrides
        translation_kwargs = kwargs.copy()
        
        # Set default temperature for translation if not provided
        if 'temperature' not in translation_kwargs:
            translation_kwargs['temperature'] = 0.3  # Lower for consistent translations
            
        # Adjust for different types of content
        if kwargs.get('creative_content', False):
            translation_kwargs['temperature'] = 0.5  # Higher for creative content
        elif kwargs.get('technical_content', False):
            translation_kwargs['temperature'] = 0.2  # Lower for technical accuracy

        # Use base class validation which handles clamping and safety limits
        return self._validate_generation_params(translation_kwargs)

    def _extract_translation(self, full_output: str, original_prompt: str) -> str:
        """Extract translation from full output.

        Args:
            full_output: Full model output
            original_prompt: Original prompt

        Returns:
            str: Extracted translation
        """
        # For NLLB models, the output is typically just the translation
        if 'nllb' in self.technical_model_name.lower():
            return full_output.strip()
        else:
            # For instruction-following models, use base class extraction
            return self._extract_response(full_output, original_prompt)

    def _post_process_translation(self,
                                  translation: str,
                                  preserve_formatting: bool,
                                  kwargs: Dict[str, Any]) -> str:
        """Post-process translation for quality and formatting.

        Args:
            translation: Raw translation
            preserve_formatting: Whether to preserve formatting
            kwargs: Additional parameters

        Returns:
            str: Post-processed translation
        """
        if not translation.strip():
            return translation

        # Basic cleanup
        translation = translation.strip()

        # Remove common translation artifacts
        translation = self._remove_translation_artifacts(translation)

        # Preserve formatting if requested
        if preserve_formatting:
            # This is a basic implementation - in practice you'd want more sophisticated formatting preservation
            pass

        return translation

    def _remove_translation_artifacts(self, translation: str) -> str:
        """Remove common translation artifacts.

        Args:
            translation: Raw translation

        Returns:
            str: Cleaned translation
        """
        # Remove common prefixes that models sometimes add
        prefixes_to_remove: List[str] = [
            "Translation:",
            "Here's the translation:",
            "The translation is:",
            "Translated text:"
        ]

        for prefix in prefixes_to_remove:
            if translation.startswith(prefix):
                translation = translation[len(prefix):].strip()

        return translation

    async def detect_language(self, text: str) -> Optional[str]:
        """Detect the language of input text.

        Args:
            text: Text to analyze

        Returns:
            Optional[str]: Detected language code or None
        """
        # This is a placeholder - in a full implementation you'd integrate
        # with a language detection library like langdetect or polyglot

        # Simple heuristic based on character patterns
        if any(ord(char) > 127 for char in text):
            # Contains non-ASCII characters
            if any('\u4e00' <= char <= '\u9fff' for char in text):
                return 'zho_Hans'  # Chinese
            elif any('\u3040' <= char <= '\u309f' for char in text):
                return 'jpn_Jpan'  # Japanese
            elif any('\uac00' <= char <= '\ud7af' for char in text):
                return 'kor_Hang'  # Korean
            elif any('\u0600' <= char <= '\u06ff' for char in text):
                return 'arb_Arab'  # Arabic
            elif any('\u0400' <= char <= '\u04ff' for char in text):
                return 'rus_Cyrl'  # Russian

        # Default to English for ASCII text
        return 'eng_Latn'

    async def warmup(self) -> bool:
        """Warm up with a translation example."""
        if not self.is_loaded():
            return False

        try:
            warmup_text = "Hello, how are you?"
            await self.generate(
                warmup_text,
                source_lang='english',
                target_lang='spanish',
                max_tokens=50
            )
            logger.info(f"Warmup completed for {self.category}")
            return True
        except Exception as e:
            logger.error(f"Warmup failed for {self.category}: {e}")
            return False

    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages.

        Returns:
            List[str]: Supported language names
        """
        return list(self.language_codes.keys())

    def get_language_pairs(self) -> List[Dict[str, str]]:
        """Get common language pairs for translation.

        Returns:
            List[Dict]: Common translation pairs
        """
        common_pairs: List[Dict[str, str]] = [
            {'source': 'english', 'target': 'spanish'},
            {'source': 'english', 'target': 'french'},
            {'source': 'english', 'target': 'german'},
            {'source': 'english', 'target': 'chinese'},
            {'source': 'spanish', 'target': 'english'},
            {'source': 'french', 'target': 'english'},
            {'source': 'german', 'target': 'english'},
            {'source': 'chinese', 'target': 'english'},
        ]

        return common_pairs

    def get_language_info(self, language: str) -> Dict[str, Any]:
        """Get information about a specific language.

        Args:
            language: Language name or code

        Returns:
            Dict containing language info
        """
        code = self._get_language_code(language)
        name = self.code_to_language.get(code or "", language)

        return {
            'name': name,
            'code': code,
            'supported': code is not None,
            'family': self._get_language_family(code) if code else None
        }

    def _get_language_family(self, language_code: str) -> str:
        """Get language family for a language code.

        Args:
            language_code: Language code

        Returns:
            str: Language family
        """
        families: Dict[str, str] = {
            'Latn': 'Latin script',
            'Cyrl': 'Cyrillic script',
            'Arab': 'Arabic script',
            'Deva': 'Devanagari script',
            'Hans': 'Chinese simplified',
            'Jpan': 'Japanese',
            'Hang': 'Korean',
            'Thai': 'Thai script',
            'Hebr': 'Hebrew script',
            'Grek': 'Greek script'
        }

        for script, family in families.items():
            if script in language_code:
                return family

        return 'Unknown'
