import asyncio
import logging
from typing import Dict, Any, Optional, AsyncGenerator, List, Tuple
from base_llm_engine import BaseLLMEngine
from model_loader import ModelLoader
from utils.persona_loader import PersonaLoader

logger = logging.getLogger(__name__)


class TranslationEngine(BaseLLMEngine):
    """Engine specialized for multilingual translation using NLLB-200."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_loader = ModelLoader()
        self.persona_loader = PersonaLoader()

        # Common language codes for NLLB-200
        self.language_codes = {
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
        self.code_to_language = {v: k for k, v in self.language_codes.items()}

    async def load_model(self) -> bool:
        """Load the translation model."""
        try:
            logger.info(f"Loading translation model: {self.technical_model_name}")
            self.model, self.tokenizer = await self.model_loader.load_model(
                self.technical_model_name,
                self.precision
            )

            if self.model is not None and self.tokenizer is not None:
                self.is_model_loaded = True
                self.load_time = asyncio.get_event_loop().time()
                logger.info(f"Successfully loaded translation model")

                # Perform warmup
                await self.warmup()
                return True
            else:
                logger.error("Failed to load translation model")
                return False

        except Exception as e:
            logger.error(f"Error loading translation model: {e}")
            return False

    async def unload_model(self) -> bool:
        """Unload the translation model."""
        try:
            if self.is_model_loaded:
                success = await self.model_loader.unload_model(self.technical_model_name)
                if success:
                    self.model = None
                    self.tokenizer = None
                    self.is_model_loaded = False
                    logger.info("Translation model unloaded")
                return success
            return True
        except Exception as e:
            logger.error(f"Error unloading translation model: {e}")
            return False

    async def generate(self, prompt: str, **kwargs) -> str:
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
        if not self.is_model_loaded:
            raise RuntimeError("Translation model not loaded")

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

            # For NLLB models, we need to set the target language
            if hasattr(self.tokenizer, 'src_lang'):
                self.tokenizer.src_lang = source_code
            if hasattr(self.tokenizer, 'tgt_lang'):
                self.tokenizer.tgt_lang = target_code

            # Build translation prompt
            translation_prompt = self._build_translation_prompt(
                prompt, source_lang, target_lang, formal_register
            )

            # Get generation parameters
            gen_params = self._get_translation_params(kwargs)

            # Tokenize input
            inputs = self.tokenizer(
                translation_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            )

            # Move to device
            if hasattr(self.model, 'device'):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Generate translation
            import torch
            with torch.no_grad():
                # For NLLB models, we might need to use generate with forced_bos_token_id
                if hasattr(self.tokenizer, 'lang_code_to_id') and target_code in self.tokenizer.lang_code_to_id:
                    forced_bos_token_id = self.tokenizer.lang_code_to_id[target_code]
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=gen_params['max_new_tokens'],
                        temperature=gen_params['temperature'],
                        do_sample=gen_params['do_sample'],
                        top_p=gen_params['top_p'],
                        forced_bos_token_id=forced_bos_token_id,
                        pad_token_id=gen_params['pad_token_id'],
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                else:
                    # Fallback for non-NLLB models
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=gen_params['max_new_tokens'],
                        temperature=gen_params['temperature'],
                        do_sample=gen_params['do_sample'],
                        top_p=gen_params['top_p'],
                        pad_token_id=gen_params['pad_token_id'],
                        eos_token_id=self.tokenizer.eos_token_id
                    )

            # Decode translation
            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            translation = self._extract_translation(full_output, translation_prompt)

            # Post-process translation
            processed_translation = self._post_process_translation(
                translation, preserve_formatting, kwargs
            )

            self.increment_generation_count()

            logger.debug(f"Generated translation: {source_lang} -> {target_lang}")
            return processed_translation

        except Exception as e:
            logger.error(f"Error generating translation: {e}")
            raise

    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Generate streaming translation.

        Args:
            prompt: Text to translate
            **kwargs: Additional parameters

        Yields:
            str: Translation chunks
        """
        if not self.is_model_loaded:
            raise RuntimeError("Translation model not loaded")

        try:
            from transformers import TextIteratorStreamer
            import torch
            from threading import Thread

            source_lang = kwargs.get('source_lang', 'english')
            target_lang = kwargs.get('target_lang', 'spanish')

            source_code = self._get_language_code(source_lang)
            target_code = self._get_language_code(target_lang)

            if hasattr(self.tokenizer, 'src_lang'):
                self.tokenizer.src_lang = source_code
            if hasattr(self.tokenizer, 'tgt_lang'):
                self.tokenizer.tgt_lang = target_code

            translation_prompt = self._build_translation_prompt(
                prompt, source_lang, target_lang, kwargs.get('formal_register', False)
            )

            gen_params = self._get_translation_params(kwargs)

            inputs = self.tokenizer(
                translation_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024
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
                'pad_token_id': gen_params['pad_token_id'],
                'eos_token_id': self.tokenizer.eos_token_id,
                'streamer': streamer
            }

            # Add forced_bos_token_id for NLLB if available
            if hasattr(self.tokenizer, 'lang_code_to_id') and target_code in self.tokenizer.lang_code_to_id:
                generation_kwargs['forced_bos_token_id'] = self.tokenizer.lang_code_to_id[target_code]

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
            logger.error(f"Error in streaming translation: {e}")
            yield f"Error: {str(e)}"

    def get_system_prompt(self) -> Optional[str]:
        """Get system prompt for translation."""
        return self.persona_loader.get_persona_for_category('translation')

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
            prompt_parts = []

            if system_prompt:
                prompt_parts.append(system_prompt)

            register_instruction = "formal" if formal_register else "natural"

            instructions = [
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
        # Parameters optimized for translation
        params = {
            'max_new_tokens': kwargs.get('max_tokens', 512),
            'temperature': 0.3,  # Lower temperature for more consistent translations
            'do_sample': True,
            'top_p': 0.9,
            'repetition_penalty': 1.1,
            'pad_token_id': self.tokenizer.eos_token_id if self.tokenizer else None
        }

        # Adjust for different types of content
        if kwargs.get('creative_content', False):
            params['temperature'] = 0.5  # Higher for creative content
        elif kwargs.get('technical_content', False):
            params['temperature'] = 0.2  # Lower for technical accuracy

        return params

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
            # For instruction-following models, extract after the prompt
            translation = self._extract_response(full_output, original_prompt)
            return translation

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
        prefixes_to_remove = [
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
        common_pairs = [
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
        name = self.code_to_language.get(code, language)

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
        families = {
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