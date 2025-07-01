import asyncio
import logging
from typing import Dict, Any, Optional, AsyncGenerator
from base_llm_engine import BaseLLMEngine
from model_loader import ModelLoader
from utils.persona_loader import PersonaLoader

logger = logging.getLogger(__name__)


class ConversationalChatEngine(BaseLLMEngine):
    """Engine for conversational chat interactions."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_loader = ModelLoader()
        self.persona_loader = PersonaLoader()
        self.conversation_history: Dict[str, list] = {}  # session_id -> messages
        self.max_history_length = 10  # Keep last 10 exchanges

    async def load_model(self) -> bool:
        """Load the conversational model."""
        try:
            logger.info(f"Loading conversational model: {self.technical_model_name}")
            self.model, self.tokenizer = await self.model_loader.load_model(
                self.technical_model_name,
                self.precision
            )

            if self.model is not None and self.tokenizer is not None:
                self.is_model_loaded = True
                self.load_time = asyncio.get_event_loop().time()
                logger.info(f"Successfully loaded conversational model")

                # Perform warmup
                await self.warmup()
                return True
            else:
                logger.error("Failed to load conversational model")
                return False

        except Exception as e:
            logger.error(f"Error loading conversational model: {e}")
            return False

    async def unload_model(self) -> bool:
        """Unload the conversational model."""
        try:
            if self.is_model_loaded:
                success = await self.model_loader.unload_model(self.technical_model_name)
                if success:
                    self.model = None
                    self.tokenizer = None
                    self.is_model_loaded = False
                    logger.info("Conversational model unloaded")
                return success
            return True
        except Exception as e:
            logger.error(f"Error unloading conversational model: {e}")
            return False

    async def generate(self, prompt: str, session_id: Optional[str] = None, **kwargs) -> str:
        """Generate conversational response.

        Args:
            prompt: User input
            session_id: Optional session ID for conversation history
            **kwargs: Additional generation parameters

        Returns:
            str: Generated response
        """
        if not self.is_model_loaded:
            raise RuntimeError("Conversational model not loaded")

        try:
            # Build conversation context
            conversation_prompt = self._build_conversation_prompt(prompt, session_id)

            # Get generation parameters
            gen_params = self._validate_generation_params(kwargs)

            # Tokenize input
            inputs = self.tokenizer(
                conversation_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096 - gen_params['max_new_tokens']  # Leave room for generation
            )

            # Move to device
            if hasattr(self.model, 'device'):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Generate response
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

            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = self._extract_response(full_response, conversation_prompt)

            # Update conversation history
            if session_id:
                self._update_conversation_history(session_id, prompt, response)

            self.increment_generation_count()

            logger.debug(f"Generated conversational response ({len(response)} chars)")
            return response

        except Exception as e:
            logger.error(f"Error generating conversational response: {e}")
            raise

    async def generate_stream(self, prompt: str, session_id: Optional[str] = None, **kwargs) -> AsyncGenerator[
        str, None]:
        """Generate streaming conversational response.

        Args:
            prompt: User input
            session_id: Optional session ID for conversation history
            **kwargs: Additional generation parameters

        Yields:
            str: Response chunks
        """
        if not self.is_model_loaded:
            raise RuntimeError("Conversational model not loaded")

        try:
            from transformers import TextIteratorStreamer
            import torch
            from threading import Thread

            # Build conversation context
            conversation_prompt = self._build_conversation_prompt(prompt, session_id)

            # Get generation parameters
            gen_params = self._validate_generation_params(kwargs)

            # Tokenize input
            inputs = self.tokenizer(
                conversation_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096 - gen_params['max_new_tokens']
            )

            # Move to device
            if hasattr(self.model, 'device'):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Setup streamer
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )

            # Generation parameters for streaming
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

            # Start generation in thread
            generation_thread = Thread(
                target=self.model.generate,
                kwargs=generation_kwargs
            )
            generation_thread.start()

            # Stream response
            full_response = ""
            for chunk in streamer:
                full_response += chunk
                yield chunk

            # Wait for generation to complete
            generation_thread.join()

            # Update conversation history
            if session_id:
                self._update_conversation_history(session_id, prompt, full_response)

            self.increment_generation_count()

        except Exception as e:
            logger.error(f"Error in streaming generation: {e}")
            yield f"Error: {str(e)}"

    def get_system_prompt(self) -> Optional[str]:
        """Get the conversational system prompt."""
        return self.persona_loader.load_conversational_persona()

    def _build_conversation_prompt(self, current_prompt: str, session_id: Optional[str] = None) -> str:
        """Build prompt with conversation history and system prompt.

        Args:
            current_prompt: Current user input
            session_id: Session ID for history

        Returns:
            str: Full conversation prompt
        """
        system_prompt = self.get_system_prompt()

        # Start with system prompt if available
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n"
        else:
            full_prompt = ""

        # Add conversation history if available
        if session_id and session_id in self.conversation_history:
            history = self.conversation_history[session_id]
            for exchange in history[-self.max_history_length:]:  # Only recent history
                full_prompt += f"User: {exchange['user']}\nAssistant: {exchange['assistant']}\n\n"

        # Add current prompt
        full_prompt += f"User: {current_prompt}\nAssistant:"

        return full_prompt

    def _update_conversation_history(self, session_id: str, user_input: str, assistant_response: str):
        """Update conversation history for a session.

        Args:
            session_id: Session identifier
            user_input: User's input
            assistant_response: Assistant's response
        """
        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = []

        self.conversation_history[session_id].append({
            'user': user_input,
            'assistant': assistant_response,
            'timestamp': asyncio.get_event_loop().time()
        })

        # Trim history if too long
        if len(self.conversation_history[session_id]) > self.max_history_length * 2:
            # Keep only recent exchanges
            self.conversation_history[session_id] = self.conversation_history[session_id][-self.max_history_length:]

    def _get_default_generation_params(self) -> Dict[str, Any]:
        """Get default generation parameters for conversational chat."""
        return {
            'max_new_tokens': 512,
            'temperature': 0.7,
            'do_sample': True,
            'top_p': 0.9,
            'repetition_penalty': 1.1,
            'pad_token_id': None
        }

    def clear_conversation_history(self, session_id: Optional[str] = None):
        """Clear conversation history.

        Args:
            session_id: Specific session to clear, or None to clear all
        """
        if session_id:
            if session_id in self.conversation_history:
                del self.conversation_history[session_id]
                logger.info(f"Cleared conversation history for session {session_id}")
        else:
            self.conversation_history.clear()
            logger.info("Cleared all conversation history")

    def get_conversation_history(self, session_id: str) -> list:
        """Get conversation history for a session.

        Args:
            session_id: Session identifier

        Returns:
            list: Conversation history
        """
        return self.conversation_history.get(session_id, [])

    def get_active_sessions(self) -> list:
        """Get list of active session IDs.

        Returns:
            list: List of session IDs
        """
        return list(self.conversation_history.keys())

    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics about conversation sessions.

        Returns:
            Dict containing session statistics
        """
        if not self.conversation_history:
            return {
                'active_sessions': 0,
                'total_exchanges': 0,
                'average_exchanges_per_session': 0
            }

        total_exchanges = sum(len(history) for history in self.conversation_history.values())

        return {
            'active_sessions': len(self.conversation_history),
            'total_exchanges': total_exchanges,
            'average_exchanges_per_session': total_exchanges / len(self.conversation_history)
        }