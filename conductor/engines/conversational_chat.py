import logging
from typing import Dict, Any, Optional, List
from conductor.engines.base_engine import BaseEngine
from conductor.model_loader import ModelLoader

logger = logging.getLogger(__name__)


class ConversationalChatEngine(BaseEngine):
    """Conversational chat engine for natural dialogue."""
    
    def __init__(self, config: Dict[str, Any], model_loader: ModelLoader, persona: str = ""):
        super().__init__(config, model_loader, persona)
        self.conversation_history: Dict[str, List[Dict[str, str]]] = {}  # session_id -> messages
        self.max_history_length = 10  # Keep last 10 exchanges

    def get_system_prompt(self) -> Optional[str]:
        """Get system prompt for conversational chat."""
        if self.persona:
            return self.persona
        return "You are a helpful, conversational AI assistant. Engage in natural dialogue while being informative, friendly, and helpful. Maintain context from previous messages in the conversation."

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate conversational response with history context."""
        session_id = kwargs.get('session_id', 'default')
        
        # Build conversation prompt with history
        conversation_prompt = self._build_conversation_prompt(prompt, session_id)
        
        # Use parent's generate method
        response = await super().generate(conversation_prompt, **kwargs)
        
        # Update conversation history
        self._update_conversation_history(session_id, prompt, response)
        
        return response

    def _build_conversation_prompt(self, user_message: str, session_id: str = 'default') -> str:
        """Build conversation prompt with history context."""
        system_prompt = self.get_system_prompt()
        
        prompt_parts: List[str] = []
        
        if system_prompt:
            prompt_parts.append(system_prompt)
        
        # Add conversation history
        history = self.conversation_history.get(session_id, [])
        if history:
            prompt_parts.append("Previous conversation:")
            for exchange in history[-self.max_history_length:]:
                prompt_parts.append(f"User: {exchange['user']}")
                prompt_parts.append(f"Assistant: {exchange['assistant']}")
        
        # Add current message
        prompt_parts.append(f"User: {user_message}")
        prompt_parts.append("Assistant:")
        
        return "\n".join(prompt_parts)

    def _update_conversation_history(self, session_id: str, user_message: str, assistant_response: str) -> None:
        """Update conversation history for session."""
        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = []
        
        self.conversation_history[session_id].append({
            'user': user_message,
            'assistant': assistant_response
        })
        
        # Trim history if too long
        if len(self.conversation_history[session_id]) > self.max_history_length:
            self.conversation_history[session_id] = self.conversation_history[session_id][-self.max_history_length:]

    def clear_conversation_history(self, session_id: Optional[str] = None) -> None:
        """Clear conversation history for specific session or all sessions."""
        if session_id:
            self.conversation_history.pop(session_id, None)
            logger.info(f"Cleared conversation history for session: {session_id}")
        else:
            self.conversation_history.clear()
            logger.info("Cleared all conversation history")

    def get_conversation_history(self, session_id: str) -> List[Dict[str, str]]:
        """Get conversation history for a session."""
        return self.conversation_history.get(session_id, [])

    async def chat(self, message: str, session_id: str = 'default', **kwargs: Any) -> str:
        """Convenience method for chat interaction."""
        return await self.generate(message, session_id=session_id, **kwargs)

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
