#!/usr/bin/env python3
"""
Prompt Classification Module

Handles intelligent classification of user prompts to route them to appropriate
specialized engines based on keyword matching and AI fallback classification.
"""

import logging
from typing import Optional, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from conductor.engines.base_engine import BaseEngine

logger = logging.getLogger(__name__)


class PromptClassificator:
    """
    Classifies user prompts to determine the most appropriate engine to handle them.
    
    Uses a hierarchical approach:
    1. Keyword-based classification (fast, reliable for common patterns)
    2. AI-based classification fallback (for complex/ambiguous cases)
    3. General reasoning fallback (last resort)
    """
    
    def __init__(self, available_categories: list[str]):
        """
        Initialize the classificator with available engine categories.
        
        Args:
            available_categories: List of available engine categories
        """
        self.available_categories: set[str] = set(available_categories)
        
        # Define keyword patterns for each category with priority ordering
        self.keyword_patterns: dict[str, dict[str, Any]] = {
            # Highest priority: Code-related
            'code_completion': {
                'keywords': ['function', 'code', 'class', 'program', 'algorithm', 'script', 'def ', 'import ', 'const ', 'var '],
                'required_keywords': ['complete', 'finish', 'continue'],
                'priority': 1
            },
            'code_generation': {
                'keywords': [
                    'function', 'code', 'class', 'program', 'algorithm', 'script', 
                    'def ', 'import ', 'const ', 'var ', 'write a function', 'write code', 
                    'create a program', 'sql query', 'query', 'database', 'select', 'insert', 
                    'update', 'delete', 'create table', 'html', 'css', 'javascript', 'python',
                    'java', 'c++', 'typescript', 'react', 'vue', 'angular'
                ],
                'exclude_keywords': ['complete', 'finish', 'continue', 'story', 'poem', 'creative', 'fiction', 'novel'],
                'priority': 7  # Lower priority to avoid conflicts with creative writing and code review
            },
            
            # High priority: Code review (moved up to have higher priority than code generation)
            'code_review_debugging': {
                'keywords': ['review', 'check', 'debug', 'error', 'bug', 'fix'],
                'priority': 2
            },
            'code_review': {  # Alias for code_review_debugging
                'keywords': ['review', 'check', 'debug', 'error', 'bug', 'fix'],
                'priority': 2
            },
            
            # High priority: Creative writing (moved up to have higher priority than code generation)
            'creative_writing': {
                'keywords': ['write a story', 'write a poem', 'story about', 'poem about', 'creative writing', 'fiction', 'novel'],
                'exclude_keywords': ['who are', 'what are', 'tell me about'],  # Exclude question patterns
                'priority': 3
            },
            
            # High priority: Translation
            'translation': {
                'keywords': ['translate', 'translation', 'from english to', 'to spanish', 'language'],
                'priority': 4
            },
            
            # High priority: Question answering
            'question_answering': {
                'keywords': [
                    'what is', 'what are', 'what was', 'what were',
                    'where is', 'where are', 'where was', 'where were', 'where in', 'where can',
                    'when is', 'when was', 'when were', 'when did',
                    'who is', 'who was', 'who were', 'who are', 'who invented', 'who created', 'who discovered',
                    'how is', 'how does', 'how did', 'how can', 'how do',
                    'why is', 'why does', 'why did', 'why do',
                    'explain', 'tell me about', 'can you tell me', 'do you know'
                ],
                'priority': 5
            },
            
            # High priority: Summarization
            'summarization': {
                'keywords': ['summarize', 'summary', 'tldr', 'brief', 'overview'],
                'priority': 6
            },
            
            # Medium priority: Math (only if no code keywords)
            'mathematical_reasoning': {
                'keywords': ['calculate', 'math', 'equation', '+', '-', '*', '/', '=', 'solve'],
                'required_keywords': ['+', '-', '*', '/', 'calculate', 'solve', 'equation'],
                'priority': 8
            },
            
            # High priority: Image generation
            'image_generation': {
                'keywords': ['generate an image', 'create an image', 'draw', 'picture of', 'image of', 'photo of', 'artwork', 'illustration', 'render', 'generate image', 'create image', 'make an image', 'produce an image', 'render an image', 'illustrate', 'visualize'],
                'priority': 3
            },
            
            # Medium priority: Conversational patterns
            'conversational_chat': {
                'keywords': ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'how are you', 'thanks', 'thank you', 'please', 'can you help'],
                'priority': 9
            }
        }

    async def classify_prompt(self, prompt: str, reasoning_engine: Optional["BaseEngine"] = None) -> str:
        """
        Classify a user prompt to determine the appropriate engine category.
        
        Args:
            prompt: The user prompt to classify
            reasoning_engine: Optional reasoning engine for AI fallback classification
            
        Returns:
            The category name of the most appropriate engine
        """
        prompt_lower = prompt.lower()
        
        # Primary: Keyword-based classification with priority order
        classified_category = self._classify_by_keywords(prompt_lower)
        if classified_category:
            logger.info(f"Classified prompt as: {classified_category} (keyword-based)")
            return classified_category
        
        # Secondary: Try AI classification for complex cases
        if reasoning_engine:
            ai_category = await self._classify_by_ai(prompt, reasoning_engine)
            if ai_category:
                logger.info(f"Classified prompt as: {ai_category} (AI-based)")
                return ai_category
        
        # Fallback: use general_reasoning for unclassified prompts
        logger.warning("Classification failed, falling back to general_reasoning")
        if "general_reasoning" in self.available_categories:
            return "general_reasoning"
        
        # Last resort: use the first available engine
        if self.available_categories:
            fallback_category = next(iter(self.available_categories))
            logger.warning(f"Using fallback engine: {fallback_category}")
            return fallback_category

        raise RuntimeError("Prompt classification failed: no suitable engine could be determined.")

    def _classify_by_keywords(self, prompt_lower: str) -> Optional[str]:
        """
        Classify prompt using keyword matching with priority ordering.
        
        Args:
            prompt_lower: Lowercase version of the prompt
            
        Returns:
            Category name if matched, None otherwise
        """
        # Sort categories by priority
        sorted_categories: list[tuple[str, dict[str, Any]]] = sorted(
            self.keyword_patterns.items(), 
            key=lambda item: item[1]['priority']
        )
        
        for category, config in sorted_categories:
            # Skip if category is not available
            if category not in self.available_categories:
                continue
                
            keywords: list[str] = config['keywords']
            required_keywords: list[str] = config.get('required_keywords', [])
            exclude_keywords: list[str] = config.get('exclude_keywords', [])
            
            # Check if any primary keywords match
            if not any(keyword in prompt_lower for keyword in keywords):
                continue
            
            # Special handling for code completion vs generation
            if category == 'code_completion':
                if any(keyword in prompt_lower for keyword in required_keywords):
                    return category
                continue
            elif category == 'code_generation':
                if any(keyword in prompt_lower for keyword in exclude_keywords):
                    continue
                return category
            
            # Special handling for mathematical_reasoning
            elif category == 'mathematical_reasoning':
                if required_keywords and any(keyword in prompt_lower for keyword in required_keywords):
                    return category
                continue
            
            # Standard keyword matching
            else:
                # Check exclude keywords
                if exclude_keywords and any(keyword in prompt_lower for keyword in exclude_keywords):
                    continue
                return category
        
        return None

    async def _classify_by_ai(self, prompt: str, reasoning_engine: "BaseEngine") -> Optional[str]:
        """
        Use AI-based classification as fallback for complex prompts.
        
        Args:
            prompt: The original prompt to classify
            reasoning_engine: The reasoning engine to use for classification
            
        Returns:
            Category name if successfully classified, None otherwise
        """
        try:
            if not reasoning_engine.is_loaded():
                return None
                
            # Create a focused list of available categories for the AI
            available_for_ai: list[str] = [cat for cat in self.available_categories 
                              if cat != 'general_reasoning']  # Exclude fallback category
            
            if not available_for_ai:
                return None
                
            categories_str = ', '.join(available_for_ai)
            classification_prompt = f"""Classify: "{prompt}"

Return only one word from: {categories_str}

Answer:"""

            classification_result = await reasoning_engine.generate(
                classification_prompt, 
                max_tokens=10,
                temperature=0.1
            )
            predicted_category = classification_result.strip().lower()

            # Validate against available categories
            if predicted_category in self.available_categories:
                return predicted_category

            # Try partial matches
            for category in self.available_categories:
                if category in predicted_category or predicted_category in category:
                    logger.info(f"Using partial match: {category} for prediction: {predicted_category}")
                    return category

        except Exception as e:
            logger.error(f"Error in AI classification: {e}")

        return None

    def classify_by_keywords(self, prompt: str) -> Optional[str]:
        """
        Public method for keyword-based classification (primarily for testing).
        
        Args:
            prompt: The prompt to classify
            
        Returns:
            Category name if matched, None otherwise
        """
        return self._classify_by_keywords(prompt.lower())

    def get_supported_categories(self) -> set[str]:
        """Get the set of categories this classificator can detect."""
        return set(self.keyword_patterns.keys())

    def update_available_categories(self, categories: list[str]) -> None:
        """Update the list of available engine categories."""
        self.available_categories = set(categories)

    def add_keyword_pattern(self, category: str, keywords: list[str], priority: int = 10,
                           required_keywords: Optional[list[str]] = None,
                           exclude_keywords: Optional[list[str]] = None) -> None:
        """
        Add or update keyword patterns for a category.
        
        Args:
            category: The engine category name
            keywords: List of keywords that indicate this category
            priority: Priority level (lower numbers = higher priority)
            required_keywords: Keywords that must be present (for refinement)
            exclude_keywords: Keywords that exclude this category
        """
        pattern_config: dict[str, Any] = {
            'keywords': keywords,
            'priority': priority
        }
        
        if required_keywords:
            pattern_config['required_keywords'] = required_keywords
        if exclude_keywords:
            pattern_config['exclude_keywords'] = exclude_keywords
            
        self.keyword_patterns[category] = pattern_config
        logger.info(f"Added/updated keyword pattern for category: {category}")
