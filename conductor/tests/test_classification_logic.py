#!/usr/bin/env python3

# Test classification logic using the new classificator module
import asyncio
import sys
import os
import pytest

# Add the project root to the path dynamically
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

@pytest.mark.asyncio
async def test_classification():
    from conductor.classificator import PromptClassificator
    
    # Available engine categories (from SETTINGS.md)
    available_categories = [
        'general_reasoning', 'code_generation', 'code_completion', 'mathematical_reasoning',
        'conversational_chat', 'translation', 'creative_writing', 'instruction_following',
        'summarization', 'question_answering', 'scientific_research', 'legal_analysis',
        'code_review', 'long_context', 'image_generation'
    ]
    
    # Create classificator instance
    classificator = PromptClassificator(available_categories)
    
    # Test prompts
    test_prompts = [
        "where in the world is carmen san diego?",
        "what is the capital of France?",
        "who is the president of the United States?",
        "write a function to calculate fibonacci",
        "translate hello to spanish",
        "who are the main characters in hamlet?",
        "hello how are you today?",
        "calculate 2 + 2",
        "write a story about dragons",
        "summarize this article",
        "review this code for bugs",
        "where can I find good pizza?",
        "how do birds fly?",
        "why do cats purr?"
    ]
    
    print("Testing classification logic with new classificator...")
    
    for prompt in test_prompts:
        print(f"\n--- Testing: '{prompt}' ---")
        
        # Test keyword-based classification only (without AI fallback)
        category = await classificator.classify_prompt(prompt, reasoning_engine=None)
        print(f"Result: {category}")

if __name__ == "__main__":
    asyncio.run(test_classification())
