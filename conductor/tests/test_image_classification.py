#!/usr/bin/env python3
"""Test image generation classification"""

import sys
import os
import pytest

# Add the project root to the path dynamically
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

@pytest.mark.asyncio
async def test_image_classification():
    from conductor.classificator import PromptClassificator
    
    # Available engine categories (including image_generation)
    available_categories = [
        'general_reasoning', 'code_generation', 'code_completion', 'mathematical_reasoning',
        'conversational_chat', 'translation', 'creative_writing', 'instruction_following',
        'summarization', 'question_answering', 'scientific_research', 'legal_analysis',
        'code_review', 'long_context', 'image_generation'
    ]
    
    # Create classificator instance
    classificator = PromptClassificator(available_categories)
    
    # Test image generation prompts
    test_prompts = [
        "generate an image of a server in the cloud with diamonds",
        "create an image of a cat",
        "draw a picture of a mountain",
        "make an image of a sunset",
        "picture of a dog",
        "image of a car",
        "illustrate a forest scene",
        "visualize a modern office",
        "render an image of a robot"
    ]
    
    print("Testing image generation classification...")
    
    for prompt in test_prompts:
        print(f"\n--- Testing: '{prompt}' ---")
        
        category = await classificator.classify_prompt(prompt, reasoning_engine=None)
        print(f"Result: {category}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_image_classification())
