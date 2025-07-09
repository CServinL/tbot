#!/usr/bin/env python3
"""Test the conductor with the new classificator"""

import asyncio
import sys
import os
import pytest

# Add the project root to the path dynamically
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

@pytest.mark.asyncio
async def test_conductor_classification():
    from conductor.conductor import Conductor
    
    # Create conductor instance
    conductor = Conductor()
    
    # Initialize without loading models (just for classification test)
    await conductor.initialize(skip_model_loading=True)
    
    # Test prompts
    test_prompts = [
        "where in the world is carmen san diego?",
        "write a function to calculate fibonacci",
        "write a story about dragons",
        "review this code for bugs"
    ]
    
    print("Testing conductor classification integration...")
    
    # Check what categories are available in the classificator
    if conductor.classificator:
        print(f"Available categories: {conductor.classificator.available_categories}")
        print(f"Supported categories: {conductor.classificator.get_supported_categories()}")
    
    for prompt in test_prompts:
        try:
            category = await conductor.classify_prompt(prompt)
            print(f"'{prompt}' -> {category}")
        except Exception as e:
            print(f"'{prompt}' -> ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(test_conductor_classification())
