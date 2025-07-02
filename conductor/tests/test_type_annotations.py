#!/usr/bin/env python3
"""Test all methods of the classificator to check for type issues"""

import sys
import os
import pytest

# Add the project root to the path dynamically
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

@pytest.mark.asyncio
async def test_all_methods():
    from conductor.classificator import PromptClassificator
    
    # Test initialization
    categories = ['code_generation', 'question_answering', 'creative_writing']
    classificator = PromptClassificator(categories)
    
    # Test get_supported_categories
    supported = classificator.get_supported_categories()
    print(f"Supported categories type: {type(supported)}")
    print(f"Supported categories: {supported}")
    
    # Test update_available_categories
    classificator.update_available_categories(['new_category', 'another_category'])
    print(f"Updated available categories: {classificator.available_categories}")
    
    # Test add_keyword_pattern
    classificator.add_keyword_pattern(
        'test_category', 
        ['test', 'example'], 
        priority=10,
        required_keywords=['required'],
        exclude_keywords=['exclude']
    )
    print("Added keyword pattern successfully")
    
    # Test classification (without AI engine)
    result = await classificator.classify_prompt("where is carmen sandiego?")
    print(f"Classification result: {result}")
    
    print("All methods tested successfully!")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_all_methods())
