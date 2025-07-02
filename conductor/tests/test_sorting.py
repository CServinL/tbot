#!/usr/bin/env python3
"""Test the actual classificator sorting"""

import sys
import os

# Add the project root to the path dynamically
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

def test_sorting():
    from conductor.classificator import PromptClassificator
    
    available_categories = ['code_generation', 'code_review', 'code_completion']
    classificator = PromptClassificator(available_categories)
    
    # Check the sorted order
    sorted_categories = sorted(
        classificator.keyword_patterns.items(), 
        key=lambda x: x[1]['priority']
    )
    
    print("Categories sorted by priority:")
    for name, config in sorted_categories:
        if name in available_categories:
            print(f"  {config['priority']}: {name}")
    
    # Test the actual classification
    prompt = "review this code for bugs"
    print(f"\nTesting classification for: '{prompt}'")
    
    category = classificator.classify_by_keywords(prompt)
    print(f"Result: {category}")

if __name__ == "__main__":
    test_sorting()
