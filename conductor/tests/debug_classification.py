#!/usr/bin/env python3
"""Debug the classification of 'review this code for bugs'"""

import sys
import os

# Add the project root to the path dynamically
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

def debug_classification():
    from conductor.classificator import PromptClassificator
    
    prompt = "review this code for bugs"
    prompt_lower = prompt.lower()
    
    print(f"Debugging prompt: '{prompt}'")
    print(f"Lowercase: '{prompt_lower}'")
    
    # Check each category manually
    categories = {
        'code_completion': {
            'keywords': ['function', 'code', 'class', 'program', 'algorithm', 'script', 'def ', 'import ', 'const ', 'var '],
            'required_keywords': ['complete', 'finish', 'continue'],
            'priority': 1
        },
        'code_generation': {
            'keywords': ['function', 'code', 'class', 'program', 'algorithm', 'script', 'def ', 'import ', 'const ', 'var ', 'write a function', 'write code', 'create a program'],
            'exclude_keywords': ['complete', 'finish', 'continue', 'story', 'poem', 'creative', 'fiction', 'novel'],
            'priority': 7
        },
        'code_review': {
            'keywords': ['review', 'check', 'debug', 'error', 'bug', 'fix'],
            'priority': 2
        }
    }
    
    for name, config in categories.items():
        keywords = config['keywords']
        required_keywords = config.get('required_keywords', [])
        exclude_keywords = config.get('exclude_keywords', [])
        priority = config['priority']
        
        matches = [kw for kw in keywords if kw in prompt_lower]
        if matches:
            print(f"\n{name} (priority {priority}):")
            print(f"  Matches: {matches}")
            
            if required_keywords:
                req_matches = [kw for kw in required_keywords if kw in prompt_lower]
                print(f"  Required matches: {req_matches}")
                if not req_matches:
                    print(f"  -> SKIPPED (no required keywords)")
                    continue
            
            if exclude_keywords:
                exc_matches = [kw for kw in exclude_keywords if kw in prompt_lower]
                print(f"  Exclude matches: {exc_matches}")
                if exc_matches:
                    print(f"  -> SKIPPED (exclude keywords found)")
                    continue
            
            print(f"  -> WOULD SELECT {name}")

if __name__ == "__main__":
    debug_classification()
