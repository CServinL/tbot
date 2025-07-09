#!/usr/bin/env python3
"""
Comprehensive Classification Test Suite

Consolidates all classification testing into a single comprehensive test battery
that includes:
- Manual keyword testing
- Real classificator validation
- Debug mode for step-by-step analysis
- Performance testing
- Edge case validation
"""

import asyncio
import sys
import os
import time
from typing import Dict, Any, Optional
import pytest

# Add the project root to the path dynamically
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)


class ClassificationTestSuite:
    """Comprehensive test suite for prompt classification."""
    
    def __init__(self):
        """Initialize the test suite."""
        self.available_categories = [
            'code_completion', 'code_review', 'code_generation', 'creative_writing',
            'translation', 'question_answering', 'summarization', 'mathematical_reasoning',
            'conversational_chat', 'general_reasoning', 'scientific_research',
            'legal_analysis', 'instruction_following', 'long_context', 'image_generation'
        ]
        
        # Test cases with expected results
        self.test_cases = [
            # Question answering
            ("where in the world is carmen san diego?", "question_answering"),
            ("what is the capital of France?", "question_answering"),
            ("who is the president of the United States?", "question_answering"),
            ("how do birds fly?", "question_answering"),
            ("why do cats purr?", "question_answering"),
            ("explain quantum physics", "question_answering"),
            ("tell me about the Renaissance", "question_answering"),
            
            # Code generation
            ("write a function to calculate fibonacci", "code_generation"),
            ("create a program that sorts an array", "code_generation"),
            ("write code to connect to a database", "code_generation"),
            ("generate a SQL query to find users", "code_generation"),
            ("create a React component", "code_generation"),
            
            # Code completion
            ("complete this function definition", "code_completion"),
            ("finish this code snippet", "code_completion"),
            ("continue this algorithm", "code_completion"),
            
            # Code review
            ("review this code for bugs", "code_review"),
            ("check this function for errors", "code_review"),
            ("debug this Python script", "code_review"),
            ("find issues in this code", "code_review"),
            
            # Creative writing
            ("write a story about dragons", "creative_writing"),
            ("create a poem about love", "creative_writing"),
            ("write a novel chapter", "creative_writing"),
            ("story about space exploration", "creative_writing"),
            
            # Translation
            ("translate hello to spanish", "translation"),
            ("convert this text from english to french", "translation"),
            ("translate this document", "translation"),
            
            # Summarization
            ("summarize this article", "summarization"),
            ("give me a brief overview", "summarization"),
            ("create a tldr", "summarization"),
            
            # Mathematical reasoning
            ("calculate 2 + 2", "mathematical_reasoning"),
            ("solve this equation: x^2 + 5x + 6 = 0", "mathematical_reasoning"),
            ("what is 15% of 200?", "mathematical_reasoning"),
            
            # Conversational
            ("hello how are you today?", "conversational_chat"),
            ("hi there!", "conversational_chat"),
            ("thank you for your help", "conversational_chat"),
            
            # Image generation
            ("generate an image of a sunset", "image_generation"),
            ("create a picture of a cat", "image_generation"),
            ("draw a landscape", "image_generation"),
        ]
    
    def test_manual_keyword_matching(self, prompt: str, verbose: bool = False) -> Dict[str, Any]:
        """Test manual keyword matching for a specific prompt."""
        prompt_lower = prompt.lower()
        
        # Define keyword patterns (simplified from classificator)
        patterns = {
            'question_answering': [
                'what is', 'what are', 'where is', 'where are', 'when is', 'when was',
                'who is', 'who was', 'how is', 'how does', 'why is', 'why does',
                'explain', 'tell me about'
            ],
            'code_generation': [
                'write a function', 'write code', 'create a program', 'generate code',
                'function', 'program', 'algorithm', 'script'
            ],
            'code_completion': ['complete', 'finish', 'continue'],
            'code_review': ['review', 'check', 'debug', 'error', 'bug', 'fix'],
            'creative_writing': ['story', 'poem', 'novel', 'creative writing', 'fiction'],
            'translation': ['translate', 'translation', 'convert'],
            'summarization': ['summarize', 'summary', 'tldr', 'brief', 'overview'],
            'mathematical_reasoning': ['calculate', 'solve', 'equation', '+', '-', '*', '/'],
            'conversational_chat': ['hello', 'hi', 'hey', 'thank you', 'how are you'],
            'image_generation': ['generate an image', 'create a picture', 'draw', 'image of']
        }
        
        matches = {}
        for category, keywords in patterns.items():
            category_matches = [kw for kw in keywords if kw in prompt_lower]
            if category_matches:
                matches[category] = category_matches
        
        if verbose:
            print(f"Manual keyword analysis for: '{prompt}'")
            for category, category_matches in matches.items():
                print(f"  {category}: {category_matches}")
            print()
        
        return matches
    
    async def test_real_classificator(self, prompt: str, verbose: bool = False) -> Optional[str]:
        """Test with the real classificator."""
        try:
            from conductor.classificator import PromptClassificator
            
            classificator = PromptClassificator(self.available_categories)
            
            # Test keyword-only classification first
            keyword_result = classificator.classify_by_keywords(prompt)
            
            # Test full classification (with AI fallback if available)
            full_result = await classificator.classify_prompt(prompt, reasoning_engine=None)
            
            if verbose:
                print(f"Real classificator results for: '{prompt}'")
                print(f"  Keyword-only: {keyword_result}")
                print(f"  Full classification: {full_result}")
                print()
            
            return full_result
            
        except Exception as e:
            if verbose:
                print(f"Error testing real classificator: {e}")
            return None
    
    def debug_classification_step_by_step(self, prompt: str) -> str:
        """Provide step-by-step debug analysis of classification."""
        prompt_lower = prompt.lower()
        
        print(f"\n🔍 Step-by-step classification debug for: '{prompt}'")
        print(f"Lowercase: '{prompt_lower}'")
        print("-" * 60)
        
        # Simplified categories with priorities from the actual classificator
        categories = {
            'code_completion': {
                'keywords': ['complete', 'finish', 'continue'],
                'required_keywords': ['complete', 'finish', 'continue'],
                'priority': 1
            },
            'code_review': {
                'keywords': ['review', 'check', 'debug', 'error', 'bug', 'fix'],
                'priority': 2
            },
            'creative_writing': {
                'keywords': ['story', 'poem', 'novel', 'creative writing', 'fiction'],
                'exclude_keywords': ['who are', 'what are', 'tell me about'],
                'priority': 3
            },
            'question_answering': {
                'keywords': [
                    'what is', 'what are', 'where is', 'where are', 'when is', 'when was',
                    'who is', 'who was', 'how is', 'how does', 'why is', 'why does',
                    'explain', 'tell me about'
                ],
                'priority': 5
            },
            'code_generation': {
                'keywords': ['function', 'code', 'write a function', 'create a program'],
                'exclude_keywords': ['complete', 'finish', 'continue', 'story', 'poem'],
                'priority': 7
            }
        }
        
        # Sort by priority
        sorted_categories = sorted(categories.items(), key=lambda x: x[1]['priority'])
        
        for name, config in sorted_categories:
            keywords = config['keywords']
            required_keywords = config.get('required_keywords', [])
            exclude_keywords = config.get('exclude_keywords', [])
            priority = config['priority']
            
            matches = [kw for kw in keywords if kw in prompt_lower]
            if not matches:
                continue
                
            print(f"{name} (priority {priority}):")
            print(f"  ✅ Keyword matches: {matches}")
            
            # Check required keywords
            if required_keywords:
                req_matches = [kw for kw in required_keywords if kw in prompt_lower]
                print(f"  🔍 Required keywords: {req_matches}")
                if not req_matches:
                    print(f"  ❌ SKIPPED (no required keywords found)")
                    continue
            
            # Check exclude keywords
            if exclude_keywords:
                exc_matches = [kw for kw in exclude_keywords if kw in prompt_lower]
                print(f"  🚫 Exclude keywords: {exc_matches}")
                if exc_matches:
                    print(f"  ❌ SKIPPED (exclude keywords found)")
                    continue
            
            print(f"  🎯 SELECTED: {name}")
            return name
        
        print("  🤷 No matches found, would use general_reasoning fallback")
        return "general_reasoning"
    
    async def run_comprehensive_tests(self, debug_mode: bool = False, verbose: bool = True):
        """Run the complete test suite."""
        print("🧪 Classification Test Suite")
        print("=" * 50)
        
        passed = 0
        failed = 0
        errors = 0
        
        start_time = time.time()
        
        for i, (prompt, expected) in enumerate(self.test_cases, 1):
            try:
                if debug_mode and i <= 3:  # Debug only first 3 cases to avoid spam
                    debug_result = self.debug_classification_step_by_step(prompt)
                
                # Test with real classificator
                actual = await self.test_real_classificator(prompt, verbose=False)
                
                # Test manual keyword matching
                manual_matches = self.test_manual_keyword_matching(prompt, verbose=False)
                
                # Determine result
                if actual == expected:
                    status = "✅ PASS"
                    passed += 1
                else:
                    status = "❌ FAIL"
                    failed += 1
                
                if verbose:
                    print(f"{i:2d}. '{prompt[:40]}{'...' if len(prompt) > 40 else ''}'")
                    print(f"    Expected: {expected}")
                    print(f"    Actual:   {actual}")
                    print(f"    Status:   {status}")
                    if manual_matches:
                        print(f"    Manual:   {list(manual_matches.keys())}")
                    print()
                
            except Exception as e:
                errors += 1
                if verbose:
                    print(f"{i:2d}. '{prompt}' - ERROR: {e}")
        
        end_time = time.time()
        
        # Results summary
        total = len(self.test_cases)
        print(f"\n📊 Test Results Summary:")
        print(f"Total tests: {total}")
        print(f"Passed: {passed} ({passed/total*100:.1f}%)")
        print(f"Failed: {failed} ({failed/total*100:.1f}%)")
        print(f"Errors: {errors} ({errors/total*100:.1f}%)")
        print(f"Time: {end_time - start_time:.2f} seconds")
        
        return {
            'total': total,
            'passed': passed,
            'failed': failed,
            'errors': errors,
            'time': end_time - start_time
        }
    
    async def test_single_prompt(self, prompt: str, debug: bool = True):
        """Test a single prompt with detailed analysis."""
        print(f"🔬 Single Prompt Analysis")
        print("=" * 40)
        
        # Manual keyword testing
        print("1️⃣ Manual Keyword Analysis:")
        manual_matches = self.test_manual_keyword_matching(prompt, verbose=True)
        
        # Real classificator testing
        print("2️⃣ Real Classificator Analysis:")
        actual = await self.test_real_classificator(prompt, verbose=True)
        
        # Debug step-by-step
        if debug:
            print("3️⃣ Step-by-step Debug:")
            debug_result = self.debug_classification_step_by_step(prompt)
        
        return {
            'manual_matches': manual_matches,
            'classificator_result': actual,
            'debug_result': debug_result if debug else None
        }


# Pytest-compatible test functions
@pytest.mark.asyncio
async def test_code_review_classification():
    """Test code review classification specifically."""
    suite = ClassificationTestSuite()
    result = await suite.test_real_classificator("review this code for bugs")
    assert result == "code_review"

@pytest.mark.asyncio
async def test_code_generation_classification():
    """Test code generation classification."""
    suite = ClassificationTestSuite()
    result = await suite.test_real_classificator("write a function to calculate fibonacci")
    assert result == "code_generation"

@pytest.mark.asyncio
async def test_question_answering_classification():
    """Test question answering classification."""
    suite = ClassificationTestSuite()
    result = await suite.test_real_classificator("where in the world is carmen san diego?")
    assert result == "question_answering"

@pytest.mark.asyncio
async def test_image_generation_classification():
    """Test image generation classification."""
    suite = ClassificationTestSuite()
    result = await suite.test_real_classificator("generate an image of a sunset")
    assert result == "image_generation"

@pytest.mark.asyncio
async def test_comprehensive_classification_suite():
    """Run the comprehensive test suite as a pytest."""
    suite = ClassificationTestSuite()
    # Test a subset for faster pytest execution
    suite.test_cases = suite.test_cases[:10]  # First 10 cases only
    results = await suite.run_comprehensive_tests(debug_mode=False, verbose=False)
    
    # Assert that most tests pass (allow some tolerance)
    success_rate = results['passed'] / results['total']
    assert success_rate >= 0.8, f"Success rate too low: {success_rate:.2f}"


async def main():
    """Main function to run tests."""
    suite = ClassificationTestSuite()
    
    # Example usage
    print("Choose test mode:")
    print("1. Single prompt test")
    print("2. Comprehensive test suite")
    print("3. Quick validation")
    
    mode = input("Enter choice (1-3, default 2): ").strip() or "2"
    
    if mode == "1":
        prompt = input("Enter prompt to test: ").strip() or "review this code for bugs"
        await suite.test_single_prompt(prompt)
    
    elif mode == "2":
        debug_mode = input("Enable debug mode? (y/n, default n): ").strip().lower() == 'y'
        await suite.run_comprehensive_tests(debug_mode=debug_mode)
    
    elif mode == "3":
        # Quick test with a few key cases
        quick_cases = [
            "review this code for bugs",
            "write a function to calculate fibonacci", 
            "where in the world is carmen san diego?",
            "translate hello to spanish"
        ]
        print("🚀 Quick Validation:")
        for prompt in quick_cases:
            result = await suite.test_real_classificator(prompt)
            print(f"  '{prompt}' → {result}")


if __name__ == "__main__":
    asyncio.run(main())
