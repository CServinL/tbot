#!/usr/bin/env python3
"""
Simple classification tests using the PromptClassificator.
"""

from conductor.classificator import PromptClassificator


def test_basic_classification():
    """Test basic keyword-based classification."""
    # Available categories for testing
    available_categories = [
        'code_generation', 'code_completion', 'code_review', 'creative_writing',
        'translation', 'question_answering', 'summarization', 'mathematical_reasoning',
        'conversational_chat', 'image_generation', 'general_reasoning'
    ]
    
    classificator = PromptClassificator(available_categories)
    
    # Test cases with expected categories
    test_cases = [
        ("write a function to sort an array", "code_generation"),
        ("complete this function def sort", "code_completion"),
        ("where in the world is carmen san diego?", "question_answering"),
        ("what are the ingredients in a cake?", "question_answering"),
        ("when did world war 2 end?", "question_answering"),
        ("who are the main characters in hamlet?", "question_answering"),
        ("hello how are you today?", "conversational_chat"),
        ("translate this to spanish", "translation"),
        ("write a story about dragons", "creative_writing"),
        ("summarize this document", "summarization"),
        ("calculate 2 + 2", "mathematical_reasoning"),
        ("generate an image of a cat", "image_generation"),
        ("review this code for bugs", "code_review"),
    ]
    
    for prompt, expected_category in test_cases:
        result = classificator.classify_by_keywords(prompt)
        assert result == expected_category, f"Expected '{expected_category}' for prompt '{prompt}', got '{result}'"


def test_priority_ordering():
    """Test that priority ordering works correctly."""
    available_categories = ['code_generation', 'code_completion', 'creative_writing']
    classificator = PromptClassificator(available_categories)
    
    # Code completion should have higher priority than code generation
    result = classificator.classify_by_keywords("complete this function def")
    assert result == "code_completion"
    
    # Creative writing should be detected even with code-like words
    result = classificator.classify_by_keywords("write a story about a programmer")
    assert result == "creative_writing"


def test_fallback_behavior():
    """Test fallback when no keywords match."""
    available_categories = ['general_reasoning']
    classificator = PromptClassificator(available_categories)
    
    # No specific keywords should result in None for keyword classification
    result = classificator.classify_by_keywords("random unclassifiable text")
    assert result is None
