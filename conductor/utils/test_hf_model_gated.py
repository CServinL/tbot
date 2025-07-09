#!/usr/bin/env python3
"""
Quick script to test which models are accessible without authentication.
Run this to verify model availability before updating SETTINGS.md
"""

import sys
from transformers import AutoTokenizer
import logging

# Reduce logging noise
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)


def test_model_access(model_name):
    """Test if a model can be accessed without authentication"""
    try:
        print(f"Testing {model_name}...", end=" ")
        # Just try to load tokenizer config - minimal download
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            # Don't actually download, just check access
        )
        print("✓ ACCESSIBLE")
        return True
    except Exception as e:
        error_str = str(e)
        if "gated" in error_str.lower():
            print("✗ GATED (requires auth)")
        elif "not found" in error_str.lower():
            print("✗ NOT FOUND")
        else:
            print(f"✗ ERROR: {error_str[:50]}...")
        return False


def main():
    """Test various open models"""
    print("Testing Model Accessibility (No Authentication)")
    print("=" * 60)

    # Models to test
    test_models = [
        # Truly open models
        "EleutherAI/gpt-j-6b",
        "EleutherAI/gpt-neo-2.7B",
        "microsoft/DialoGPT-large",
        "Salesforce/codet5p-770m-py",
        "Salesforce/codet5p-220m",
        "google/flan-t5-large",
        "facebook/nllb-200-1.3B",

        # Previously thought open (now gated)
        "mistralai/Mistral-7B-Instruct-v0.2",
        "codellama/CodeLlama-7b-hf",
        "codellama/CodeLlama-7b-Instruct-hf",

        # Known gated
        "meta-llama/Llama-3.1-8B-Instruct",
    ]

    accessible_models = []
    gated_models = []

    for model in test_models:
        if test_model_access(model):
            accessible_models.append(model)
        else:
            gated_models.append(model)

    print("\n" + "=" * 60)
    print(f"✓ ACCESSIBLE MODELS ({len(accessible_models)}):")
    for model in accessible_models:
        print(f"  - {model}")

    print(f"\n✗ GATED/UNAVAILABLE MODELS ({len(gated_models)}):")
    for model in gated_models:
        print(f"  - {model}")

    print("\n" + "=" * 60)
    print("RECOMMENDATION:")
    if accessible_models:
        print("Use accessible models for immediate testing:")
        for model in accessible_models[:3]:  # Show top 3
            print(f"  {model}")
    else:
        print("No accessible models found. You may need authentication.")

    print("\nTo use gated models:")
    print("1. huggingface-cli login")
    print("2. Request access at model pages")
    print("3. Wait for approval")


if __name__ == "__main__":
    main()