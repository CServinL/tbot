#!/usr/bin/env python3
"""
Conductor LLM Server Usage Examples

This script demonstrates both intelligent routing and direct endpoint usage.
"""

import asyncio
import aiohttp
import json
from typing import Dict, Any


class ConductorClient:
    """Simple client for interacting with Conductor LLM Server."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    async def intelligent_generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Use intelligent routing - conductor will classify and route automatically."""
        async with aiohttp.ClientSession() as session:
            data = {"prompt": prompt, **kwargs}
            async with session.post(f"{self.base_url}/generate", json=data) as response:
                return await response.json()

    async def chat_completion(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Direct chat completion endpoint."""
        async with aiohttp.ClientSession() as session:
            data = {"prompt": prompt, **kwargs}
            async with session.post(f"{self.base_url}/chat/completions", json=data) as response:
                return await response.json()

    async def code_completion(self, code: str, language: str = "python", **kwargs) -> Dict[str, Any]:
        """Direct code completion endpoint."""
        async with aiohttp.ClientSession() as session:
            data = {"prompt": code, "language": language, **kwargs}
            async with session.post(f"{self.base_url}/code/completion", json=data) as response:
                return await response.json()

    async def code_generation(self, prompt: str, language: str = "python", **kwargs) -> Dict[str, Any]:
        """Direct code generation endpoint."""
        async with aiohttp.ClientSession() as session:
            data = {"prompt": prompt, "language": language, **kwargs}
            async with session.post(f"{self.base_url}/code/generation", json=data) as response:
                return await response.json()

    async def translate(self, text: str, source_lang: str, target_lang: str, **kwargs) -> Dict[str, Any]:
        """Direct translation endpoint."""
        async with aiohttp.ClientSession() as session:
            data = {
                "prompt": text,
                "extra_params": {
                    "source_lang": source_lang,
                    "target_lang": target_lang
                },
                **kwargs
            }
            async with session.post(f"{self.base_url}/translate", json=data) as response:
                return await response.json()

    async def solve_math(self, problem: str, **kwargs) -> Dict[str, Any]:
        """Direct mathematical reasoning endpoint."""
        async with aiohttp.ClientSession() as session:
            data = {"prompt": problem, **kwargs}
            async with session.post(f"{self.base_url}/math/solve", json=data) as response:
                return await response.json()

    async def research_analyze(self, query: str, **kwargs) -> Dict[str, Any]:
        """Direct scientific research endpoint."""
        async with aiohttp.ClientSession() as session:
            data = {"prompt": query, **kwargs}
            async with session.post(f"{self.base_url}/research/analyze", json=data) as response:
                return await response.json()

    async def legal_analyze(self, text: str, **kwargs) -> Dict[str, Any]:
        """Direct legal analysis endpoint."""
        async with aiohttp.ClientSession() as session:
            data = {"prompt": text, **kwargs}
            async with session.post(f"{self.base_url}/legal/analyze", json=data) as response:
                return await response.json()

    async def summarize(self, text: str, **kwargs) -> Dict[str, Any]:
        """Direct summarization endpoint."""
        async with aiohttp.ClientSession() as session:
            data = {"prompt": text, **kwargs}
            async with session.post(f"{self.base_url}/summarize", json=data) as response:
                return await response.json()


async def demonstrate_intelligent_routing():
    """Demonstrate intelligent routing where conductor classifies prompts automatically."""

    client = ConductorClient()

    print("=== Intelligent Routing Examples ===\n")

    # Examples that should be automatically classified
    test_prompts = [
        {
            "prompt": "Write a Python function to calculate the factorial of a number",
            "expected_category": "code_generation"
        },
        {
            "prompt": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
            "expected_category": "code_completion"
        },
        {
            "prompt": "Solve this equation: 2x + 5 = 13",
            "expected_category": "mathematical_reasoning"
        },
        {
            "prompt": "Translate 'Hello, how are you?' to Spanish",
            "expected_category": "translation"
        },
        {
            "prompt": "Write a short story about a robot learning to love",
            "expected_category": "creative_writing"
        },
        {
            "prompt": "How does photosynthesis work in plants?",
            "expected_category": "question_answering"
        },
        {
            "prompt": "Analyze the methodology used in this research paper: [abstract text]",
            "expected_category": "scientific_research"
        },
        {
            "prompt": "Review this employment contract for potential legal issues",
            "expected_category": "legal_analysis"
        }
    ]

    for test in test_prompts:
        try:
            print(f"Prompt: {test['prompt'][:60]}...")
            print(f"Expected: {test['expected_category']}")

            response = await client.intelligent_generate(test['prompt'], max_tokens=100)

            print(f"Classified as: {response['category']}")
            print(f"Match: {'✓' if response['category'] == test['expected_category'] else '✗'}")
            print(f"Response: {response['response'][:100]}...\n")

        except Exception as e:
            print(f"Error: {e}\n")

        await asyncio.sleep(0.5)  # Be nice to the server


async def demonstrate_direct_endpoints():
    """Demonstrate direct endpoint usage without classification."""

    client = ConductorClient()

    print("=== Direct Endpoint Examples ===\n")

    try:
        # Code completion
        print("1. Code Completion:")
        code_prompt = "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right ="
        response = await client.code_completion(code_prompt, max_tokens=50)
        print(f"Completion: {response['response']}\n")

        # Code generation
        print("2. Code Generation:")
        response = await client.code_generation(
            "Create a FastAPI server with a health check endpoint",
            language="python",
            max_tokens=300
        )
        print(f"Generated code: {response['response'][:200]}...\n")

        # Translation
        print("3. Translation:")
        response = await client.translate(
            "The weather is beautiful today",
            source_lang="english",
            target_lang="spanish"
        )
        print(f"Translation: {response['response']}\n")

        # Math solving
        print("4. Math Solving:")
        response = await client.solve_math(
            "Find the derivative of f(x) = 3x² + 2x - 1",
            extra_params={"show_work": True}
        )
        print(f"Solution: {response['response'][:200]}...\n")

        # Research analysis
        print("5. Research Analysis:")
        response = await client.research_analyze(
            "What are the current challenges in quantum computing?",
            max_tokens=400
        )
        print(f"Analysis: {response['response'][:200]}...\n")

        # Summarization
        print("6. Summarization:")
        long_text = """
        Artificial intelligence has made remarkable progress in recent years, with large language models 
        demonstrating unprecedented capabilities in understanding and generating human-like text. These 
        models, trained on vast amounts of data, can perform a wide variety of tasks including translation, 
        summarization, question answering, and code generation. However, they also present new challenges 
        in terms of safety, alignment, and responsible deployment. As we continue to develop more powerful 
        AI systems, it becomes increasingly important to ensure they are beneficial, safe, and aligned 
        with human values.
        """
        response = await client.summarize(
            long_text,
            extra_params={"summary_type": "brief", "length": "short"}
        )
        print(f"Summary: {response['response']}\n")

    except Exception as e:
        print(f"Error in direct endpoints: {e}")


async def compare_routing_methods():
    """Compare intelligent routing vs direct endpoints for the same prompts."""

    client = ConductorClient()

    print("=== Routing Method Comparison ===\n")

    # Code generation example
    code_prompt = "Create a binary search function in Python"

    print("Code Generation Prompt:", code_prompt)
    print()

    try:
        # Intelligent routing
        print("Via Intelligent Routing:")
        intelligent_response = await client.intelligent_generate(code_prompt, max_tokens=300)
        print(f"Classified as: {intelligent_response['category']}")
        print(f"Generation time: {intelligent_response['generation_time']:.3f}s")
        print()

        # Direct endpoint
        print("Via Direct Endpoint:")
        direct_response = await client.code_generation(code_prompt, max_tokens=300)
        print(f"Category: {direct_response['category']}")
        print(f"Generation time: {direct_response['generation_time']:.3f}s")
        print()

        # Compare responses
        print("Response similarity:",
              "High" if len(set(intelligent_response['response'].split()) &
                            set(direct_response['response'].split())) > 10 else "Low")

    except Exception as e:
        print(f"Error in comparison: {e}")


async def main():
    """Run all examples."""

    print("Conductor LLM Server Usage Examples")
    print("=" * 50)
    print()

    try:
        # Test server availability
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8000/health") as response:
                if response.status == 200:
                    print("✓ Conductor server is running\n")
                else:
                    print("✗ Conductor server not responding properly")
                    return
    except Exception as e:
        print(f"✗ Cannot connect to Conductor server: {e}")
        print("Make sure the server is running: python conductor.py")
        return

    # Run demonstrations
    await demonstrate_intelligent_routing()
    await demonstrate_direct_endpoints()
    await compare_routing_methods()

    print("Examples completed!")


if __name__ == "__main__":
    asyncio.run(main())