#!/usr/bin/env python3
"""
Test script for dynamic model switching functionality
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add conductor to path
sys.path.insert(0, str(Path(__file__).parent))

from conductor.conductor import Conductor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_dynamic_switching():
    """Test the dynamic model switching functionality"""
    logger.info("=== Testing Dynamic Model Switching ===")
    
    # Initialize conductor
    conductor = Conductor()
    success = await conductor.initialize(skip_model_loading=False)
    
    if not success:
        logger.error("Failed to initialize conductor")
        return False
        
    # Test 1: Check initial state
    logger.info("\n=== Test 1: Initial State ===")
    status = conductor.get_model_status()
    logger.info(f"Default model: {status['default_model']}")
    logger.info(f"Current active model: {status['current_active_model']}")
    logger.info(f"Loaded models: {status['loaded_models']}")
    logger.info(f"Loaded engines: {status['loaded_engines']}")
    
    # Test 2: General reasoning (should use default model)
    logger.info("\n=== Test 2: General Reasoning (Default Model) ===")
    category, response = await conductor.generate("What is the capital of France?")
    logger.info(f"Category: {category}")
    logger.info(f"Response: {response[:100]}...")
    
    status = conductor.get_model_status()
    logger.info(f"Active model after general reasoning: {status['current_active_model']}")
    
    # Test 3: Code generation (should switch to specialized model if different)
    logger.info("\n=== Test 3: Code Generation (Specialized Model) ===")
    category, response = await conductor.generate("Write a Python function to calculate fibonacci numbers")
    logger.info(f"Category: {category}")
    logger.info(f"Response: {response[:100]}...")
    
    status = conductor.get_model_status()
    logger.info(f"Active model after code generation: {status['current_active_model']}")
    logger.info(f"Loaded models: {status['loaded_models']}")
    
    # Test 4: Another general query (should switch back to default)
    logger.info("\n=== Test 4: Another General Query (Back to Default) ===")
    category, response = await conductor.generate("Explain photosynthesis")
    logger.info(f"Category: {category}")
    logger.info(f"Response: {response[:100]}...")
    
    status = conductor.get_model_status()
    logger.info(f"Active model after general query: {status['current_active_model']}")
    logger.info(f"Loaded models: {status['loaded_models']}")
    
    # Test 5: Mathematical reasoning (specialized model)
    logger.info("\n=== Test 5: Mathematical Reasoning (Specialized Model) ===")
    category, response = await conductor.generate("Calculate 15 * 23 + 45")
    logger.info(f"Category: {category}")
    logger.info(f"Response: {response[:100]}...")
    
    status = conductor.get_model_status()
    logger.info(f"Active model after math: {status['current_active_model']}")
    
    # Test 6: Force switch back to default
    logger.info("\n=== Test 6: Force Switch to Default ===")
    success = await conductor.force_switch_to_default()
    logger.info(f"Force switch successful: {success}")
    
    status = conductor.get_model_status()
    logger.info(f"Active model after force switch: {status['current_active_model']}")
    
    # Test 7: Cleanup specialized models
    logger.info("\n=== Test 7: Cleanup Specialized Models ===")
    cleaned_count = await conductor.cleanup_specialized_models()
    logger.info(f"Cleaned up {cleaned_count} specialized models")
    
    status = conductor.get_model_status()
    logger.info(f"Final loaded models: {status['loaded_models']}")
    logger.info(f"Final active model: {status['current_active_model']}")
    
    logger.info("\n=== Dynamic Model Switching Test Complete ===")
    return True


async def main():
    """Main test function"""
    try:
        success = await test_dynamic_switching()
        if success:
            logger.info("✓ All tests completed successfully")
        else:
            logger.error("✗ Tests failed")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Test error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
