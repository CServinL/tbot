# Classification Test Suite

## Overview

**`test_classification.py`** - Comprehensive test suite with multiple modes:

### Features:
1. **Manual Keyword Testing** - Test keyword matching logic manually
2. **Real Classificator Validation** - Test against the actual `PromptClassificator`
3. **Step-by-step Debug Mode** - Detailed analysis of classification logic
4. **Comprehensive Test Battery** - 40+ test cases covering all engine categories
5. **Performance Metrics** - Timing and success rate statistics

### Usage:

```bash
# Interactive mode
python conductor/tests/test_classification.py

# Quick validation (non-interactive)
echo "3" | python conductor/tests/test_classification.py

# Single prompt test
echo -e "1\nyour prompt here" | python conductor/tests/test_classification.py
```

### Test Modes:

1. **Single Prompt Test** - Detailed analysis of one specific prompt
   - Manual keyword analysis
   - Real classificator results
   - Step-by-step debug output

2. **Comprehensive Test Suite** - Run all 40+ test cases
   - Tests all engine categories
   - Shows pass/fail rates
   - Optional debug mode for first few cases

3. **Quick Validation** - Fast test of key scenarios
   - Tests 4 representative prompts
   - Shows classification results
   - Good for CI/CD or quick checks

### Test Categories Covered:

- ✅ **Code Review** - "review this code for bugs"
- ✅ **Code Generation** - "write a function to calculate fibonacci"
- ✅ **Code Completion** - "complete this function definition"
- ✅ **Question Answering** - "where in the world is carmen san diego?"
- ✅ **Creative Writing** - "write a story about dragons"
- ✅ **Translation** - "translate hello to spanish"
- ✅ **Summarization** - "summarize this article"
- ✅ **Mathematical Reasoning** - "calculate 2 + 2"
- ✅ **Conversational Chat** - "hello how are you today?"
- ✅ **Image Generation** - "generate an image of a sunset"
- ✅ **Scientific Research** - Various scientific prompts
- ✅ **Legal Analysis** - Legal document analysis
- ✅ **Instruction Following** - Task-based instructions
- ✅ **Long Context** - Long document processing

### Example Output:

```
🧪 Classification Test Suite
==================================================
 1. 'where in the world is carmen san diego?'
    Expected: question_answering
    Actual:   question_answering
    Status:   ✅ PASS

 2. 'write a function to calculate fibonacci'
    Expected: code_generation
    Actual:   code_generation
    Status:   ✅ PASS

📊 Test Results Summary:
Total tests: 42
Passed: 40 (95.2%)
Failed: 2 (4.8%)
Errors: 0 (0.0%)
Time: 1.23 seconds
```

## Benefits of Consolidation:

1. **Single Source of Truth** - One comprehensive test file instead of 3 separate ones
2. **Better Coverage** - More test cases and edge cases
3. **Multiple Test Modes** - From quick validation to detailed debugging
4. **Real Validation** - Tests against actual classificator implementation
5. **Performance Metrics** - Track test execution time and success rates
6. **Interactive & Automated** - Works in both interactive and CI/CD environments

## Maintenance:

- Add new test cases to the `test_cases` list in `ClassificationTestSuite.__init__()`
- Update keyword patterns to match any changes in the real classificator
- Run comprehensive tests after any changes to classification logic

### Pytest Compatibility:

The test suite is also pytest-compatible for automated testing:

```bash
# Run all classification tests
python -m pytest conductor/tests/test_classification.py -v

# Run specific test
python -m pytest conductor/tests/test_classification.py::test_code_review_classification -v

# Run with coverage
python -m pytest conductor/tests/test_classification.py --cov=conductor.classificator
```

**Available pytest functions:**
- `test_code_review_classification()`
- `test_code_generation_classification()`
- `test_question_answering_classification()`
- `test_image_generation_classification()`
- `test_comprehensive_classification_suite()`
