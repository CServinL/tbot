# 🧪 Stable Diffusion MCP Server - Test Suite

A comprehensive test suite for the Stable Diffusion MCP Server with ComfyUI compatibility.

## 🚀 Quick Start

### Prerequisites
```bash
# Install required packages
pip install torch diffusers pillow

# Optional: Install coverage for test coverage analysis
pip install coverage
```

### Make Script Executable
```bash
chmod +x run_tests.sh
```

### Run All Tests
```bash
# Run all tests with auto-detected Python
./run_tests.sh

# Run with specific Python version
./run_tests.sh -p /path/to/python
```

## 🎯 Test Types

### Unit Tests
```bash
# Run only unit tests
./run_tests.sh unit

# Run HTTP API tests specifically
./run_tests.sh http

# Run ComfyUI tests specifically  
./run_tests.sh comfyui
```

### Integration Tests
```bash
# Run integration tests
./run_tests.sh integration

# Run with coverage analysis
./run_tests.sh --coverage integration
```

### Live Server Tests
```bash
# Start your server first
python main.py --model test/model --http-port 8999 --device cpu

# Then run live tests
./run_tests.sh --live live

# Or set environment variable
RUN_LIVE_TESTS=1 python -m unittest test_integration.TestLiveServerIntegration
```

## 🛠️ Advanced Usage

### Coverage Analysis
```bash
# Run all tests with coverage
./run_tests.sh --coverage

# Run specific tests with coverage
./run_tests.sh --coverage integration
```

### Verbose Output
```bash
# Run with detailed output
./run_tests.sh --verbose

# Combine flags
./run_tests.sh -v -c integration
```

### Direct Python Testing
```bash
# Run individual test files
python -m unittest test_http_api.py
python -m unittest test_comfyui_api.py
python -m unittest test_integration.py

# Run with the test runner
python run_tests.py --http --comfyui
python test_config.py --coverage
```

### Force Specific Python Version
```bash
# Use virtual environment Python
./run_tests.sh -p /home/user/project/.venv/bin/python

# Use system Python
./run_tests.sh -p python3.11
```

## 📋 Command Reference

### Test Runner Options
```bash
./run_tests.sh [OPTIONS] [TEST_TYPE]

# Test Types:
#   all         - Run all tests (default)
#   unit        - Run only unit tests  
#   integration - Run only integration tests
#   http        - Run only HTTP API tests
#   comfyui     - Run only ComfyUI tests
#   live        - Run only live server tests

# Options:
#   -v, --verbose       - Verbose output
#   -c, --coverage      - Run with coverage analysis
#   -l, --live          - Enable live tests
#   -p, --python CMD    - Python command to use
#   -h, --help          - Show help message
```

### Environment Variables
```bash
RUN_LIVE_TESTS=1           # Enable live tests
PYTHON_CMD=/path/to/python # Override Python command  
TEST_TIMEOUT=60            # Test timeout in seconds
```

## 🎯 Key Features

### 📊 Comprehensive Coverage
- **HTTP API**: All REST endpoints with error handling
- **ComfyUI Compatibility**: Workflow parsing and execution
- **Model Management**: Loading, switching, caching operations
- **Image Generation**: Parameter validation and optimization

### 🔧 Realistic Testing
- **Mock Components**: No GPU dependencies for CI/CD
- **Actual Workflows**: Real ComfyUI workflow examples
- **Error Scenarios**: Network failures, model errors, invalid inputs
- **Live Integration**: Tests against running server instances

### 👨‍💻 Developer Friendly
- **Clear Output**: Colored, organized test results
- **Fast Execution**: Mocked dependencies for speed
- **Easy Debugging**: Verbose modes and detailed error messages
- **Coverage Reports**: HTML reports for missing test coverage

## 📊 Test Statistics

| Metric | Count |
|--------|-------|
| Test Methods | ~40 |
| Test Assertions | ~200 |
| Error Scenarios | ~15 |
| ComfyUI Workflows | Multiple complex examples |
| Test Files | 4 main test files |

### Test Coverage Areas
- ✅ **HTTP API Endpoints** - All REST API routes
- ✅ **ComfyUI Workflows** - Complex multi-node parsing
- ✅ **Model Operations** - Loading, switching, caching
- ✅ **Image Generation** - Parameters and validation
- ✅ **Error Handling** - Comprehensive error scenarios
- ✅ **Queue Management** - FIFO execution and history

## 🔧 Troubleshooting

### Common Issues

#### Python Not Found
```bash
# Check available Python versions
which python3
which python3.11

# Use specific Python version
./run_tests.sh -p $(which python3.11)
```

#### Missing Dependencies
```bash
# Install missing packages
pip install torch diffusers pillow coverage

# Or in virtual environment
source .venv/bin/activate
pip install torch diffusers pillow coverage
```

#### Shell Script Issues
```bash
# If run_tests.sh fails, run tests directly
python -m unittest discover -s . -p "test_*.py"

# Or run individual test files
python test_comfyui_api.py
python test_http_api.py
```

#### Live Tests Failing
```bash
# Ensure server is running on correct port
python main.py --http-port 8999 --device cpu

# Check server health
curl http://127.0.0.1:8999/health

# Then run live tests
RUN_LIVE_TESTS=1 ./run_tests.sh live
```

## 🚀 CI/CD Integration

The test suite is designed for CI/CD environments:

```yaml
# Example GitHub Actions workflow
- name: Run Tests
  run: |
    pip install torch diffusers pillow coverage
    chmod +x run_tests.sh
    ./run_tests.sh --coverage

- name: Upload Coverage
  run: |
    pip install codecov
    codecov
```

## 📁 Test File Structure

```
tests/
├── run_tests.sh              # Main test runner script
├── test_config.py            # Test configuration and utilities
├── test_http_api.py          # HTTP API endpoint tests
├── test_comfyui_api.py       # ComfyUI compatibility tests
├── test_integration.py       # Integration and live server tests
└── README.md                 # This documentation
```

## 🤝 Contributing

When adding new tests:

1. **Follow naming conventions**: `test_*.py` for test files
2. **Use descriptive test names**: `test_function_specific_scenario`
3. **Include docstrings**: Explain what each test validates
4. **Mock external dependencies**: Keep tests fast and reliable
5. **Test error conditions**: Include failure scenarios
6. **Update documentation**: Add new test types to this README

## 📝 Examples

### Testing a New Feature
```python
# In test_http_api.py
def test_new_endpoint_success(self):
    """Test successful response from new endpoint"""
    # Test implementation here
    
def test_new_endpoint_validation(self):
    """Test input validation on new endpoint"""
    # Test implementation here
```

### Running Specific Test Methods
```bash
# Run single test method
python -m unittest test_http_api.TestHTTPAPIEndpoints.test_generate_endpoint_success

# Run test class
python -m unittest test_comfyui_api.TestComfyUIAdapter
```

---

The test suite follows Python testing best practices and provides comprehensive coverage of your server's functionality. It's designed to be both CI/CD ready and developer-friendly for local testing and debugging.