#!/bin/bash
# Test runner script for Stable Diffusion MCP Server

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Default values
VERBOSE=false
COVERAGE=false
LIVE_TESTS=false
TEST_TYPE="all"
PYTHON_CMD=""

# Auto-detect Python command
detect_python() {
    # Check for virtual environment first
    if [ -n "$VIRTUAL_ENV" ]; then
        if [ -f "$VIRTUAL_ENV/bin/python" ]; then
            echo "$VIRTUAL_ENV/bin/python"
            return
        fi
    fi

    # Check for .venv in current directory
    if [ -f ".venv/bin/python" ]; then
        echo ".venv/bin/python"
        return
    fi

    # Check for common Python commands
    for cmd in python3.11 python3.10 python3.9 python3.8 python3 python; do
        if command -v "$cmd" >/dev/null 2>&1; then
            echo "$cmd"
            return
        fi
    done

    echo ""
}

# Function to print colored output
print_color() {
    local color=$1
    local message=$2
    printf "%b%s%b\n" "$color" "$message" "$NC"
}

# Function to print usage
usage() {
    cat << 'EOF'
Usage: ./run_tests.sh [OPTIONS] [TEST_TYPE]

Test runner for Stable Diffusion MCP Server

TEST_TYPE:
    all         Run all tests (default)
    unit        Run only unit tests
    integration Run only integration tests
    http        Run only HTTP API tests
    comfyui     Run only ComfyUI tests
    live        Run only live server tests

OPTIONS:
    -v, --verbose       Verbose output
    -c, --coverage      Run with coverage analysis
    -l, --live          Enable live tests (requires running server)
    -p, --python CMD    Python command to use
    -h, --help          Show this help message

EXAMPLES:
    ./run_tests.sh                          # Run all tests
    ./run_tests.sh unit                     # Run only unit tests
    ./run_tests.sh --coverage               # Run all tests with coverage
    ./run_tests.sh --live live              # Run live tests
    ./run_tests.sh -v -c integration        # Run integration tests with verbose output and coverage

ENVIRONMENT VARIABLES:
    RUN_LIVE_TESTS=1           # Enable live tests
    PYTHON_CMD=/path/to/python # Override Python command
    TEST_TIMEOUT=60            # Test timeout in seconds

EOF
}

# Parse command line arguments
while [ $# -gt 0 ]; do
    case $1 in
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -c|--coverage)
            COVERAGE=true
            shift
            ;;
        -l|--live)
            LIVE_TESTS=true
            export RUN_LIVE_TESTS=1
            shift
            ;;
        -p|--python)
            PYTHON_CMD="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        unit|integration|http|comfyui|live|all)
            TEST_TYPE="$1"
            shift
            ;;
        *)
            print_color "$RED" "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Auto-detect Python if not specified
if [ -z "$PYTHON_CMD" ]; then
    PYTHON_CMD=$(detect_python)
fi

# Check if Python is available
if [ -z "$PYTHON_CMD" ] || ! command -v "$PYTHON_CMD" >/dev/null 2>&1; then
    print_color "$RED" "Error: Python not found"
    print_color "$YELLOW" "Available Python in your system:"
    for cmd in python3.11 python3.10 python3.9 python3.8 python3 python; do
        if command -v "$cmd" >/dev/null 2>&1; then
            print_color "$YELLOW" "  - $cmd ($(command -v "$cmd"))"
        fi
    done
    exit 1
fi

print_color "$GREEN" "Using Python: $PYTHON_CMD"

# Check if required packages are installed
check_dependencies() {
    print_color "$BLUE" "🔍 Checking dependencies..."

    # Check for required Python packages
    required_packages="torch diffusers PIL"
    missing_packages=""

    for package in $required_packages; do
        if ! "$PYTHON_CMD" -c "import $package" 2>/dev/null; then
            if [ -z "$missing_packages" ]; then
                missing_packages="$package"
            else
                missing_packages="$missing_packages $package"
            fi
        fi
    done

    if [ -n "$missing_packages" ]; then
        print_color "$YELLOW" "⚠️  Missing packages: $missing_packages"
        print_color "$YELLOW" "   Install with: pip install torch diffusers pillow"
    fi

    # Check for coverage if requested
    if [ "$COVERAGE" = true ]; then
        if ! "$PYTHON_CMD" -c "import coverage" 2>/dev/null; then
            print_color "$YELLOW" "⚠️  Coverage package not found"
            print_color "$YELLOW" "   Install with: pip install coverage"
            COVERAGE=false
        fi
    fi
}

# Function to run specific test type
run_tests() {
    local test_type=$1
    local test_cmd=""

    case $test_type in
        unit)
            print_color "$BLUE" "🧪 Running Unit Tests..."
            test_cmd="$PYTHON_CMD run_tests.py --http --comfyui"
            ;;
        integration)
            print_color "$BLUE" "🔗 Running Integration Tests..."
            test_cmd="$PYTHON_CMD -m unittest test_integration.py"
            ;;
        http)
            print_color "$BLUE" "🌐 Running HTTP API Tests..."
            test_cmd="$PYTHON_CMD run_tests.py --http"
            ;;
        comfyui)
            print_color "$BLUE" "🎨 Running ComfyUI Tests..."
            test_cmd="$PYTHON_CMD run_tests.py --comfyui"
            ;;
        live)
            print_color "$BLUE" "🚀 Running Live Server Tests..."
            if [ "$LIVE_TESTS" != true ]; then
                print_color "$YELLOW" "⚠️  Live tests require --live flag or RUN_LIVE_TESTS=1"
                return 1
            fi
            test_cmd="$PYTHON_CMD -m unittest test_integration.TestLiveServerIntegration"
            ;;
        all)
            print_color "$BLUE" "🎯 Running All Tests..."
            if [ "$COVERAGE" = true ]; then
                test_cmd="$PYTHON_CMD test_config.py --coverage"
            else
                test_cmd="$PYTHON_CMD run_tests.py"
            fi
            ;;
        *)
            print_color "$RED" "Unknown test type: $test_type"
            return 1
            ;;
    esac

    # Add verbose flag if requested
    if [ "$VERBOSE" = true ]; then
        test_cmd="$test_cmd --verbose"
    fi

    # Run the test command
    print_color "$BLUE" "Running: $test_cmd"
    echo "----------------------------------------"

    if eval "$test_cmd"; then
        print_color "$GREEN" "✅ Tests passed!"
        return 0
    else
        print_color "$RED" "❌ Tests failed!"
        return 1
    fi
}

# Function to setup test environment
setup_test_env() {
    print_color "$BLUE" "🛠️  Setting up test environment..."

    # Set environment variables
    export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
    export PYTORCH_DISABLE_CUDA_MALLOC_WARNING=1

    # Create test directories if they don't exist
    mkdir -p "$PROJECT_DIR/test_output" "$PROJECT_DIR/test_models"

    # Set live tests environment variable if requested
    if [ "$LIVE_TESTS" = true ]; then
        export RUN_LIVE_TESTS=1
    fi
}

# Function to cleanup
cleanup() {
    print_color "$BLUE" "🧹 Cleaning up..."

    # Remove test output directories
    if [ -d "$PROJECT_DIR/test_output" ]; then
        rm -rf "$PROJECT_DIR/test_output"
    fi

    if [ -d "$PROJECT_DIR/test_models" ]; then
        rm -rf "$PROJECT_DIR/test_models"
    fi

    # Remove Python cache
    find "$PROJECT_DIR" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    find "$PROJECT_DIR" -name "*.pyc" -delete 2>/dev/null || true
}

# Function to start test server for live tests
start_test_server() {
    if [ "$TEST_TYPE" = "live" ] || ([ "$TEST_TYPE" = "all" ] && [ "$LIVE_TESTS" = true ]); then
        print_color "$BLUE" "🚀 Starting test server..."

        # Check if server is already running
        if curl -s "http://127.0.0.1:8999/health" >/dev/null 2>&1; then
            print_color "$GREEN" "✅ Test server already running"
        else
            print_color "$YELLOW" "⚠️  Test server not running. Start with:"
            print_color "$YELLOW" "   $PYTHON_CMD main.py --model test/model --http-port 8999 --device cpu"
            print_color "$YELLOW" "   Then run tests again"
            return 1
        fi
    fi
    return 0
}

# Function to generate test report
generate_report() {
    local test_result=$1

    print_color "$BLUE" "📊 Generating test report..."

    echo "========================================"
    echo "STABLE DIFFUSION MCP SERVER TEST REPORT"
    echo "========================================"
    echo "Test Type: $TEST_TYPE"
    echo "Python Version: $($PYTHON_CMD --version 2>&1)"
    echo "Python Path: $(command -v "$PYTHON_CMD")"
    echo "Date: $(date)"
    echo "Coverage: $([ "$COVERAGE" = true ] && echo "Enabled" || echo "Disabled")"
    echo "Live Tests: $([ "$LIVE_TESTS" = true ] && echo "Enabled" || echo "Disabled")"
    echo "----------------------------------------"

    if [ $test_result -eq 0 ]; then
        print_color "$GREEN" "Result: ✅ PASSED"
    else
        print_color "$RED" "Result: ❌ FAILED"
    fi

    echo "========================================"

    # Show coverage report location if generated
    if [ "$COVERAGE" = true ] && [ -d "htmlcov" ]; then
        print_color "$BLUE" "📈 Coverage report: file://$(pwd)/htmlcov/index.html"
    fi
}

# Main execution
main() {
    print_color "$GREEN" "🧪 Stable Diffusion MCP Server Test Runner"
    print_color "$GREEN" "==========================================="

    # Change to script directory (where tests are located)
    cd "$SCRIPT_DIR"

    # Setup
    setup_test_env
    check_dependencies

    # Start test server if needed
    if ! start_test_server; then
        exit 1
    fi

    # Run tests
    if run_tests "$TEST_TYPE"; then
        test_result=0
    else
        test_result=1
    fi

    # Generate report
    generate_report $test_result

    # Cleanup
    cleanup

    exit $test_result
}

# Trap to ensure cleanup happens even if script is interrupted
trap cleanup EXIT

# Run main function
main "$@"