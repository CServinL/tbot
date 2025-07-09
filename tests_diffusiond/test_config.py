#!/usr/bin/env python3
"""
Test configuration and utilities for Stable Diffusion MCP Server tests
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import logging

# Test configuration
TEST_CONFIG = {
    "test_model_id": "test/mock-model",
    "test_device": "cpu",  # Use CPU for testing to avoid GPU dependencies
    "test_precision": "fp32",  # Use fp32 for CPU testing
    "test_output_dir": None,  # Will be set to temp directory
    "test_models_dir": None,  # Will be set to temp directory
    "test_port": 8999,
    "test_host": "127.0.0.1",
    "enable_live_tests": False,  # Set to True to enable live server tests
    "test_timeout": 30,  # Timeout for HTTP requests in tests
}


class TestEnvironment:
    """Test environment setup and teardown"""

    def __init__(self):
        self.temp_dirs = []
        self.original_env = {}

    def __enter__(self):
        """Set up test environment"""
        # Create temporary directories
        TEST_CONFIG["test_output_dir"] = tempfile.mkdtemp(prefix="test_output_")
        TEST_CONFIG["test_models_dir"] = tempfile.mkdtemp(prefix="test_models_")

        self.temp_dirs.extend([
            TEST_CONFIG["test_output_dir"],
            TEST_CONFIG["test_models_dir"]
        ])

        # Set environment variables for testing
        test_env_vars = {
            "PYTORCH_DISABLE_CUDA_MALLOC_WARNING": "1",
            "TRANSFORMERS_OFFLINE": "1",  # Prevent automatic downloads during tests
            "HF_DATASETS_OFFLINE": "1",
            "DIFFUSERS_OFFLINE": "1",
        }

        for key, value in test_env_vars.items():
            self.original_env[key] = os.environ.get(key)
            os.environ[key] = value

        # Configure logging for tests
        logging.getLogger().setLevel(logging.WARNING)  # Reduce noise in tests

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up test environment"""
        import shutil

        # Clean up temporary directories
        for temp_dir in self.temp_dirs:
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    print(f"Warning: Could not clean up {temp_dir}: {e}")

        # Restore environment variables
        for key, original_value in self.original_env.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value


class MockTorchComponents:
    """Mock PyTorch components for testing without GPU dependencies"""

    @staticmethod
    def create_mock_pipeline():
        """Create a mock Stable Diffusion pipeline"""
        mock_pipeline = Mock()

        # Mock pipeline components
        mock_pipeline.unet = Mock()
        mock_pipeline.vae = Mock()
        mock_pipeline.text_encoder = Mock()
        mock_pipeline.text_encoder_2 = Mock()
        mock_pipeline.tokenizer = Mock()
        mock_pipeline.scheduler = Mock()

        # Mock component configurations
        mock_pipeline.text_encoder.config = Mock()
        mock_pipeline.text_encoder.config.hidden_size = 768
        mock_pipeline.tokenizer.model_max_length = 77
        mock_pipeline.scheduler.config = {}

        # Mock device properties
        mock_parameters = Mock()
        mock_parameters.device = "cpu"
        mock_pipeline.unet.parameters.return_value = [mock_parameters]

        # Mock generation
        mock_image = Mock()
        mock_image.mode = "RGB"
        mock_image.save = Mock()

        mock_result = Mock()
        mock_result.images = [mock_image]
        mock_pipeline.return_value = mock_result

        return mock_pipeline

    @staticmethod
    def create_mock_torch():
        """Create mock torch module"""
        mock_torch = Mock()

        # Mock basic torch functions
        mock_torch.float32 = "float32"
        mock_torch.float16 = "float16"
        mock_torch.bfloat16 = "bfloat16"

        # Mock CUDA functions
        mock_torch.cuda = Mock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.cuda.device_count.return_value = 0

        # Mock MPS functions
        mock_torch.backends = Mock()
        mock_torch.backends.mps = Mock()
        mock_torch.backends.mps.is_available.return_value = False

        # Mock Generator
        mock_generator = Mock()
        mock_generator.manual_seed.return_value = mock_generator
        mock_torch.Generator.return_value = mock_generator

        # Mock tensor operations
        mock_torch.zeros.return_value = Mock()
        mock_torch.randint.return_value = Mock(item=Mock(return_value=12345))
        mock_torch.no_grad.return_value.__enter__ = Mock()
        mock_torch.no_grad.return_value.__exit__ = Mock()

        return mock_torch


class BaseTestCase(unittest.TestCase):
    """Base test case with common setup"""

    def setUp(self):
        """Set up common test fixtures"""
        self.test_env = TestEnvironment()
        self.test_env.__enter__()

        # Create mock components
        self.mock_torch = MockTorchComponents.create_mock_torch()
        self.mock_pipeline = MockTorchComponents.create_mock_pipeline()

        # Standard mocks for SD server
        self.mock_sd_server = Mock()
        self.mock_sd_server.logger = Mock()
        self.mock_sd_server.model_ready = True
        self.mock_sd_server.model_loading = False
        self.mock_sd_server.model_id = TEST_CONFIG["test_model_id"]
        self.mock_sd_server.device = TEST_CONFIG["test_device"]
        self.mock_sd_server.precision = TEST_CONFIG["test_precision"]
        self.mock_sd_server.attention_precision = TEST_CONFIG["test_precision"]
        self.mock_sd_server.is_sdxl = False
        self.mock_sd_server.output_dir = TEST_CONFIG["test_output_dir"]

        # Mock generation result
        self.mock_generation_result = {
            "images": ["/images/test_image_001.png"],
            "metadata": {
                "prompt": "test prompt",
                "negative_prompt": "test negative",
                "width": 512,
                "height": 512,
                "num_inference_steps": 20,
                "guidance_scale": 7.5,
                "seed": 12345
            },
            "generation_time": 3.2,
            "request_id": "test123"
        }
        self.mock_sd_server.generate_image.return_value = self.mock_generation_result

    def tearDown(self):
        """Clean up test fixtures"""
        self.test_env.__exit__(None, None, None)


def skip_if_no_gpu(test_func):
    """Decorator to skip tests that require GPU"""

    def wrapper(self):
        try:
            import torch
            if not torch.cuda.is_available():
                self.skipTest("GPU not available")
        except ImportError:
            self.skipTest("PyTorch not available")
        return test_func(self)

    return wrapper


def skip_if_no_internet(test_func):
    """Decorator to skip tests that require internet"""

    def wrapper(self):
        import socket
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
        except OSError:
            self.skipTest("Internet connection not available")
        return test_func(self)

    return wrapper


def skip_live_tests(test_func):
    """Decorator to skip live server tests unless explicitly enabled"""

    def wrapper(self):
        if not (TEST_CONFIG["enable_live_tests"] or os.environ.get("RUN_LIVE_TESTS")):
            self.skipTest("Live tests disabled. Set RUN_LIVE_TESTS=1 to enable")
        return test_func(self)

    return wrapper


class TestSuiteBuilder:
    """Builder for creating test suites"""

    @staticmethod
    def build_unit_test_suite():
        """Build unit test suite"""
        from test_http_api import TestHTTPAPIEndpoints
        from test_comfyui_api import TestComfyUIAdapter

        suite = unittest.TestSuite()
        suite.addTest(unittest.makeSuite(TestHTTPAPIEndpoints))
        suite.addTest(unittest.makeSuite(TestComfyUIAdapter))

        return suite

    @staticmethod
    def build_integration_test_suite():
        """Build integration test suite"""
        from test_http_api import TestHTTPAPIIntegration
        from test_comfyui_api import TestComfyUIHTTPIntegration
        from test_integration import TestModelLoader, TestImageGenerator, TestStableDiffusionServer

        suite = unittest.TestSuite()
        suite.addTest(unittest.makeSuite(TestHTTPAPIIntegration))
        suite.addTest(unittest.makeSuite(TestComfyUIHTTPIntegration))
        suite.addTest(unittest.makeSuite(TestModelLoader))
        suite.addTest(unittest.makeSuite(TestImageGenerator))
        suite.addTest(unittest.makeSuite(TestStableDiffusionServer))

        return suite

    @staticmethod
    def build_live_test_suite():
        """Build live test suite (requires running server)"""
        from test_integration import TestLiveServerIntegration

        suite = unittest.TestSuite()
        suite.addTest(unittest.makeSuite(TestLiveServerIntegration))

        return suite

    @staticmethod
    def build_all_tests_suite():
        """Build complete test suite"""
        all_suite = unittest.TestSuite()
        all_suite.addTest(TestSuiteBuilder.build_unit_test_suite())
        all_suite.addTest(TestSuiteBuilder.build_integration_test_suite())

        # Only add live tests if enabled
        if TEST_CONFIG["enable_live_tests"] or os.environ.get("RUN_LIVE_TESTS"):
            all_suite.addTest(TestSuiteBuilder.build_live_test_suite())

        return all_suite


def run_with_coverage():
    """Run tests with coverage analysis"""
    try:
        import coverage

        # Start coverage
        cov = coverage.Coverage(
            source=['mcp_server', 'http_server', 'model_loader',
                    'image_generator', 'comfyui_adapter', 'comfyui_http_handler'],
            omit=['*/tests/*', '*/test_*']
        )
        cov.start()

        # Run tests
        suite = TestSuiteBuilder.build_all_tests_suite()
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)

        # Stop coverage and generate report
        cov.stop()
        cov.save()

        print("\n" + "=" * 50)
        print("COVERAGE REPORT")
        print("=" * 50)
        cov.report()

        # Generate HTML report
        try:
            cov.html_report(directory='htmlcov')
            print("\nDetailed HTML coverage report generated in 'htmlcov' directory")
        except Exception as e:
            print(f"Could not generate HTML report: {e}")

        return result.wasSuccessful()

    except ImportError:
        print("Coverage.py not installed. Install with: pip install coverage")
        return False


if __name__ == "__main__":
    # Enable live tests if requested
    if len(sys.argv) > 1 and sys.argv[1] == "--live":
        TEST_CONFIG["enable_live_tests"] = True
        sys.argv.remove("--live")

    # Run with coverage if requested
    if len(sys.argv) > 1 and sys.argv[1] == "--coverage":
        success = run_with_coverage()
        sys.exit(0 if success else 1)

    # Default: run all tests
    with TestEnvironment():
        suite = TestSuiteBuilder.build_all_tests_suite()
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)