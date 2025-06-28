#!/usr/bin/env python3
"""
Integration tests for the complete Stable Diffusion MCP Server
These tests require actual server components to be running
"""

import json
import unittest
import requests
import time
import threading
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_server import StableDiffusionServer
from http_server import HTTPServerManager
from model_loader import ModelLoader
from image_generator import ImageGenerator
import torch


class TestModelLoader(unittest.TestCase):
    """Test cases for model loading functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_cache_dir = "/tmp/test_model_cache"
        self.test_device = "cpu"  # Use CPU for testing to avoid GPU dependencies
        self.test_dtype = torch.float32
        self.logger = Mock()

        # Create test directory
        import pathlib
        pathlib.Path(self.test_cache_dir).mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        if os.path.exists(self.test_cache_dir):
            shutil.rmtree(self.test_cache_dir)

    def test_detect_sdxl_model(self):
        """Test SDXL model detection"""
        loader = ModelLoader(
            cache_dir=self.test_cache_dir,
            device=self.test_device,
            torch_dtype=self.test_dtype,
            is_sdxl=False,
            logger=self.logger
        )

        # Test SDXL models
        sdxl_models = [
            "SG161222/RealVisXL_V4.0",
            "stabilityai/stable-diffusion-xl-base-1.0",
            "lykon/dreamshaper-xl-1-0",
            "RunDiffusion/Juggernaut-XL-v9"
        ]

        for model in sdxl_models:
            self.assertTrue(loader.detect_sdxl_model(model), f"Failed to detect {model} as SDXL")

        # Test non-SDXL models
        non_sdxl_models = [
            "SG161222/Realistic_Vision_V2.0",
            "runwayml/stable-diffusion-v1-5",
            "lykon/dreamshaper-8",
            "lykon/dreamshaper-7"
        ]

        for model in non_sdxl_models:
            self.assertFalse(loader.detect_sdxl_model(model), f"Incorrectly detected {model} as SDXL")

    def test_is_model_cached(self):
        """Test model cache detection"""
        loader = ModelLoader(
            cache_dir=self.test_cache_dir,
            device=self.test_device,
            torch_dtype=self.test_dtype,
            is_sdxl=False,
            logger=self.logger
        )

        # Test non-existent model
        self.assertFalse(loader.is_model_cached("nonexistent/model"))

        # Create fake cache structure
        import pathlib
        model_id = "test/model"
        cache_name = f"models--{model_id.replace('/', '--')}"
        model_cache_path = pathlib.Path(self.test_cache_dir) / "hub" / cache_name / "snapshots" / "abc123"
        model_cache_path.mkdir(parents=True, exist_ok=True)

        # Create required files
        (model_cache_path / "model_index.json").touch()
        (model_cache_path / "unet").mkdir(exist_ok=True)
        (model_cache_path / "unet" / "config.json").touch()
        (model_cache_path / "text_encoder").mkdir(exist_ok=True)
        (model_cache_path / "text_encoder" / "config.json").touch()

        # Test cached model
        self.assertTrue(loader.is_model_cached(model_id))

    @patch('model_loader.StableDiffusionPipeline')
    @patch('model_loader.ACCELERATE_AVAILABLE', False)
    def test_load_model_basic(self, mock_pipeline_class):
        """Test basic model loading"""
        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline

        loader = ModelLoader(
            cache_dir=self.test_cache_dir,
            device=self.test_device,
            torch_dtype=self.test_dtype,
            is_sdxl=False,
            logger=self.logger
        )

        # Mock successful loading
        with patch.object(loader, '_move_to_device', return_value=mock_pipeline), \
                patch.object(loader, '_apply_optimizations', return_value=mock_pipeline), \
                patch.object(loader, '_run_validation_test'):
            result = loader.load_model("test/model")
            self.assertEqual(result, mock_pipeline)

    def test_clear_cache(self):
        """Test cache clearing functionality"""
        loader = ModelLoader(
            cache_dir=self.test_cache_dir,
            device=self.test_device,
            torch_dtype=self.test_dtype,
            is_sdxl=False,
            logger=self.logger
        )

        # Create fake cache
        import pathlib
        hub_cache = pathlib.Path(self.test_cache_dir) / "hub"
        hub_cache.mkdir(parents=True, exist_ok=True)
        (hub_cache / "test_file.txt").touch()

        self.assertTrue(hub_cache.exists())
        self.assertTrue((hub_cache / "test_file.txt").exists())

        # Clear cache
        result = loader.clear_cache()

        self.assertTrue(result)
        self.assertTrue(hub_cache.exists())  # Directory should be recreated
        self.assertFalse((hub_cache / "test_file.txt").exists())  # File should be gone


class TestImageGenerator(unittest.TestCase):
    """Test cases for image generation functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_output_dir = "/tmp/test_images"
        self.test_device = "cpu"
        self.test_dtype = torch.float32
        self.logger = Mock()

        # Create test directory
        import pathlib
        pathlib.Path(self.test_output_dir).mkdir(parents=True, exist_ok=True)

        # Mock pipeline
        self.mock_pipeline = Mock()
        self.mock_pipeline.scheduler = Mock()
        self.mock_pipeline.scheduler.config = {}
        self.mock_pipeline.tokenizer = Mock()
        self.mock_pipeline.text_encoder = Mock()
        self.mock_pipeline.text_encoder.config = Mock()
        self.mock_pipeline.text_encoder.config.hidden_size = 768
        self.mock_pipeline.tokenizer.model_max_length = 77

        # Mock successful generation
        mock_image = Mock()
        mock_image.mode = "RGB"
        mock_image.save = Mock()

        mock_result = Mock()
        mock_result.images = [mock_image]
        self.mock_pipeline.return_value = mock_result

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        if os.path.exists(self.test_output_dir):
            shutil.rmtree(self.test_output_dir)

    def test_validate_dimensions(self):
        """Test dimension validation"""
        generator = ImageGenerator(
            pipeline=self.mock_pipeline,
            device=self.test_device,
            torch_dtype=self.test_dtype,
            is_sdxl=False,
            output_dir=self.test_output_dir,
            logger=self.logger
        )

        # Test valid dimensions
        width, height = generator._validate_dimensions(512, 512)
        self.assertEqual(width, 512)
        self.assertEqual(height, 512)

        # Test dimensions not divisible by 8
        width, height = generator._validate_dimensions(513, 515)
        self.assertEqual(width, 512)  # Should be rounded down
        self.assertEqual(height, 512)  # Should be rounded down

        # Test very large dimensions
        width, height = generator._validate_dimensions(2048, 2048)
        self.assertLessEqual(width * height, 1024 * 1024)  # Should be scaled down

    def test_needs_chunking(self):
        """Test prompt chunking detection"""
        generator = ImageGenerator(
            pipeline=self.mock_pipeline,
            device=self.test_device,
            torch_dtype=self.test_dtype,
            is_sdxl=False,
            output_dir=self.test_output_dir,
            logger=self.logger
        )

        # Mock tokenizer
        def mock_encode(text):
            # Simulate token length based on text length
            return list(range(len(text.split())))

        generator.pipeline.tokenizer.encode = mock_encode

        # Short prompt should not need chunking
        short_prompt = "a beautiful landscape"
        self.assertFalse(generator._needs_chunking(short_prompt))

        # Long prompt should need chunking
        long_prompt = " ".join(["word"] * 100)  # 100 words
        self.assertTrue(generator._needs_chunking(long_prompt))

        # Empty prompt should not need chunking
        self.assertFalse(generator._needs_chunking(""))
        self.assertFalse(generator._needs_chunking("   "))

    @patch('torch.Generator')
    def test_setup_generator(self, mock_generator_class):
        """Test random generator setup"""
        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator

        generator = ImageGenerator(
            pipeline=self.mock_pipeline,
            device=self.test_device,
            torch_dtype=self.test_dtype,
            is_sdxl=False,
            output_dir=self.test_output_dir,
            logger=self.logger
        )

        # Test with seed
        result = generator._setup_generator(12345)
        mock_generator_class.assert_called_with(device=self.test_device)
        mock_generator.manual_seed.assert_called_with(12345)

        # Test without seed
        with patch('torch.randint', return_value=Mock(item=Mock(return_value=54321))):
            result = generator._setup_generator(None)
            mock_generator.manual_seed.assert_called_with(54321)

    @patch('time.time')
    @patch('uuid.uuid4')
    def test_generate_image_basic(self, mock_uuid, mock_time):
        """Test basic image generation"""
        # Mock time and UUID
        mock_time.side_effect = [1000.0, 1003.5]  # Start and end time
        mock_uuid.return_value = Mock(__str__=Mock(return_value="abc123def"))

        generator = ImageGenerator(
            pipeline=self.mock_pipeline,
            device=self.test_device,
            torch_dtype=self.test_dtype,
            is_sdxl=False,
            output_dir=self.test_output_dir,
            logger=self.logger
        )

        # Mock image saving
        with patch.object(generator, '_save_images', return_value=["/images/test.png"]):
            result = generator.generate_image("test prompt")

        # Verify result structure
        self.assertIn("images", result)
        self.assertIn("metadata", result)
        self.assertIn("generation_time", result)
        self.assertIn("request_id", result)

        # Verify metadata
        metadata = result["metadata"]
        self.assertEqual(metadata["prompt"], "test prompt")
        self.assertIn("negative_prompt", metadata)
        self.assertIn("width", metadata)
        self.assertIn("height", metadata)

        # Verify generation time
        self.assertEqual(result["generation_time"], 3.5)

    def test_save_images(self):
        """Test image saving functionality"""
        generator = ImageGenerator(
            pipeline=self.mock_pipeline,
            device=self.test_device,
            torch_dtype=self.test_dtype,
            is_sdxl=False,
            output_dir=self.test_output_dir,
            logger=self.logger
        )

        # Mock images
        mock_images = []
        for i in range(2):
            mock_image = Mock()
            mock_image.mode = "RGB"
            mock_image.save = Mock()
            mock_images.append(mock_image)

        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "20241227_120000"

            result = generator._save_images(mock_images, "test123")

        # Verify paths
        self.assertEqual(len(result), 2)
        self.assertTrue(all(path.startswith("/images/") for path in result))
        self.assertTrue(all("test123" in path for path in result))

        # Verify save was called
        for mock_image in mock_images:
            mock_image.save.assert_called_once()


class TestStableDiffusionServer(unittest.TestCase):
    """Test cases for the main server class"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_output_dir = "/tmp/test_server_output"
        self.test_models_dir = "/tmp/test_server_models"

        # Create test directories
        import pathlib
        pathlib.Path(self.test_output_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.test_models_dir).mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        for dir_path in [self.test_output_dir, self.test_models_dir]:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)

    def test_device_detection(self):
        """Test device detection logic"""
        server = StableDiffusionServer(
            model_id="test/model",
            device="auto",
            output_dir=self.test_output_dir,
            models_dir=self.test_models_dir
        )

        # Test auto detection (will fall back to CPU in test environment)
        device = server._get_device("auto")
        self.assertIn(device, ["cpu", "cuda", "mps"])

        # Test explicit CPU
        device = server._get_device("cpu")
        self.assertEqual(device, "cpu")

        # Test invalid CUDA request (should fall back to CPU)
        with patch('torch.cuda.is_available', return_value=False):
            device = server._get_device("cuda")
            self.assertEqual(device, "cpu")

    def test_torch_dtype_conversion(self):
        """Test torch dtype conversion"""
        server = StableDiffusionServer(
            model_id="test/model",
            device="cpu",
            output_dir=self.test_output_dir,
            models_dir=self.test_models_dir
        )

        # Test fp32
        dtype = server._get_torch_dtype("fp32")
        self.assertEqual(dtype, torch.float32)

        # Test fp16 on CPU (should fall back to fp32)
        dtype = server._get_torch_dtype("fp16")
        self.assertEqual(dtype, torch.float32)

        # Test bf16 on CPU (should fall back to fp32)
        dtype = server._get_torch_dtype("bf16")
        self.assertEqual(dtype, torch.float32)

    def test_sdxl_detection(self):
        """Test SDXL model detection"""
        server = StableDiffusionServer(
            model_id="test/model",
            device="cpu",
            output_dir=self.test_output_dir,
            models_dir=self.test_models_dir
        )

        # Test SDXL models
        self.assertTrue(server._detect_sdxl_model("SG161222/RealVisXL_V4.0"))
        self.assertTrue(server._detect_sdxl_model("stabilityai/stable-diffusion-xl-base-1.0"))
        self.assertTrue(server._detect_sdxl_model("lykon/dreamshaper-xl-1-0"))

        # Test non-SDXL models
        self.assertFalse(server._detect_sdxl_model("SG161222/Realistic_Vision_V2.0"))
        self.assertFalse(server._detect_sdxl_model("runwayml/stable-diffusion-v1-5"))

    def test_get_available_models(self):
        """Test getting available models"""
        with patch.object(StableDiffusionServer, '_initialize_components'), \
                patch.object(StableDiffusionServer, '_setup_environment'), \
                patch.object(StableDiffusionServer, 'setup_logging'):
            server = StableDiffusionServer(
                model_id="test/model",
                device="cpu",
                output_dir=self.test_output_dir,
                models_dir=self.test_models_dir
            )

            # Mock model loader
            server.model_loader = Mock()
            server.model_loader.is_model_cached.return_value = False

            models = server.get_available_models()

            # Verify structure
            self.assertIn("current_model", models)
            self.assertIn("current_model_ready", models)
            self.assertIn("available_models", models)

            # Verify current model
            self.assertEqual(models["current_model"], "test/model")
            self.assertEqual(models["current_model_ready"], False)  # Not loaded yet

            # Verify available models list
            self.assertIsInstance(models["available_models"], list)
            self.assertGreater(len(models["available_models"]), 0)

    @patch('threading.Thread')
    def test_generate_image_not_ready(self, mock_thread):
        """Test image generation when model is not ready"""
        with patch.object(StableDiffusionServer, '_initialize_components'), \
                patch.object(StableDiffusionServer, '_setup_environment'), \
                patch.object(StableDiffusionServer, 'setup_logging'):
            server = StableDiffusionServer(
                model_id="test/model",
                device="cpu",
                output_dir=self.test_output_dir,
                models_dir=self.test_models_dir
            )

            # Model not ready
            server.model_ready = False
            server.model_loading = False

            # Should raise RuntimeError and start loading
            with self.assertRaises(RuntimeError) as context:
                server.generate_image("test prompt")

            self.assertIn("Model not ready", str(context.exception))
            mock_thread.assert_called_once()

    @patch('gc.collect')
    @patch('torch.cuda.empty_cache')
    def test_switch_model(self, mock_empty_cache, mock_gc):
        """Test model switching"""
        with patch.object(StableDiffusionServer, '_initialize_components'), \
                patch.object(StableDiffusionServer, '_setup_environment'), \
                patch.object(StableDiffusionServer, 'setup_logging'), \
                patch('threading.Thread'):
            server = StableDiffusionServer(
                model_id="old/model",
                device="cpu",
                output_dir=self.test_output_dir,
                models_dir=self.test_models_dir
            )

            # Mock components
            server.pipeline = Mock()
            server.image_generator = Mock()
            server.model_loader = Mock()
            server.model_ready = True
            server.model_loading = False

            # Test switching to same model
            result = server.switch_model("old/model")
            self.assertTrue(result["success"])
            self.assertIn("Already using", result["message"])

            # Test switching to new model
            result = server.switch_model("new/model")
            self.assertTrue(result["success"])
            self.assertIn("Switching to", result["message"])
            self.assertEqual(result["old_model"], "old/model")
            self.assertEqual(result["new_model"], "new/model")
            self.assertEqual(server.model_id, "new/model")

            # Verify cleanup was called
            mock_gc.assert_called()

    def test_clear_model_cache(self):
        """Test model cache clearing"""
        with patch.object(StableDiffusionServer, '_initialize_components'), \
                patch.object(StableDiffusionServer, '_setup_environment'), \
                patch.object(StableDiffusionServer, 'setup_logging'):
            server = StableDiffusionServer(
                model_id="test/model",
                device="cpu",
                output_dir=self.test_output_dir,
                models_dir=self.test_models_dir
            )

            # Mock components
            server.pipeline = Mock()
            server.image_generator = Mock()
            server.model_loader = Mock()
            server.model_loader.clear_cache.return_value = True
            server.model_ready = True

            result = server.clear_model_cache()

            self.assertTrue(result["success"])
            self.assertTrue(result["cache_cleared"])
            self.assertFalse(result["model_ready"])
            self.assertEqual(result["current_model"], "test/model")

            # Verify model loader clear_cache was called
            server.model_loader.clear_cache.assert_called_once()


class TestLiveServerIntegration(unittest.TestCase):
    """Live integration tests that start an actual server"""

    @classmethod
    def setUpClass(cls):
        """Set up a test server instance"""
        cls.test_port = 8999
        cls.base_url = f"http://127.0.0.1:{cls.test_port}"

        # Skip if we can't import required modules
        try:
            import requests
        except ImportError:
            raise unittest.SkipTest("requests library not available")

    def setUp(self):
        """Set up for each test"""
        # Skip live tests by default unless specifically enabled
        if not os.environ.get('RUN_LIVE_TESTS'):
            self.skipTest("Live tests disabled. Set RUN_LIVE_TESTS=1 to enable")

    def test_server_health_check(self):
        """Test server health check endpoint"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            self.assertEqual(response.status_code, 200)

            data = response.json()
            self.assertIn("status", data)
            self.assertIn("model_id", data)
            self.assertIn("device", data)

        except requests.exceptions.ConnectionError:
            self.skipTest("Test server not running")

    def test_server_models_endpoint(self):
        """Test server models endpoint"""
        try:
            response = requests.get(f"{self.base_url}/models", timeout=5)
            self.assertEqual(response.status_code, 200)

            data = response.json()
            self.assertIn("models", data)
            models_info = data["models"]
            self.assertIn("current_model", models_info)
            self.assertIn("available_models", models_info)

        except requests.exceptions.ConnectionError:
            self.skipTest("Test server not running")

    def test_server_cors_headers(self):
        """Test CORS headers on live server"""
        try:
            response = requests.options(f"{self.base_url}/generate", timeout=5)
            self.assertEqual(response.status_code, 200)

            headers = response.headers
            self.assertEqual(headers.get("Access-Control-Allow-Origin"), "*")
            self.assertIn("POST", headers.get("Access-Control-Allow-Methods", ""))

        except requests.exceptions.ConnectionError:
            self.skipTest("Test server not running")

    def test_server_generation_endpoint_validation(self):
        """Test generation endpoint validation on live server"""
        try:
            # Test missing prompt
            response = requests.post(
                f"{self.base_url}/generate",
                json={},
                timeout=30
            )
            self.assertEqual(response.status_code, 400)

            # Test empty prompt
            response = requests.post(
                f"{self.base_url}/generate",
                json={"prompt": ""},
                timeout=30
            )
            self.assertEqual(response.status_code, 400)

        except requests.exceptions.ConnectionError:
            self.skipTest("Test server not running")


if __name__ == '__main__':
    # Set up test environment
    os.environ.setdefault('PYTORCH_DISABLE_CUDA_MALLOC_WARNING', '1')

    # Run tests
    unittest.main()