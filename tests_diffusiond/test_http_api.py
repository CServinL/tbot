#!/usr/bin/env python3
"""
Unit tests for the HTTP API endpoints
"""

import json
import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from io import BytesIO
from http.server import HTTPServer
import threading
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from http_server import StableDiffusionHTTPHandler, HTTPServerManager
from mcp_server import StableDiffusionServer


class MockRequest:
    """Mock HTTP request for testing"""

    def __init__(self, method="GET", path="/", headers=None, body=b""):
        self.method = method
        self.path = path
        self.headers = headers or {}
        self.body = body
        self.rfile = BytesIO(body)
        self.wfile = BytesIO()


class TestHTTPAPIEndpoints(unittest.TestCase):
    """Test cases for HTTP API endpoints"""

    def setUp(self):
        """Set up test fixtures"""
        # Mock SD server
        self.mock_sd_server = Mock()
        self.mock_sd_server.logger = Mock()
        self.mock_sd_server.model_ready = True
        self.mock_sd_server.model_loading = False
        self.mock_sd_server.model_id = "test_model"
        self.mock_sd_server.device = "cuda"
        self.mock_sd_server.precision = "fp16"
        self.mock_sd_server.attention_precision = "fp16"
        self.mock_sd_server.is_sdxl = False
        self.mock_sd_server.output_dir = "/tmp/test_images"

        # Mock generation response
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
            "request_id": "abc123"
        }
        self.mock_sd_server.generate_image.return_value = self.mock_generation_result

        # Mock available models
        self.mock_models = {
            "current_model": "test_model",
            "current_model_ready": True,
            "available_models": [
                {"model_id": "test_model", "cached": True, "type": "SD 1.5"},
                {"model_id": "another_model", "cached": False, "type": "SDXL"}
            ]
        }
        self.mock_sd_server.get_available_models.return_value = self.mock_models

    def create_handler(self, method="GET", path="/", headers=None, body=b""):
        """Create a handler instance for testing"""

        # Create a mock request
        class MockHTTPHandler(StableDiffusionHTTPHandler):
            def __init__(self, sd_server, method, path, headers, body):
                self.sd_server = sd_server
                self.command = method
                self.path = path
                self.headers = headers or {}
                self.rfile = BytesIO(body)
                self.wfile = BytesIO()
                self._response_code = None
                self._response_headers = {}
                self._response_body = b""

            def send_response(self, code):
                self._response_code = code

            def send_header(self, name, value):
                self._response_headers[name] = value

            def end_headers(self):
                pass

            def address_string(self):
                return "127.0.0.1"

            def log_message(self, format, *args):
                pass

        return MockHTTPHandler(self.mock_sd_server, method, path, headers, body)

    def test_health_check_endpoint(self):
        """Test /health endpoint"""
        handler = self.create_handler("GET", "/health")
        handler.handle_health_check()

        # Verify response code
        self.assertEqual(handler._response_code, 200)
        self.assertEqual(handler._response_headers.get('Content-Type'), 'application/json')

    def test_models_endpoint(self):
        """Test /models endpoint"""
        handler = self.create_handler("GET", "/models")
        handler.handle_list_models()

        # Verify response code
        self.assertEqual(handler._response_code, 200)
        self.assertEqual(handler._response_headers.get('Content-Type'), 'application/json')

        # Verify models endpoint was called
        self.mock_sd_server.get_available_models.assert_called_once()

    def test_generate_endpoint_success(self):
        """Test successful image generation"""
        request_data = {
            "prompt": "a beautiful sunset over mountains",
            "negative_prompt": "blurry, low quality",
            "width": 768,
            "height": 512,
            "num_inference_steps": 25,
            "guidance_scale": 8.0,
            "seed": 42
        }

        body = json.dumps(request_data).encode('utf-8')
        headers = {"Content-Length": str(len(body))}

        handler = self.create_handler("POST", "/generate", headers, body)
        handler.handle_generate()

        # Verify response code
        self.assertEqual(handler._response_code, 200)

        # Verify generation was called with correct parameters
        self.mock_sd_server.generate_image.assert_called_once()
        args, kwargs = self.mock_sd_server.generate_image.call_args

        self.assertEqual(kwargs["prompt"] if "prompt" in kwargs else args[0], "a beautiful sunset over mountains")
        self.assertEqual(kwargs.get("negative_prompt"), "blurry, low quality")
        self.assertEqual(kwargs.get("width"), 768)
        self.assertEqual(kwargs.get("height"), 512)
        self.assertEqual(kwargs.get("num_inference_steps"), 25)
        self.assertEqual(kwargs.get("guidance_scale"), 8.0)
        self.assertEqual(kwargs.get("seed"), 42)

    def test_generate_endpoint_missing_prompt(self):
        """Test generation with missing prompt"""
        request_data = {
            "width": 512,
            "height": 512
        }

        body = json.dumps(request_data).encode('utf-8')
        headers = {"Content-Length": str(len(body))}

        handler = self.create_handler("POST", "/generate", headers, body)
        handler.handle_generate()

        # Verify error response
        self.assertEqual(handler._response_code, 400)

    def test_generate_endpoint_empty_prompt(self):
        """Test generation with empty prompt"""
        request_data = {
            "prompt": "",
            "width": 512,
            "height": 512
        }

        body = json.dumps(request_data).encode('utf-8')
        headers = {"Content-Length": str(len(body))}

        handler = self.create_handler("POST", "/generate", headers, body)
        handler.handle_generate()

        # Verify error response
        self.assertEqual(handler._response_code, 400)

    def test_generate_endpoint_model_not_ready(self):
        """Test generation when model is not ready"""
        # Mock model not ready
        self.mock_sd_server.generate_image.side_effect = RuntimeError(
            "Model not ready. Please wait for model loading to complete.")

        request_data = {
            "prompt": "test prompt",
            "width": 512,
            "height": 512
        }

        body = json.dumps(request_data).encode('utf-8')
        headers = {"Content-Length": str(len(body))}

        handler = self.create_handler("POST", "/generate", headers, body)
        handler.handle_generate()

        # Verify service unavailable response
        self.assertEqual(handler._response_code, 503)

    def test_generate_endpoint_invalid_json(self):
        """Test generation with invalid JSON"""
        body = b"invalid json content"
        headers = {"Content-Length": str(len(body))}

        handler = self.create_handler("POST", "/generate", headers, body)
        handler.handle_generate()

        # Verify error response
        self.assertEqual(handler._response_code, 400)

    def test_switch_model_endpoint(self):
        """Test model switching endpoint"""
        request_data = {
            "model_id": "new_test_model",
            "precision": "fp32"
        }

        # Mock switch model response
        switch_result = {
            "success": True,
            "message": "Switching to model new_test_model",
            "old_model": "test_model",
            "new_model": "new_test_model",
            "model_type": "SD 1.5",
            "precision": "fp32",
            "attention_precision": "fp16",
            "loading": True
        }
        self.mock_sd_server.switch_model.return_value = switch_result

        body = json.dumps(request_data).encode('utf-8')
        headers = {"Content-Length": str(len(body))}

        handler = self.create_handler("POST", "/switch-model", headers, body)
        handler.handle_switch_model()

        # Verify response
        self.assertEqual(handler._response_code, 200)

        # Verify switch_model was called
        self.mock_sd_server.switch_model.assert_called_once_with("new_test_model", "fp32", "fp16")

    def test_switch_model_missing_model_id(self):
        """Test model switching with missing model_id"""
        request_data = {
            "precision": "fp32"
        }

        body = json.dumps(request_data).encode('utf-8')
        headers = {"Content-Length": str(len(body))}

        handler = self.create_handler("POST", "/switch-model", headers, body)
        handler.handle_switch_model()

        # Verify error response
        self.assertEqual(handler._response_code, 400)

    def test_switch_model_while_loading(self):
        """Test model switching while another model is loading"""
        self.mock_sd_server.switch_model.side_effect = RuntimeError(
            "Cannot switch models while another model is loading")

        request_data = {
            "model_id": "new_test_model"
        }

        body = json.dumps(request_data).encode('utf-8')
        headers = {"Content-Length": str(len(body))}

        handler = self.create_handler("POST", "/switch-model", headers, body)
        handler.handle_switch_model()

        # Verify service unavailable response
        self.assertEqual(handler._response_code, 503)

    def test_clear_cache_endpoint(self):
        """Test cache clearing endpoint"""
        # Mock clear cache response
        clear_result = {
            "success": True,
            "message": "Model cache cleared successfully",
            "cache_cleared": True,
            "current_model": "test_model",
            "model_ready": False,
            "note": "Use /switch-model to reload the current model with fresh files"
        }
        self.mock_sd_server.clear_model_cache.return_value = clear_result

        handler = self.create_handler("POST", "/clear-cache")
        handler.handle_clear_cache()

        # Verify response
        self.assertEqual(handler._response_code, 200)

        # Verify clear_model_cache was called
        self.mock_sd_server.clear_model_cache.assert_called_once()

    def test_cors_headers(self):
        """Test CORS headers are properly set"""
        handler = self.create_handler("OPTIONS", "/generate")
        handler.do_OPTIONS()

        # Verify CORS headers
        self.assertEqual(handler._response_code, 200)
        self.assertEqual(handler._response_headers.get('Access-Control-Allow-Origin'), '*')
        self.assertEqual(handler._response_headers.get('Access-Control-Allow-Methods'), 'GET, POST, OPTIONS')
        self.assertEqual(handler._response_headers.get('Access-Control-Allow-Headers'), 'Content-Type')

    def test_404_for_unknown_endpoint(self):
        """Test 404 response for unknown endpoints"""
        handler = self.create_handler("GET", "/unknown-endpoint")
        handler.do_GET()

        self.assertEqual(handler._response_code, 404)

    def test_method_not_allowed(self):
        """Test 405 response for wrong HTTP methods"""
        # Test POST to GET-only endpoint
        handler = self.create_handler("POST", "/health")
        handler.do_POST()

        self.assertEqual(handler._response_code, 404)  # Falls through to 404 since POST /health isn't handled

    @patch('builtins.open')
    @patch('os.path.exists')
    def test_serve_image_success(self, mock_exists, mock_open):
        """Test serving an existing image"""
        # Mock file exists and can be read
        mock_exists.return_value = True
        mock_file = Mock()
        mock_file.read.side_effect = [b"fake_image_data", b""]  # Return data then empty to stop loop
        mock_file.__enter__.return_value = mock_file
        mock_open.return_value = mock_file

        # Mock file stats
        with patch('pathlib.Path.stat') as mock_stat:
            mock_stat.return_value.st_size = 1024

            handler = self.create_handler("GET", "/images/test_image.png")
            handler.serve_image("test_image.png")

            # Verify response
            self.assertEqual(handler._response_code, 200)
            self.assertEqual(handler._response_headers.get('Content-Type'), 'image/png')

    def test_serve_image_not_found(self):
        """Test serving a non-existent image"""
        with patch('pathlib.Path.exists', return_value=False):
            handler = self.create_handler("GET", "/images/nonexistent.png")
            handler.serve_image("nonexistent.png")

            # Verify 404 response
            self.assertEqual(handler._response_code, 404)


class TestHTTPServerManager(unittest.TestCase):
    """Test cases for HTTP server manager"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_sd_server = Mock()
        self.mock_sd_server.logger = Mock()

    def test_create_handler_factory(self):
        """Test handler factory creation"""
        manager = HTTPServerManager(self.mock_sd_server, "127.0.0.1", 8000)
        factory = manager.create_handler_factory()

        # Test that factory creates handler with sd_server
        handler = factory(None, ("127.0.0.1", 12345), None)
        self.assertEqual(handler.sd_server, self.mock_sd_server)

    def test_server_initialization(self):
        """Test server initialization parameters"""
        manager = HTTPServerManager(self.mock_sd_server, "0.0.0.0", 9000)

        self.assertEqual(manager.sd_server, self.mock_sd_server)
        self.assertEqual(manager.host, "0.0.0.0")
        self.assertEqual(manager.port, 9000)
        self.assertIsNone(manager.http_server)


class TestHTTPAPIIntegration(unittest.TestCase):
    """Integration tests for HTTP API"""

    def setUp(self):
        """Set up integration test fixtures"""
        # Create a more complete mock SD server
        self.mock_sd_server = Mock()
        self.mock_sd_server.logger = Mock()
        self.mock_sd_server.model_ready = True
        self.mock_sd_server.model_loading = False
        self.mock_sd_server.model_id = "test_model"
        self.mock_sd_server.device = "cuda"
        self.mock_sd_server.precision = "fp16"
        self.mock_sd_server.attention_precision = "fp16"
        self.mock_sd_server.is_sdxl = False
        self.mock_sd_server.output_dir = "/tmp/test_images"

    def test_full_generation_workflow(self):
        """Test complete image generation workflow"""
        # Mock a complete generation response
        generation_result = {
            "images": ["/images/sunset_20241227_120000_abc123_0.png"],
            "metadata": {
                "prompt": "beautiful sunset over mountains, golden hour lighting",
                "negative_prompt": "blurry, low quality, dark, night",
                "width": 768,
                "height": 512,
                "num_inference_steps": 30,
                "guidance_scale": 8.5,
                "seed": 98765,
                "sampler": "DPM++ 2M Karras",
                "eta": 0.0,
                "clip_skip": 1
            },
            "generation_time": 4.7,
            "request_id": "abc123"
        }
        self.mock_sd_server.generate_image.return_value = generation_result

        # Test request data
        request_data = {
            "prompt": "beautiful sunset over mountains, golden hour lighting",
            "negative_prompt": "blurry, low quality, dark, night",
            "width": 768,
            "height": 512,
            "num_inference_steps": 30,
            "guidance_scale": 8.5,
            "seed": 98765,
            "sampler": "DPM++ 2M Karras"
        }

        # Create handler and process request
        body = json.dumps(request_data).encode('utf-8')
        headers = {"Content-Length": str(len(body))}

        class TestHandler(StableDiffusionHTTPHandler):
            def __init__(self, sd_server, body, headers):
                self.sd_server = sd_server
                self.command = "POST"
                self.path = "/generate"
                self.headers = headers
                self.rfile = BytesIO(body)
                self.wfile = BytesIO()
                self._response_code = None
                self._response_headers = {}
                self._response_body = None

            def send_response(self, code):
                self._response_code = code

            def send_header(self, name, value):
                self._response_headers[name] = value

            def end_headers(self):
                pass

            def address_string(self):
                return "127.0.0.1"

            def log_message(self, format, *args):
                pass

            def send_json_response(self, code, data):
                self._response_code = code
                self._response_headers['Content-Type'] = 'application/json'
                self._response_body = json.dumps(data)

            def send_json_error(self, code, message):
                self._response_code = code
                self._response_headers['Content-Type'] = 'application/json'
                self._response_body = json.dumps({"error": message})

        handler = TestHandler(self.mock_sd_server, body, headers)
        handler.handle_generate()

        # Verify successful response
        self.assertEqual(handler._response_code, 200)
        self.assertEqual(handler._response_headers.get('Content-Type'), 'application/json')

        # Verify generation was called correctly
        self.mock_sd_server.generate_image.assert_called_once()
        args, kwargs = self.mock_sd_server.generate_image.call_args

        # Check all parameters were passed correctly
        expected_prompt = "beautiful sunset over mountains, golden hour lighting"
        actual_prompt = kwargs["prompt"] if "prompt" in kwargs else args[0]
        self.assertEqual(actual_prompt, expected_prompt)

        self.assertEqual(kwargs.get("negative_prompt"), "blurry, low quality, dark, night")
        self.assertEqual(kwargs.get("width"), 768)
        self.assertEqual(kwargs.get("height"), 512)
        self.assertEqual(kwargs.get("num_inference_steps"), 30)
        self.assertEqual(kwargs.get("guidance_scale"), 8.5)
        self.assertEqual(kwargs.get("seed"), 98765)
        self.assertEqual(kwargs.get("sampler"), "DPM++ 2M Karras")

        # Verify response contains expected data
        response_data = json.loads(handler._response_body)
        self.assertIn("images", response_data)
        self.assertIn("metadata", response_data)
        self.assertIn("generation_time", response_data)
        self.assertIn("request_id", response_data)

    def test_model_management_workflow(self):
        """Test complete model management workflow"""
        # Test getting available models
        models_response = {
            "current_model": "SG161222/Realistic_Vision_V2.0",
            "current_model_ready": True,
            "available_models": [
                {"model_id": "SG161222/Realistic_Vision_V2.0", "cached": True, "type": "SD 1.5"},
                {"model_id": "SG161222/RealVisXL_V4.0", "cached": False, "type": "SDXL"},
                {"model_id": "stabilityai/stable-diffusion-xl-base-1.0", "cached": False, "type": "SDXL"}
            ]
        }
        self.mock_sd_server.get_available_models.return_value = models_response

        # Test switch model
        switch_response = {
            "success": True,
            "message": "Switching to model SG161222/RealVisXL_V4.0",
            "old_model": "SG161222/Realistic_Vision_V2.0",
            "new_model": "SG161222/RealVisXL_V4.0",
            "model_type": "SDXL",
            "precision": "fp16",
            "attention_precision": "fp32",
            "loading": True
        }
        self.mock_sd_server.switch_model.return_value = switch_response

        # Test clear cache
        clear_response = {
            "success": True,
            "message": "Model cache cleared successfully",
            "cache_cleared": True,
            "current_model": "SG161222/RealVisXL_V4.0",
            "model_ready": False,
            "note": "Use /switch-model to reload the current model with fresh files"
        }
        self.mock_sd_server.clear_model_cache.return_value = clear_response

        # Create handler for testing each endpoint
        class TestHandler(StableDiffusionHTTPHandler):
            def __init__(self, sd_server):
                self.sd_server = sd_server
                self._responses = []

            def send_json_response(self, code, data):
                self._responses.append({"code": code, "data": data})

        handler = TestHandler(self.mock_sd_server)

        # Test models endpoint
        handler.handle_list_models()
        models_result = handler._responses[-1]
        self.assertEqual(models_result["code"], 200)
        self.assertIn("models", models_result["data"])

        # Test switch model endpoint
        handler.rfile = BytesIO(
            json.dumps({"model_id": "SG161222/RealVisXL_V4.0", "attention_precision": "fp32"}).encode())
        handler.headers = {"Content-Length": "100"}
        handler.handle_switch_model()
        switch_result = handler._responses[-1]
        self.assertEqual(switch_result["code"], 200)
        self.assertTrue(switch_result["data"]["success"])

        # Test clear cache endpoint
        handler.handle_clear_cache()
        clear_result = handler._responses[-1]
        self.assertEqual(clear_result["code"], 200)
        self.assertTrue(clear_result["data"]["success"])

        # Verify all methods were called
        self.mock_sd_server.get_available_models.assert_called_once()
        self.mock_sd_server.switch_model.assert_called_once()
        self.mock_sd_server.clear_model_cache.assert_called_once()


if __name__ == '__main__':
    unittest.main()