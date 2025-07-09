#!/usr/bin/env python3
"""
HTTP REST API server for Stable Diffusion MCP Server
Now includes ComfyUI compatibility
"""

import json
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

from comfyui_http_handler import ComfyUIHTTPHandler


class StableDiffusionHTTPHandler(BaseHTTPRequestHandler, ComfyUIHTTPHandler):
    """HTTP request handler for the Stable Diffusion server with ComfyUI compatibility"""

    def __init__(self, sd_server, *args, **kwargs):
        self.sd_server = sd_server
        super().__init__(*args, **kwargs)

    def log_message(self, format, *args):
        """Override to use our logger"""
        self.sd_server.logger.info(f"{self.address_string()} - {format % args}")

    def do_GET(self):
        """Handle GET requests"""
        # Try ComfyUI routes first
        if self.handle_comfyui_routes():
            return

        # Fall back to original endpoints
        parsed_path = urlparse(self.path)
        path = parsed_path.path

        if path == '/health':
            self.handle_health_check()
        elif path == '/models':
            self.handle_list_models()
        elif path.startswith('/images/'):
            filename = path.split('/')[-1]
            self.serve_image(filename)
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        """Handle POST requests"""
        # Try ComfyUI routes first
        if self.handle_comfyui_routes():
            return

        # Fall back to original endpoints
        parsed_path = urlparse(self.path)
        path = parsed_path.path

        if path == '/generate':
            self.handle_generate()
        elif path == '/switch-model':
            self.handle_switch_model()
        elif path == '/clear-cache':
            self.handle_clear_cache()
        else:
            self.send_error(404, "Not Found")

    def do_OPTIONS(self):
        """Handle OPTIONS requests for CORS"""
        self.send_response(200)
        self.send_cors_headers()
        self.end_headers()

    def send_cors_headers(self):
        """Send CORS headers"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def handle_health_check(self):
        """Handle health check endpoint"""
        status = {
            "status": "healthy" if self.sd_server.model_ready else "loading" if self.sd_server.model_loading else "starting",
            "model_id": self.sd_server.model_id,
            "model_type": "SDXL" if self.sd_server.is_sdxl else "SD 1.5/2.x",
            "device": self.sd_server.device,
            "precision": self.sd_server.precision,
            "attention_precision": self.sd_server.attention_precision,
            "model_ready": self.sd_server.model_ready,
            "model_loading": self.sd_server.model_loading
        }

        self.send_json_response(200, status)

# The duplicate definition of StableDiffusionHTTPHandler has been removed to preserve ComfyUI route support.
        """Handle OPTIONS requests for CORS"""
        self.send_response(200)
        self.send_cors_headers()
        self.end_headers()

    def send_cors_headers(self):
        """Send CORS headers"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def handle_health_check(self):
        """Handle health check endpoint"""
        status = {
            "status": "healthy" if self.sd_server.model_ready else "loading" if self.sd_server.model_loading else "starting",
            "model_id": self.sd_server.model_id,
            "model_type": "SDXL" if self.sd_server.is_sdxl else "SD 1.5/2.x",
            "device": self.sd_server.device,
            "precision": self.sd_server.precision,
            "attention_precision": self.sd_server.attention_precision,
            "model_ready": self.sd_server.model_ready,
            "model_loading": self.sd_server.model_loading
        }

        self.send_json_response(200, status)

    def handle_list_models(self):
        """Handle model listing endpoint"""
        models = self.sd_server.get_available_models()
        self.send_json_response(200, {"models": models})

    def handle_generate(self):
        """Handle image generation endpoint"""
        try:
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self.send_json_error(400, "Missing request body")
                return

            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))

            prompt = data.get("prompt")
            if not prompt or not prompt.strip():
                self.send_json_error(400, "Missing or empty 'prompt' parameter")
                return

            # Remove prompt from data to avoid duplicate argument
            generation_params = data.copy()
            generation_params.pop("prompt", None)

            # Generate the image
            result = self.sd_server.generate_image(prompt, **generation_params)

            if result is None:
                self.send_json_error(500, "Image generation returned null result")
                return

            self.send_json_response(200, result)

        except RuntimeError as e:
            self.send_json_error(503, str(e))
        except json.JSONDecodeError:
            self.send_json_error(400, "Invalid JSON in request body")
        except Exception as e:
            self.sd_server.logger.error(f"Generation endpoint error: {e}")
            import traceback
            traceback.print_exc()
            self.send_json_error(500, f"Internal server error: {str(e)}")

    def handle_switch_model(self):
        """Handle model switching endpoint"""
        try:
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self.send_json_error(400, "Missing request body")
                return

            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))

            new_model_id = data.get("model_id")
            if not new_model_id:
                self.send_json_error(400, "Missing required 'model_id' parameter")
                return

            # Optional precision parameters
            precision = data.get("precision", self.sd_server.precision)
            attention_precision = data.get("attention_precision", self.sd_server.attention_precision)

            # Switch the model
            result = self.sd_server.switch_model(new_model_id, precision, attention_precision)
            self.send_json_response(200, result)

        except RuntimeError as e:
            self.send_json_error(503, str(e))
        except json.JSONDecodeError:
            self.send_json_error(400, "Invalid JSON in request body")
        except Exception as e:
            self.sd_server.logger.error(f"Model switch endpoint error: {e}")
            self.send_json_error(500, str(e))

    def handle_clear_cache(self):
        """Handle cache clearing endpoint"""
        try:
            result = self.sd_server.clear_model_cache()
            self.send_json_response(200, result)
        except Exception as e:
            self.sd_server.logger.error(f"Cache clear endpoint error: {e}")
            self.send_json_error(500, str(e))

    def serve_image(self, filename):
        """Serve generated images"""
        file_path = self.sd_server.output_dir / filename

        if not file_path.exists():
            self.send_error(404, "Image not found")
            return

        try:
            with open(file_path, 'rb') as f:
                self.send_response(200)
                self.send_header('Content-Type', 'image/png')
                self.send_cors_headers()
                self.send_header('Content-Length', str(file_path.stat().st_size))
                self.end_headers()

                # Send file in chunks
                while True:
                    chunk = f.read(8192)
                    if not chunk:
                        break
                    self.wfile.write(chunk)

        except Exception as e:
            self.sd_server.logger.error(f"Error serving image {filename}: {e}")
            self.send_error(500, "Internal Server Error")

    def send_json_response(self, code, data):
        """Send JSON response"""
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_cors_headers()
        self.end_headers()

        response_data = json.dumps(data).encode('utf-8')
        self.wfile.write(response_data)

    def send_json_error(self, code, message):
        """Send JSON error response"""
        error_data = {"error": message}
        self.send_json_response(code, error_data)


class HTTPServerManager:
    """Manages the HTTP server lifecycle"""

    def __init__(self, sd_server, host="127.0.0.1", port=8000):
        self.sd_server = sd_server
        self.host = host
        self.port = port
        self.http_server = None

    def create_handler_factory(self):
        """Create a handler factory for the HTTP server"""

        def handler_factory(*args, **kwargs):
            return StableDiffusionHTTPHandler(self.sd_server, *args, **kwargs)

        return handler_factory

    def start_server(self):
        """Start the HTTP server"""
        handler_factory = self.create_handler_factory()
        self.http_server = HTTPServer((self.host, self.port), handler_factory)

        self._print_server_info()

        try:
            self.http_server.serve_forever()
        except KeyboardInterrupt:
            print("🛑 Shutting down server...")
            self.http_server.shutdown()

    def _print_server_info(self):
        """Print server startup information"""
        print(f"🌐 HTTP server listening on {self.host}:{self.port}")
        print("🔗 Available endpoints:")
        print(f"   📊 Status: http://{self.host}:{self.port}/health")
        print(f"   🎨 Generate: POST http://{self.host}:{self.port}/generate")
        print(f"   🔄 Switch Model: POST http://{self.host}:{self.port}/switch-model")
        print(f"   📋 List Models: http://{self.host}:{self.port}/models")
        print(f"   🗑️  Clear Cache: POST http://{self.host}:{self.port}/clear-cache")
        print(f"   🖼️  Images: http://{self.host}:{self.port}/images/FILENAME.png")
        print()
        print("🎯 Ready to generate images!")
        print(
            f"   Generate: curl -X POST http://{self.host}:{self.port}/generate -H 'Content-Type: application/json' -d '{{\"prompt\": \"a beautiful sunset\"}}'")
        print(
            f"   Switch Model: curl -X POST http://{self.host}:{self.port}/switch-model -H 'Content-Type: application/json' -d '{{\"model_id\": \"runwayml/stable-diffusion-v1-5\"}}'")
        print(f"   Clear Cache: curl -X POST http://{self.host}:{self.port}/clear-cache")
        print()