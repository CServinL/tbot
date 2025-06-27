"""
Stable Diffusion MCP Server Module for tbot

Diffusers-only implementation with SG161222/Realistic_Vision_V2.0 model.
"""

import asyncio
import json
import logging
import os
import socket
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import base64

# Check for required dependencies on import
try:
    import torch
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    from PIL import Image

    # Check accelerate availability
    try:
        import accelerate
        ACCELERATE_AVAILABLE = True
        print(f"✅ Accelerate version: {accelerate.__version__}")
    except ImportError:
        ACCELERATE_AVAILABLE = False
        print("⚠️  Accelerate not available - using basic loading")

except ImportError as e:
    missing_dep = str(e).split("'")[1] if "'" in str(e) else str(e)
    print(f"Error: Missing required dependency: {missing_dep}")
    print("Please install: pip install torch diffusers pillow accelerate")
    sys.exit(1)

# MCP Protocol implementation
class MCPMessage:
    """Base class for MCP messages"""
    def __init__(self, id: str, method: str, params: Dict[str, Any] = None):
        self.id = id
        self.method = method
        self.params = params or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": self.id,
            "method": self.method,
            "params": self.params
        }

class MCPResponse:
    """MCP response message"""
    def __init__(self, id: str, result: Any = None, error: Dict[str, Any] = None):
        self.id = id
        self.result = result
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        response = {"jsonrpc": "2.0", "id": self.id}
        if self.error:
            response["error"] = self.error
        else:
            response["result"] = self.result
        return response

@dataclass
class ImageGenerationRequest:
    """Request for image generation"""
    prompt: str
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    seed: Optional[int] = None
    num_images: int = 1

@dataclass
class ImageGenerationResult:
    """Result of image generation"""
    images: List[str]  # File paths
    metadata: Dict[str, Any]
    generation_time: float
    request_id: str

class SimpleWebSocketServer:
    """Minimal WebSocket server implementation"""

    def __init__(self, host: str = "127.0.0.1", port: int = 8001, sd_server=None):
        self.host = host
        self.port = port
        self.sd_server = sd_server
        self.clients = []
        self.running = False
        self.server_socket = None

    def start(self):
        """Start the WebSocket server"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        self.running = True

        print(f"🌐 WebSocket server listening on {self.host}:{self.port}")

        while self.running:
            try:
                client_socket, address = self.server_socket.accept()
                client_thread = threading.Thread(
                    target=self.handle_client,
                    args=(client_socket, address)
                )
                client_thread.daemon = True
                client_thread.start()
            except OSError:
                break

    def handle_client(self, client_socket, address):
        """Handle WebSocket client connection"""
        try:
            # Simple WebSocket handshake
            request = client_socket.recv(1024).decode('utf-8')
            if 'Upgrade: websocket' in request:
                # Extract WebSocket key
                key = None
                for line in request.split('\n'):
                    if 'Sec-WebSocket-Key:' in line:
                        key = line.split(': ')[1].strip()
                        break

                if key:
                    # Simple handshake response
                    response = (
                        "HTTP/1.1 101 Switching Protocols\r\n"
                        "Upgrade: websocket\r\n"
                        "Connection: Upgrade\r\n"
                        f"Sec-WebSocket-Accept: {self._generate_accept_key(key)}\r\n"
                        "\r\n"
                    )
                    client_socket.send(response.encode('utf-8'))

                    # Handle messages
                    while self.running:
                        try:
                            message = self._receive_frame(client_socket)
                            if message:
                                if self.sd_server:
                                    response_data = self.sd_server.process_mcp_message(json.loads(message))
                                    self._send_frame(client_socket, json.dumps(response_data.to_dict()))
                        except:
                            break
        except Exception as e:
            print(f"Client handling error: {e}")
        finally:
            client_socket.close()

    def _generate_accept_key(self, key: str) -> str:
        """Generate WebSocket accept key"""
        import hashlib
        magic_string = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
        combined = key + magic_string
        return base64.b64encode(hashlib.sha1(combined.encode()).digest()).decode()

    def _receive_frame(self, socket) -> str:
        """Receive WebSocket frame (simplified)"""
        try:
            data = socket.recv(2)
            if len(data) < 2:
                return None

            payload_length = data[1] & 0x7F
            if payload_length == 126:
                data = socket.recv(2)
                payload_length = int.from_bytes(data, 'big')
            elif payload_length == 127:
                data = socket.recv(8)
                payload_length = int.from_bytes(data, 'big')

            mask_key = socket.recv(4)
            payload = socket.recv(payload_length)

            # Unmask payload
            unmasked = bytearray()
            for i in range(len(payload)):
                unmasked.append(payload[i] ^ mask_key[i % 4])

            return unmasked.decode('utf-8')
        except:
            return None

    def _send_frame(self, socket, message: str):
        """Send WebSocket frame"""
        try:
            message_bytes = message.encode('utf-8')
            frame = bytearray()
            frame.append(0x81)  # Text frame

            if len(message_bytes) < 126:
                frame.append(len(message_bytes))
            elif len(message_bytes) < 65536:
                frame.append(126)
                frame.extend(len(message_bytes).to_bytes(2, 'big'))
            else:
                frame.append(127)
                frame.extend(len(message_bytes).to_bytes(8, 'big'))

            frame.extend(message_bytes)
            socket.send(frame)
        except:
            pass

    def stop(self):
        """Stop the server"""
        self.running = False
        if self.server_socket:
            self.server_socket.close()

class HTTPRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler"""

    def __init__(self, server_instance, *args, **kwargs):
        self.server_instance = server_instance
        super().__init__(*args, **kwargs)

    def do_GET(self):
        """Handle GET requests"""
        path = urlparse(self.path).path

        if path == "/":
            self._send_json_response({
                "message": "Stable Diffusion MCP Server",
                "status": "ready" if self.server_instance.is_ready() else "loading",
                "model": "SG161222/Realistic_Vision_V2.0"
            })
        elif path == "/health":
            self._send_json_response({
                "status": "ready" if self.server_instance.is_ready() else "loading",
                "model_loaded": self.server_instance.pipeline is not None,
                "device": self.server_instance.device,
                "model_id": self.server_instance.model_id
            })
        elif path.startswith("/images/"):
            filename = path.split("/")[-1]
            self._serve_image(filename)
        else:
            self._send_error_response(404, "Not found")

    def do_POST(self):
        """Handle POST requests"""
        if self.path == "/generate":
            if not self.server_instance.is_ready():
                self._send_error_response(503, "Model still loading, please wait")
                return

            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                request_data = json.loads(post_data.decode('utf-8'))

                request = ImageGenerationRequest(**request_data)
                result = self.server_instance.generate_images(request)

                self._send_json_response(asdict(result))
            except Exception as e:
                self._send_error_response(500, str(e))
        else:
            self._send_error_response(404, "Not found")

    def _send_json_response(self, data: Dict[str, Any]):
        """Send JSON response"""
        response = json.dumps(data).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(response)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(response)

    def _send_error_response(self, code: int, message: str):
        """Send error response"""
        self.send_response(code)
        self.send_header('Content-Type', 'text/plain')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(message.encode('utf-8'))

    def _serve_image(self, filename: str):
        """Serve image file"""
        file_path = self.server_instance.output_dir / filename
        if file_path.exists():
            with open(file_path, 'rb') as f:
                content = f.read()

            self.send_response(200)
            self.send_header('Content-Type', 'image/png')
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(content)
        else:
            self._send_error_response(404, "Image not found")

    def log_message(self, format, *args):
        """Override to reduce logging noise"""
        pass

class StableDiffusionServer:
    """Stable Diffusion server with diffusers backend"""

    def __init__(self,
                 model_id: str = "SG161222/Realistic_Vision_V2.0",
                 device: str = "auto",
                 output_dir: str = "./generated_images",
                 models_dir: str = "./models",
                 precision: str = "fp16",
                 attention_precision: str = "fp16",
                 http_host: str = "127.0.0.1",
                 http_port: int = 8000,
                 ws_host: str = "127.0.0.1",
                 ws_port: int = 8001):

        self.model_id = model_id
        self.device = self._get_device(device)
        self.precision = precision
        self.attention_precision = attention_precision

        # Validate precision settings
        valid_precisions = ["fp16", "fp32", "bf16"]
        if self.precision not in valid_precisions:
            raise ValueError(f"Invalid precision '{self.precision}'. Must be one of: {valid_precisions}")
        if self.attention_precision not in valid_precisions:
            raise ValueError(f"Invalid attention_precision '{self.attention_precision}'. Must be one of: {valid_precisions}")

        # Determine torch dtype from precision
        self.torch_dtype = self._get_torch_dtype(self.precision)
        self.attention_dtype = self._get_torch_dtype(self.attention_precision)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Setup models directory for caching
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.cache_dir = self.models_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)

        self.http_host = http_host
        self.http_port = http_port
        self.ws_host = ws_host
        self.ws_port = ws_port

        self.pipeline = None
        self.model_loading = False
        self.model_ready = False

        self.setup_logging()
        self._setup_environment()

        print("🚀 Starting Stable Diffusion MCP Server")
        print(f"📦 Model: {self.model_id}")
        print(f"🔧 Device: {self.device}")
        print(f"⚡ Precision: {self.precision} (torch: {self.torch_dtype})")
        print(f"🧠 Attention Precision: {self.attention_precision} (torch: {self.attention_dtype})")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"💾 Models cache directory: {self.cache_dir}")

        # Check if model is already cached
        if self._is_model_cached():
            print("✅ Model found in local cache - startup will be faster!")
        else:
            print("⏳ Model not cached - first run will download ~5-7GB")

    def _get_device(self, device: str) -> str:
        """Determine the best device to use"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"

    def _get_torch_dtype(self, precision: str):
        """Convert precision string to torch dtype"""
        precision_map = {
            "fp16": torch.float16,
            "fp32": torch.float32,
            "bf16": torch.bfloat16 if hasattr(torch, 'bfloat16') else torch.float16
        }
        dtype = precision_map[precision]

        # Check device compatibility
        if precision == "bf16" and self.device == "cpu":
            print(f"⚠️  Warning: bf16 not optimal on CPU, falling back to fp32")
            return torch.float32
        elif precision == "fp16" and self.device == "cpu":
            print(f"⚠️  Warning: fp16 not optimal on CPU, falling back to fp32")
            return torch.float32

        return dtype
        return device

        self.logger = logging.getLogger(__name__)

    def _setup_environment(self):
        """Setup environment variables for local model caching"""
        # Set Hugging Face cache to local models directory
        os.environ['HF_HOME'] = str(self.cache_dir)
        os.environ['TRANSFORMERS_CACHE'] = str(self.cache_dir)
        os.environ['HF_HUB_CACHE'] = str(self.cache_dir / "hub")

        # Create subdirectories
        (self.cache_dir / "hub").mkdir(exist_ok=True)
        (self.cache_dir / "transformers").mkdir(exist_ok=True)

        self.logger.info(f"Model cache configured: {self.cache_dir}")

    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        """Convert precision string to torch dtype"""
        precision_map = {
            "fp16": torch.float16,
            "fp32": torch.float32,
            "bf16": torch.bfloat16 if hasattr(torch, 'bfloat16') else torch.float16
        }
        dtype = precision_map[precision]

        # Check device compatibility
        if precision == "bf16" and self.device == "cpu":
            print(f"⚠️  Warning: bf16 not optimal on CPU, falling back to fp32")
            return torch.float32
        elif precision == "fp16" and self.device == "cpu":
            print(f"⚠️  Warning: fp16 not optimal on CPU, falling back to fp32")
            return torch.float32

        return dtype
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _is_model_cached(self) -> bool:
        """Check if the model is already cached locally"""
        try:
            # Check for the model in the hub cache
            hub_cache = self.cache_dir / "hub"
            model_cache_name = f"models--{self.model_id.replace('/', '--')}"
            model_path = hub_cache / model_cache_name

            if model_path.exists():
                # Check if it has the main model files
                snapshot_dir = model_path / "snapshots"
                if snapshot_dir.exists():
                    snapshots = list(snapshot_dir.iterdir())
                    if snapshots:
                        # Check the latest snapshot for key files
                        latest_snapshot = max(snapshots, key=lambda x: x.stat().st_mtime)
                        key_files = [
                            "model_index.json",
                            "unet/config.json",
                            "text_encoder/config.json",
                        ]

                        if all((latest_snapshot / f).exists() for f in key_files):
                            return True

            # Also check for single file format
            single_file_path = self.models_dir / f"{self.model_id.replace('/', '_')}.safetensors"
            if single_file_path.exists():
                return True

            return False
        except Exception as e:
            # If logging isn't set up yet, just return False
            if hasattr(self, 'logger'):
                self.logger.debug(f"Cache check failed: {e}")
            return False
        """Check if the model is already cached locally"""
        try:
            # Check for the model in the hub cache
            hub_cache = self.cache_dir / "hub"
            model_cache_name = f"models--{self.model_id.replace('/', '--')}"
            model_path = hub_cache / model_cache_name

            if model_path.exists():
                # Check if it has the main model files
                snapshot_dir = model_path / "snapshots"
                if snapshot_dir.exists():
                    snapshots = list(snapshot_dir.iterdir())
                    if snapshots:
                        # Check the latest snapshot for key files
                        latest_snapshot = max(snapshots, key=lambda x: x.stat().st_mtime)
                        key_files = [
                            "model_index.json",
                            "unet/config.json",
                            "text_encoder/config.json",
                        ]

                        if all((latest_snapshot / f).exists() for f in key_files):
                            return True

            # Also check for single file format
            single_file_path = self.models_dir / f"{self.model_id.replace('/', '_')}.safetensors"
            if single_file_path.exists():
                return True

            return False
        except Exception as e:
            self.logger.debug(f"Cache check failed: {e}")
            return False

    def is_ready(self) -> bool:
        """Check if the server is ready to generate images"""
        return self.model_ready and self.pipeline is not None

    def load_model(self):
        """Load the Stable Diffusion model with progress indicators"""
        if self.model_loading or self.model_ready:
            return

        self.model_loading = True

        try:
            print("⏳ Starting model loading process...")
            print(f"📦 Target model: {self.model_id}")
            print(f"🎯 Device: {self.device}")
            print(f"💾 Data type: {'float16' if self.device == 'cuda' else 'float32'}")
            print()
            print("🔍 Step 1: Checking model repository structure...")

            start_time = time.time()

            # Load pipeline with progress
            # This model has checkpoint files, not standard diffusers format
            try:
                print("📥 Step 2: Attempting standard diffusers loading...")
                print(f"   📂 Using local cache: {self.cache_dir}")
                print("   ⏳ This may download several GB of data on first run...")
                print(f"   🔧 Accelerate available: {ACCELERATE_AVAILABLE}")

                # Prepare loading kwargs based on accelerate availability
                loading_kwargs = {
                    "cache_dir": str(self.cache_dir),
                    "torch_dtype": self.torch_dtype,
                    "safety_checker": None,
                    "requires_safety_checker": False,
                    "use_safetensors": False,  # This model uses .ckpt files
                }

                # Add memory-efficient options only if accelerate is available
                if ACCELERATE_AVAILABLE:
                    loading_kwargs.update({
                        "low_cpu_mem_usage": True,
                        "device_map": "balanced" if self.device == "cuda" else None
                    })
                    print("   💾 Using memory-efficient loading for 8GB RAM")
                else:
                    print("   ⚠️  Using basic loading (may use more RAM)")

                # First try standard diffusers loading with local cache
                self.pipeline = StableDiffusionPipeline.from_pretrained(
                    self.model_id,
                    **loading_kwargs
                )
                print("✅ Step 2: Standard loading successful!")

            except (OSError, EnvironmentError) as e:
                if "safetensors" in str(e):
                    print("⚠️  Step 2: Standard loading failed (expected for this model)")
                    print("🔄 Step 3: Trying single-file checkpoint loading...")
                    print("   📁 Loading from: Realistic_Vision_V2.0.safetensors")
                    print(f"   💾 Saving to local cache: {self.cache_dir}")
                    print("   ⏳ Downloading checkpoint file (this may take several minutes)...")

                    # Prepare single-file loading kwargs
                    single_file_kwargs = {
                        "cache_dir": str(self.cache_dir),
                        "torch_dtype": self.torch_dtype,
                        "safety_checker": None,
                        "requires_safety_checker": False,
                        "use_safetensors": True,
                    }

                    # Add memory-efficient options only if accelerate is available
                    if ACCELERATE_AVAILABLE:
                        single_file_kwargs.update({
                            "low_cpu_mem_usage": True,
                            "device_map": "balanced" if self.device == "cuda" else None
                        })

                    # Try loading from single file (checkpoint) with local cache
                    self.pipeline = StableDiffusionPipeline.from_single_file(
                        f"https://huggingface.co/{self.model_id}/resolve/main/Realistic_Vision_V2.0.safetensors",
                        **single_file_kwargs
                    )
                    print("✅ Step 3: Single-file loading successful!")
                else:
                    print(f"❌ Step 2: Unexpected error: {e}")
                    raise

            print()
            print("🔄 Step 4: Optimizing pipeline components...")
            print("   🧠 Setting up DPM++ Multistep scheduler...")

            # Optimize scheduler
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipeline.scheduler.config
            )
            print("   ✅ Scheduler optimization complete")

            # Move to device
            print(f"   📱 Moving model components to {self.device}...")
            print("   ⏳ This step may take 30-60 seconds...")
            self.pipeline = self.pipeline.to(self.device)
            print(f"   ✅ Model successfully moved to {self.device}")

            print()
            print("🚀 Step 5: Applying performance optimizations...")

            # Enable optimizations
            if hasattr(self.pipeline, "enable_attention_slicing"):
                print("   🔧 Enabling attention slicing for memory efficiency...")
                self.pipeline.enable_attention_slicing()
                print("   ✅ Attention slicing enabled")

            # Apply attention precision settings
            if self.attention_precision != self.precision:
                print(f"   🧠 Setting attention precision to {self.attention_precision}...")
                if hasattr(self.pipeline.unet, "set_attn_processor"):
                    # Set custom attention precision if different from model precision
                    if self.attention_precision == "fp32":
                        print("   ⚡ Using fp32 attention for better quality")
                        # Force attention computations to fp32
                        for module in self.pipeline.unet.modules():
                            if hasattr(module, 'dtype'):
                                if 'attn' in str(type(module)).lower():
                                    module.to(dtype=self.attention_dtype)
                print(f"   ✅ Attention precision set to {self.attention_precision}")

            if self.device == "cuda":
                if hasattr(self.pipeline, "enable_model_cpu_offload"):
                    print("   💾 Skipping CPU offload for better CUDA performance...")
                    # Don't use CPU offload on CUDA for better performance
                    pass
                if hasattr(self.pipeline, "enable_xformers_memory_efficient_attention"):
                    try:
                        print("   ⚡ Attempting xFormers optimization...")
                        self.pipeline.enable_xformers_memory_efficient_attention()
                        print("   ✅ xFormers optimization enabled")
                    except Exception as xf_error:
                        print(f"   ⚠️  xFormers not available: {xf_error}")

            load_time = time.time() - start_time
            print()
            print("🧪 Step 6: Running validation test...")
            print("   🎨 Generating small test image to verify functionality...")
            test_start = time.time()

            with torch.no_grad():
                print("   ⏳ Generating 64x64 test image...")
                test_image = self.pipeline(
                    "test image",
                    num_inference_steps=1,
                    width=64,
                    height=64,
                    generator=torch.Generator(device=self.device).manual_seed(42)
                ).images[0]

            test_time = time.time() - test_start
            print(f"   ✅ Test generation completed in {test_time:.1f}s")

            self.model_ready = True
            self.model_loading = False

            print()
            print("🎉 MODEL LOADING COMPLETE!")
            print(f"⏱️  Total loading time: {load_time:.1f}s")
            print(f"🎯 Validation time: {test_time:.1f}s")
            print("🟢 Server is ready to generate images!")
            print("=" * 50)

        except Exception as e:
            self.model_loading = False
            self.model_ready = False
            print(f"❌ Failed to load model: {e}")
            self.logger.error(f"Model loading failed: {e}")
            raise

    def generate_images(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        """Generate images using Stable Diffusion"""
        if not self.is_ready():
            raise RuntimeError("Model not ready. Please wait for loading to complete.")

        start_time = time.time()
        request_id = str(uuid.uuid4())

        # Set random seed if not provided
        if request.seed is None:
            request.seed = torch.randint(0, 2**32 - 1, (1,)).item()

        generator = torch.Generator(device=self.device).manual_seed(request.seed)

        try:
            print(f"🎨 Generating {request.num_images} image(s): '{request.prompt[:50]}...'")

            # Generate images
            with torch.autocast(self.device, enabled=(self.device != "cpu")):
                result = self.pipeline(
                    prompt=request.prompt,
                    negative_prompt=request.negative_prompt,
                    width=request.width,
                    height=request.height,
                    num_inference_steps=request.num_inference_steps,
                    guidance_scale=request.guidance_scale,
                    generator=generator,
                    num_images_per_prompt=request.num_images
                )
                images = result.images

            # Save images
            image_paths = []
            for i, image in enumerate(images):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{timestamp}_{request_id}_{i}.png"
                file_path = self.output_dir / filename
                image.save(file_path)
                image_paths.append(str(file_path))

            generation_time = time.time() - start_time

            result = ImageGenerationResult(
                images=image_paths,
                metadata={
                    "prompt": request.prompt,
                    "negative_prompt": request.negative_prompt,
                    "width": request.width,
                    "height": request.height,
                    "num_inference_steps": request.num_inference_steps,
                    "guidance_scale": request.guidance_scale,
                    "seed": request.seed,
                    "model_id": self.model_id,
                    "device": self.device
                },
                generation_time=generation_time,
                request_id=request_id
            )

            print(f"✅ Generated {len(images)} image(s) in {generation_time:.2f}s")
            return result

        except Exception as e:
            print(f"❌ Failed to generate images: {e}")
            self.logger.error(f"Generation failed: {e}")
            raise

    def process_mcp_message(self, message_data: Dict[str, Any]) -> MCPResponse:
        """Process MCP messages"""
        try:
            msg_id = message_data.get("id")
            method = message_data.get("method")
            params = message_data.get("params", {})

            if method == "generate_image":
                if not self.is_ready():
                    return MCPResponse(
                        id=msg_id,
                        error={
                            "code": -32002,
                            "message": "Model still loading, please wait"
                        }
                    )

                request = ImageGenerationRequest(**params)
                result = self.generate_images(request)

                return MCPResponse(
                    id=msg_id,
                    result={
                        "images": [f"/images/{Path(img).name}" for img in result.images],
                        "metadata": result.metadata,
                        "generation_time": result.generation_time,
                        "request_id": result.request_id
                    }
                )

            elif method == "get_status":
                return MCPResponse(
                    id=msg_id,
                    result={
                        "status": "ready" if self.is_ready() else ("loading" if self.model_loading else "error"),
                        "model_loaded": self.pipeline is not None,
                        "model_ready": self.model_ready,
                        "device": self.device,
                        "model_id": self.model_id
                    }
                )

            elif method == "list_capabilities":
                return MCPResponse(
                    id=msg_id,
                    result={
                        "capabilities": [
                            "generate_image",
                            "get_status",
                            "list_capabilities"
                        ],
                        "version": "1.0.0",
                        "model": self.model_id
                    }
                )

            else:
                return MCPResponse(
                    id=msg_id,
                    error={
                        "code": -32601,
                        "message": f"Method not found: {method}"
                    }
                )

        except Exception as e:
            return MCPResponse(
                id=message_data.get("id"),
                error={
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            )

    def run(self):
        """Run the server"""
        # Start model loading in background thread
        load_thread = threading.Thread(target=self.load_model)
        load_thread.daemon = True
        load_thread.start()

        print(f"🌐 Starting HTTP server on {self.http_host}:{self.http_port}")
        print(f"🔌 Starting WebSocket server on {self.ws_host}:{self.ws_port}")

        # Create WebSocket server
        ws_server = SimpleWebSocketServer(self.ws_host, self.ws_port, self)

        # Start WebSocket server in separate thread
        ws_thread = threading.Thread(target=ws_server.start)
        ws_thread.daemon = True
        ws_thread.start()

        # Create HTTP server
        def handler(*args, **kwargs):
            return HTTPRequestHandler(self, *args, **kwargs)

        http_server = HTTPServer((self.http_host, self.http_port), handler)

        print("⏳ Server starting... (model loading in background)")
        print("🔗 HTTP Endpoints:")
        print(f"   📊 Status: http://{self.http_host}:{self.http_port}/health")
        print(f"   🎨 Generate: POST http://{self.http_host}:{self.http_port}/generate")
        print(f"🔌 WebSocket MCP: ws://{self.ws_host}:{self.ws_port}/mcp")
        print()

        try:
            http_server.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Shutting down server...")
            ws_server.stop()
            http_server.shutdown()
            print("👋 Server stopped")

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Stable Diffusion MCP Server (Diffusers)")
    parser.add_argument("--model", default="SG161222/Realistic_Vision_V2.0",
                       help="Stable Diffusion model to use")
    parser.add_argument("--device", default="auto",
                       help="Device to use (auto, cuda, mps, cpu)")
    parser.add_argument("--output-dir", default="./generated_images",
                       help="Directory to save generated images")
    parser.add_argument("--precision", default="fp16", choices=["fp16", "fp32", "bf16"],
                       help="Model precision (fp16=faster/less memory, fp32=higher quality/more memory)")
    parser.add_argument("--attention-precision", default="fp16", choices=["fp16", "fp32", "bf16"],
                       help="Attention layer precision (fp32 can improve quality at cost of speed)")
    parser.add_argument("--models-dir", default="./models",
                       help="Directory to cache downloaded models")
    parser.add_argument("--http-host", default="127.0.0.1",
                       help="HTTP server host")
    parser.add_argument("--http-port", type=int, default=8000,
                       help="HTTP server port")
    parser.add_argument("--ws-host", default="127.0.0.1",
                       help="WebSocket server host")
    parser.add_argument("--ws-port", type=int, default=8001,
                       help="WebSocket server port")

    args = parser.parse_args()

    server = StableDiffusionServer(
        model_id=args.model,
        device=args.device,
        output_dir=args.output_dir,
        models_dir=args.models_dir,
        precision=args.precision,
        attention_precision=args.attention_precision,
        http_host=args.http_host,
        http_port=args.http_port,
        ws_host=args.ws_host,
        ws_port=args.ws_port
    )

    server.run()

if __name__ == "__main__":
    main()