#!/usr/bin/env python3
"""
Main Stable Diffusion MCP Server implementation
"""

import logging
import os
import time
from pathlib import Path
from threading import Thread

from dependencies import torch
from diffusiond.model_loader import ModelLoader
from image_generator import ImageGenerator
from http_server import HTTPServerManager


class StableDiffusionServer:
    """Main Stable Diffusion MCP Server class"""

    def __init__(self,
                 model_id: str = "SG161222/Realistic_Vision_V2.0",
                 device: str = "auto",
                 output_dir: str = "./generated_images",
                 models_dir: str = "./models",
                 precision: str = "fp16",
                 attention_precision: str = "fp16",
                 http_host: str = "127.0.0.1",
                 http_port: int = 8000,
                 load_on_startup: bool = False  # <-- new argument
                 ):

        self.model_id = model_id
        self.device = self._get_device(device)
        self.precision = precision
        self.attention_precision = attention_precision

        # Detect if this is an SDXL model
        self.is_sdxl = self._detect_sdxl_model(model_id)
        if self.is_sdxl:
            print(f"🔍 Detected SDXL model - using StableDiffusionXLPipeline")
        else:
            print(f"🔍 Detected SD 1.5/2.x model - using StableDiffusionPipeline")

        # Validate precision settings
        valid_precisions = ["fp16", "fp32", "bf16"]
        if self.precision not in valid_precisions:
            raise ValueError(f"Invalid precision '{self.precision}'. Must be one of: {valid_precisions}")
        if self.attention_precision not in valid_precisions:
            raise ValueError(
                f"Invalid attention_precision '{self.attention_precision}'. Must be one of: {valid_precisions}")

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
        self.load_on_startup = load_on_startup

        # Initialize components
        self.pipeline = None
        self.model_loading = False
        self.model_ready = False
        self.model_loader = None
        self.image_generator = None
        self.http_server_manager = None

        self.setup_logging()
        self._setup_environment()
        self._initialize_components()

        print("🚀 Starting Stable Diffusion MCP Server")
        print(f"📦 Model: {self.model_id}")
        print(f"🎭 Model Type: {'SDXL' if self.is_sdxl else 'SD 1.5/2.x'}")
        print(f"🔧 Device: {self.device}")
        print(f"⚡ Precision: {self.precision} (torch: {self.torch_dtype})")
        print(f"🧠 Attention Precision: {self.attention_precision} (torch: {self.attention_dtype})")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"💾 Models cache directory: {self.cache_dir}")
        print(f"🌐 HTTP server: {self.http_host}:{self.http_port}")

        # Check if model is already cached
        if hasattr(self.model_loader, "is_model_cached") and callable(getattr(self.model_loader, "is_model_cached")):
            if self.model_loader.is_model_cached(self.model_id):
                print("✅ Model found in local cache - startup will be faster!")
            else:
                cache_size = "~12-15GB" if self.is_sdxl else "~5-7GB"
                print(f"⏳ Model not cached - first run will download {cache_size}")
        else:
            print("ℹ️ Model cache check not available in this ModelLoader implementation.")

    def _get_device(self, device: str) -> str:
        """Determine the best device to use"""
        if device == "auto":
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
                print(f"🎮 CUDA detected: {device_name} ({vram_gb:.1f}GB VRAM)")
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                print("🍎 MPS (Apple Silicon) detected")
                return "mps"
            else:
                print("💻 Using CPU (no GPU acceleration available)")
                return "cpu"

        # Validate manual device selection
        if device == "cuda" and not torch.cuda.is_available():
            print("⚠️  CUDA requested but not available, falling back to CPU")
            return "cpu"
        elif device == "mps" and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            print("⚠️  MPS requested but not available, falling back to CPU")
            return "cpu"
        elif device == "cpu":
            print("💻 Using CPU (manual selection)")

        return device

    def _get_torch_dtype(self, precision: str):
        """Convert precision string to torch dtype"""
        precision_map = {
            "fp16": torch.float16,
            "fp32": torch.float32,
            "bf16": torch.bfloat16 if hasattr(torch, 'bfloat16') else torch.float16
        }
        dtype = precision_map[precision]

        # Check device compatibility - artifacts often caused by wrong precision
        if precision == "bf16" and self.device == "cpu":
            print(f"⚠️  Warning: bf16 not optimal on CPU, falling back to fp32")
            return torch.float32
        elif precision == "fp16" and self.device == "cpu":
            print(f"⚠️  Warning: fp16 not optimal on CPU, falling back to fp32")
            return torch.float32
        elif precision == "fp16" and self.device == "mps":
            print(f"⚠️  Warning: fp16 may cause artifacts on MPS, consider using fp32")

        return dtype

    def _detect_sdxl_model(self, model_id: str) -> bool:
        """Detect if this is an SDXL model based on the model ID"""
        sdxl_indicators = [
            "xl", "XL", "sdxl", "SDXL",
            "realvisxl", "RealVisXL", "realvision_xl",
            "juggernautxl", "JuggernautXL",
            "dreamshaper-xl", "dreamshaper_xl", "DreamShaperXL"
        ]

        model_lower = model_id.lower()
        return any(indicator.lower() in model_lower for indicator in sdxl_indicators)

    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
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

    def _initialize_components(self):
        """Initialize server components"""
        # Use ModelLoader from diffusiond, pass all required arguments
        self.model_loader = ModelLoader(
            cache_dir=self.cache_dir,
            device=self.device,
            torch_dtype=self.torch_dtype,
            is_sdxl=self.is_sdxl,
            logger=self.logger
        )

        self.http_server_manager = HTTPServerManager(
            sd_server=self,
            host=self.http_host,
            port=self.http_port
        )

    def load_model(self):
        """Load the Stable Diffusion model"""
        if self.model_loading or self.model_ready:
            return

        self.model_loading = True

        try:
            # Always use the ModelLoader interface
            pipeline = self.model_loader.ensure_model_loaded(self.model_id)
            self.pipeline = pipeline

            self.image_generator = ImageGenerator(
                pipeline=self.pipeline,
                device=self.device,
                torch_dtype=self.torch_dtype,
                is_sdxl=self.is_sdxl,
                output_dir=self.output_dir,
                logger=self.logger
            )

            self.model_ready = True

        except Exception as e:
            self.logger.error(f"Model loading failed: {e}")
            raise
        finally:
            self.model_loading = False

    def run(self):
        """Start the HTTP server and model loading"""
        # Only load model at startup if requested
        if self.load_on_startup:
            Thread(target=self.load_model, daemon=True).start()
            print("⏳ Server starting... (model loading in background)")
        else:
            print("⏳ Server starting... (model will be loaded on first image generation request)")
        print()
        # Start HTTP server (this will block)
        self.http_server_manager.start_server()

    def generate_image(self, prompt: str, **kwargs):
        """Generate an image using the loaded model"""
        # Lazy load: load model if not ready
        wait_start = time.time()
        max_wait = 300  # 5 minutes in seconds
        while not self.model_ready:
            if not self.model_loading:
                Thread(target=self.load_model, daemon=True).start()
            if time.time() - wait_start > max_wait:
                return {
                    "error": "Model loading timed out. Try again later.",
                    "status": "timeout"
                }
            time.sleep(0.2)

        result = self.image_generator.generate_image(prompt, **kwargs)

        # If not load_on_startup, offload model after generation
        if not self.load_on_startup:
            self._offload_model()

        return result

    def _offload_model(self):
        """Offload/unload the model to free memory (used in lazy mode)"""
        self.model_loader.offload_model()
        self.pipeline = None
        self.image_generator = None
        self.model_ready = False
        import gc
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
            print("🧹 GPU cache cleared")

    def get_available_models(self):
        """Get list of cached models and popular models"""
        cached_models = []

        # Scan cache for already downloaded models
        hub_cache = self.cache_dir / "hub"
        if hub_cache.exists():
            for model_dir in hub_cache.iterdir():
                if model_dir.is_dir() and model_dir.name.startswith("models--"):
                    # Convert cache name back to model ID
                    model_id = model_dir.name[8:].replace("--", "/")
                    cached_models.append({
                        "model_id": model_id,
                        "cached": True,
                        "type": "SDXL" if self._detect_sdxl_model(model_id) else "SD 1.5/2.x"
                    })

        # Popular models list
        popular_models = [
            {"model_id": "SG161222/Realistic_Vision_V2.0", "type": "SD 1.5", "cached": False},
            {"model_id": "SG161222/RealVisXL_V4.0", "type": "SDXL", "cached": False},
            {"model_id": "stabilityai/stable-diffusion-xl-base-1.0", "type": "SDXL", "cached": False},
            {"model_id": "runwayml/stable-diffusion-v1-5", "type": "SD 1.5", "cached": False},
            {"model_id": "lykon/dreamshaper-8", "type": "SD 1.5", "cached": False},
            {"model_id": "lykon/dreamshaper-7", "type": "SD 1.5", "cached": False},
            {"model_id": "lykon/dreamshaper-xl-1-0", "type": "SDXL", "cached": False},
        ]

        # Mark popular models as cached if they exist
        cached_ids = {model["model_id"] for model in cached_models}
        for model in popular_models:
            model["cached"] = model["model_id"] in cached_ids

        # Combine and deduplicate
        all_models = {model["model_id"]: model for model in cached_models + popular_models}

        return {
            "current_model": self.model_id,
            "current_model_ready": self.model_ready,
            "available_models": list(all_models.values())
        }

    def switch_model(self, new_model_id: str, precision: str = None, attention_precision: str = None):
        """Switch to a different model"""
        if self.model_loading:
            raise RuntimeError("Cannot switch models while another model is loading")

        if new_model_id == self.model_id:
            return {
                "success": True,
                "message": f"Already using model {new_model_id}",
                "model_id": self.model_id,
                "model_ready": self.model_ready
            }

        # Clear current model to free memory
        if self.pipeline is not None:
            print(f"🔄 Unloading current model: {self.model_id}")
            del self.pipeline
            self.pipeline = None
            self.image_generator = None

            # Force garbage collection to free GPU memory
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("🧹 GPU cache cleared")

        # Update model configuration
        old_model_id = self.model_id
        self.model_id = new_model_id
        self.is_sdxl = self._detect_sdxl_model(new_model_id)

        # Update precision if provided
        if precision:
            self.precision = precision
            self.torch_dtype = self._get_torch_dtype(precision)
        if attention_precision:
            self.attention_precision = attention_precision
            self.attention_dtype = self._get_torch_dtype(attention_precision)

        # Update model loader with new settings
        self.model_loader.torch_dtype = self.torch_dtype
        self.model_loader.is_sdxl = self.is_sdxl

        # Reset model state
        self.model_ready = False

        print(f"🔄 Switching from {old_model_id} to {new_model_id}")
        print(f"🎭 New model type: {'SDXL' if self.is_sdxl else 'SD 1.5/2.x'}")

        # Start loading new model in background
        Thread(target=self.load_model, daemon=True).start()

        return {
            "success": True,
            "message": f"Switching to model {new_model_id}",
            "old_model": old_model_id,
            "new_model": new_model_id,
            "model_type": "SDXL" if self.is_sdxl else "SD 1.5/2.x",
            "precision": self.precision,
            "attention_precision": self.attention_precision,
            "loading": True
        }

    def clear_model_cache(self):
        """Clear all cached models and force fresh downloads"""
        # Clear current model to free memory
        if self.pipeline is not None:
            print("🔄 Unloading current model for cache clear")
            del self.pipeline
            self.pipeline = None
            self.image_generator = None

            # Force garbage collection
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("🧹 GPU cache cleared")

        # Clear the cache using model loader
        cache_cleared = self.model_loader.clear_cache()

        # Reset model state
        self.model_ready = False

        return {
            "success": True,
            "message": "Model cache cleared successfully",
            "cache_cleared": cache_cleared,
            "current_model": self.model_id,
            "model_ready": False,
            "note": "Use /switch-model to reload the current model with fresh files"
        }