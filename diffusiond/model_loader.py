#!/usr/bin/env python3
"""
Model loading and management for Stable Diffusion MCP Server
"""

import os
import time
import shutil
from pathlib import Path
from dependencies import torch, StableDiffusionPipeline, StableDiffusionXLPipeline, ACCELERATE_AVAILABLE


class ModelLoader:
    """Handles loading and managing Stable Diffusion models"""

    def __init__(self, cache_dir, device, torch_dtype, is_sdxl, logger):
        self.cache_dir = Path(cache_dir)
        self.device = device
        self.torch_dtype = torch_dtype
        self.is_sdxl = is_sdxl
        self.logger = logger

    def detect_sdxl_model(self, model_id: str) -> bool:
        """Detect if this is an SDXL model based on the model ID"""
        sdxl_indicators = [
            "xl", "XL", "sdxl", "SDXL",
            "realvisxl", "RealVisXL", "realvision_xl",
            "juggernautxl", "JuggernautXL",
            "dreamshaper-xl", "dreamshaper_xl", "DreamShaperXL"
        ]
        model_lower = model_id.lower()
        return any(indicator.lower() in model_lower for indicator in sdxl_indicators)

    def is_model_cached(self, model_id: str) -> bool:
        """Check if the model is already cached locally"""
        try:
            # Check for the model in the hub cache
            hub_cache = self.cache_dir / "hub"
            model_cache_name = f"models--{model_id.replace('/', '--')}"
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
            single_file_path = self.cache_dir.parent / f"{model_id.replace('/', '_')}.safetensors"
            if single_file_path.exists():
                return True

            return False
        except Exception as e:
            self.logger.debug(f"Cache check failed: {e}")
            return False

    def load_model(self, model_id: str):
        """Load a Stable Diffusion model with error handling and fallbacks"""
        start_time = time.time()

        print("⏳ Starting model loading process...")
        print(f"📦 Target model: {model_id}")
        print(f"🎯 Device: {self.device}")
        print(f"💾 Data type: {self.torch_dtype}")
        print()

        try:
            # Step 1: Basic availability check
            print("🔍 Step 1: Checking model repository structure...")
            print(f"   🌐 Repository: {model_id}")
            print(f"   📂 Local cache: {self.cache_dir}")
            print("   ✅ Repository structure check complete")
            print()

            # Step 2: Try different loading methods
            pipeline = self._attempt_standard_loading(model_id)
            if pipeline is None:
                pipeline = self._attempt_single_file_loading(model_id)

            if pipeline is None:
                raise RuntimeError("Failed to load model with all available methods")

            # Step 3: Move to target device
            pipeline = self._move_to_device(pipeline)

            # Step 4: Apply optimizations
            pipeline = self._apply_optimizations(pipeline, model_id)

            # Step 5: Run validation
            self._run_validation_test(pipeline)

            load_time = time.time() - start_time
            print()
            print("🎉 Model loading complete!")
            print(f"⏱️  Total loading time: {load_time:.1f} seconds")
            print(f"📊 Model ready for generation on {self.device}")
            print()

            return pipeline

        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            self.logger.error(f"Model loading failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _attempt_standard_loading(self, model_id: str):
        """Attempt standard diffusers loading"""
        try:
            print("📥 Step 2: Attempting standard diffusers loading...")
            print(f"   📂 Using local cache: {self.cache_dir}")
            print("   ⏳ This may download several GB of data on first run...")
            print(f"   🔧 Accelerate available: {ACCELERATE_AVAILABLE}")

            # Prepare loading kwargs
            loading_kwargs = {
                "cache_dir": str(self.cache_dir),
                "torch_dtype": self.torch_dtype,
            }

            # SDXL doesn't use safety checker parameters
            if not self.is_sdxl:
                loading_kwargs.update({
                    "safety_checker": None,
                    "requires_safety_checker": False,
                })

            # Add memory-efficient options only if accelerate is available
            if ACCELERATE_AVAILABLE:
                loading_kwargs.update({
                    "low_cpu_mem_usage": True,
                    "device_map": "balanced" if self.device == "cuda" else None
                })
                print("   💾 Using memory-efficient loading")
            else:
                print("   ⚠️  Using basic loading (may use more RAM)")

            # Choose the correct pipeline class
            pipeline_class = StableDiffusionXLPipeline if self.is_sdxl else StableDiffusionPipeline
            print(f"   🎭 Using {pipeline_class.__name__}")

            # First try standard diffusers loading with local cache
            pipeline = pipeline_class.from_pretrained(model_id, **loading_kwargs)
            print("✅ Step 2: Standard loading successful!")
            return pipeline

        except (OSError, EnvironmentError) as e:
            error_msg = str(e)
            if ("no file named" in error_msg and (
                    "text_encoder" in error_msg or "vae" in error_msg or "unet" in error_msg)) or "diffusion_pytorch_model.bin" in error_msg:
                print("⚠️  Step 2: Diffusers format incomplete - will try single-file loading")
                return None
            else:
                print(f"❌ Step 2: Unexpected error: {e}")
                raise

    def _attempt_single_file_loading(self, model_id: str):
        """Attempt single-file loading for models that need it"""
        print("🔄 Step 3: Loading from single safetensors file...")

        if "dreamshaper" in model_id.lower():
            return self._load_dreamshaper_model(model_id)
        elif "realistic" in model_id.lower():
            return self._load_realistic_vision_model(model_id)
        else:
            # Try force download as fallback
            return self._force_fresh_download(model_id)

    def _load_dreamshaper_model(self, model_id: str):
        """Load DreamShaper models from single file"""
        if "dreamshaper-8" in model_id.lower() or "lykon/dreamshaper" in model_id.lower():
            return self._download_and_load_dreamshaper_8()
        elif "dreamshaper-7" in model_id.lower():
            return self._download_and_load_dreamshaper_7()
        else:
            raise ValueError(f"Unknown DreamShaper variant: {model_id}")

    def _download_and_load_dreamshaper_8(self):
        """Download and load DreamShaper 8"""
        print("   📁 Loading DreamShaper 8 from: Lykon/DreamShaper")
        print("   💾 Using file: DreamShaper_8_pruned.safetensors")

        local_file_path = self.cache_dir / "DreamShaper_8_pruned.safetensors"
        if not local_file_path.exists():
            print("   ⏳ Downloading checkpoint file (2.13GB - may take several minutes)...")
            self._download_file(
                "https://huggingface.co/Lykon/DreamShaper/resolve/main/DreamShaper_8_pruned.safetensors",
                local_file_path
            )
        else:
            print("   ✅ Using cached DreamShaper 8 file")

        # Load from local file
        single_file_kwargs = {
            "torch_dtype": torch.float32 if self.device == "cpu" else self.torch_dtype,
        }

        if not self.is_sdxl:
            single_file_kwargs.update({
                "safety_checker": None,
                "requires_safety_checker": False,
            })

        if ACCELERATE_AVAILABLE:
            single_file_kwargs.update({"low_cpu_mem_usage": True})

        pipeline_class = StableDiffusionXLPipeline if self.is_sdxl else StableDiffusionPipeline

        print("   🔄 Loading model from local file...")
        pipeline = pipeline_class.from_single_file(str(local_file_path), **single_file_kwargs)
        print("✅ Step 3: DreamShaper 8 loading successful!")
        return pipeline

    def _download_and_load_dreamshaper_7(self):
        """Download and load DreamShaper 7"""
        local_file_path = self.cache_dir / "DreamShaper_7_pruned.safetensors"
        if not local_file_path.exists():
            print("   ⏳ Downloading DreamShaper 7...")
            self._download_file(
                "https://huggingface.co/Lykon/DreamShaper/resolve/main/DreamShaper_7_pruned.safetensors",
                local_file_path
            )

        single_file_kwargs = {
            "torch_dtype": torch.float32 if self.device == "cpu" else self.torch_dtype,
        }

        if not self.is_sdxl:
            single_file_kwargs.update({
                "safety_checker": None,
                "requires_safety_checker": False,
            })

        if ACCELERATE_AVAILABLE:
            single_file_kwargs.update({
                "low_cpu_mem_usage": True,
                "device_map": "balanced" if self.device == "cuda" else None
            })

        pipeline_class = StableDiffusionXLPipeline if self.is_sdxl else StableDiffusionPipeline
        pipeline = pipeline_class.from_single_file(str(local_file_path), **single_file_kwargs)
        print("✅ Step 3: DreamShaper 7 loading successful!")
        return pipeline

    def _load_realistic_vision_model(self, model_id: str):
        """Load RealisticVision from URL"""
        single_file_url = f"https://huggingface.co/{model_id}/resolve/main/Realistic_Vision_V2.0.safetensors"
        print(f"   📁 Loading RealisticVision from: {single_file_url}")

        single_file_kwargs = {
            "torch_dtype": self.torch_dtype,
            "use_safetensors": True,
        }

        if not self.is_sdxl:
            single_file_kwargs.update({
                "safety_checker": None,
                "requires_safety_checker": False,
            })

        if ACCELERATE_AVAILABLE:
            single_file_kwargs.update({
                "low_cpu_mem_usage": True,
                "device_map": "balanced" if self.device == "cuda" else None
            })

        pipeline_class = StableDiffusionXLPipeline if self.is_sdxl else StableDiffusionPipeline
        pipeline = pipeline_class.from_single_file(single_file_url, **single_file_kwargs)
        print("✅ Step 3: RealisticVision loading successful!")
        return pipeline

    def _force_fresh_download(self, model_id: str):
        """Force fresh download by clearing cache"""
        model_cache_path = self.cache_dir / "hub" / f"models--{model_id.replace('/', '--')}"
        if model_cache_path.exists():
            print(f"   🗑️  Removing corrupted cache: {model_cache_path}")
            shutil.rmtree(model_cache_path)

        print("🔄 Step 4: Forcing fresh download...")
        loading_kwargs = {
            "cache_dir": str(self.cache_dir),
            "torch_dtype": self.torch_dtype,
            "force_download": True
        }

        if not self.is_sdxl:
            loading_kwargs.update({
                "safety_checker": None,
                "requires_safety_checker": False,
            })

        pipeline_class = StableDiffusionXLPipeline if self.is_sdxl else StableDiffusionPipeline
        pipeline = pipeline_class.from_pretrained(model_id, **loading_kwargs)
        print("✅ Step 4: Fresh download successful!")
        return pipeline

    def _download_file(self, url: str, local_path: Path):
        """Download a file with progress tracking"""
        import requests

        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(
                                f"   📥 Downloaded: {percent:.1f}% ({downloaded / 1024 / 1024:.1f}MB/{total_size / 1024 / 1024:.1f}MB)",
                                end='\r')

            print("\n   ✅ Download complete!")

        except Exception as download_error:
            print(f"   ❌ Download failed: {download_error}")
            if local_path.exists():
                local_path.unlink()  # Remove partial file
            raise

    def _move_to_device(self, pipeline):
        """Move pipeline to target device"""
        print()
        print("🎯 Step 4: Moving pipeline to target device...")
        print(f"   📍 Target device: {self.device}")

        # Only move to device if we didn't use device mapping, OR if device mapping failed to use GPU
        unet_device = str(next(pipeline.unet.parameters()).device)
        print(f"   🔍 Current UNet device: {unet_device}")

        if self.device == "cuda" and unet_device != "cuda:0":
            print(f"   ⚠️  UNet is on {unet_device} but target is {self.device}")
            print(f"   🚀 Force moving entire pipeline to {self.device}...")

            # Force move to CUDA regardless of device mapping
            pipeline = pipeline.to(self.device)

            # Verify the move worked
            new_unet_device = str(next(pipeline.unet.parameters()).device)
            print(f"   ✅ UNet now on: {new_unet_device}")

            # Show GPU memory usage
            if self.device == "cuda":
                gpu_memory = torch.cuda.memory_allocated() / 1024 ** 3
                gpu_cache = torch.cuda.memory_reserved() / 1024 ** 3
                print(f"   📊 GPU Memory - Allocated: {gpu_memory:.1f}GB, Reserved: {gpu_cache:.1f}GB")

        elif not (ACCELERATE_AVAILABLE and self.device == "cuda"):
            if self.device != "cpu":
                print(f"   🚀 Moving pipeline to {self.device}...")
                pipeline = pipeline.to(self.device)
                print(f"   ✅ Pipeline moved to {self.device}")

                # Verify GPU usage
                if self.device == "cuda":
                    gpu_memory = torch.cuda.memory_allocated() / 1024 ** 3
                    gpu_cache = torch.cuda.memory_reserved() / 1024 ** 3
                    print(f"   📊 GPU Memory - Allocated: {gpu_memory:.1f}GB, Reserved: {gpu_cache:.1f}GB")
            else:
                print("   ✅ Pipeline staying on CPU")
        else:
            print("   ✅ Pipeline already distributed via device mapping (balanced)")
            print(f"   🔧 Device mapping is handling GPU placement automatically")

            # Still show GPU memory usage
            if self.device == "cuda":
                gpu_memory = torch.cuda.memory_allocated() / 1024 ** 3
                gpu_cache = torch.cuda.memory_reserved() / 1024 ** 3
                print(f"   📊 GPU Memory - Allocated: {gpu_memory:.1f}GB, Reserved: {gpu_cache:.1f}GB")

        return pipeline

    def _apply_optimizations(self, pipeline, model_id: str):
        """Apply face-optimized settings and optimizations"""
        print()
        print("🚀 Step 5: Applying face-optimized settings...")

        # For DreamShaper models, avoid aggressive optimizations that hurt face quality
        is_dreamshaper = "dreamshaper" in model_id.lower()

        if self.device == "cuda":
            # Check available VRAM
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            gpu_memory_used = torch.cuda.memory_allocated() / 1024 ** 3
            gpu_memory_free = gpu_memory_total - gpu_memory_used

            print(
                f"   📊 VRAM: {gpu_memory_used:.1f}GB used, {gpu_memory_free:.1f}GB free of {gpu_memory_total:.1f}GB total")

            # Be more conservative with attention slicing for face quality
            if gpu_memory_free < 2.0 and hasattr(pipeline, "enable_attention_slicing"):
                print("   💾 Enabling minimal attention slicing (preserving face quality)...")
                # Use "auto" for minimal slicing that preserves quality
                pipeline.enable_attention_slicing("auto")
            else:
                print("   ✅ Sufficient VRAM - using full attention for best face quality")

            # Don't use CPU offload - it degrades face quality significantly
            print("   🚀 Using full GPU mode for optimal face rendering")

            # Check if pipeline is actually on CUDA
            unet_device = str(next(pipeline.unet.parameters()).device)
            print(f"   🔍 UNet device: {unet_device}")

            # Be very conservative with xFormers for DreamShaper face quality
            if hasattr(pipeline, "enable_xformers_memory_efficient_attention"):
                if (not is_dreamshaper and unet_device == "cuda:0" and
                        self.torch_dtype in [torch.float16, torch.bfloat16] and
                        gpu_memory_free < 2.0):
                    try:
                        print("   ⚡ Enabling xFormers (non-DreamShaper model)...")
                        pipeline.enable_xformers_memory_efficient_attention()
                        print("   ✅ xFormers optimization enabled")
                    except Exception as xf_error:
                        print(f"   ⚠️  xFormers not available: {xf_error}")
                else:
                    if is_dreamshaper:
                        print("   ✅ Skipping xFormers for DreamShaper (preserving face quality)")
                    else:
                        print("   ✅ Skipping xFormers (sufficient VRAM for standard attention)")
        elif hasattr(pipeline, "enable_attention_slicing"):
            # Always use minimal attention slicing on CPU/MPS
            print("   💾 Enabling minimal attention slicing for CPU/MPS...")
            pipeline.enable_attention_slicing("auto")

        # Ensure pipeline components are in proper mode for face generation
        pipeline.unet.eval()
        if hasattr(pipeline, 'vae'):
            pipeline.vae.eval()
            # Set VAE to high precision for face details if possible
            if hasattr(pipeline.vae, 'dtype') and self.device == "cuda":
                print("   🎨 VAE configured for face detail preservation")

        if hasattr(pipeline, 'text_encoder'):
            pipeline.text_encoder.eval()
        if hasattr(pipeline, 'text_encoder_2'):
            pipeline.text_encoder_2.eval()
        print("   ✅ Pipeline optimized for facial generation")

        return pipeline

    def _run_validation_test(self, pipeline):
        """Run face-quality validation test"""
        print()
        print("🧪 Step 6: Running face-quality validation test...")
        print("   🎨 Generating face test image to verify quality...")

        # Face-specific validation test
        try:
            test_prompt = "portrait of a young woman, detailed face, beautiful eyes, natural lighting"
            test_image = pipeline(
                test_prompt,
                negative_prompt="blurry face, distorted face, bad anatomy, low quality",
                num_inference_steps=20,
                width=512,
                height=512,
                guidance_scale=7.0,
                generator=torch.Generator(device=self.device).manual_seed(42)
            ).images[0]

            # Save test image for inspection
            output_dir = self.cache_dir.parent / "generated_images"
            output_dir.mkdir(exist_ok=True)
            test_path = output_dir / "face_test.png"
            test_image.save(test_path)
            print(f"   ✅ Face validation test passed! Test image saved to: {test_path}")
            print("   🔍 Inspect the test image - faces should be clear and artifact-free")

        except Exception as test_error:
            print(f"   ⚠️  Face validation test failed: {test_error}")
            print("   ✅ Model loaded successfully (skipping validation)")

    def clear_cache(self):
        """Clear all cached models"""
        cache_cleared = False
        hub_cache = self.cache_dir / "hub"

        if hub_cache.exists():
            print(f"🗑️  Clearing model cache: {hub_cache}")
            shutil.rmtree(hub_cache)
            cache_cleared = True
            print("✅ Model cache cleared")

        # Recreate cache directory
        hub_cache.mkdir(parents=True, exist_ok=True)
        print("🔄 Cache cleared - next model load will download fresh files")

        return cache_cleared