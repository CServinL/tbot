#!/usr/bin/env python3
"""
Image generation implementation with multiple CLIP passes for long prompts
"""

import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from dependencies import torch, SCHEDULER_MAP


class ImageGenerator:
    """Handles image generation with support for long prompts via multiple CLIP passes"""

    def __init__(self, pipeline, device, torch_dtype, is_sdxl, output_dir, logger):
        self.pipeline = pipeline
        self.device = device
        self.torch_dtype = torch_dtype
        self.is_sdxl = is_sdxl
        self.output_dir = Path(output_dir)
        self.logger = logger

        # Ensure output directory exists
        self.output_dir.mkdir(exist_ok=True)

    def generate_image(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate an image using the loaded model with support for long prompts"""

        # Extract generation parameters with defaults optimized for facial quality
        negative_prompt = kwargs.get("negative_prompt",
                                     "blurry, low quality, distorted, ugly, bad anatomy, worst quality, low quality, normal quality, lowres, "
                                     "bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, jpeg artifacts, signature, "
                                     "watermark, username, bad face, deformed face, mutated face, disfigured face, face mutation, "
                                     "asymmetric face, malformed face, weird face, strange face, scary face, distorted eyes, "
                                     "cross-eyed, asymmetric eyes, deformed eyes, malformed eyes, extra eyes, missing eyes, "
                                     "bad mouth, deformed mouth, malformed mouth, extra mouth, missing mouth, bad nose, deformed nose, "
                                     "malformed nose, extra nose, missing nose, bad teeth, deformed teeth, extra teeth, missing teeth")

        width = kwargs.get("width", 1024 if self.is_sdxl else 512)
        height = kwargs.get("height", 1024 if self.is_sdxl else 512)
        num_inference_steps = kwargs.get("num_inference_steps", kwargs.get("steps", 25))
        guidance_scale = kwargs.get("guidance_scale", kwargs.get("cfg_scale", 7.0))
        num_images = kwargs.get("num_images", 1)
        seed = kwargs.get("seed")

        # Advanced parameters - optimized for facial quality
        sampler = kwargs.get("sampler", kwargs.get("scheduler", "DPMSolverMultistep"))
        eta = kwargs.get("eta", 0.0)
        clip_skip = kwargs.get("clip_skip", 1)  # Don't skip CLIP layers for faces

        # Hi-res fix parameters
        enable_hires_fix = kwargs.get("enable_hires_fix", False)
        hires_steps = kwargs.get("hires_steps", 20)
        hires_upscale = kwargs.get("hires_upscale", 2.0)
        hires_upscaler = kwargs.get("hires_upscaler", "Latent")
        denoising_strength = kwargs.get("denoising_strength", 0.7)

        # Validate and fix dimensions to prevent artifacts
        width, height = self._validate_dimensions(width, height)

        # Handle long prompts - use multiple CLIP passes instead of truncation
        if self._needs_chunking(prompt) or self._needs_chunking(negative_prompt):
            self.logger.info("Using multiple CLIP passes for long prompt(s)")

        # Increase steps for complex prompts
        if len(prompt) > 200:  # Long detailed prompt
            if num_inference_steps < 30:
                num_inference_steps = max(30, num_inference_steps)
                self.logger.info(f"Increased steps to {num_inference_steps} for detailed prompt")

        # Set up scheduler based on sampler parameter
        if sampler != "DPMSolverMultistep":
            self._set_scheduler(sampler)

        # Set seed for reproducibility
        generator = self._setup_generator(seed)
        if seed is None:
            seed = torch.randint(0, 2 ** 32, (1,)).item()

        start_time = time.time()
        request_id = str(uuid.uuid4())[:8]

        self.logger.info(f"Generating image: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        self.logger.info(
            f"Parameters: {width}x{height}, steps={num_inference_steps}, cfg={guidance_scale}, sampler={sampler}, seed={seed}")

        try:
            # Ensure we're in no_grad context for inference
            with torch.no_grad():
                # Check if we need multiple CLIP passes for long prompts
                if self._needs_chunking(prompt) or self._needs_chunking(negative_prompt):
                    # Use multiple CLIP passes with blended embeddings
                    prompt_embeds, negative_prompt_embeds = self._encode_long_prompts(prompt, negative_prompt)

                    # Prepare generation kwargs with pre-computed embeddings
                    generation_kwargs = {
                        "prompt_embeds": prompt_embeds,
                        "negative_prompt_embeds": negative_prompt_embeds,
                        "width": width,
                        "height": height,
                        "num_inference_steps": num_inference_steps,
                        "guidance_scale": guidance_scale,
                        "num_images_per_prompt": num_images,
                        "generator": generator,
                        "eta": eta,
                    }
                else:
                    # Standard generation with short prompts
                    generation_kwargs = {
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "width": width,
                        "height": height,
                        "num_inference_steps": num_inference_steps,
                        "guidance_scale": guidance_scale,
                        "num_images_per_prompt": num_images,
                        "generator": generator,
                        "eta": eta,
                    }

                # Handle clip skip for models that support it
                if clip_skip > 1 and hasattr(self.pipeline, 'text_encoder'):
                    self.logger.info(f"Using CLIP skip: {clip_skip}")

                # Generate the image
                if enable_hires_fix:
                    # Two-stage generation: low-res then upscale
                    self.logger.info(
                        f"Hi-res fix enabled: {width}x{height} -> {int(width * hires_upscale)}x{int(height * hires_upscale)}")

                    # First stage: generate at lower resolution
                    base_width = int(width / hires_upscale)
                    base_height = int(height / hires_upscale)

                    base_kwargs = generation_kwargs.copy()
                    base_kwargs.update({
                        "width": base_width,
                        "height": base_height,
                        "num_inference_steps": max(10, num_inference_steps // 2)
                    })

                    base_result = self.pipeline(**base_kwargs)

                    # For now, fallback to base generation at full resolution
                    self.logger.warning(
                        "Hi-res fix requested but img2img pipeline not implemented, using base generation")
                    result = self.pipeline(**generation_kwargs)
                else:
                    # Standard generation
                    result = self.pipeline(**generation_kwargs)

            # Save images with proper handling
            image_paths = self._save_images(result.images, request_id)

            generation_time = time.time() - start_time
            self.logger.info(f"Image generation complete: {generation_time:.1f}s, {len(image_paths)} images")

            return {
                "images": image_paths,
                "metadata": {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "width": width,
                    "height": height,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "seed": seed,
                    "sampler": sampler,
                    "eta": eta,
                    "clip_skip": clip_skip,
                    "enable_hires_fix": enable_hires_fix,
                    "hires_steps": hires_steps if enable_hires_fix else None,
                    "hires_upscale": hires_upscale if enable_hires_fix else None,
                    "denoising_strength": denoising_strength if enable_hires_fix else None,
                },
                "generation_time": round(generation_time, 2),
                "request_id": request_id
            }

        except Exception as e:
            self.logger.error(f"Image generation failed: {e}")
            raise RuntimeError(f"Image generation failed: {e}")

    def _validate_dimensions(self, width: int, height: int) -> tuple:
        """Validate and fix dimensions to prevent artifacts"""
        # Ensure divisible by 8
        if width % 8 != 0:
            width = (width // 8) * 8
            self.logger.warning(f"Width adjusted to {width} (must be divisible by 8)")
        if height % 8 != 0:
            height = (height // 8) * 8
            self.logger.warning(f"Height adjusted to {height} (must be divisible by 8)")

        # Validate reasonable dimensions - be more permissive but maintain aspect ratio
        max_pixels = 1536 * 1536 if self.is_sdxl else 1024 * 1024
        if width * height > max_pixels:
            # Maintain aspect ratio when scaling
            aspect_ratio = width / height
            max_width = int((max_pixels * aspect_ratio) ** 0.5)
            max_height = int(max_pixels / max_width)

            # Ensure divisible by 8
            width = (max_width // 8) * 8
            height = (max_height // 8) * 8

            self.logger.warning(f"Dimensions scaled to {width}x{height} maintaining aspect ratio")

        return width, height

    def _setup_generator(self, seed):
        """Setup random generator with seed"""
        if seed is not None:
            return torch.Generator(device=self.device).manual_seed(seed)
        else:
            seed = torch.randint(0, 2 ** 32, (1,)).item()
            return torch.Generator(device=self.device).manual_seed(seed)

    def _needs_chunking(self, prompt: str) -> bool:
        """Check if a prompt needs chunking due to length"""
        if not prompt or not prompt.strip():
            return False

        tokenizer = self.pipeline.tokenizer
        tokens = tokenizer.encode(prompt)
        return len(tokens) > 75  # Leave room for start/end tokens

    def _encode_long_prompts(self, prompt: str, negative_prompt: str = "") -> tuple:
        """
        Encode long prompts using multiple CLIP passes and blend embeddings.
        This prevents truncation artifacts while handling unlimited prompt length.
        """
        tokenizer = self.pipeline.tokenizer
        text_encoder = self.pipeline.text_encoder

        def chunk_prompt(text: str, max_tokens: int = 75) -> list:
            """Split prompt into overlapping chunks that preserve context"""
            if not text.strip():
                return [""]

            # Tokenize the full prompt
            tokens = tokenizer.encode(text, add_special_tokens=False)

            if len(tokens) <= max_tokens:
                return [text]

            # Split into overlapping chunks to preserve context
            chunks = []
            overlap = 10  # Overlap tokens to maintain context

            for i in range(0, len(tokens), max_tokens - overlap):
                chunk_tokens = tokens[i:i + max_tokens]
                chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                chunks.append(chunk_text.strip())

            # Ensure we don't have empty chunks
            chunks = [chunk for chunk in chunks if chunk.strip()]

            return chunks if chunks else [""]

        # Chunk both prompts
        pos_chunks = chunk_prompt(prompt)
        neg_chunks = chunk_prompt(negative_prompt)

        # Log chunking info
        if len(pos_chunks) > 1:
            self.logger.info(f"Split positive prompt into {len(pos_chunks)} chunks")
        if len(neg_chunks) > 1:
            self.logger.info(f"Split negative prompt into {len(neg_chunks)} chunks")

        # Ensure we have the same number of chunks (pad with empty strings)
        max_chunks = max(len(pos_chunks), len(neg_chunks))
        while len(pos_chunks) < max_chunks:
            pos_chunks.append("")
        while len(neg_chunks) < max_chunks:
            neg_chunks.append("")

        def encode_chunk(text: str) -> torch.Tensor:
            """Encode a single text chunk"""
            if not text.strip():
                # Return zero embedding for empty chunks
                hidden_size = text_encoder.config.hidden_size
                max_length = tokenizer.model_max_length
                return torch.zeros(
                    (1, max_length, hidden_size),
                    dtype=self.torch_dtype,
                    device=self.device
                )

            # Tokenize and encode
            text_inputs = tokenizer(
                text,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )

            with torch.no_grad():
                text_embeddings = text_encoder(text_inputs.input_ids.to(self.device))[0]

            return text_embeddings

        # Encode all chunks
        pos_embeddings = []
        neg_embeddings = []

        for pos_chunk, neg_chunk in zip(pos_chunks, neg_chunks):
            pos_emb = encode_chunk(pos_chunk)
            neg_emb = encode_chunk(neg_chunk)

            pos_embeddings.append(pos_emb)
            neg_embeddings.append(neg_emb)

        if len(pos_embeddings) > 1:
            # Blend embeddings using weighted average
            # Give more weight to earlier chunks (usually more important)
            weights = torch.tensor([1.0 / (i + 1) for i in range(len(pos_embeddings))], device=self.device)
            weights = weights / weights.sum()

            # Weighted average of embeddings
            final_pos_emb = torch.zeros_like(pos_embeddings[0])
            final_neg_emb = torch.zeros_like(neg_embeddings[0])

            for i, (pos_emb, neg_emb) in enumerate(zip(pos_embeddings, neg_embeddings)):
                final_pos_emb += weights[i] * pos_emb
                final_neg_emb += weights[i] * neg_emb

            self.logger.info(f"Blended {len(pos_embeddings)} embeddings with weighted average")
        else:
            final_pos_emb = pos_embeddings[0]
            final_neg_emb = neg_embeddings[0]

        return final_pos_emb, final_neg_emb

    def _set_scheduler(self, sampler_name: str):
        """Set the scheduler based on sampler name"""
        try:
            if sampler_name in SCHEDULER_MAP:
                scheduler_class = SCHEDULER_MAP[sampler_name]

                # Get current scheduler config
                scheduler_config = self.pipeline.scheduler.config

                # Create new scheduler with Karras sigmas if requested
                if "Karras" in sampler_name:
                    new_scheduler = scheduler_class.from_config(
                        scheduler_config,
                        use_karras_sigmas=True
                    )
                else:
                    new_scheduler = scheduler_class.from_config(scheduler_config)

                self.pipeline.scheduler = new_scheduler
                self.logger.info(f"Scheduler set to: {sampler_name}")
            else:
                self.logger.warning(f"Unknown sampler '{sampler_name}', using default DPMSolverMultistep")

        except Exception as e:
            self.logger.error(f"Failed to set scheduler '{sampler_name}': {e}")
            # Keep current scheduler on error

    def _save_images(self, images, request_id: str) -> list:
        """Save generated images and return their paths"""
        image_paths = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for i, image in enumerate(images):
            filename = f"{timestamp}_{request_id}_{i}.png"
            image_path = self.output_dir / filename

            # Ensure image is in RGB mode before saving
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Save with high quality settings
            image.save(image_path, format='PNG', optimize=False, compress_level=1)

            # Return relative path for web serving
            image_paths.append(f"/images/{filename}")

        return image_paths