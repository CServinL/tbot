#!/usr/bin/env python3
"""
ComfyUI HTTP endpoints for Stable Diffusion MCP Server
Adds ComfyUI-compatible routes to the existing HTTP server
"""

import json
from urllib.parse import urlparse, parse_qs


class ComfyUIHTTPHandler:
    """Mixin class to add ComfyUI endpoints to the existing HTTP handler"""

    def handle_comfyui_routes(self):
        """Route ComfyUI-specific endpoints"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        query_params = parse_qs(parsed_path.query)

        # ComfyUI API endpoints
        if path == '/prompt':
            if self.command == 'POST':
                self.handle_comfyui_prompt()
            else:
                self.send_error(405, "Method Not Allowed")

        elif path == '/queue':
            if self.command == 'GET':
                self.handle_comfyui_queue()
            elif self.command == 'POST':
                self.handle_comfyui_queue_action()
            else:
                self.send_error(405, "Method Not Allowed")

        elif path == '/history':
            if self.command == 'GET':
                prompt_id = query_params.get('prompt_id', [None])[0]
                self.handle_comfyui_history(prompt_id)
            elif self.command == 'POST':
                self.handle_comfyui_history_action()
            else:
                self.send_error(405, "Method Not Allowed")

        elif path == '/system_stats':
            self.handle_comfyui_system_stats()

        elif path == '/embeddings':
            self.handle_comfyui_embeddings()

        elif path == '/extensions':
            self.handle_comfyui_extensions()

        elif path == '/object_info':
            self.handle_comfyui_object_info()

        elif path.startswith('/view'):
            # Handle image viewing (similar to our /images endpoint)
            self.handle_comfyui_view()

        else:
            return False  # Not a ComfyUI route

        return True  # Was a ComfyUI route

    def handle_comfyui_prompt(self):
        """Handle ComfyUI /prompt endpoint"""
        try:
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self.send_json_error(400, "Missing request body")
                return

            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))

            # Process through ComfyUI adapter
            result = self.sd_server.comfyui_adapter.submit_prompt(data)
            self.send_json_response(200, result)

        except json.JSONDecodeError:
            self.send_json_error(400, "Invalid JSON in request body")
        except Exception as e:
            self.sd_server.logger.error(f"ComfyUI prompt endpoint error: {e}")
            self.send_json_error(500, f"Internal server error: {str(e)}")

    def handle_comfyui_queue(self):
        """Handle ComfyUI /queue GET endpoint"""
        try:
            result = self.sd_server.comfyui_adapter.get_queue_status()
            self.send_json_response(200, result)
        except Exception as e:
            self.sd_server.logger.error(f"ComfyUI queue endpoint error: {e}")
            self.send_json_error(500, str(e))

    def handle_comfyui_queue_action(self):
        """Handle ComfyUI /queue POST endpoint (actions like clear, interrupt)"""
        try:
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length) if content_length > 0 else b'{}'
            data = json.loads(body.decode('utf-8'))

            action = data.get("clear", False)
            interrupt = data.get("interrupt", False)

            if action:
                result = self.sd_server.comfyui_adapter.clear_queue()
                self.send_json_response(200, result)
            elif interrupt:
                result = self.sd_server.comfyui_adapter.interrupt_execution()
                self.send_json_response(200, result)
            else:
                self.send_json_error(400, "Unknown queue action")

        except json.JSONDecodeError:
            self.send_json_error(400, "Invalid JSON in request body")
        except Exception as e:
            self.sd_server.logger.error(f"ComfyUI queue action error: {e}")
            self.send_json_error(500, str(e))

    def handle_comfyui_history(self, prompt_id=None):
        """Handle ComfyUI /history endpoint"""
        try:
            result = self.sd_server.comfyui_adapter.get_history(prompt_id)
            self.send_json_response(200, result)
        except Exception as e:
            self.sd_server.logger.error(f"ComfyUI history endpoint error: {e}")
            self.send_json_error(500, str(e))

    def handle_comfyui_history_action(self):
        """Handle ComfyUI /history POST endpoint (clear history)"""
        try:
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length) if content_length > 0 else b'{}'
            data = json.loads(body.decode('utf-8'))

            if data.get("clear", False):
                # Clear history
                self.sd_server.comfyui_adapter.history.clear()
                self.send_json_response(200, {"cleared": True})
            else:
                self.send_json_error(400, "Unknown history action")

        except json.JSONDecodeError:
            self.send_json_error(400, "Invalid JSON in request body")
        except Exception as e:
            self.sd_server.logger.error(f"ComfyUI history action error: {e}")
            self.send_json_error(500, str(e))

    def handle_comfyui_system_stats(self):
        """Handle ComfyUI /system_stats endpoint"""
        try:
            result = self.sd_server.comfyui_adapter.get_system_stats()
            self.send_json_response(200, result)
        except Exception as e:
            self.sd_server.logger.error(f"ComfyUI system stats error: {e}")
            self.send_json_error(500, str(e))

    def handle_comfyui_embeddings(self):
        """Handle ComfyUI /embeddings endpoint"""
        try:
            result = self.sd_server.comfyui_adapter.get_embeddings()
            self.send_json_response(200, result)
        except Exception as e:
            self.sd_server.logger.error(f"ComfyUI embeddings error: {e}")
            self.send_json_error(500, str(e))

    def handle_comfyui_extensions(self):
        """Handle ComfyUI /extensions endpoint"""
        try:
            result = self.sd_server.comfyui_adapter.get_extensions()
            self.send_json_response(200, result)
        except Exception as e:
            self.sd_server.logger.error(f"ComfyUI extensions error: {e}")
            self.send_json_error(500, str(e))

    def handle_comfyui_object_info(self):
        """Handle ComfyUI /object_info endpoint"""
        # Return basic node information that ComfyUI clients expect
        object_info = {
            "CheckpointLoaderSimple": {
                "input": {
                    "required": {
                        "ckpt_name": ["CHECKPOINT"]
                    }
                },
                "output": ["MODEL", "CLIP", "VAE"],
                "output_is_list": [False, False, False],
                "output_name": ["MODEL", "CLIP", "VAE"],
                "name": "CheckpointLoaderSimple",
                "display_name": "Load Checkpoint",
                "description": "Loads a checkpoint file",
                "category": "loaders"
            },
            "CLIPTextEncode": {
                "input": {
                    "required": {
                        "text": ["STRING", {"multiline": True}],
                        "clip": ["CLIP"]
                    }
                },
                "output": ["CONDITIONING"],
                "output_is_list": [False],
                "output_name": ["CONDITIONING"],
                "name": "CLIPTextEncode",
                "display_name": "CLIP Text Encode (Prompt)",
                "description": "Encodes a text prompt using CLIP",
                "category": "conditioning"
            },
            "KSampler": {
                "input": {
                    "required": {
                        "model": ["MODEL"],
                        "seed": ["INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}],
                        "steps": ["INT", {"default": 20, "min": 1, "max": 10000}],
                        "cfg": ["FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1}],
                        "sampler_name": [
                            ["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral", "lms", "dpm_fast",
                             "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_2m", "dpmpp_2m_karras",
                             "dpmpp_sde_karras", "ddim", "plms", "ddpm"]],
                        "scheduler": [["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"]],
                        "positive": ["CONDITIONING"],
                        "negative": ["CONDITIONING"],
                        "latent_image": ["LATENT"],
                        "denoise": ["FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}]
                    }
                },
                "output": ["LATENT"],
                "output_is_list": [False],
                "output_name": ["LATENT"],
                "name": "KSampler",
                "display_name": "KSampler",
                "description": "Samples latent images using K-diffusion samplers",
                "category": "sampling"
            },
            "EmptyLatentImage": {
                "input": {
                    "required": {
                        "width": ["INT", {"default": 512, "min": 16, "max": 8192, "step": 8}],
                        "height": ["INT", {"default": 512, "min": 16, "max": 8192, "step": 8}],
                        "batch_size": ["INT", {"default": 1, "min": 1, "max": 4096}]
                    }
                },
                "output": ["LATENT"],
                "output_is_list": [False],
                "output_name": ["LATENT"],
                "name": "EmptyLatentImage",
                "display_name": "Empty Latent Image",
                "description": "Creates an empty latent image",
                "category": "latent"
            },
            "VAEDecode": {
                "input": {
                    "required": {
                        "samples": ["LATENT"],
                        "vae": ["VAE"]
                    }
                },
                "output": ["IMAGE"],
                "output_is_list": [False],
                "output_name": ["IMAGE"],
                "name": "VAEDecode",
                "display_name": "VAE Decode",
                "description": "Decodes latent images to pixel images",
                "category": "latent"
            },
            "SaveImage": {
                "input": {
                    "required": {
                        "images": ["IMAGE"],
                        "filename_prefix": ["STRING", {"default": "ComfyUI"}]
                    }
                },
                "output": [],
                "output_is_list": [],
                "output_name": [],
                "name": "SaveImage",
                "display_name": "Save Image",
                "description": "Saves images to disk",
                "category": "image"
            }
        }

        self.send_json_response(200, object_info)

    def handle_comfyui_view(self):
        """Handle ComfyUI /view endpoint (similar to our /images endpoint)"""
        # Extract parameters from query string
        parsed_path = urlparse(self.path)
        query_params = parse_qs(parsed_path.query)

        filename = query_params.get('filename', [None])[0]
        subfolder = query_params.get('subfolder', [''])[0]
        type_param = query_params.get('type', ['output'])[0]

        if not filename:
            self.send_error(400, "Missing filename parameter")
            return

        # Serve the image (reuse our existing image serving logic)
        self.serve_image(filename)