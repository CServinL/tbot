#!/usr/bin/env python3
"""
ComfyUI API compatibility layer for Stable Diffusion MCP Server
Provides ComfyUI-compatible endpoints for seamless integration with ComfyUI workflows
"""

import json
import time
import uuid
from collections import deque
from datetime import datetime
from typing import Dict, Any, Optional


class ComfyUIAdapter:
    """Adapter to provide ComfyUI-compatible API endpoints"""

    def __init__(self, sd_server):
        self.sd_server = sd_server

        # Queue and history management
        self.queue = deque()
        self.history = {}
        self.current_execution = None
        self.execution_counter = 0

        # Default ComfyUI workflow structure
        self.default_workflow = {
            "1": {
                "inputs": {
                    "ckpt_name": "model.safetensors"
                },
                "class_type": "CheckpointLoaderSimple"
            },
            "2": {
                "inputs": {
                    "text": "beautiful landscape",
                    "clip": ["1", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "3": {
                "inputs": {
                    "text": "blurry, low quality",
                    "clip": ["1", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "4": {
                "inputs": {
                    "width": 512,
                    "height": 512,
                    "batch_size": 1
                },
                "class_type": "EmptyLatentImage"
            },
            "5": {
                "inputs": {
                    "seed": -1,
                    "steps": 20,
                    "cfg": 7.5,
                    "sampler_name": "dpmpp_2m",
                    "scheduler": "karras",
                    "denoise": 1.0,
                    "model": ["1", 0],
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                    "latent_image": ["4", 0]
                },
                "class_type": "KSampler"
            },
            "6": {
                "inputs": {
                    "samples": ["5", 0],
                    "vae": ["1", 2]
                },
                "class_type": "VAEDecode"
            },
            "7": {
                "inputs": {
                    "filename_prefix": "ComfyUI",
                    "images": ["6", 0]
                },
                "class_type": "SaveImage"
            }
        }

    def submit_prompt(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle ComfyUI /prompt endpoint"""
        try:
            # Extract workflow and client_id
            workflow = data.get("prompt", self.default_workflow)
            client_id = data.get("client_id", str(uuid.uuid4()))

            # Generate execution ID
            self.execution_counter += 1
            prompt_id = str(self.execution_counter)

            # Parse ComfyUI workflow to extract generation parameters
            generation_params = self._parse_comfyui_workflow(workflow)

            # Add to queue
            queue_item = {
                "prompt_id": prompt_id,
                "client_id": client_id,
                "workflow": workflow,
                "generation_params": generation_params,
                "status": "queued",
                "timestamp": datetime.now().isoformat()
            }

            self.queue.append(queue_item)

            # Start execution if not already running
            if self.current_execution is None:
                self._execute_next_in_queue()

            return {
                "prompt_id": prompt_id,
                "number": len(self.queue),
                "node_errors": {}
            }

        except Exception as e:
            return {
                "error": str(e),
                "node_errors": {}
            }

    def get_queue_status(self) -> Dict[str, Any]:
        """Handle ComfyUI /queue endpoint"""
        queue_running = []
        queue_pending = []

        # Current execution
        if self.current_execution:
            queue_running.append([
                self.current_execution["prompt_id"],
                self.current_execution["workflow"]
            ])

        # Pending queue items
        for item in self.queue:
            if item["status"] == "queued":
                queue_pending.append([
                    item["prompt_id"],
                    item["workflow"]
                ])

        return {
            "queue_running": queue_running,
            "queue_pending": queue_pending
        }

    def get_history(self, prompt_id: Optional[str] = None) -> Dict[str, Any]:
        """Handle ComfyUI /history endpoint"""
        if prompt_id:
            return {prompt_id: self.history.get(prompt_id, {})}
        else:
            return self.history

    def clear_queue(self) -> Dict[str, Any]:
        """Clear the generation queue"""
        cleared_count = len(self.queue)
        self.queue.clear()

        return {
            "deleted": list(range(cleared_count))
        }

    def interrupt_execution(self) -> Dict[str, Any]:
        """Interrupt current execution"""
        if self.current_execution:
            self.current_execution["status"] = "interrupted"
            self.current_execution = None
            return {"interrupted": True}
        return {"interrupted": False}

    def _parse_comfyui_workflow(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Parse ComfyUI workflow and extract generation parameters"""
        params = {
            "prompt": "beautiful landscape",
            "negative_prompt": "blurry, low quality",
            "width": 512,
            "height": 512,
            "num_inference_steps": 20,
            "guidance_scale": 7.5,
            "seed": None,
            "sampler": "DPMSolverMultistep"
        }

        try:
            # Look for common ComfyUI node types and extract parameters
            for node_id, node in workflow.items():
                class_type = node.get("class_type", "")
                inputs = node.get("inputs", {})

                # Extract positive prompt
                if class_type == "CLIPTextEncode":
                    text = inputs.get("text", "")
                    if text and "negative" not in text.lower() and "bad" not in text.lower():
                        params["prompt"] = text
                    elif text:
                        params["negative_prompt"] = text

                # Extract sampler settings
                elif class_type == "KSampler":
                    if "steps" in inputs:
                        params["num_inference_steps"] = inputs["steps"]
                    if "cfg" in inputs:
                        params["guidance_scale"] = inputs["cfg"]
                    if "seed" in inputs and inputs["seed"] != -1:
                        params["seed"] = inputs["seed"]
                    if "sampler_name" in inputs:
                        params["sampler"] = self._convert_comfyui_sampler(inputs["sampler_name"])

                # Extract dimensions
                elif class_type == "EmptyLatentImage":
                    if "width" in inputs:
                        params["width"] = inputs["width"]
                    if "height" in inputs:
                        params["height"] = inputs["height"]

                # Handle text input nodes
                elif class_type in ["String", "Text"]:
                    text = inputs.get("text", "")
                    if text and len(text) > len(params["prompt"]):
                        params["prompt"] = text

        except Exception as e:
            self.sd_server.logger.warning(f"Error parsing ComfyUI workflow: {e}")

        return params

    def _convert_comfyui_sampler(self, comfyui_sampler: str) -> str:
        """Convert ComfyUI sampler names to our sampler names"""
        sampler_mapping = {
            "euler": "Euler",
            "euler_ancestral": "Euler a",
            "heun": "Heun",
            "dpm_2": "DPM2",
            "dpm_2_ancestral": "DPM2 a",
            "lms": "LMS",
            "dpm_fast": "DPMSolverMultistep",
            "dpm_adaptive": "DPMSolverMultistep",
            "dpmpp_2s_ancestral": "DPM++ SDE",
            "dpmpp_sde": "DPM++ SDE",
            "dpmpp_2m": "DPM++ 2M",
            "dpmpp_2m_karras": "DPM++ 2M Karras",
            "dpmpp_sde_karras": "DPM++ SDE Karras",
            "ddim": "DDIM",
            "plms": "PNDM",
            "ddpm": "DDPM"
        }

        return sampler_mapping.get(comfyui_sampler.lower(), "DPMSolverMultistep")

    def _execute_next_in_queue(self):
        """Execute the next item in the queue"""
        if not self.queue or self.current_execution is not None:
            return

        # Get next item from queue
        queue_item = self.queue.popleft()
        self.current_execution = queue_item
        self.current_execution["status"] = "running"

        try:
            # Execute generation
            prompt_id = queue_item["prompt_id"]
            generation_params = queue_item["generation_params"]

            self.sd_server.logger.info(f"Executing ComfyUI prompt {prompt_id}")

            # Generate image using our server
            result = self.sd_server.generate_image(**generation_params)

            # Convert result to ComfyUI format
            comfyui_result = self._convert_result_to_comfyui_format(result, queue_item)

            # Add to history
            self.history[prompt_id] = comfyui_result

            # Mark as completed
            self.current_execution["status"] = "completed"
            self.current_execution = None

            # Execute next item if available
            if self.queue:
                self._execute_next_in_queue()

        except Exception as e:
            self.sd_server.logger.error(f"ComfyUI execution failed: {e}")

            # Mark as failed and add to history
            error_result = {
                "status": {
                    "status_str": "error",
                    "completed": True,
                    "messages": [["execution_error", {"exception_message": str(e)}]]
                },
                "outputs": {}
            }

            self.history[queue_item["prompt_id"]] = error_result
            self.current_execution = None

            # Continue with next item
            if self.queue:
                self._execute_next_in_queue()

    def _convert_result_to_comfyui_format(self, result: Dict[str, Any], queue_item: Dict[str, Any]) -> Dict[str, Any]:
        """Convert our generation result to ComfyUI format"""
        prompt_id = queue_item["prompt_id"]

        # Extract image information
        images = result.get("images", [])
        metadata = result.get("metadata", {})

        # Create ComfyUI-style outputs
        outputs = {}

        if images:
            # ComfyUI expects outputs organized by node ID
            # We'll use node "7" as the SaveImage node (standard in workflows)
            outputs["7"] = {
                "images": []
            }

            for i, image_path in enumerate(images):
                # Extract filename from path
                filename = image_path.split("/")[-1] if "/" in image_path else image_path

                outputs["7"]["images"].append({
                    "filename": filename,
                    "subfolder": "",
                    "type": "output"
                })

        return {
            "status": {
                "status_str": "success",
                "completed": True,
                "messages": []
            },
            "outputs": outputs,
            "meta": {
                "prompt_id": prompt_id,
                "workflow_execution_time": result.get("generation_time", 0),
                "generation_params": metadata
            }
        }

    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics in ComfyUI format"""
        return {
            "system": {
                "os": "linux",
                "python_version": "3.x",
                "embedded_python": False
            },
            "devices": [
                {
                    "name": self.sd_server.device,
                    "type": self.sd_server.device.upper(),
                    "index": 0,
                    "vram_total": 0,
                    "vram_free": 0,
                    "torch_vram_total": 0,
                    "torch_vram_free": 0
                }
            ],
            "queue": {
                "pending": len(self.queue),
                "running": 1 if self.current_execution else 0
            }
        }

    def get_embeddings(self) -> Dict[str, Any]:
        """Get available embeddings/textual inversions"""
        return {
            "loaded": {},
            "available": {}
        }

    def get_extensions(self) -> Dict[str, Any]:
        """Get available extensions"""
        return {
            "extensions": [],
            "disabled": []
        }