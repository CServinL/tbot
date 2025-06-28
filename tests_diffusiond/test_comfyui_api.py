#!/usr/bin/env python3
"""
Unit tests for the ComfyUI API compatibility layer
"""

import json
import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from collections import deque

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comfyui_adapter import ComfyUIAdapter


class TestComfyUIAdapter(unittest.TestCase):
    """Test cases for ComfyUI adapter functionality"""

    def setUp(self):
        """Set up test fixtures"""
        # Mock SD server
        self.mock_sd_server = Mock()
        self.mock_sd_server.logger = Mock()
        self.mock_sd_server.model_ready = True
        self.mock_sd_server.device = "cuda"
        self.mock_sd_server.generate_image = Mock()

        # Create adapter
        self.adapter = ComfyUIAdapter(self.mock_sd_server)

    def test_submit_prompt_basic_workflow(self):
        """Test submitting a basic ComfyUI workflow"""
        # Basic ComfyUI workflow
        workflow = {
            "2": {
                "inputs": {"text": "a beautiful landscape", "clip": ["1", 1]},
                "class_type": "CLIPTextEncode"
            },
            "3": {
                "inputs": {"text": "blurry, low quality", "clip": ["1", 1]},
                "class_type": "CLIPTextEncode"
            },
            "4": {
                "inputs": {"width": 768, "height": 512, "batch_size": 1},
                "class_type": "EmptyLatentImage"
            },
            "5": {
                "inputs": {
                    "seed": 12345,
                    "steps": 25,
                    "cfg": 8.0,
                    "sampler_name": "dpmpp_2m",
                    "scheduler": "karras",
                    "model": ["1", 0],
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                    "latent_image": ["4", 0]
                },
                "class_type": "KSampler"
            }
        }

        request_data = {
            "prompt": workflow,
            "client_id": "test_client_123"
        }

        # Mock successful generation
        mock_result = {
            "images": ["/images/test_image.png"],
            "metadata": {"prompt": "a beautiful landscape"},
            "generation_time": 3.5
        }
        self.mock_sd_server.generate_image.return_value = mock_result

        # Submit prompt
        result = self.adapter.submit_prompt(request_data)

        # Verify response structure
        self.assertIn("prompt_id", result)
        self.assertIn("number", result)
        self.assertIn("node_errors", result)
        self.assertEqual(result["node_errors"], {})

        # Verify queue was populated
        self.assertEqual(len(self.adapter.queue), 1)

        # Verify parameters were parsed correctly
        queue_item = self.adapter.queue[0]
        params = queue_item["generation_params"]
        self.assertEqual(params["prompt"], "a beautiful landscape")
        self.assertEqual(params["negative_prompt"], "blurry, low quality")
        self.assertEqual(params["width"], 768)
        self.assertEqual(params["height"], 512)
        self.assertEqual(params["num_inference_steps"], 25)
        self.assertEqual(params["guidance_scale"], 8.0)
        self.assertEqual(params["seed"], 12345)
        self.assertEqual(params["sampler"], "DPM++ 2M")

    def test_parse_comfyui_workflow_empty(self):
        """Test parsing empty or minimal workflow"""
        empty_workflow = {}
        params = self.adapter._parse_comfyui_workflow(empty_workflow)

        # Should return defaults
        self.assertEqual(params["prompt"], "beautiful landscape")
        self.assertEqual(params["negative_prompt"], "blurry, low quality")
        self.assertEqual(params["width"], 512)
        self.assertEqual(params["height"], 512)
        self.assertEqual(params["num_inference_steps"], 20)
        self.assertEqual(params["guidance_scale"], 7.5)

    def test_convert_comfyui_samplers(self):
        """Test ComfyUI sampler name conversion"""
        sampler_tests = [
            ("euler", "Euler"),
            ("euler_ancestral", "Euler a"),
            ("dpmpp_2m", "DPM++ 2M"),
            ("dpmpp_2m_karras", "DPM++ 2M Karras"),
            ("dpmpp_sde", "DPM++ SDE"),
            ("dpmpp_sde_karras", "DPM++ SDE Karras"),
            ("ddim", "DDIM"),
            ("unknown_sampler", "DPMSolverMultistep")  # Default fallback
        ]

        for comfyui_name, expected_name in sampler_tests:
            result = self.adapter._convert_comfyui_sampler(comfyui_name)
            self.assertEqual(result, expected_name,
                             f"Failed to convert {comfyui_name} to {expected_name}, got {result}")

    def test_queue_management(self):
        """Test queue status and management"""
        # Initially empty
        status = self.adapter.get_queue_status()
        self.assertEqual(len(status["queue_running"]), 0)
        self.assertEqual(len(status["queue_pending"]), 0)

        # Add items to queue
        workflow = self.adapter.default_workflow
        for i in range(3):
            request_data = {
                "prompt": workflow,
                "client_id": f"client_{i}"
            }
            self.adapter.submit_prompt(request_data)

        # Check queue status
        status = self.adapter.get_queue_status()
        self.assertEqual(len(status["queue_pending"]), 3)

    def test_clear_queue(self):
        """Test clearing the queue"""
        # Add items to queue
        workflow = self.adapter.default_workflow
        for i in range(5):
            request_data = {
                "prompt": workflow,
                "client_id": f"client_{i}"
            }
            self.adapter.submit_prompt(request_data)

        # Clear queue
        result = self.adapter.clear_queue()

        # Verify queue is cleared
        self.assertEqual(len(self.adapter.queue), 0)
        self.assertEqual(len(result["deleted"]), 5)

    def test_execution_flow(self):
        """Test the execution flow from queue to history"""
        # Mock successful generation
        mock_result = {
            "images": ["/images/test_image.png"],
            "metadata": {
                "prompt": "test prompt",
                "width": 512,
                "height": 512,
                "num_inference_steps": 20
            },
            "generation_time": 3.5,
            "request_id": "abc123"
        }

        queue_item = {
            "prompt_id": "test_prompt_1",
            "client_id": "test_client",
            "workflow": {},
            "generation_params": {}
        }

        # Convert result - Fixed variable name from sd_result to mock_result
        comfyui_result = self.adapter._convert_result_to_comfyui_format(mock_result, queue_item)

        # Verify structure
        self.assertIn("status", comfyui_result)
        self.assertIn("outputs", comfyui_result)
        self.assertIn("meta", comfyui_result)

        # Verify status
        status = comfyui_result["status"]
        self.assertEqual(status["status_str"], "success")
        self.assertTrue(status["completed"])
        self.assertEqual(status["messages"], [])

        # Verify outputs
        outputs = comfyui_result["outputs"]
        self.assertIn("7", outputs)  # SaveImage node
        self.assertIn("images", outputs["7"])

        image_info = outputs["7"]["images"][0]
        self.assertEqual(image_info["filename"], "test_image.png")
        self.assertEqual(image_info["subfolder"], "")
        self.assertEqual(image_info["type"], "output")

        # Verify metadata
        meta = comfyui_result["meta"]
        self.assertEqual(meta["prompt_id"], "test_prompt_1")
        self.assertEqual(meta["workflow_execution_time"], 3.5)

    def test_complex_workflow_parsing(self):
        """Test parsing a complex ComfyUI workflow with multiple nodes"""
        complex_workflow = {
            "1": {
                "inputs": {"ckpt_name": "model.safetensors"},
                "class_type": "CheckpointLoaderSimple"
            },
            "2": {
                "inputs": {
                    "text": "masterpiece, highly detailed portrait of a wizard, magical atmosphere, fantasy art",
                    "clip": ["1", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "3": {
                "inputs": {
                    "text": "blurry, low quality, bad anatomy, worst quality, low res",
                    "clip": ["1", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "4": {
                "inputs": {"width": 1024, "height": 768, "batch_size": 2},
                "class_type": "EmptyLatentImage"
            },
            "5": {
                "inputs": {
                    "seed": 42,
                    "steps": 30,
                    "cfg": 7.5,
                    "sampler_name": "dpmpp_2m_karras",
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
                    "filename_prefix": "wizard_portrait",
                    "images": ["6", 0]
                },
                "class_type": "SaveImage"
            }
        }

        # Parse the workflow
        params = self.adapter._parse_comfyui_workflow(complex_workflow)

        # Verify complex parsing
        self.assertEqual(params["prompt"],
                         "masterpiece, highly detailed portrait of a wizard, magical atmosphere, fantasy art")
        self.assertEqual(params["negative_prompt"], "blurry, low quality, bad anatomy, worst quality, low res")
        self.assertEqual(params["width"], 1024)
        self.assertEqual(params["height"], 768)
        self.assertEqual(params["num_inference_steps"], 30)
        self.assertEqual(params["guidance_scale"], 7.5)
        self.assertEqual(params["seed"], 42)
        self.assertEqual(params["sampler"], "DPM++ 2M Karras")

    def test_workflow_with_custom_nodes(self):
        """Test handling workflows with unknown/custom node types"""
        workflow_with_custom = {
            "1": {
                "inputs": {"ckpt_name": "model.safetensors"},
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
                    "some_param": "value",
                    "input_node": ["2", 0]
                },
                "class_type": "CustomNodeType"  # Unknown node type
            },
            "4": {
                "inputs": {
                    "width": 512,
                    "height": 512,
                    "batch_size": 1
                },
                "class_type": "EmptyLatentImage"
            }
        }

        # Should not crash on unknown node types
        params = self.adapter._parse_comfyui_workflow(workflow_with_custom)

        # Should still extract known parameters
        self.assertEqual(params["prompt"], "beautiful landscape")
        self.assertEqual(params["width"], 512)
        self.assertEqual(params["height"], 512)

    def test_multiple_text_nodes(self):
        """Test handling multiple text input nodes"""
        workflow_multi_text = {
            "1": {
                "inputs": {"text": "first text input"},
                "class_type": "CLIPTextEncode"
            },
            "2": {
                "inputs": {"text": "much longer and more detailed text input with lots of description"},
                "class_type": "CLIPTextEncode"
            },
            "3": {
                "inputs": {"text": "negative prompt text"},
                "class_type": "CLIPTextEncode"
            }
        }

        params = self.adapter._parse_comfyui_workflow(workflow_multi_text)

        # Should pick the longest text as the main prompt
        self.assertEqual(params["prompt"], "much longer and more detailed text input with lots of description")

    def test_queue_execution_order(self):
        """Test that queue executes in FIFO order"""
        # Mock generation to track execution order
        executed_prompts = []

        def mock_generate(**kwargs):
            executed_prompts.append(kwargs.get("prompt", "unknown"))
            return {
                "images": ["/images/test.png"],
                "metadata": {"prompt": kwargs.get("prompt", "unknown")},
                "generation_time": 1.0
            }

        self.mock_sd_server.generate_image.side_effect = mock_generate

        # Submit multiple prompts
        prompts = ["first prompt", "second prompt", "third prompt"]
        prompt_ids = []

        for i, prompt_text in enumerate(prompts):
            workflow = {
                "2": {
                    "inputs": {"text": prompt_text, "clip": ["1", 1]},
                    "class_type": "CLIPTextEncode"
                }
            }
            request_data = {
                "prompt": workflow,
                "client_id": f"client_{i}"
            }
            result = self.adapter.submit_prompt(request_data)
            prompt_ids.append(result["prompt_id"])

        # Execute all queued items
        while self.adapter.queue or self.adapter.current_execution:
            if self.adapter.current_execution is None:
                self.adapter._execute_next_in_queue()
            else:
                # Simulate completion
                self.adapter.current_execution = None
                if self.adapter.queue:
                    self.adapter._execute_next_in_queue()

        # Verify execution order
        self.assertEqual(executed_prompts, prompts)

        # Verify all prompts are in history
        for prompt_id in prompt_ids:
            self.assertIn(prompt_id, self.adapter.history)

    def test_execution_success_workflow(self):
        """Test successful execution workflow from queue to history"""
        # Mock successful generation
        mock_result = {
            "images": ["/images/test_image.png"],
            "metadata": {
                "prompt": "test prompt",
                "width": 512,
                "height": 512,
                "num_inference_steps": 20
            },
            "generation_time": 2.1,
            "request_id": "test123"
        }
        self.mock_sd_server.generate_image.return_value = mock_result

        # Submit a prompt
        workflow = self.adapter.default_workflow
        request_data = {
            "prompt": workflow,
            "client_id": "test_client"
        }

        result = self.adapter.submit_prompt(request_data)
        prompt_id = result["prompt_id"]

        # Execute (this would normally happen automatically)
        self.adapter._execute_next_in_queue()

        # Verify execution completed
        self.assertIsNone(self.adapter.current_execution)
        self.assertEqual(len(self.adapter.queue), 0)

        # Verify history was updated
        self.assertIn(prompt_id, self.adapter.history)
        history_item = self.adapter.history[prompt_id]

        self.assertEqual(history_item["status"]["status_str"], "success")
        self.assertTrue(history_item["status"]["completed"])
        self.assertIn("outputs", history_item)

    def test_execution_error_handling(self):
        """Test handling of execution errors"""
        # Mock generation error
        self.mock_sd_server.generate_image.side_effect = RuntimeError("Model failed")

        # Submit a prompt
        workflow = self.adapter.default_workflow
        request_data = {
            "prompt": workflow,
            "client_id": "test_client"
        }

        result = self.adapter.submit_prompt(request_data)
        prompt_id = result["prompt_id"]

        # Execute (this would normally happen automatically)
        self.adapter._execute_next_in_queue()

        # Verify error was handled
        self.assertIsNone(self.adapter.current_execution)
        self.assertIn(prompt_id, self.adapter.history)

        history_item = self.adapter.history[prompt_id]
        self.assertEqual(history_item["status"]["status_str"], "error")
        self.assertTrue(history_item["status"]["completed"])
        self.assertIn("exception_message", str(history_item["status"]["messages"]))

    def test_get_history_specific_prompt(self):
        """Test getting history for a specific prompt ID"""
        # Add some history
        self.adapter.history["prompt_1"] = {"status": "completed"}
        self.adapter.history["prompt_2"] = {"status": "running"}

        # Get specific prompt history
        result = self.adapter.get_history("prompt_1")

        self.assertIn("prompt_1", result)
        self.assertNotIn("prompt_2", result)
        self.assertEqual(result["prompt_1"]["status"], "completed")

    def test_get_history_all(self):
        """Test getting all history"""
        # Add some history
        self.adapter.history["prompt_1"] = {"status": "completed"}
        self.adapter.history["prompt_2"] = {"status": "running"}

        # Get all history
        result = self.adapter.get_history()

        self.assertIn("prompt_1", result)
        self.assertIn("prompt_2", result)
        self.assertEqual(len(result), 2)

    def test_interrupt_execution(self):
        """Test interrupting current execution"""
        # Start an execution
        workflow = self.adapter.default_workflow
        request_data = {
            "prompt": workflow,
            "client_id": "test_client"
        }

        result = self.adapter.submit_prompt(request_data)

        # Simulate execution start
        self.adapter.current_execution = self.adapter.queue[0]
        self.adapter.current_execution["status"] = "running"

        # Interrupt
        interrupt_result = self.adapter.interrupt_execution()

        self.assertTrue(interrupt_result["interrupted"])
        self.assertIsNone(self.adapter.current_execution)

    def test_interrupt_no_execution(self):
        """Test interrupting when nothing is running"""
        result = self.adapter.interrupt_execution()
        self.assertFalse(result["interrupted"])

    def test_system_stats(self):
        """Test system statistics endpoint"""
        stats = self.adapter.get_system_stats()

        self.assertIn("system", stats)
        self.assertIn("devices", stats)
        self.assertIn("queue", stats)

        # Check device info
        device_info = stats["devices"][0]
        self.assertEqual(device_info["name"], "cuda")
        self.assertEqual(device_info["type"], "CUDA")

        # Check queue info
        queue_info = stats["queue"]
        self.assertEqual(queue_info["pending"], 0)
        self.assertEqual(queue_info["running"], 0)

    def test_embeddings_endpoint(self):
        """Test embeddings endpoint"""
        result = self.adapter.get_embeddings()

        self.assertIn("loaded", result)
        self.assertIn("available", result)
        self.assertEqual(result["loaded"], {})
        self.assertEqual(result["available"], {})

    def test_extensions_endpoint(self):
        """Test extensions endpoint"""
        result = self.adapter.get_extensions()

        self.assertIn("extensions", result)
        self.assertIn("disabled", result)
        self.assertEqual(result["extensions"], [])
        self.assertEqual(result["disabled"], [])

    def test_convert_result_to_comfyui_format(self):
        """Test converting generation results to ComfyUI format"""
        # Mock generation result
        sd_result = {
            "images": ["/images/20241227_120000_abc123_0.png"],
            "metadata": {
                "prompt": "test prompt",
                "width": 512,
                "height": 512,
                "num_inference_steps": 20,
                "guidance_scale": 7.5,
                "seed": 12345
            },
            "generation_time": 2.5,
            "request_id": "abc123"
        }

        queue_item = {
            "prompt_id": "test_prompt_1",
            "client_id": "test_client",
            "workflow": {},
            "generation_params": {}
        }

        # Convert result
        comfyui_result = self.adapter._convert_result_to_comfyui_format(sd_result, queue_item)

        # Verify structure
        self.assertIn("status", comfyui_result)
        self.assertIn("outputs", comfyui_result)
        self.assertIn("meta", comfyui_result)

        # Verify status
        status = comfyui_result["status"]
        self.assertEqual(status["status_str"], "success")
        self.assertTrue(status["completed"])
        self.assertEqual(status["messages"], [])

        # Verify outputs
        outputs = comfyui_result["outputs"]
        self.assertIn("7", outputs)  # SaveImage node
        self.assertIn("images", outputs["7"])

        image_info = outputs["7"]["images"][0]
        self.assertEqual(image_info["filename"], "20241227_120000_abc123_0.png")
        self.assertEqual(image_info["subfolder"], "")
        self.assertEqual(image_info["type"], "output")

        # Verify metadata
        meta = comfyui_result["meta"]
        self.assertEqual(meta["prompt_id"], "test_prompt_1")
        self.assertEqual(meta["workflow_execution_time"], 2.5)


class TestComfyUIHTTPIntegration(unittest.TestCase):
    """Integration tests for ComfyUI HTTP endpoints"""

    def setUp(self):
        """Set up integration test fixtures"""
        # Mock SD server with ComfyUI adapter
        self.mock_sd_server = Mock()
        self.mock_sd_server.logger = Mock()
        self.mock_sd_server.comfyui_adapter = ComfyUIAdapter(self.mock_sd_server)

        # Mock generation
        mock_result = {
            "images": ["/images/test_image.png"],
            "metadata": {"prompt": "test prompt"},
            "generation_time": 2.5
        }
        self.mock_sd_server.generate_image.return_value = mock_result

    def test_comfyui_prompt_endpoint_integration(self):
        """Test full integration of ComfyUI prompt endpoint"""
        # Create a realistic ComfyUI workflow
        workflow = {
            "1": {
                "inputs": {"ckpt_name": "model.safetensors"},
                "class_type": "CheckpointLoaderSimple"
            },
            "2": {
                "inputs": {
                    "text": "a majestic dragon flying over mountains",
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
                "inputs": {"width": 768, "height": 512, "batch_size": 1},
                "class_type": "EmptyLatentImage"
            },
            "5": {
                "inputs": {
                    "seed": 98765,
                    "steps": 25,
                    "cfg": 8.0,
                    "sampler_name": "euler_ancestral",
                    "scheduler": "normal",
                    "denoise": 1.0,
                    "model": ["1", 0],
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                    "latent_image": ["4", 0]
                },
                "class_type": "KSampler"
            }
        }

        request_data = {
            "prompt": workflow,
            "client_id": "integration_test_client"
        }

        # Submit through adapter
        result = self.mock_sd_server.comfyui_adapter.submit_prompt(request_data)

        # Verify submission
        self.assertIn("prompt_id", result)
        self.assertEqual(result["node_errors"], {})

        # Execute the queued item
        self.mock_sd_server.comfyui_adapter._execute_next_in_queue()

        # Verify generation was called with correct parameters
        self.mock_sd_server.generate_image.assert_called_once()
        args, kwargs = self.mock_sd_server.generate_image.call_args

        self.assertEqual(kwargs["prompt"], "a majestic dragon flying over mountains")
        self.assertEqual(kwargs["negative_prompt"], "blurry, low quality")
        self.assertEqual(kwargs["width"], 768)
        self.assertEqual(kwargs["height"], 512)
        self.assertEqual(kwargs["num_inference_steps"], 25)
        self.assertEqual(kwargs["guidance_scale"], 8.0)
        self.assertEqual(kwargs["seed"], 98765)
        self.assertEqual(kwargs["sampler"], "Euler a")

        # Verify result is in history
        prompt_id = result["prompt_id"]
        history = self.mock_sd_server.comfyui_adapter.get_history(prompt_id)

        self.assertIn(prompt_id, history)
        history_item = history[prompt_id]
        self.assertEqual(history_item["status"]["status_str"], "success")
        self.assertTrue(history_item["status"]["completed"])

    def test_queue_and_history_workflow(self):
        """Test complete queue and history workflow"""
        adapter = self.mock_sd_server.comfyui_adapter

        # Submit multiple requests
        workflows = []
        prompt_ids = []

        for i in range(3):
            workflow = {
                "2": {
                    "inputs": {"text": f"prompt {i}", "clip": ["1", 1]},
                    "class_type": "CLIPTextEncode"
                }
            }
            workflows.append(workflow)

            request_data = {
                "prompt": workflow,
                "client_id": f"client_{i}"
            }
            result = adapter.submit_prompt(request_data)
            prompt_ids.append(result["prompt_id"])

        # Check queue status
        queue_status = adapter.get_queue_status()
        self.assertEqual(len(queue_status["queue_pending"]), 3)
        self.assertEqual(len(queue_status["queue_running"]), 0)

        # Execute all items
        for _ in range(3):
            if adapter.queue:
                adapter._execute_next_in_queue()

        # Check final queue status
        final_queue_status = adapter.get_queue_status()
        self.assertEqual(len(final_queue_status["queue_pending"]), 0)
        self.assertEqual(len(final_queue_status["queue_running"]), 0)

        # Check history
        full_history = adapter.get_history()
        self.assertEqual(len(full_history), 3)

        for prompt_id in prompt_ids:
            self.assertIn(prompt_id, full_history)
            history_item = full_history[prompt_id]
            self.assertEqual(history_item["status"]["status_str"], "success")

    def test_error_handling_in_workflow(self):
        """Test error handling during workflow execution"""
        adapter = self.mock_sd_server.comfyui_adapter

        # Mock generation failure
        self.mock_sd_server.generate_image.side_effect = Exception("GPU out of memory")

        # Submit a workflow
        workflow = {
            "2": {
                "inputs": {"text": "test prompt", "clip": ["1", 1]},
                "class_type": "CLIPTextEncode"
            }
        }

        request_data = {
            "prompt": workflow,
            "client_id": "error_test_client"
        }

        result = adapter.submit_prompt(request_data)
        prompt_id = result["prompt_id"]

        # Execute the queued item (should handle error gracefully)
        adapter._execute_next_in_queue()

        # Verify error was handled
        self.assertIsNone(adapter.current_execution)
        self.assertIn(prompt_id, adapter.history)

        history_item = adapter.history[prompt_id]
        self.assertEqual(history_item["status"]["status_str"], "error")
        self.assertTrue(history_item["status"]["completed"])

    def test_adapter_system_endpoints(self):
        """Test ComfyUI adapter system endpoints"""
        adapter = self.mock_sd_server.comfyui_adapter

        # Test system stats
        stats = adapter.get_system_stats()
        self.assertIn("system", stats)
        self.assertIn("devices", stats)
        self.assertIn("queue", stats)

        # Test embeddings
        embeddings = adapter.get_embeddings()
        self.assertIn("loaded", embeddings)
        self.assertIn("available", embeddings)

        # Test extensions
        extensions = adapter.get_extensions()
        self.assertIn("extensions", extensions)
        self.assertIn("disabled", extensions)

        # Verify default empty responses
        self.assertEqual(embeddings["loaded"], {})
        self.assertEqual(embeddings["available"], {})
        self.assertEqual(extensions["extensions"], [])
        self.assertEqual(extensions["disabled"], [])


if __name__ == '__main__':
    unittest.main()