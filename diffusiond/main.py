#!/usr/bin/env python3
"""
Main entry point for the Stable Diffusion MCP Server
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path so modules can find each other
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

import argparse
from mcp_server import StableDiffusionServer


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(description="Stable Diffusion MCP Server")
    parser.add_argument("--model", default="SG161222/Realistic_Vision_V2.0",
                        help="HuggingFace model ID to use")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"],
                        help="Device to run inference on")
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
    parser.add_argument("--load-on-startup", action="store_true",
                        help="If set, load the model at startup and keep it loaded (default: lazy loading)")

    args = parser.parse_args()

    # Create and run the server
    server = StableDiffusionServer(
        model_id=args.model,
        device=args.device,
        output_dir=args.output_dir,
        models_dir=args.models_dir,
        precision=args.precision,
        attention_precision=args.attention_precision,
        http_host=args.http_host,
        http_port=args.http_port,
        load_on_startup=args.load_on_startup
    )

    server.run()


if __name__ == "__main__":
    main()