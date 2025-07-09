#!/usr/bin/env python3
"""
Dependency management and imports for Stable Diffusion MCP Server
"""

import sys
import warnings

# Suppress specific warnings that don't affect functionality
warnings.filterwarnings("ignore", message=".*CLIPFeatureExtractor.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*feature_extraction_clip.*", category=FutureWarning)


def check_dependencies():
    """Check and import all required dependencies"""
    try:
        import torch
        from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, DPMSolverMultistepScheduler
        from PIL import Image

        # Check accelerate availability
        try:
            import accelerate
            accelerate_available = True
            print(f"✅ Accelerate version: {accelerate.__version__}")
        except ImportError:
            accelerate_available = False
            print("⚠️  Accelerate not available - using basic loading")

        return {
            'torch': torch,
            'StableDiffusionPipeline': StableDiffusionPipeline,
            'StableDiffusionXLPipeline': StableDiffusionXLPipeline,
            'DPMSolverMultistepScheduler': DPMSolverMultistepScheduler,
            'Image': Image,
            'accelerate_available': accelerate_available
        }

    except ImportError as e:
        missing_dep = str(e).split("'")[1] if "'" in str(e) else str(e)
        print(f"Error: Missing required dependency: {missing_dep}")
        print("Please install: pip install torch diffusers pillow accelerate")
        sys.exit(1)


def get_diffusers_schedulers():
    """Import and return all available diffusers schedulers"""
    try:
        from diffusers import (
            DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler,
            EulerDiscreteScheduler, EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler,
            KDPM2DiscreteScheduler, KDPM2AncestralDiscreteScheduler,
            HeunDiscreteScheduler, DDPMScheduler
        )

        return {
            "DPMSolverMultistep": DPMSolverMultistepScheduler,
            "DPM++ SDE Karras": DPMSolverMultistepScheduler,
            "DPM++ 2M Karras": DPMSolverMultistepScheduler,
            "DPM++ 2M": DPMSolverMultistepScheduler,
            "DPM++ SDE": DPMSolverSinglestepScheduler,
            "DDIM": DDIMScheduler,
            "PNDM": PNDMScheduler,
            "LMS": LMSDiscreteScheduler,
            "Euler": EulerDiscreteScheduler,
            "Euler a": EulerAncestralDiscreteScheduler,
            "Heun": HeunDiscreteScheduler,
            "DPM2": KDPM2DiscreteScheduler,
            "DPM2 a": KDPM2AncestralDiscreteScheduler,
            "DDPM": DDPMScheduler,
        }
    except ImportError as e:
        print(f"Error importing schedulers: {e}")
        return {}


# Make dependencies available as module-level variables
deps = check_dependencies()
torch = deps['torch']
StableDiffusionPipeline = deps['StableDiffusionPipeline']
StableDiffusionXLPipeline = deps['StableDiffusionXLPipeline']
DPMSolverMultistepScheduler = deps['DPMSolverMultistepScheduler']
Image = deps['Image']
ACCELERATE_AVAILABLE = deps['accelerate_available']

# Scheduler mapping
SCHEDULER_MAP = get_diffusers_schedulers()