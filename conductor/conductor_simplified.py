#!/usr/bin/env python3
"""
Working Conductor LLM Server - Fixed Version
Save this as conductor/conductor.py and run it directly
"""

import asyncio
import logging
import sys
import argparse
import json
import time
import os
from pathlib import Path
from typing import Dict, Any, Optional, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class SimpleConfigParser:
    """Simple config parser for SETTINGS.md"""

    def __init__(self, config_path: str = "conductor/SETTINGS.md"):
        self.config_path = Path(config_path)

    def parse_settings(self) -> Dict[str, Dict[str, Any]]:
        """Parse SETTINGS.md file"""
        # Default configuration if file doesn't exist
        default_config = {
            'conversational_chat': {
                'category': 'conversational_chat',
                'model_name': 'Mistral 7B',
                'technical_model_name': 'mistralai/Mistral-7B-Instruct-v0.3',
                'vram_requirement': '~7GB',
                'stay_loaded': True,
                'precision': 'FP16'
            },
            'code_generation': {
                'category': 'code_generation',
                'model_name': 'Mistral 7B',
                'technical_model_name': 'mistralai/Mistral-7B-Instruct-v0.3',
                'vram_requirement': '~7GB',
                'stay_loaded': True,
                'precision': 'FP16'
            },
            'code_completion': {
                'category': 'code_completion',
                'model_name': 'CodeLlama 7B',
                'technical_model_name': 'codellama/CodeLlama-7b-hf',
                'vram_requirement': '~4GB',
                'stay_loaded': False,
                'precision': '4-bit'
            },
            'mathematical_reasoning': {
                'category': 'mathematical_reasoning',
                'model_name': 'Mistral 7B',
                'technical_model_name': 'mistralai/Mistral-7B-Instruct-v0.3',
                'vram_requirement': '~7GB',
                'stay_loaded': True,
                'precision': 'FP16'
            }
        }

        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}, using defaults")
            return default_config

        try:
            content = self.config_path.read_text(encoding='utf-8')
            parsed = self._parse_table(content)
            return parsed if parsed else default_config
        except Exception as e:
            logger.error(f"Error parsing config: {e}, using defaults")
            return default_config

    def _parse_table(self, content: str) -> Dict[str, Dict[str, Any]]:
        """Parse markdown table from content"""
        lines = content.split('\n')
        configs = {}

        in_table = False
        for line in lines:
            line = line.strip()
            if '| Category/Area |' in line:
                in_table = True
                continue
            elif line.startswith('|---'):
                continue
            elif in_table and line.startswith('|') and '**' in line:
                try:
                    parts = [p.strip() for p in line.split('|')[1:-1]]
                    if len(parts) >= 5:
                        category = parts[0].replace('**', '').strip()
                        model_name = parts[1].strip()
                        vram_req = parts[2].strip()
                        technical_name = parts[3].strip().replace('`', '')
                        stay_loaded = parts[4].strip().lower() == 'true'
                        precision = parts[5].strip() if len(parts) > 5 else 'FP16'

                        category_key = category.lower().replace(' ', '_').replace('/', '_')
                        configs[category_key] = {
                            'category': category_key,
                            'model_name': model_name,
                            'technical_model_name': technical_name,
                            'vram_requirement': vram_req,
                            'stay_loaded': stay_loaded,
                            'precision': precision
                        }
                except Exception as e:
                    logger.warning(f"Error parsing line: {line}, error: {e}")
                    continue

        return configs

    def get_conversational_persona(self) -> str:
        return "You are a helpful, accurate, and honest assistant."


class SimpleModelLoader:
    """Simple model loader"""

    def __init__(self):
        self.loaded_models: Dict[str, tuple] = {}
        self.device = None
        self._torch = None
        self._transformers = None

    def _init_torch(self):
        """Initialize torch"""
        if self._torch is None:
            try:
                import torch
                self._torch = torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"Using device: {self.device}")
            except ImportError:
                logger.error("PyTorch not available")
                raise
        return self._torch

    def _init_transformers(self):
        """Initialize transformers"""
        if self._transformers is None:
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM
                self._transformers = {'AutoTokenizer': AutoTokenizer, 'AutoModelForCausalLM': AutoModelForCausalLM}
            except ImportError:
                logger.error("Transformers not available")
                raise
        return self._transformers

    async def load_model(self, model_name: str, precision: str = "FP16") -> tuple:
        """Load model"""
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]

        logger.info(f"Loading {model_name} with {precision}")

        try:
            torch = self._init_torch()
            transformers = self._init_transformers()

            # Load tokenizer
            tokenizer = transformers['AutoTokenizer'].from_pretrained(
                model_name, trust_remote_code=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Model kwargs
            kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16,
                "device_map": "auto" if self.device == "cuda" else None,
                "low_cpu_mem_usage": True
            }

            # Add quantization for 4-bit
            if precision == "4-bit" and self.device == "cuda":
                try:
                    from transformers import BitsAndBytesConfig
                    kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                except ImportError:
                    logger.warning("BitsAndBytesConfig not available")

            # Load model
            model = transformers['AutoModelForCausalLM'].from_pretrained(model_name, **kwargs)

            self.loaded_models[model_name] = (model, tokenizer)
            logger.info(f"✓ Loaded {model_name}")
            return model, tokenizer

        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            return None, None

    def is_model_loaded(self, model_name: str) -> bool:
        return model_name in self.loaded_models

    def get_memory_status(self) -> Dict[str, Any]:
        if self._torch and self.device == "cuda":
            total = self._torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            used = self._torch.cuda.memory_allocated() / (1024 ** 3)
            return {
                'total_memory_gb': total,
                'used_memory_gb': used,
                'available_memory_gb': total - used,
                'device': self.device,
                'loaded_model_count': len(self.loaded_models)
            }
        return {
            'total_memory_gb': 16.0,
            'used_memory_gb': 8.0,
            'available_memory_gb': 8.0,
            'device': 'cpu',
            'loaded_model_count': len(self.loaded_models)
        }


class SimpleEngine:
    """Simple engine implementation"""

    def __init__(self, config: Dict[str, Any], model_loader: SimpleModelLoader, persona: str = ""):
        self.config = config
        self.category = config.get('category', 'unknown')
        self.technical_model_name = config.get('technical_model_name', '')
        self.model_loader = model_loader
        self.persona = persona
        self.generation_count = 0

    async def load_model(self) -> bool:
        model, tokenizer = await self.model_loader.load_model(
            self.technical_model_name,
            self.config.get('precision', 'FP16')
        )
        return model is not None and tokenizer is not None

    def is_loaded(self) -> bool:
        return self.model_loader.is_model_loaded(self.technical_model_name)

    @property
    def model(self):
        if self.technical_model_name in self.model_loader.loaded_models:
            return self.model_loader.loaded_models[self.technical_model_name][0]
        return None

    @property
    def tokenizer(self):
        if self.technical_model_name in self.model_loader.loaded_models:
            return self.model_loader.loaded_models[self.technical_model_name][1]
        return None

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate response"""
        if not self.model or not self.tokenizer:
            return f"Model not loaded for {self.category}"

        try:
            # Simple prompt preparation
            if self.persona and len(self.persona) < 100:
                full_prompt = f"{self.persona}\n\nUser: {prompt}\nAssistant:"
            else:
                full_prompt = f"User: {prompt}\nAssistant:"

            # Tokenize
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )

            # Move to device
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate
            max_tokens = min(kwargs.get('max_tokens', 100), 150)
            torch = self.model_loader._init_torch()

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs.get('attention_mask'),
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )

            # Decode
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            self.generation_count += 1
            return response.strip()

        except Exception as e:
            logger.error(f"Generation error in {self.category}: {e}")
            return f"Error: {str(e)}"


class SimpleClassifier:
    """Simple rule-based classifier"""

    def __init__(self, categories: List[str]):
        self.categories = set(categories)

    def classify(self, prompt: str) -> str:
        """Classify prompt using simple rules"""
        prompt_lower = prompt.lower()

        # Code keywords
        if any(word in prompt_lower for word in ['code', 'function', 'program', 'script', 'python', 'javascript']):
            if 'complete' in prompt_lower or '...' in prompt or 'def ' in prompt:
                return 'code_completion' if 'code_completion' in self.categories else 'code_generation'
            return 'code_generation' if 'code_generation' in self.categories else 'conversational_chat'

        # Math keywords
        if any(word in prompt_lower for word in ['math', 'calculate', 'solve', '+', '-', '*', '/', 'equation']):
            return 'mathematical_reasoning' if 'mathematical_reasoning' in self.categories else 'conversational_chat'

        # Translation keywords
        if any(word in prompt_lower for word in ['translate', 'spanish', 'french', 'german', 'chinese']):
            return 'translation' if 'translation' in self.categories else 'conversational_chat'

        # Creative keywords
        if any(word in prompt_lower for word in ['story', 'poem', 'creative', 'fiction', 'character']):
            return 'creative_writing' if 'creative_writing' in self.categories else 'conversational_chat'

        # Default
        return 'conversational_chat'


class SimpleConductor:
    """Simple conductor implementation"""

    def __init__(self, config_path: str = "conductor/SETTINGS.md"):
        self.config_parser = SimpleConfigParser(config_path)
        self.model_loader = SimpleModelLoader()
        self.engines: Dict[str, SimpleEngine] = {}
        self.engine_configs: Dict[str, Dict[str, Any]] = {}
        self.classifier = None

    async def initialize(self) -> bool:
        """Initialize conductor"""
        try:
            logger.info("=== Initializing Simple Conductor ===")

            # Parse config
            self.engine_configs = self.config_parser.parse_settings()
            if not self.engine_configs:
                logger.error("No engine configs found")
                return False

            # Initialize classifier
            self.classifier = SimpleClassifier(list(self.engine_configs.keys()))

            # Load stay-loaded models
            await self._load_models()

            logger.info("=== Simple Conductor Ready ===")
            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False

    async def _load_models(self):
        """Load models that should stay loaded"""
        stay_loaded = [
            (cat, config) for cat, config in self.engine_configs.items()
            if config.get('stay_loaded', False)
        ]

        # Group by model to avoid duplicates
        models_to_load = {}
        for category, config in stay_loaded:
            model_name = config['technical_model_name']
            if model_name not in models_to_load:
                models_to_load[model_name] = []
            models_to_load[model_name].append((category, config))

        # Load each model once
        for model_name, categories in models_to_load.items():
            logger.info(f"Loading {model_name}")
            success = False

            for category, config in categories:
                engine = SimpleEngine(config, self.model_loader,
                                      self.config_parser.get_conversational_persona())
                if await engine.load_model():
                    self.engines[category] = engine
                    success = True

            if success:
                logger.info(f"✓ {model_name} loaded for {len(categories)} engines")
            else:
                logger.error(f"✗ Failed to load {model_name}")

    async def generate(self, prompt: str, category: Optional[str] = None, **kwargs) -> tuple[str, str]:
        """Generate response"""
        if not category:
            category = self.classifier.classify(prompt)

        engine = self.engines.get(category)
        if not engine:
            engine = self.engines.get('conversational_chat')
            if not engine:
                return category, "No engines available"

        response = await engine.generate(prompt, **kwargs)
        return category, response

    def get_status(self) -> Dict[str, Any]:
        """Get status"""
        return {
            'engines_loaded': len(self.engines),
            'categories_configured': len(self.engine_configs),
            'memory_status': self.model_loader.get_memory_status()
        }


async def cli_interface(conductor: SimpleConductor):
    """CLI interface"""
    print("\n=== Simple Conductor CLI ===")
    print("Commands: /status, /engines, /memory, /test, /quit")
    print("Or type any prompt to generate a response.\n")

    while True:
        try:
            user_input = input("> ").strip()

            if user_input.lower() in ['/quit', '/exit']:
                break
            elif user_input == '/status':
                status = conductor.get_status()
                print(json.dumps(status, indent=2))
            elif user_input == '/engines':
                print(f"Loaded engines: {list(conductor.engines.keys())}")
            elif user_input == '/memory':
                memory = conductor.model_loader.get_memory_status()
                print(f"Memory: {memory['used_memory_gb']:.1f}GB / {memory['total_memory_gb']:.1f}GB")
                print(f"Device: {memory['device']}")
            elif user_input == '/test':
                start = time.time()
                category, response = await conductor.generate("Hello!")
                end = time.time()
                print(f"Test ({end - start:.2f}s): {response}")
            elif user_input:
                start = time.time()
                category, response = await conductor.generate(user_input)
                end = time.time()
                print(f"\n[{category}] ({end - start:.2f}s): {response}\n")

        except (KeyboardInterrupt, EOFError):
            break
        except Exception as e:
            print(f"Error: {e}")

    print("Goodbye!")


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Simple Conductor")
    parser.add_argument("--config", default="conductor/SETTINGS.md")
    args = parser.parse_args()

    conductor = SimpleConductor(args.config)

    if await conductor.initialize():
        await cli_interface(conductor)
    else:
        print("Failed to initialize")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())