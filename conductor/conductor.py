#!/usr/bin/env python3
"""
Conductor LLM Server - Main Entry Point

A modular LLM server system that routes different types of tasks to specialized models
and engines based on configuration defined in SETTINGS.md.
"""

import logging
import asyncio
import sys
import argparse
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

# Import the new classificator
from conductor.classificator import PromptClassificator

# Import dependencies loader with fallback
try:
    from dependencies_loader import DependenciesLoader  # type: ignore[import-not-found]
except ImportError:
    try:
        from conductor.dependencies_loader import DependenciesLoader
    except ImportError:
        class DependenciesLoaderFallback:
            def __init__(self) -> None: 
                pass

            async def ensure_dependencies(self, model_name: str, precision: str = "FP16") -> bool: 
                return True

            def check_system_requirements(self) -> Dict[str, Any]: 
                return {'platform': 'unknown', 'memory_gb': 16.0}

            def get_missing_packages(self, packages: List[str]) -> List[str]: 
                return []

            def get_installation_instructions(self, packages: List[str]) -> Dict[str, str]: 
                return {}

            def validate_model_requirements(self, model_name: str, precision: str) -> Dict[str, Any]:
                return {'can_load': True, 'warnings': [], 'recommendations': []}

            def get_dependency_report(self) -> Dict[str, Any]: 
                return {'system': {}, 'packages': {}, 'recommendations': []}

        DependenciesLoader = DependenciesLoaderFallback

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

from conductor.utils.config_parser import ConfigParser
from conductor.model_loader import ModelLoader

def _default_dict() -> Dict[str, Any]:
    return {}

def _default_list() -> List[str]:
    return []

@dataclass
class ModelInfo:
    technical_name: str
    max_context_window: int = 2048
    max_new_tokens: int = 1024
    description: str = ""
    special_flags: Dict[str, Any] = field(default_factory=_default_dict)
    stop_patterns: List[str] = field(default_factory=_default_list)

    # Add more fields as needed (e.g., quantization, architecture, etc.)

from conductor.engines.base_engine import BaseEngine
from conductor.mcp_server import MCPServer

# Specialized Engine Classes
from conductor.engines import (
    ConversationalChatEngine,
    CodeCompletionEngine,
    CodeGenerationEngine,
    MathematicalReasoningEngine,
    TranslationEngine,
    GeneralReasoningEngine,
    CreativeWritingEngine,
    InstructionFollowingEngine,
    SummarizationEngine,
    QuestionAnsweringEngine,
    ScientificResearchEngine,
    LegalAnalysisEngine,
    CodeReviewEngine,
    LongContextEngine,
    ImageGenerationEngine,
)

# Import full engine implementations from their respective modules
class Conductor:
    """Main conductor with MCP support and efficient model sharing"""

    def __init__(self, config_path: str = "conductor/SETTINGS.md", models_dir: str = "./models"):
        self.config_parser = ConfigParser(config_path)
        self.model_loader = ModelLoader(models_dir)
        # Ensure model_info_dir is set and absolute
        if not hasattr(self.model_loader, "model_info_dir") or not getattr(self.model_loader, "model_info_dir", None):
            setattr(self.model_loader, "model_info_dir", str(Path(__file__).parent / "model_info"))
        else:
            current_dir = getattr(self.model_loader, "model_info_dir")
            setattr(self.model_loader, "model_info_dir", str(Path(current_dir).resolve()))
        # Patch: set the expected naming convention for ModelInfo files
        setattr(self.model_loader, "model_info_naming", "double_underscore_py")
        self.dependencies_loader = DependenciesLoader()
        self.engines: Dict[str, BaseEngine] = {}
        self.engine_configs: Dict[str, Dict[str, Any]] = {}
        self.persona = ""
        self.mcp_server: Optional[MCPServer] = None
        
        # Initialize the prompt classificator
        self.classificator: Optional[PromptClassificator] = None
        
        # Dynamic model switching state
        self.default_model_name: Optional[str] = None
        self.current_active_model: Optional[str] = None
        self.last_specialized_model: Optional[str] = None

        # Engine class mapping
        self.engine_classes: Dict[str, type] = {
            'conversational_chat': ConversationalChatEngine,
            'code_completion': CodeCompletionEngine,
            'code_generation': CodeGenerationEngine,
            'mathematical_reasoning': MathematicalReasoningEngine,
            'translation': TranslationEngine,
            'general_reasoning': GeneralReasoningEngine,
            'creative_writing': CreativeWritingEngine,
            'instruction_following': InstructionFollowingEngine,
            'summarization': SummarizationEngine,
            'question_answering': QuestionAnsweringEngine,
            'scientific_research': ScientificResearchEngine,
            'legal_analysis': LegalAnalysisEngine,
            'code_review': CodeReviewEngine,
            'long_context': LongContextEngine,
            'image_generation': ImageGenerationEngine,
        }
        # Use diffusiond_url from settings if present, else env, else default
        self.diffusiond_url = (
            self.config_parser.get_diffusiond_url()
            or os.environ.get("DIFFUSIOND_URL")
            or "http://127.0.0.1:8000"
        )

    async def initialize(self, skip_model_loading: bool = False) -> bool:
        """Initialize the conductor"""
        try:
            logger.info("=== Initializing Conductor with MCP ===")

            # Check system requirements
            system_info = self.dependencies_loader.check_system_requirements()
            logger.info("=== System Requirements ===")
            logger.info(f"Platform: {system_info['platform']}")
            logger.info(f"Python: {system_info['python_version']}")
            logger.info(f"Memory: {system_info['memory_gb']:.1f} GB")
            logger.info(f"CUDA available: {system_info['cuda_available']}")
            if system_info['cuda_available']:
                logger.info(f"GPU count: {system_info['gpu_count']}")

            # Parse configuration
            self.engine_configs = self.config_parser.parse_settings()
            if not self.engine_configs:
                logger.error("Failed to parse configuration")
                return False

            self.persona = self.config_parser.get_conversational_persona()
            logger.info(f"Loaded {len(self.engine_configs)} engine configurations")
            
            # Initialize the prompt classificator with available engine categories
            self.classificator = PromptClassificator(list(self.engine_configs.keys()))
            logger.info("✓ Prompt classificator initialized")

            # Initialize MCP server
            self.mcp_server = MCPServer()  # type: ignore[no-untyped-call]
            logger.info("✓ MCP server initialized")

            # Debug: print model_info search path and technical names
            logger.info(f"Model info search path: {getattr(self.model_loader, 'model_info_dir', 'unknown')}")
            logger.info(f"Configured technical_model_names: {[config['technical_model_name'] for config in self.engine_configs.values()]}")

            for category, config in self.engine_configs.items():
                technical_name = config['technical_model_name']
                # Exception: skip ModelInfo registration for Image Generation engine
                if category == "image_generation" or technical_name == "diffusiond":
                    logger.info(f"Skipping ModelInfo registration for image generation engine: {category}")
                    continue
                model_info = self.model_loader.get_model_info(technical_name)
                if model_info is not None:
                    self.model_loader.register_model_info(model_info)
                else:
                    logger.warning(
                        f"No ModelInfo found for {technical_name}, checked in: {getattr(self.model_loader, 'model_info_dir', 'unknown')} "
                        f"(expected: {getattr(self.model_loader, 'model_info_dir', 'unknown')}/{technical_name.replace('/', '--')}.py)"
                    )

            # Only load default model (general_reasoning) on startup
            if not skip_model_loading:
                await self._ensure_dependencies()
                await self._load_default_model()

            logger.info("=== Conductor with MCP initialized ===")
            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False

    async def _ensure_dependencies(self):
        """Ensure dependencies for general_reasoning only (image_generation is external)"""
        logger.info("Checking dependencies for general_reasoning model...")
        config = self.engine_configs.get("general_reasoning")
        if not config:
            logger.error("No general_reasoning engine configured.")
            return
        technical_name = config['technical_model_name']
        precision = config.get('precision', 'FP16')
        deps_ok = await self.dependencies_loader.ensure_dependencies(technical_name, precision)
        if not deps_ok:
            logger.warning(f"Some dependencies missing for {technical_name}")
            missing_packages = self.dependencies_loader.get_missing_packages(['torch', 'transformers'])
            if 'bitsandbytes' not in missing_packages and precision in ['4-bit', '8-bit']:
                missing_packages.append('bitsandbytes')
            if missing_packages:
                instructions = self.dependencies_loader.get_installation_instructions(missing_packages)
                logger.info("Install missing packages:")
                for _pkg, instruction in instructions.items():
                    logger.info(f"  {instruction}")
        validation = self.dependencies_loader.validate_model_requirements(technical_name, precision)
        if not validation['can_load']:
            logger.error(f"Cannot load {technical_name}: insufficient resources")
            for warning in validation['warnings']:
                logger.warning(f"  {warning}")
            for rec in validation['recommendations']:
                logger.info(f"  Recommendation: {rec}")
        else:
            if validation['warnings']:
                for warning in validation['warnings']:
                    logger.warning(f"  {warning}")

    async def _load_default_model(self):
        """Load only the default model (general_reasoning) at startup"""
        # Determine the default model (general_reasoning)
        config = self.engine_configs.get("general_reasoning")
        if not config:
            logger.error("No general_reasoning engine configured.")
            return
            
        technical_name = config['technical_model_name']
        self.default_model_name = technical_name
        self.current_active_model = technical_name
        
        logger.info(f"Loading default model: {technical_name}")
        model, tokenizer = await self.model_loader.load_model(
            technical_name,
            config.get('precision', 'FP16')
        )
        if model and tokenizer:
            await self._create_engine_for_category("general_reasoning", config)
            logger.info("✓ Default model loaded and general_reasoning engine created")
        else:
            logger.error(f"✗ Failed to load default model: {technical_name}")

        # Create image_generation engine (no model loading required)
        img_config = self.engine_configs.get("image_generation")
        if img_config:
            await self._create_engine_for_category("image_generation", img_config)
            logger.info("✓ image_generation engine created")

    async def _switch_to_specialized_model(self, category: str) -> bool:
        """
        Switch to a specialized model for the given category.
        
        Args:
            category: The engine category requiring a specialized model
            
        Returns:
            bool: True if model switch was successful
        """
        config = self.engine_configs.get(category)
        if not config:
            logger.error(f"No configuration found for category: {category}")
            return False
            
        technical_name = config['technical_model_name']
        
        # Skip if already using the correct model
        if self.current_active_model == technical_name:
            logger.debug(f"Already using correct model for {category}: {technical_name}")
            return True
            
        # Skip model switching for image generation
        if category == "image_generation":
            logger.debug("Skipping model switch for image_generation")
            return True
            
        logger.info(f"Switching to specialized model for {category}: {technical_name}")
        
        # Unload current model if it's not the default and different from target
        if (self.current_active_model and 
            self.current_active_model != self.default_model_name and 
            self.current_active_model != technical_name):
            self.model_loader.unload_model(self.current_active_model)
            
        # Load the specialized model if not already loaded
        if not self.model_loader.is_model_loaded(technical_name):
            model, tokenizer = await self.model_loader.load_model(
                technical_name,
                config.get('precision', 'FP16')
            )
            if not (model and tokenizer):
                logger.error(f"Failed to load specialized model: {technical_name}")
                return False
                
        self.last_specialized_model = self.current_active_model
        self.current_active_model = technical_name
        logger.info(f"✓ Successfully switched to specialized model: {technical_name}")
        return True

    async def _switch_back_to_default_model(self) -> bool:
        """
        Switch back to the default model after processing with a specialized model.
        
        Returns:
            bool: True if switch was successful
        """
        if not self.default_model_name:
            logger.warning("No default model defined, cannot switch back")
            return False
            
        # Skip if already using default model
        if self.current_active_model == self.default_model_name:
            logger.debug("Already using default model")
            return True
            
        logger.info(f"Switching back to default model: {self.default_model_name}")
        
        # Unload current specialized model if it's not the default
        if (self.current_active_model and 
            self.current_active_model != self.default_model_name):
            self.model_loader.unload_model(self.current_active_model)
            
        # Ensure default model is loaded
        if not self.model_loader.is_model_loaded(self.default_model_name):
            config = self.engine_configs.get("general_reasoning")
            if config:
                model, tokenizer = await self.model_loader.load_model(
                    self.default_model_name,
                    config.get('precision', 'FP16')
                )
                if not (model and tokenizer):
                    logger.error(f"Failed to reload default model: {self.default_model_name}")
                    return False
                    
        self.current_active_model = self.default_model_name
        logger.info(f"✓ Successfully switched back to default model: {self.default_model_name}")
        return True

    async def return_to_normal_models(self, keep_models: List[str] = []) -> None:
        """
        Legacy method - use cleanup_specialized_models instead.
        Unload all models except the default model.
        """
        logger.info("Using legacy return_to_normal_models, consider using cleanup_specialized_models")
        await self.cleanup_specialized_models()
        await self._switch_back_to_default_model()

    async def generate(self, prompt: str, category: str = "", **kwargs: Any) -> Any:
        """
        Generate a response for the given prompt with dynamic model switching.
        If category is not specified, classify the prompt and route to the appropriate engine.
        Returns (category_used, response).
        """
        if not category:
            category = await self.classify_prompt(prompt)
            
        # Track if we need to switch back to default model after generation
        need_switch_back = (category != "general_reasoning" and 
                           category != "image_generation" and
                           self.current_active_model != self.default_model_name)
        
        try:
            engine = await self.get_engine(category)
            if engine is None:
                logger.error(f"No engine loaded for category '{category}'.")
                return category, f"Error: Unable to load engine for category '{category}'. (Insufficient memory or missing model)"
                
            response = await engine.generate(prompt, **kwargs)
            
            # Switch back to default model if we used a specialized model
            if need_switch_back:
                await self._switch_back_to_default_model()
                
            return category, response
            
        except Exception as e:
            logger.error(f"Error during generation for category {category}: {e}")
            # Ensure we switch back to default model even if generation failed
            if need_switch_back:
                await self._switch_back_to_default_model()
            return category, f"Error during generation: {str(e)}"

    async def classify_prompt(self, prompt: str) -> str:
        """Classify a user prompt using the dedicated classificator module."""
        if not self.classificator:
            # Fallback if classificator is not initialized
            logger.warning("Classificator not initialized, falling back to general_reasoning")
            return "general_reasoning" if "general_reasoning" in self.engine_configs else next(iter(self.engine_configs.keys()))
        
        # Get reasoning engine for AI fallback classification
        reasoning_engine = await self.get_engine('general_reasoning') if 'general_reasoning' in self.engines else None
        
        # Use the classificator to classify the prompt
        return await self.classificator.classify_prompt(prompt, reasoning_engine)

    async def get_engine(self, category: str) -> Optional["BaseEngine"]:
        """
        Get an engine for the specified category, loading specialized models on-demand.
        
        Args:
            category: The engine category
            
        Returns:
            BaseEngine instance or None if failed
        """
        # Return existing engine if available
        if category in self.engines:
            # For specialized engines, ensure the correct model is loaded
            if category != "general_reasoning" and category != "image_generation":
                config = self.engine_configs.get(category)
                if config:
                    technical_name = config['technical_model_name']
                    if self.current_active_model != technical_name:
                        success = await self._switch_to_specialized_model(category)
                        if not success:
                            logger.error(f"Failed to switch to specialized model for {category}")
                            return None
            return self.engines[category]
        
        # Create engine on-demand if configuration exists
        if category in self.engine_configs:
            config = self.engine_configs[category]
            
            # For specialized engines, switch to the appropriate model first
            if category != "general_reasoning" and category != "image_generation":
                success = await self._switch_to_specialized_model(category)
                if not success:
                    logger.error(f"Failed to switch to specialized model for {category}")
                    return None
                    
            await self._create_engine_for_category(category, config)
            return self.engines.get(category)
        
        return None

    async def _create_engine_for_category(self, category: str, config: Dict[str, Any]):
        """
        Create and register an engine instance for the given category and config.
        """
        if category not in self.engine_classes:
            raise RuntimeError(f"No engine class registered for category '{category}'. Please ensure a full engine implementation is available.")
        engine_class = self.engine_classes[category]

        # Always use the correct technical_model_name for model sharing
        technical_name = config.get('technical_model_name')
        if technical_name is None:
            raise RuntimeError(f"No technical_model_name found in config for category '{category}'")

        # Only use -- for ModelInfo file lookup, not for model loading or engine config
        # Do NOT rewrite technical_model_name in config for engine/model loading
        # Just ensure the model is loaded, then create the engine

        # Try to load the model if not loaded (except for image_generation)
        if category != "image_generation" and not self.model_loader.is_model_loaded(technical_name):
            model, tokenizer = await self.model_loader.load_model(
                technical_name,
                config.get('precision', 'FP16')
            )
            if not (model and tokenizer):
                logger.error(f"✗ Cannot create engine {category}: shared model not available")
                return

        # Special handling for image_generation engine
        if category == "image_generation":
            config = dict(config)
            config["diffusiond_url"] = self.diffusiond_url
            engine = engine_class(config, self.model_loader, self.persona, diffusiond_url=self.diffusiond_url)
            self.engines[category] = engine
            logger.info(f"✓ Created engine: {category} (image_generation, no model preloading)")
            return
        elif 'conversational' in category or 'general_reasoning' in category:
            engine = engine_class(config, self.model_loader, self.persona)
        else:
            engine = engine_class(config, self.model_loader)

        self.engines[category] = engine
        logger.info(f"✓ Created engine: {category}")

    def get_model_status(self) -> Dict[str, Any]:
        """
        Get current model switching status and loaded models information.
        
        Returns:
            Dict containing model status information
        """
        return {
            'default_model': self.default_model_name,
            'current_active_model': self.current_active_model,
            'last_specialized_model': self.last_specialized_model,
            'loaded_models': list(self.model_loader.loaded_models.keys()),
            'memory_status': self.model_loader.get_memory_status(),
            'loaded_engines': list(self.engines.keys())
        }

    async def force_switch_to_default(self) -> bool:
        """
        Force switch back to default model, useful for manual control.
        
        Returns:
            bool: True if successful
        """
        return await self._switch_back_to_default_model()

    async def cleanup_specialized_models(self) -> int:
        """
        Cleanup all specialized models except the default model.
        
        Returns:
            int: Number of models cleaned up
        """
        if not self.default_model_name:
            logger.warning("No default model defined")
            return 0
            
        keep_models = [self.default_model_name]
        # Also keep image generation external service
        return await self.model_loader.cleanup_unused_models(keep_models=keep_models)


def main_sync() -> None:
    import asyncio
    asyncio.run(main())


async def main() -> None:
    # ...existing code for main() that sets up the CLI and calls cli_interface...
    parser = argparse.ArgumentParser(description="Conductor LLM Server CLI")
    parser.add_argument("--config", default="conductor/SETTINGS.md", help="Config file path")
    parser.add_argument("--skip-model-loading", action="store_true", help="Skip loading models")
    parser.add_argument("--log-level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO',
                        help="Logging level")
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    conductor = Conductor(args.config)
    success = await conductor.initialize(skip_model_loading=args.skip_model_loading)
    if not success:
        logger.error("Failed to initialize conductor")
        sys.exit(1)

    # Import cli_interface here to avoid NameError
    from conductor.conductor_cli import cli_interface
    await cli_interface(conductor)


def print_object_members(obj: Any) -> None:
    for attr in dir(obj):
        try:
            value = getattr(obj, attr)
        except Exception:
            value = "<error>"
        print(f"{attr}: {value}")


if __name__ == "__main__":
    asyncio.run(main())
