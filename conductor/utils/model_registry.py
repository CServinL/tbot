import time
import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from threading import Lock
from enum import Enum

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model status enumeration."""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    UNLOADING = "unloading"
    ERROR = "error"


@dataclass
class ModelInfo:
    """Information about a registered model."""
    model_name: str
    technical_name: str
    categories: List[str]
    precision: str
    stay_loaded: bool
    vram_requirement: str
    status: ModelStatus
    load_time: Optional[float] = None
    last_used: Optional[float] = None
    memory_usage_gb: float = 0.0
    generation_count: int = 0
    error_message: Optional[str] = None
    engine_instance: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding engine instance."""
        data = asdict(self)
        data['status'] = self.status.value
        data.pop('engine_instance', None)  # Don't serialize engine instance
        return data


class ModelRegistry:
    """Registry for tracking model instances and their states."""

    def __init__(self):
        self.models: Dict[str, ModelInfo] = {}
        self.category_to_model: Dict[str, str] = {}
        self.model_lock = Lock()
        self.access_history: List[Dict[str, Any]] = []
        self.max_history_size = 1000

    def register_model(self,
                       model_name: str,
                       technical_name: str,
                       categories: List[str],
                       precision: str,
                       stay_loaded: bool,
                       vram_requirement: str) -> bool:
        """Register a model in the registry.

        Args:
            model_name: Display name of the model
            technical_name: HuggingFace model identifier
            categories: List of categories this model serves
            precision: Model precision (FP16, FP32, 4-bit)
            stay_loaded: Whether model should stay in memory
            vram_requirement: Memory requirement string

        Returns:
            bool: True if registered successfully
        """
        with self.model_lock:
            try:
                model_info = ModelInfo(
                    model_name=model_name,
                    technical_name=technical_name,
                    categories=categories,
                    precision=precision,
                    stay_loaded=stay_loaded,
                    vram_requirement=vram_requirement,
                    status=ModelStatus.UNLOADED
                )

                self.models[technical_name] = model_info

                # Map categories to models
                for category in categories:
                    self.category_to_model[category] = technical_name

                logger.info(f"Registered model {technical_name} for categories: {categories}")
                return True

            except Exception as e:
                logger.error(f"Failed to register model {technical_name}: {e}")
                return False

    def get_model_for_category(self, category: str) -> Optional[str]:
        """Get technical model name for a category.

        Args:
            category: Category name

        Returns:
            Optional[str]: Technical model name or None if not found
        """
        return self.category_to_model.get(category)

    def get_model_info(self, technical_name: str) -> Optional[ModelInfo]:
        """Get model information.

        Args:
            technical_name: Technical model name

        Returns:
            Optional[ModelInfo]: Model info or None if not found
        """
        return self.models.get(technical_name)

    def update_model_status(self, technical_name: str, status: ModelStatus, error_message: Optional[str] = None):
        """Update model status.

        Args:
            technical_name: Technical model name
            status: New status
            error_message: Error message if status is ERROR
        """
        with self.model_lock:
            if technical_name in self.models:
                self.models[technical_name].status = status
                if error_message:
                    self.models[technical_name].error_message = error_message

                # Record status change
                self._record_access(technical_name, 'status_change', {'status': status.value})

    def set_model_loaded(self,
                         technical_name: str,
                         engine_instance: Any,
                         memory_usage_gb: float = 0.0):
        """Mark model as loaded and set engine instance.

        Args:
            technical_name: Technical model name
            engine_instance: Engine instance
            memory_usage_gb: Actual memory usage
        """
        with self.model_lock:
            if technical_name in self.models:
                model_info = self.models[technical_name]
                model_info.status = ModelStatus.LOADED
                model_info.engine_instance = engine_instance
                model_info.load_time = time.time()
                model_info.memory_usage_gb = memory_usage_gb
                model_info.error_message = None

                self._record_access(technical_name, 'loaded', {'memory_gb': memory_usage_gb})
                logger.info(f"Model {technical_name} marked as loaded ({memory_usage_gb:.1f}GB)")

    def set_model_unloaded(self, technical_name: str):
        """Mark model as unloaded.

        Args:
            technical_name: Technical model name
        """
        with self.model_lock:
            if technical_name in self.models:
                model_info = self.models[technical_name]
                model_info.status = ModelStatus.UNLOADED
                model_info.engine_instance = None
                model_info.load_time = None
                model_info.memory_usage_gb = 0.0

                self._record_access(technical_name, 'unloaded')
                logger.info(f"Model {technical_name} marked as unloaded")

    def record_generation(self, technical_name: str):
        """Record a generation event for a model.

        Args:
            technical_name: Technical model name
        """
        with self.model_lock:
            if technical_name in self.models:
                model_info = self.models[technical_name]
                model_info.generation_count += 1
                model_info.last_used = time.time()

                self._record_access(technical_name, 'generation')

    def get_loaded_models(self) -> List[ModelInfo]:
        """Get list of currently loaded models.

        Returns:
            List[ModelInfo]: List of loaded model info
        """
        return [
            model_info for model_info in self.models.values()
            if model_info.status == ModelStatus.LOADED
        ]

    def get_stay_loaded_models(self) -> List[str]:
        """Get list of models that should stay loaded.

        Returns:
            List[str]: List of technical model names
        """
        return [
            technical_name for technical_name, model_info in self.models.items()
            if model_info.stay_loaded
        ]

    def get_on_demand_models(self) -> List[str]:
        """Get list of on-demand models.

        Returns:
            List[str]: List of technical model names
        """
        return [
            technical_name for technical_name, model_info in self.models.items()
            if not model_info.stay_loaded
        ]

    def get_model_engine(self, technical_name: str) -> Optional[Any]:
        """Get engine instance for a model.

        Args:
            technical_name: Technical model name

        Returns:
            Optional[Any]: Engine instance or None
        """
        model_info = self.models.get(technical_name)
        return model_info.engine_instance if model_info else None

    def get_categories_for_model(self, technical_name: str) -> List[str]:
        """Get categories served by a model.

        Args:
            technical_name: Technical model name

        Returns:
            List[str]: List of categories
        """
        model_info = self.models.get(technical_name)
        return model_info.categories if model_info else []

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get total memory usage statistics.

        Returns:
            Dict containing memory usage info
        """
        loaded_models = self.get_loaded_models()
        total_memory = sum(model.memory_usage_gb for model in loaded_models)

        return {
            'total_memory_gb': total_memory,
            'loaded_model_count': len(loaded_models),
            'models': [
                {
                    'technical_name': model.technical_name,
                    'memory_gb': model.memory_usage_gb,
                    'precision': model.precision
                }
                for model in loaded_models
            ]
        }

    def get_activity_stats(self) -> Dict[str, Any]:
        """Get model activity statistics.

        Returns:
            Dict containing activity stats
        """
        total_generations = sum(model.generation_count for model in self.models.values())

        # Most used models
        most_used = sorted(
            self.models.values(),
            key=lambda m: m.generation_count,
            reverse=True
        )[:5]

        # Recently used models
        recently_used = [
            model for model in self.models.values()
            if model.last_used and (time.time() - model.last_used) < 3600  # Last hour
        ]

        return {
            'total_generations': total_generations,
            'active_models': len([m for m in self.models.values() if m.generation_count > 0]),
            'most_used': [
                {
                    'technical_name': model.technical_name,
                    'generation_count': model.generation_count,
                    'categories': model.categories
                }
                for model in most_used
            ],
            'recently_used_count': len(recently_used)
        }

    def get_all_models(self) -> Dict[str, Dict[str, Any]]:
        """Get all registered models and their info.

        Returns:
            Dict mapping technical names to model info dicts
        """
        return {
            technical_name: model_info.to_dict()
            for technical_name, model_info in self.models.items()
        }

    def find_models_by_status(self, status: ModelStatus) -> List[str]:
        """Find models by status.

        Args:
            status: Model status to search for

        Returns:
            List[str]: List of technical model names
        """
        return [
            technical_name for technical_name, model_info in self.models.items()
            if model_info.status == status
        ]

    def get_least_used_loaded_model(self) -> Optional[str]:
        """Get the least used loaded model (for cleanup).

        Returns:
            Optional[str]: Technical model name or None
        """
        loaded_models = [
            model for model in self.models.values()
            if model.status == ModelStatus.LOADED and not model.stay_loaded
        ]

        if not loaded_models:
            return None

        # Sort by last used time (oldest first), then by generation count
        least_used = min(
            loaded_models,
            key=lambda m: (m.last_used or 0, m.generation_count)
        )

        return least_used.technical_name

    def _record_access(self, technical_name: str, action: str, metadata: Optional[Dict] = None):
        """Record model access for analytics.

        Args:
            technical_name: Technical model name
            action: Action performed
            metadata: Additional metadata
        """
        access_record = {
            'timestamp': time.time(),
            'model': technical_name,
            'action': action,
            'metadata': metadata or {}
        }

        self.access_history.append(access_record)

        # Trim history if too large
        if len(self.access_history) > self.max_history_size:
            self.access_history = self.access_history[-self.max_history_size:]

    def get_registry_stats(self) -> Dict[str, Any]:
        """Get overall registry statistics.

        Returns:
            Dict containing registry stats
        """
        statuses = {}
        for status in ModelStatus:
            statuses[status.value] = len(self.find_models_by_status(status))

        return {
            'total_registered_models': len(self.models),
            'total_categories': len(self.category_to_model),
            'status_breakdown': statuses,
            'memory_usage': self.get_memory_usage(),
            'activity_stats': self.get_activity_stats(),
            'access_history_size': len(self.access_history)
        }

    def clear_registry(self):
        """Clear the entire registry (use with caution)."""
        with self.model_lock:
            self.models.clear()
            self.category_to_model.clear()
            self.access_history.clear()
            logger.warning("Model registry cleared")

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on registry.

        Returns:
            Dict containing health status
        """
        loaded_count = len(self.find_models_by_status(ModelStatus.LOADED))
        error_count = len(self.find_models_by_status(ModelStatus.ERROR))

        health_status = "healthy"
        if error_count > 0:
            health_status = "degraded"
        if loaded_count == 0:
            health_status = "no_models_loaded"

        return {
            'status': health_status,
            'loaded_models': loaded_count,
            'error_models': error_count,
            'total_memory_gb': self.get_memory_usage()['total_memory_gb'],
            'registry_size': len(self.models)
        }