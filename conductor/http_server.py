import asyncio
import logging
import json
import time
import uuid
from typing import Dict, Any, List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from conductor.model_loader import ModelLoader
from conductor.utils.config_parser import ConfigParser
from conductor.utils.model_registry import ModelRegistry, ModelStatus
from conductor.utils.persona_loader import PersonaLoader
from conductor.dependencies_loader import DependenciesLoader

logger = logging.getLogger(__name__)


# Pydantic models for API requests/responses
class GenerationRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt for generation")
    category: Optional[str] = Field(None, description="Task category (auto-detected if not provided)")
    session_id: Optional[str] = Field(None, description="Session ID for conversation history")
    stream: bool = Field(False, description="Whether to stream the response")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(None, description="Generation temperature")
    top_p: Optional[float] = Field(None, description="Top-p sampling parameter")
    language: Optional[str] = Field(None, description="Programming/natural language hint")
    extra_params: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters")


class GenerationResponse(BaseModel):
    response: str = Field(..., description="Generated response")
    category: str = Field(..., description="Task category used")
    model: str = Field(..., description="Model used for generation")
    session_id: Optional[str] = Field(None, description="Session ID if applicable")
    generation_time: float = Field(..., description="Generation time in seconds")
    tokens_generated: Optional[int] = Field(None, description="Number of tokens generated")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Overall health status")
    loaded_models: int = Field(..., description="Number of loaded models")
    total_memory_gb: float = Field(..., description="Total memory usage in GB")
    uptime_seconds: float = Field(..., description="Server uptime in seconds")
    last_updated: float = Field(..., description="Last update timestamp")


class ModelInfo(BaseModel):
    technical_name: str
    category: str
    status: str
    memory_gb: float
    generation_count: int
    stay_loaded: bool


class ConductorHTTPServer:
    """HTTP API server for the Conductor LLM system."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8000):
        self.host = host
        self.port = port
        self.app = FastAPI(
            title="Conductor LLM Server",
            description="Multi-model LLM server with specialized engines",
            version="1.0.0"
        )

        # Core components
        self.config_parser = ConfigParser()
        self.model_registry = ModelRegistry()
        self.persona_loader = PersonaLoader()
        self.model_loader = ModelLoader()
        self.dependencies_loader = DependenciesLoader()

        # Engine instances
        self.engines: Dict[str, Any] = {}
        self.engine_configs: Dict[str, Dict[str, Any]] = {}

        # Server state
        self.start_time = time.time()
        self.request_count = 0
        self.active_sessions: Dict[str, Dict[str, Any]] = {}

        # Setup FastAPI
        self._setup_middleware()
        self._setup_routes()

    def _setup_middleware(self):
        """Setup FastAPI middleware."""
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Request logging middleware
        @self.app.middleware("http")
        async def log_requests(request: Request, call_next):
            start_time = time.time()
            self.request_count += 1

            response = await call_next(request)

            process_time = time.time() - start_time
            logger.info(
                f"{request.method} {request.url.path} - "
                f"Status: {response.status_code} - "
                f"Time: {process_time:.3f}s"
            )

            return response

    def _setup_routes(self):
        """Setup FastAPI routes."""

        @self.app.get("/", response_model=Dict[str, str])
        async def root():
            """Root endpoint with basic server info."""
            return {
                "service": "Conductor LLM Server",
                "version": "1.0.0",
                "status": "running",
                "docs": "/docs"
            }

        @self.app.get("/health", response_model=HealthResponse)
        async def health():
            """Health check endpoint."""
            memory_status = self.model_registry.get_memory_usage()
            uptime = time.time() - self.start_time

            return HealthResponse(
                status="healthy",
                loaded_models=memory_status['loaded_model_count'],
                total_memory_gb=memory_status['total_memory_gb'],
                uptime_seconds=uptime,
                last_updated=time.time()
            )

        @self.app.post("/generate", response_model=GenerationResponse)
        async def generate(request: GenerationRequest):
            """Generate response using intelligent routing."""
            try:
                start_time = time.time()

                # If category is specified, use it directly
                if hasattr(request, 'category') and request.category:
                    category = request.category
                else:
                    # Use general reasoning to classify the prompt
                    category = await self._classify_prompt(request.prompt)

                # Get engine for determined category
                engine = await self._get_engine_for_category(category)
                if not engine:
                    raise HTTPException(
                        status_code=400,
                        detail=f"No engine available for category: {category}"
                    )

                # Prepare generation parameters
                gen_params = {
                    'session_id': request.session_id,
                    **request.extra_params
                }

                if request.max_tokens:
                    gen_params['max_tokens'] = request.max_tokens
                if request.temperature:
                    gen_params['temperature'] = request.temperature
                if request.top_p:
                    gen_params['top_p'] = request.top_p
                if request.language:
                    gen_params['language'] = request.language

                # Generate response
                response_text = await engine.generate(request.prompt, **gen_params)

                generation_time = time.time() - start_time

                return GenerationResponse(
                    response=response_text,
                    category=category,
                    model=engine.technical_model_name,
                    session_id=request.session_id,
                    generation_time=generation_time,
                    tokens_generated=len(response_text.split())  # Rough estimate
                )

            except Exception as e:
                logger.error(f"Error in generation: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # Specialized endpoints (direct routing, no classification)

        @self.app.post("/chat/completions")
        async def chat_completions(request: GenerationRequest):
            """Direct chat completions (OpenAI-compatible)."""
            request.category = "conversational_chat"
            return await generate(request)

        @self.app.post("/code/completion")
        async def code_completion(request: GenerationRequest):
            """Direct code completion endpoint."""
            try:
                start_time = time.time()

                engine = await self._get_engine_for_category("code_completion")
                if not engine:
                    raise HTTPException(status_code=400, detail="Code completion engine not available")

                gen_params = {**request.extra_params}
                if request.max_tokens:
                    gen_params['max_tokens'] = request.max_tokens
                if request.language:
                    gen_params['language'] = request.language

                response_text = await engine.generate(request.prompt, **gen_params)
                generation_time = time.time() - start_time

                return GenerationResponse(
                    response=response_text,
                    category="code_completion",
                    model=engine.technical_model_name,
                    session_id=request.session_id,
                    generation_time=generation_time,
                    tokens_generated=len(response_text.split())
                )

            except Exception as e:
                logger.error(f"Error in code completion: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/code/generation")
        async def code_generation(request: GenerationRequest):
            """Direct code generation endpoint."""
            try:
                start_time = time.time()

                engine = await self._get_engine_for_category("code_generation")
                if not engine:
                    raise HTTPException(status_code=400, detail="Code generation engine not available")

                gen_params = {**request.extra_params}
                if request.max_tokens:
                    gen_params['max_tokens'] = request.max_tokens
                if request.temperature:
                    gen_params['temperature'] = request.temperature
                if request.language:
                    gen_params['language'] = request.language

                response_text = await engine.generate(request.prompt, **gen_params)
                generation_time = time.time() - start_time

                return GenerationResponse(
                    response=response_text,
                    category="code_generation",
                    model=engine.technical_model_name,
                    session_id=request.session_id,
                    generation_time=generation_time,
                    tokens_generated=len(response_text.split())
                )

            except Exception as e:
                logger.error(f"Error in code generation: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/translate")
        async def translate(request: GenerationRequest):
            """Direct translation endpoint."""
            try:
                start_time = time.time()

                engine = await self._get_engine_for_category("translation")
                if not engine:
                    raise HTTPException(status_code=400, detail="Translation engine not available")

                gen_params = {**request.extra_params}
                if request.max_tokens:
                    gen_params['max_tokens'] = request.max_tokens

                response_text = await engine.generate(request.prompt, **gen_params)
                generation_time = time.time() - start_time

                return GenerationResponse(
                    response=response_text,
                    category="translation",
                    model=engine.technical_model_name,
                    session_id=request.session_id,
                    generation_time=generation_time,
                    tokens_generated=len(response_text.split())
                )

            except Exception as e:
                logger.error(f"Error in translation: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/math/solve")
        async def math_solve(request: GenerationRequest):
            """Direct mathematical reasoning endpoint."""
            try:
                start_time = time.time()

                engine = await self._get_engine_for_category("mathematical_reasoning")
                if not engine:
                    raise HTTPException(status_code=400, detail="Mathematical reasoning engine not available")

                gen_params = {**request.extra_params}
                if request.max_tokens:
                    gen_params['max_tokens'] = request.max_tokens

                response_text = await engine.generate(request.prompt, **gen_params)
                generation_time = time.time() - start_time

                return GenerationResponse(
                    response=response_text,
                    category="mathematical_reasoning",
                    model=engine.technical_model_name,
                    session_id=request.session_id,
                    generation_time=generation_time,
                    tokens_generated=len(response_text.split())
                )

            except Exception as e:
                logger.error(f"Error in math solving: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/research/analyze")
        async def research_analyze(request: GenerationRequest):
            """Direct scientific research endpoint."""
            try:
                start_time = time.time()

                engine = await self._get_engine_for_category("scientific_research")
                if not engine:
                    raise HTTPException(status_code=400, detail="Scientific research engine not available")

                gen_params = {**request.extra_params}
                if request.max_tokens:
                    gen_params['max_tokens'] = request.max_tokens

                response_text = await engine.generate(request.prompt, **gen_params)
                generation_time = time.time() - start_time

                return GenerationResponse(
                    response=response_text,
                    category="scientific_research",
                    model=engine.technical_model_name,
                    session_id=request.session_id,
                    generation_time=generation_time,
                    tokens_generated=len(response_text.split())
                )

            except Exception as e:
                logger.error(f"Error in research analysis: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/legal/analyze")
        async def legal_analyze(request: GenerationRequest):
            """Direct legal analysis endpoint."""
            try:
                start_time = time.time()

                engine = await self._get_engine_for_category("legal_analysis")
                if not engine:
                    raise HTTPException(status_code=400, detail="Legal analysis engine not available")

                gen_params = {**request.extra_params}
                if request.max_tokens:
                    gen_params['max_tokens'] = request.max_tokens

                response_text = await engine.generate(request.prompt, **gen_params)
                generation_time = time.time() - start_time

                return GenerationResponse(
                    response=response_text,
                    category="legal_analysis",
                    model=engine.technical_model_name,
                    session_id=request.session_id,
                    generation_time=generation_time,
                    tokens_generated=len(response_text.split())
                )

            except Exception as e:
                logger.error(f"Error in legal analysis: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/summarize")
        async def summarize(request: GenerationRequest):
            """Direct summarization endpoint."""
            try:
                start_time = time.time()

                engine = await self._get_engine_for_category("summarization")
                if not engine:
                    raise HTTPException(status_code=400, detail="Summarization engine not available")

                gen_params = {**request.extra_params}
                if request.max_tokens:
                    gen_params['max_tokens'] = request.max_tokens

                response_text = await engine.generate(request.prompt, **gen_params)
                generation_time = time.time() - start_time

                return GenerationResponse(
                    response=response_text,
                    category="summarization",
                    model=engine.technical_model_name,
                    session_id=request.session_id,
                    generation_time=generation_time,
                    tokens_generated=len(response_text.split())
                )

            except Exception as e:
                logger.error(f"Error in summarization: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/generate/stream")
        async def generate_stream(request: GenerationRequest):
            """Generate streaming response."""
            try:
                if not request.stream:
                    raise HTTPException(
                        status_code=400,
                        detail="Stream parameter must be True for streaming endpoint"
                    )

                engine = await self._get_engine_for_category(request.category)
                if not engine:
                    raise HTTPException(
                        status_code=400,
                        detail=f"No engine available for category: {request.category}"
                    )

                # Prepare generation parameters
                gen_params = {
                    'session_id': request.session_id,
                    **request.extra_params
                }

                if request.max_tokens:
                    gen_params['max_tokens'] = request.max_tokens
                if request.temperature:
                    gen_params['temperature'] = request.temperature
                if request.top_p:
                    gen_params['top_p'] = request.top_p
                if request.language:
                    gen_params['language'] = request.language

                async def stream_generator():
                    try:
                        async for chunk in engine.generate_stream(request.prompt, **gen_params):
                            # Format as Server-Sent Events
                            data = {
                                "chunk": chunk,
                                "category": request.category,
                                "model": engine.technical_model_name,
                                "session_id": request.session_id
                            }
                            yield f"data: {json.dumps(data)}\n\n"

                        # Send end marker
                        yield f"data: {json.dumps({'done': True})}\n\n"

                    except Exception as e:
                        error_data = {"error": str(e)}
                        yield f"data: {json.dumps(error_data)}\n\n"

                return StreamingResponse(
                    stream_generator(),
                    media_type="text/plain",
                    headers={"Cache-Control": "no-cache"}
                )

            except Exception as e:
                logger.error(f"Error in streaming generation: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/models", response_model=List[ModelInfo])
        async def list_models():
            """List all registered models and their status."""
            models_info = []

            for technical_name, model_info in self.model_registry.get_all_models().items():
                models_info.append(ModelInfo(
                    technical_name=technical_name,
                    category=', '.join(model_info['categories']),
                    status=model_info['status'],
                    memory_gb=model_info['memory_usage_gb'],
                    generation_count=model_info['generation_count'],
                    stay_loaded=model_info['stay_loaded']
                ))

            return models_info

        @self.app.post("/models/{technical_name}/load")
        async def load_model(technical_name: str, background_tasks: BackgroundTasks):
            """Load a specific model."""
            try:
                model_info = self.model_registry.get_model_info(technical_name)
                if not model_info:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Model not found: {technical_name}"
                    )

                if model_info.status == ModelStatus.LOADED:
                    return {"message": f"Model {technical_name} already loaded"}

                # Load in background
                background_tasks.add_task(self._load_model_background, technical_name)

                return {"message": f"Loading model {technical_name} in background"}

            except Exception as e:
                logger.error(f"Error loading model {technical_name}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/models/{technical_name}/unload")
        async def unload_model(technical_name: str):
            """Unload a specific model."""
            try:
                model_info = self.model_registry.get_model_info(technical_name)
                if not model_info:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Model not found: {technical_name}"
                    )

                if model_info.status != ModelStatus.LOADED:
                    return {"message": f"Model {technical_name} not loaded"}

                engine = self.model_registry.get_model_engine(technical_name)
                if engine:
                    success = await engine.unload_model()
                    if success:
                        self.model_registry.set_model_unloaded(technical_name)
                        return {"message": f"Model {technical_name} unloaded successfully"}
                    else:
                        raise HTTPException(
                            status_code=500,
                            detail=f"Failed to unload model {technical_name}"
                        )
                else:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Engine not found for model {technical_name}"
                    )

            except Exception as e:
                logger.error(f"Error unloading model {technical_name}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/categories")
        async def list_categories():
            """List available task categories."""
            return {
                "categories": list(self.engine_configs.keys()),
                "category_info": {
                    category: {
                        "model": config['technical_model_name'],
                        "precision": config['precision'],
                        "stay_loaded": config['stay_loaded']
                    }
                    for category, config in self.engine_configs.items()
                }
            }

        @self.app.get("/sessions")
        async def list_sessions():
            """List active conversation sessions."""
            return {
                "active_sessions": len(self.active_sessions),
                "sessions": list(self.active_sessions.keys())
            }

        @self.app.delete("/sessions/{session_id}")
        async def clear_session(session_id: str):
            """Clear a specific conversation session."""
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]

                # Clear history in conversational engines
                for engine in self.engines.values():
                    if hasattr(engine, 'clear_conversation_history'):
                        engine.clear_conversation_history(session_id)

                return {"message": f"Session {session_id} cleared"}
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"Session not found: {session_id}"
                )

        @self.app.get("/stats")
        async def get_stats():
            """Get server statistics."""
            uptime = time.time() - self.start_time
            memory_stats = self.model_registry.get_memory_usage()
            activity_stats = self.model_registry.get_activity_stats()

            return {
                "uptime_seconds": uptime,
                "request_count": self.request_count,
                "active_sessions": len(self.active_sessions),
                "memory_usage": memory_stats,
                "activity": activity_stats,
                "engine_count": len(self.engines)
            }

        @self.app.post("/reload-config")
        async def reload_config():
            """Reload configuration from SETTINGS.md."""
            try:
                success = self.config_parser.reload_settings()
                if success:
                    # Re-register models with new config
                    await self._register_models()
                    return {"message": "Configuration reloaded successfully"}
                else:
                    raise HTTPException(
                        status_code=500,
                        detail="Failed to reload configuration"
                    )
            except Exception as e:
                logger.error(f"Error reloading config: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    async def initialize(self):
        """Initialize the server and load models."""
        try:
            logger.info("Initializing Conductor HTTP Server...")

            # Parse configuration
            self.engine_configs = self.config_parser.parse_settings()
            if not self.engine_configs:
                raise RuntimeError("Failed to parse engine configurations")

            # Set up persona loader
            conversational_persona = self.config_parser.get_conversational_persona()
            self.persona_loader.set_conversational_persona(conversational_persona)

            # Register models
            await self._register_models()

            # Load stay-loaded models
            await self._load_persistent_models()

            logger.info("Conductor HTTP Server initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize server: {e}")
            raise

    async def _register_models(self):
        """Register all models from configuration."""
        for category, config in self.engine_configs.items():
            success = self.model_registry.register_model(
                model_name=config['model_name'],
                technical_name=config['technical_model_name'],
                categories=[category],
                precision=config['precision'],
                stay_loaded=config['stay_loaded'],
                vram_requirement=config['vram_requirement']
            )

            if success:
                logger.info(f"Registered model for {category}: {config['technical_model_name']}")
            else:
                logger.error(f"Failed to register model for {category}")

    async def _load_persistent_models(self):
        """Load models that should stay loaded."""
        stay_loaded_models = self.model_registry.get_stay_loaded_models()

        for technical_name in stay_loaded_models:
            await self._load_model_background(technical_name)

    async def _classify_prompt(self, prompt: str) -> str:
        """Use general reasoning engine to classify the prompt and determine appropriate category.

        Args:
            prompt: User's input prompt

        Returns:
            str: Determined category for routing
        """
        try:
            # Get the general reasoning engine for classification
            reasoning_engine = await self._get_engine_for_category("general_reasoning")

            if not reasoning_engine:
                logger.warning("General reasoning engine not available, falling back to conversational_chat")
                return "conversational_chat"

            # Build classification prompt
            classification_prompt = f"""Analyze this user prompt and determine which specialized AI engine would be most appropriate to handle it.

Available engine categories:
- conversational_chat: General conversation, greetings, casual questions
- code_completion: Code completion, filling in partial code snippets
- code_generation: Creating new code, algorithms, full programs
- mathematical_reasoning: Math problems, equations, calculations, proofs
- translation: Translating between languages
- creative_writing: Stories, poems, creative content, fiction
- instruction_following: Step-by-step tasks, following complex instructions
- summarization: Summarizing documents, articles, long texts
- question_answering: Factual questions, explanations, informational queries
- scientific_research: Scientific analysis, research papers, academic content
- legal_analysis: Legal documents, contracts, legal advice
- code_review: Reviewing code quality, security, best practices
- long_context: Processing very long documents, complex analysis

User prompt: "{prompt}"

Respond with ONLY the category name (e.g., "code_generation" or "mathematical_reasoning"). Choose the most specific and appropriate category."""

            # Generate classification
            classification_result = await reasoning_engine.generate(
                classification_prompt,
                max_tokens=20,
                temperature=0.3,  # Low temperature for consistent classification
                reasoning_type='analytical'
            )

            # Extract category from response
            predicted_category = classification_result.strip().lower()

            # Validate against available categories
            available_categories = set(self.engine_configs.keys())

            if predicted_category in available_categories:
                logger.info(f"Classified prompt as: {predicted_category}")
                return predicted_category

            # Fallback: try to find partial matches
            for category in available_categories:
                if category in predicted_category or predicted_category in category:
                    logger.info(f"Classified prompt as: {category} (partial match)")
                    return category

            # Final fallback
            logger.warning(f"Could not classify prompt '{prompt[:50]}...', defaulting to conversational_chat")
            return "conversational_chat"

        except Exception as e:
            logger.error(f"Error in prompt classification: {e}")
            # Fallback to conversational chat if classification fails
            return "conversational_chat"
        """Load a model in the background."""
        try:
            self.model_registry.update_model_status(technical_name, ModelStatus.LOADING)

            # Create engine for the model
            engine = await self._create_engine_for_model(technical_name)
            if not engine:
                self.model_registry.update_model_status(
                    technical_name, ModelStatus.ERROR, "Failed to create engine"
                )
                return

            # Load the model
            success = await engine.load_model()
            if success:
                # Get memory usage
                memory_usage = engine.get_memory_usage()['memory_gb']

                # Register successful load
                self.model_registry.set_model_loaded(technical_name, engine, memory_usage)

                # Store engine reference
                categories = self.model_registry.get_categories_for_model(technical_name)
                for category in categories:
                    self.engines[category] = engine

                logger.info(f"Successfully loaded model {technical_name}")
            else:
                self.model_registry.update_model_status(
                    technical_name, ModelStatus.ERROR, "Model loading failed"
                )
                logger.error(f"Failed to load model {technical_name}")

        except Exception as e:
            self.model_registry.update_model_status(
                technical_name, ModelStatus.ERROR, str(e)
            )
            logger.error(f"Error loading model {technical_name}: {e}")

    async def _create_engine_for_model(self, technical_name: str):
        """Create appropriate engine for a model."""
        categories = self.model_registry.get_categories_for_model(technical_name)
        if not categories:
            return None

        # Use first category to determine engine type
        category = categories[0]
        config = self.engine_configs.get(category)
        if not config:
            return None

        # Import and create appropriate engine
        engine_class = self._get_engine_class(category)
        if not engine_class:
            return None

        return engine_class(config)

    def _get_engine_class(self, category: str):
        """Get engine class for a category."""
        engine_mapping = {
            'conversational_chat': 'ConversationalChatEngine',
            'code_completion': 'CodeCompletionEngine',
            'code_generation': 'CodeGenerationEngine',
            'mathematical_reasoning': 'MathematicalReasoningEngine',
            'translation': 'TranslationEngine',
            # Add more mappings as needed
        }

        engine_class_name = engine_mapping.get(category)
        if not engine_class_name:
            # Default to conversational for unknown categories
            engine_class_name = 'ConversationalChatEngine'

        try:
            # Dynamic import based on category
            if category == 'conversational_chat':
                from engines.conversational_chat import ConversationalChatEngine
                return ConversationalChatEngine
            elif category == 'code_completion':
                from engines.code_completion import CodeCompletionEngine
                return CodeCompletionEngine
            elif category == 'code_generation':
                from engines.code_generation import CodeGenerationEngine
                return CodeGenerationEngine
            elif category == 'mathematical_reasoning':
                from engines.mathematical_reasoning import MathematicalReasoningEngine
                return MathematicalReasoningEngine
            elif category == 'translation':
                from engines.translation import TranslationEngine
                return TranslationEngine
            else:
                # Default fallback
                from engines.conversational_chat import ConversationalChatEngine
                return ConversationalChatEngine

        except ImportError as e:
            logger.error(f"Failed to import engine for {category}: {e}")
            return None

    async def _get_engine_for_category(self, category: str):
        """Get engine for a specific category."""
        if category in self.engines:
            return self.engines[category]

        # Try to load engine on demand
        technical_name = self.model_registry.get_model_for_category(category)
        if technical_name:
            model_info = self.model_registry.get_model_info(technical_name)
            if model_info and model_info.status != ModelStatus.LOADED:
                await self._load_model_background(technical_name)
                return self.engines.get(category)

        return None

    async def run(self):
        """Run the HTTP server."""
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info",
            access_log=True
        )

        server = uvicorn.Server(config)
        await server.serve()


# Standalone function to create and run server
async def create_and_run_server(host: str = "0.0.0.0", port: int = 8000):
    """Create and run the HTTP server."""
    server = ConductorHTTPServer(host, port)
    await server.initialize()
    await server.run()


if __name__ == "__main__":
    asyncio.run(create_and_run_server())

# Compare this file's HTTP server implementation to the one in conductor.py:
# - conductor.py defines `async def start_http_server(conductor: Conductor, host: str = "localhost", port: int = 8000):`
#   which sets up a FastAPI app, defines endpoints for /generate, /mcp, /status, /health, /engines, /dependencies,
#   and starts the server with uvicorn.
# - conductor/http_server.py may contain a similar FastAPI or other HTTP server implementation, or it may be a stub.

# Key points to compare:
# - Endpoints: conductor.py's version has endpoints for /generate, /mcp, /status, /health, /engines, /dependencies.
# - Startup: conductor.py's version uses FastAPI and uvicorn, and logs startup info.
# - conductor/http_server.py may be less complete, missing endpoints, or not used at all.

# Recommendation:
# If the HTTP server logic in conductor.py is more complete, move it to conductor/http_server.py,
# and import and use it in conductor.py. Remove any duplicate or outdated code.

# Example (if you want to use the more complete version everywhere):
# 1. Move the full start_http_server function from conductor.py to conductor/http_server.py.
# 2. In conductor.py, replace the function with:
#    from conductor.http_server import start_http_server