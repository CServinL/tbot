import asyncio
import logging
import json
import time
from typing import Dict, Any, List, Optional, AsyncGenerator
from dataclasses import dataclass, asdict
from pathlib import Path

from conductor.utils.config_parser import ConfigParser
from conductor.utils.model_registry import ModelRegistry, ModelStatus
from conductor.utils.persona_loader import PersonaLoader
from conductor.model_loader import ModelLoader
from conductor.dependencies_loader import DependenciesLoader

logger = logging.getLogger(__name__)


@dataclass
class MCPRequest:
    """MCP request structure."""
    id: str
    method: str
    params: Dict[str, Any]
    jsonrpc: str = "2.0"


@dataclass
class MCPResponse:
    """MCP response structure."""
    id: str
    result: Any = None
    error: Optional[Dict[str, Any]] = None
    jsonrpc: str = "2.0"


@dataclass
class MCPNotification:
    """MCP notification structure."""
    method: str
    params: Dict[str, Any]
    jsonrpc: str = "2.0"


@dataclass
class ResourceInfo:
    """Information about an available resource."""
    uri: str
    name: str
    description: str
    mimeType: str


@dataclass
class ToolInfo:
    """Information about an available tool."""
    name: str
    description: str
    inputSchema: Dict[str, Any]


class MCPServer:
    """Model Context Protocol server for Conductor LLM system."""

    def __init__(self):
        # Core components
        self.config_parser = ConfigParser()
        self.model_registry = ModelRegistry()
        self.persona_loader = PersonaLoader()
        self.model_loader = ModelLoader()
        self.dependencies_loader = DependenciesLoader()

        # Server state
        self.engines: Dict[str, Any] = {}
        self.engine_configs: Dict[str, Dict[str, Any]] = {}
        self.client_capabilities: Dict[str, Any] = {}
        self.notifications_enabled = True

        # MCP protocol handlers
        self.method_handlers = {
            # Initialization
            'initialize': self._handle_initialize,
            'initialized': self._handle_initialized,

            # Resource management
            'resources/list': self._handle_list_resources,
            'resources/read': self._handle_read_resource,

            # Tool management
            'tools/list': self._handle_list_tools,
            'tools/call': self._handle_call_tool,

            # Completion and generation
            'completion/complete': self._handle_completion,
            'generation/generate': self._handle_generation,
            'generation/stream': self._handle_stream_generation,

            # Model management
            'models/list': self._handle_list_models,
            'models/load': self._handle_load_model,
            'models/unload': self._handle_unload_model,
            'models/status': self._handle_model_status,

            # Session management
            'sessions/create': self._handle_create_session,
            'sessions/list': self._handle_list_sessions,
            'sessions/clear': self._handle_clear_session,

            # System operations
            'system/health': self._handle_health_check,
            'system/stats': self._handle_system_stats,
            'system/config': self._handle_config_operations,
        }

    async def initialize(self):
        """Initialize the MCP server."""
        try:
            logger.info("Initializing Conductor MCP Server...")

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

            logger.info("Conductor MCP Server initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize MCP server: {e}")
            raise

    async def handle_request(self, request_data: str) -> str:
        """Handle incoming MCP request.

        Args:
            request_data: JSON string containing MCP request

        Returns:
            str: JSON response string
        """
        try:
            # Parse request
            request_dict = json.loads(request_data)

            # Handle notification (no response needed)
            if 'id' not in request_dict:
                await self._handle_notification(request_dict)
                return ""

            # Create request object
            request = MCPRequest(
                id=request_dict['id'],
                method=request_dict['method'],
                params=request_dict.get('params', {}),
                jsonrpc=request_dict.get('jsonrpc', '2.0')
            )

            # Handle request
            response = await self._handle_request(request)

            # Return JSON response
            return json.dumps(asdict(response))

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in request: {e}")
            error_response = MCPResponse(
                id="unknown",
                error={"code": -32700, "message": "Parse error"}
            )
            return json.dumps(asdict(error_response))

        except Exception as e:
            logger.error(f"Error handling request: {e}")
            error_response = MCPResponse(
                id=request_dict.get('id', 'unknown'),
                error={"code": -32603, "message": "Internal error", "data": str(e)}
            )
            return json.dumps(asdict(error_response))

    async def _handle_request(self, request: MCPRequest) -> MCPResponse:
        """Handle an MCP request and return response."""
        handler = self.method_handlers.get(request.method)

        if not handler:
            return MCPResponse(
                id=request.id,
                error={
                    "code": -32601,
                    "message": f"Method not found: {request.method}"
                }
            )

        try:
            result = await handler(request.params)
            return MCPResponse(id=request.id, result=result)
        except Exception as e:
            logger.error(f"Error in handler {request.method}: {e}")
            return MCPResponse(
                id=request.id,
                error={
                    "code": -32603,
                    "message": "Internal error",
                    "data": str(e)
                }
            )

    async def _handle_notification(self, notification_dict: Dict[str, Any]):
        """Handle MCP notification."""
        method = notification_dict.get('method')
        params = notification_dict.get('params', {})

        if method == 'notifications/initialized':
            logger.info("Client initialized")
        elif method == 'notifications/cancelled':
            # Handle request cancellation
            request_id = params.get('id')
            logger.info(f"Request cancelled: {request_id}")
        else:
            logger.warning(f"Unknown notification method: {method}")

    # MCP Method Handlers

    async def _handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle initialize request."""
        self.client_capabilities = params.get('capabilities', {})

        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "resources": {
                    "subscribe": True,
                    "listChanged": True
                },
                "tools": {
                    "listChanged": True
                },
                "completion": {
                    "supports": ["generation", "code", "translation"]
                },
                "generation": {
                    "streaming": True,
                    "categories": list(self.engine_configs.keys())
                }
            },
            "serverInfo": {
                "name": "Conductor LLM Server",
                "version": "1.0.0",
                "description": "Multi-model LLM server with specialized engines"
            }
        }

    async def _handle_initialized(self, params: Dict[str, Any]) -> None:
        """Handle initialized notification."""
        logger.info("MCP client initialization complete")
        return None

    async def _handle_list_resources(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List available resources."""
        resources = [
            ResourceInfo(
                uri="conductor://models",
                name="Model Registry",
                description="Information about all registered models",
                mimeType="application/json"
            ),
            ResourceInfo(
                uri="conductor://config",
                name="Configuration",
                description="Current server configuration",
                mimeType="text/markdown"
            ),
            ResourceInfo(
                uri="conductor://health",
                name="Health Status",
                description="Current server health and statistics",
                mimeType="application/json"
            ),
            ResourceInfo(
                uri="conductor://engines",
                name="Engine Status",
                description="Status of all LLM engines",
                mimeType="application/json"
            )
        ]

        return {
            "resources": [asdict(resource) for resource in resources]
        }

    async def _handle_read_resource(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Read a specific resource."""
        uri = params.get('uri')

        if uri == "conductor://models":
            models = self.model_registry.get_all_models()
            return {
                "contents": [
                    {
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": json.dumps(models, indent=2)
                    }
                ]
            }

        elif uri == "conductor://config":
            config_summary = self.config_parser.get_configuration_summary()
            return {
                "contents": [
                    {
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": json.dumps(config_summary, indent=2)
                    }
                ]
            }

        elif uri == "conductor://health":
            health_data = await self._get_health_data()
            return {
                "contents": [
                    {
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": json.dumps(health_data, indent=2)
                    }
                ]
            }

        elif uri == "conductor://engines":
            engine_status = await self._get_engine_status()
            return {
                "contents": [
                    {
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": json.dumps(engine_status, indent=2)
                    }
                ]
            }

        else:
            raise ValueError(f"Unknown resource URI: {uri}")

    async def _handle_list_tools(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List available tools."""
        tools = [
            ToolInfo(
                name="generate_text",
                description="Generate text using specified category engine",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string", "description": "Input prompt"},
                        "category": {"type": "string", "description": "Task category"},
                        "max_tokens": {"type": "integer", "description": "Maximum tokens"},
                        "temperature": {"type": "number", "description": "Generation temperature"},
                        "session_id": {"type": "string", "description": "Session ID for conversations"}
                    },
                    "required": ["prompt", "category"]
                }
            ),
            ToolInfo(
                name="complete_code",
                description="Complete code using code completion engine",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Code context"},
                        "language": {"type": "string", "description": "Programming language"},
                        "max_tokens": {"type": "integer", "description": "Maximum completion tokens"}
                    },
                    "required": ["code"]
                }
            ),
            ToolInfo(
                name="translate_text",
                description="Translate text between languages",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to translate"},
                        "source_lang": {"type": "string", "description": "Source language"},
                        "target_lang": {"type": "string", "description": "Target language"},
                        "formal_register": {"type": "boolean", "description": "Use formal language"}
                    },
                    "required": ["text", "source_lang", "target_lang"]
                }
            ),
            ToolInfo(
                name="solve_math",
                description="Solve mathematical problems with step-by-step reasoning",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "problem": {"type": "string", "description": "Mathematical problem"},
                        "math_context": {"type": "string", "description": "Type of mathematics"},
                        "show_work": {"type": "boolean", "description": "Show step-by-step work"},
                        "use_latex": {"type": "boolean", "description": "Use LaTeX formatting"}
                    },
                    "required": ["problem"]
                }
            )
        ]

        return {
            "tools": [asdict(tool) for tool in tools]
        }

    async def _handle_call_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Call a specific tool."""
        tool_name = params.get('name')
        arguments = params.get('arguments', {})

        if tool_name == "generate_text":
            result = await self._tool_generate_text(arguments)
        elif tool_name == "complete_code":
            result = await self._tool_complete_code(arguments)
        elif tool_name == "translate_text":
            result = await self._tool_translate_text(arguments)
        elif tool_name == "solve_math":
            result = await self._tool_solve_math(arguments)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

        return {"content": [{"type": "text", "text": result}]}

    async def _handle_completion(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle completion request."""
        prompt = params.get('prompt', '')
        category = params.get('category', 'conversational_chat')

        engine = await self._get_engine_for_category(category)
        if not engine:
            raise ValueError(f"No engine available for category: {category}")

        response = await engine.generate(prompt, **params)

        return {
            "completion": {
                "values": [response],
                "total": 1,
                "hasMore": False
            }
        }

    async def _handle_generation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle generation request."""
        prompt = params.get('prompt', '')
        category = params.get('category', 'conversational_chat')

        engine = await self._get_engine_for_category(category)
        if not engine:
            raise ValueError(f"No engine available for category: {category}")

        response = await engine.generate(prompt, **params)

        return {
            "response": response,
            "category": category,
            "model": engine.technical_model_name,
            "metadata": {
                "generation_count": engine.generation_count,
                "model_loaded": engine.is_loaded()
            }
        }

    async def _handle_stream_generation(self, params: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Handle streaming generation request."""
        prompt = params.get('prompt', '')
        category = params.get('category', 'conversational_chat')

        engine = await self._get_engine_for_category(category)
        if not engine:
            raise ValueError(f"No engine available for category: {category}")

        async for chunk in engine.generate_stream(prompt, **params):
            yield chunk

    async def _handle_list_models(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List all models."""
        models = self.model_registry.get_all_models()
        return {"models": models}

    async def _handle_load_model(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Load a specific model."""
        technical_name = params.get('technical_name')
        if not technical_name:
            raise ValueError("technical_name parameter required")

        await self._load_model_background(technical_name)
        return {"message": f"Model {technical_name} loading initiated"}

    async def _handle_unload_model(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Unload a specific model."""
        technical_name = params.get('technical_name')
        if not technical_name:
            raise ValueError("technical_name parameter required")

        model_info = self.model_registry.get_model_info(technical_name)
        if not model_info or model_info.status != ModelStatus.LOADED:
            raise ValueError(f"Model {technical_name} not loaded")

        engine = self.model_registry.get_model_engine(technical_name)
        if engine:
            success = await engine.unload_model()
            if success:
                self.model_registry.set_model_unloaded(technical_name)
                return {"message": f"Model {technical_name} unloaded successfully"}

        raise ValueError(f"Failed to unload model {technical_name}")

    async def _handle_model_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get status of specific model or all models."""
        technical_name = params.get('technical_name')

        if technical_name:
            model_info = self.model_registry.get_model_info(technical_name)
            if not model_info:
                raise ValueError(f"Model not found: {technical_name}")
            return {"model": model_info.to_dict()}
        else:
            models = self.model_registry.get_all_models()
            return {"models": models}

    async def _handle_create_session(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new conversation session."""
        import uuid
        session_id = str(uuid.uuid4())
        return {"session_id": session_id}

    async def _handle_list_sessions(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List active sessions."""
        # This would be implemented with actual session tracking
        return {"sessions": [], "count": 0}

    async def _handle_clear_session(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Clear a conversation session."""
        session_id = params.get('session_id')
        if not session_id:
            raise ValueError("session_id parameter required")

        # Clear session in engines
        for engine in self.engines.values():
            if hasattr(engine, 'clear_conversation_history'):
                engine.clear_conversation_history(session_id)

        return {"message": f"Session {session_id} cleared"}

    async def _handle_health_check(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get system health status."""
        return await self._get_health_data()

    async def _handle_system_stats(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get system statistics."""
        memory_stats = self.model_registry.get_memory_usage()
        activity_stats = self.model_registry.get_activity_stats()
        registry_stats = self.model_registry.get_registry_stats()

        return {
            "memory": memory_stats,
            "activity": activity_stats,
            "registry": registry_stats,
            "engines": len(self.engines)
        }

    async def _handle_config_operations(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle configuration operations."""
        operation = params.get('operation', 'get')

        if operation == 'get':
            return self.config_parser.get_configuration_summary()
        elif operation == 'reload':
            success = self.config_parser.reload_settings()
            if success:
                await self._register_models()
                return {"message": "Configuration reloaded successfully"}
            else:
                raise ValueError("Failed to reload configuration")
        elif operation == 'validate':
            validation_result = self.config_parser.validate_configuration()
            return {"validation": validation_result}
        else:
            raise ValueError(f"Unknown operation: {operation}")

    # Tool implementations

    async def _tool_generate_text(self, args: Dict[str, Any]) -> str:
        """Generate text tool implementation."""
        prompt = args.get('prompt', '')
        category = args.get('category', 'conversational_chat')

        engine = await self._get_engine_for_category(category)
        if not engine:
            raise ValueError(f"No engine available for category: {category}")

        return await engine.generate(prompt, **args)

    async def _tool_complete_code(self, args: Dict[str, Any]) -> str:
        """Complete code tool implementation."""
        code = args.get('code', '')
        language = args.get('language')

        engine = await self._get_engine_for_category('code_completion')
        if not engine:
            raise ValueError("Code completion engine not available")

        return await engine.generate(code, language=language, **args)

    async def _tool_translate_text(self, args: Dict[str, Any]) -> str:
        """Translate text tool implementation."""
        text = args.get('text', '')
        source_lang = args.get('source_lang', 'english')
        target_lang = args.get('target_lang', 'spanish')

        engine = await self._get_engine_for_category('translation')
        if not engine:
            raise ValueError("Translation engine not available")

        return await engine.generate(
            text,
            source_lang=source_lang,
            target_lang=target_lang,
            **args
        )

    async def _tool_solve_math(self, args: Dict[str, Any]) -> str:
        """Solve math tool implementation."""
        problem = args.get('problem', '')

        engine = await self._get_engine_for_category('mathematical_reasoning')
        if not engine:
            raise ValueError("Mathematical reasoning engine not available")

        return await engine.generate(problem, **args)

    # Helper methods (similar to HTTP server)

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

    async def _load_persistent_models(self):
        """Load models that should stay loaded."""
        stay_loaded_models = self.model_registry.get_stay_loaded_models()

        for technical_name in stay_loaded_models:
            await self._load_model_background(technical_name)

    async def _load_model_background(self, technical_name: str):
        """Load a model in the background."""
        try:
            self.model_registry.update_model_status(technical_name, ModelStatus.LOADING)

            engine = await self._create_engine_for_model(technical_name)
            if not engine:
                self.model_registry.update_model_status(
                    technical_name, ModelStatus.ERROR, "Failed to create engine"
                )
                return

            success = await engine.load_model()
            if success:
                memory_usage = engine.get_memory_usage()['memory_gb']
                self.model_registry.set_model_loaded(technical_name, engine, memory_usage)

                categories = self.model_registry.get_categories_for_model(technical_name)
                for category in categories:
                    self.engines[category] = engine

                logger.info(f"Successfully loaded model {technical_name}")
            else:
                self.model_registry.update_model_status(
                    technical_name, ModelStatus.ERROR, "Model loading failed"
                )

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

        category = categories[0]
        config = self.engine_configs.get(category)
        if not config:
            return None

        engine_class = self._get_engine_class(category)
        if not engine_class:
            return None

        return engine_class(config)

    def _get_engine_class(self, category: str):
        """Get engine class for a category."""
        try:
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

    async def _get_health_data(self) -> Dict[str, Any]:
        """Get health data for the system."""
        memory_stats = self.model_registry.get_memory_usage()
        registry_health = self.model_registry.health_check()

        return {
            "status": registry_health['status'],
            "loaded_models": registry_health['loaded_models'],
            "error_models": registry_health['error_models'],
            "total_memory_gb": memory_stats['total_memory_gb'],
            "engines_active": len(self.engines),
            "timestamp": time.time()
        }

    async def _get_engine_status(self) -> Dict[str, Any]:
        """Get status of all engines."""
        engine_status = {}

        for category, engine in self.engines.items():
            engine_status[category] = await engine.health_check()

        return {
            "engines": engine_status,
            "total_engines": len(self.engines),
            "categories_configured": len(self.engine_configs)
        }


# Standalone function to create MCP server
def create_mcp_server() -> MCPServer:
    """Create MCP server instance."""
    return MCPServer()


if __name__ == "__main__":
    # Example usage
    async def main():
        server = MCPServer()
        await server.initialize()

        # Example request
        test_request = {
            "id": "1",
            "method": "tools/call",
            "params": {
                "name": "generate_text",
                "arguments": {
                    "prompt": "Hello, how are you?",
                    "category": "conversational_chat"
                }
            }
        }

        response = await server.handle_request(json.dumps(test_request))
        print(f"Response: {response}")


    asyncio.run(main())

MCPServer = MCPServer