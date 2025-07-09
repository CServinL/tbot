#!/usr/bin/env python3
"""
HTTP Client for communicating with Conductor server using standard library
"""

import json
import logging
import urllib.request
import urllib.parse
import urllib.error
from typing import Dict, Any, Optional, Union, Literal, TypedDict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Type definitions for API responses
class HealthResponse(TypedDict):
    status: str
    loaded_models: int
    total_memory_gb: float
    uptime_seconds: float


class GenerationResponse(TypedDict):
    response: str
    category: str
    model: str
    session_id: Optional[str]
    generation_time: float
    tokens_generated: Optional[int]


class ErrorResponse(TypedDict):
    type: Literal['text']
    response: str
    error: str
    category: Literal['error']
    model: Literal['none']


class ConductorHealthStatus(TypedDict, total=False):
    status: Literal['healthy', 'error', 'unknown']
    error: Optional[str]
    # Allow additional fields from health response
    loaded_models: int
    total_memory_gb: float
    uptime_seconds: float


class HealthCheckResult(TypedDict):
    conductor: ConductorHealthStatus


# HTTP method types
HttpMethod = Literal['GET', 'POST', 'PUT', 'DELETE']

# Generation parameter types
GenerationParams = Union[int, float, str, bool]


@dataclass
class ServerConfig:
    """Configuration for a server endpoint"""
    base_url: str
    timeout: int = 30
    headers: Optional[Dict[str, str]] = None


class ConductorClient:
    """HTTP client for Conductor LLM server using standard library"""
    
    def __init__(self, config: ServerConfig) -> None:
        self.config = config
    
    def health_check(self) -> Dict[str, Any]:
        """Check server health"""
        url = urllib.parse.urljoin(self.config.base_url, "/health")
        return self._make_request("GET", url)
    
    def generate(self, 
                prompt: str, 
                category: Optional[str] = None,
                session_id: Optional[str] = None,
                stream: bool = False,
                **kwargs: GenerationParams) -> Dict[str, Any]:
        """Generate text response"""
        url = urllib.parse.urljoin(self.config.base_url, "/generate")
        
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "stream": stream,
            "extra_params": kwargs
        }
        
        if category:
            payload["category"] = category
        if session_id:
            payload["session_id"] = session_id
        
        # Add optional parameters
        for param in ['max_tokens', 'temperature', 'top_p', 'language']:
            if param in kwargs:
                payload[param] = kwargs[param]
        
        return self._make_request("POST", url, data=payload)
    
    def classify(self, prompt: str) -> str:
        """Classify a prompt to determine the appropriate category"""
        # Use generate with a simple classification request
        response = self.generate(
            prompt=prompt,
            category="general_reasoning",
            max_tokens=10
        )
        return response.get('category', 'general_reasoning')
    
    def get_models(self) -> Dict[str, Any]:
        """Get list of available models"""
        url = urllib.parse.urljoin(self.config.base_url, "/models")
        return self._make_request("GET", url)
    
    def _make_request(self, method: HttpMethod, url: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make HTTP request using urllib"""
        headers: Dict[str, str] = self.config.headers.copy() if self.config.headers else {}
        headers['Content-Type'] = 'application/json'
        
        request_data: Optional[bytes] = None
        if data:
            request_data = json.dumps(data).encode('utf-8')
        
        req = urllib.request.Request(
            url=url,
            data=request_data,
            headers=headers,
            method=method
        )
        
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout) as response:
                response_data = response.read().decode('utf-8')
                return json.loads(response_data)
        
        except urllib.error.HTTPError as e:
            error_msg = f"HTTP {e.code}: {e.reason}"
            try:
                error_data = e.read().decode('utf-8')
                error_detail = json.loads(error_data)
                error_msg += f" - {error_detail.get('detail', error_data)}"
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
            raise RuntimeError(error_msg)
        
        except urllib.error.URLError as e:
            raise RuntimeError(f"Connection error: {e.reason}")
        
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON response: {e}")


class TBotClient:
    """HTTP client for TBot CLI - communicates only with Conductor"""
    
    def __init__(self, 
                 conductor_url: str = "http://localhost:8001",
                 timeout: int = 30) -> None:
        self.conductor_config = ServerConfig(base_url=conductor_url, timeout=timeout)
        self.conductor_client = ConductorClient(self.conductor_config)
    
    def process_prompt(self, prompt: str, **kwargs: GenerationParams) -> Union[Dict[str, Any], ErrorResponse]:
        """Process a prompt via Conductor (handles both text and image routing internally)"""
        try:
            # Extract specific parameters that go to generate() directly
            category = kwargs.pop('category', None) if 'category' in kwargs else None
            session_id = kwargs.pop('session_id', None) if 'session_id' in kwargs else None
            stream = kwargs.pop('stream', False) if 'stream' in kwargs else False
            
            # Ensure extracted parameters have correct types
            if category is not None and not isinstance(category, str):
                category = str(category)
            if session_id is not None and not isinstance(session_id, str):
                session_id = str(session_id)
            if not isinstance(stream, bool):
                stream = bool(stream)
            
            result = self.conductor_client.generate(
                prompt=prompt,
                category=category,
                session_id=session_id,
                stream=stream,
                **kwargs  # Remaining kwargs go as generation parameters
            )
            result['type'] = 'text'  # Conductor handles image routing internally
            return result
        except Exception as e:
            logger.error(f"Conductor error: {e}")
            error_response: ErrorResponse = {
                'type': 'text',
                'response': f"Sorry, I encountered an error: {str(e)}",
                'error': str(e),
                'category': 'error',
                'model': 'none'
            }
            return error_response
    
    def health_check(self) -> Dict[str, Any]:
        """Check health of Conductor (which includes Diffusiond status)"""
        results: Dict[str, Any] = {
            'conductor': {'status': 'unknown', 'error': None}
        }
        
        # Check conductor
        try:
            conductor_health = self.conductor_client.health_check()
            results['conductor'] = {
                'status': 'healthy',
                'error': None,
                **conductor_health
            }
        except Exception as e:
            results['conductor'] = {
                'status': 'error',
                'error': str(e)
            }
        
        return results
