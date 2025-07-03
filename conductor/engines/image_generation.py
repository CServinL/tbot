import logging
import json
import urllib3
from typing import Dict, Any, Optional
from conductor.engines.base_engine import BaseEngine
from conductor.model_loader import ModelLoader

logger = logging.getLogger(__name__)

class ImageGenerationEngine(BaseEngine):
    """Engine for text-to-image or image generation tasks."""

    def __init__(
        self,
        config: Dict[str, Any],
        model_loader: ModelLoader,
        persona: str = "",
        diffusiond_url: Optional[str] = None
    ):
        super().__init__(config, model_loader, persona)
        self.diffusiond_url = diffusiond_url

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        logger.info(f"Requesting image generation from diffusiond for prompt: {prompt}")
        try:
            image = await self._call_diffusiond_api("generate", prompt=prompt, **kwargs)
            return image
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            import traceback
            logger.error(f"Detailed error: {traceback.format_exc()}")
            return f"Error: {e}"

    def get_system_prompt(self) -> Optional[str]:
        """Get system prompt for image generation."""
        if self.persona:
            return self.persona
        return "Generate detailed image descriptions for text-to-image models."

    async def _call_diffusiond_api(self, endpoint: str, **payload: Any) -> str:
        """
        Call the diffusiond API using urllib3 instead of aiohttp.
        """
        if not self.diffusiond_url:
            raise RuntimeError("diffusiond_url is not configured")
        url = f"{self.diffusiond_url.rstrip('/')}/{endpoint.lstrip('/')}"
        http = urllib3.PoolManager()
        headers = {"Content-Type": "application/json"}
        encoded_data = json.dumps(payload).encode("utf-8")
        response = http.request(
            "POST",
            url,
            body=encoded_data,
            headers=headers,
            timeout=urllib3.Timeout(connect=10.0, read=60.0),
            retries=False,
        )
        if response.status != 200:
            raise RuntimeError(f"Diffusiond API error: {response.status} {response.data.decode('utf-8')}")
        return response.data.decode("utf-8")
