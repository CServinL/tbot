import re
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ConfigParser:
    """Parse SETTINGS.md configuration file"""

    def __init__(self, config_path: str = "conductor/SETTINGS.md"):
        self.config_path = Path(config_path)
        self._settings = {}
        self._persona = ""

    def parse_settings(self) -> Dict[str, Dict[str, Any]]:
        """Parse SETTINGS.md and extract model configurations and diffusiond settings"""
        if not self.config_path.exists():
            logger.error(f"Configuration file not found: {self.config_path}")
            return {}

        try:
            content = self.config_path.read_text(encoding='utf-8')
            configs = self._parse_markdown_table(content)
            # Parse diffusiond settings if present
            self.diffusiond_settings = self._parse_diffusiond_settings(content)
            return configs
        except Exception as e:
            logger.error(f"Failed to parse configuration: {e}")
            return {}

    def _parse_markdown_table(self, content: str) -> Dict[str, Dict[str, Any]]:
        """Extract configuration from markdown table"""
        lines = content.split('\n')
        configs = {}

        # Extract persona
        persona_start = False
        persona_lines = []
        for line in lines:
            if '## Conversational Persona' in line:
                persona_start = True
                continue
            elif persona_start and line.startswith('##'):
                break
            elif persona_start and line.strip():
                persona_lines.append(line)
        self._persona = '\n'.join(persona_lines).strip()

        # Extract table data
        in_table = False
        for line in lines:
            line = line.strip()
            if not line:
                continue

            if '| Category/Area |' in line:
                in_table = True
                continue
            elif line.startswith('|---'):
                continue
            elif in_table and line.startswith('|') and '**' in line:
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

        logger.info(f"Parsed {len(configs)} model configurations")
        return configs

    def _parse_diffusiond_settings(self, content: str) -> dict:
        """Extract diffusiond settings from SETTINGS.md (look for a section like ## Diffusiond Settings)"""
        lines = content.split('\n')
        settings = {}
        in_section = False
        for line in lines:
            if line.strip().lower().startswith("## diffusiond"):
                in_section = True
                continue
            if in_section:
                if line.strip().startswith("##"):
                    break
                if ":" in line:
                    key, value = line.split(":", 1)
                    settings[key.strip()] = value.strip()
        return settings

    def get_conversational_persona(self) -> str:
        """
        Extracts the conversational persona prompt from the SETTINGS.md file.
        Only the content inside the first ``` block after 'Conversational Persona' is returned.
        """
        with open(self.config_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Find the section header
        persona_section = re.split(r"^##\s*Conversational Persona\s*$", content, flags=re.MULTILINE)
        if len(persona_section) < 2:
            return ""

        # Look for the first code block (``` ... ```) after the section header
        after_header = persona_section[1]
        code_block_match = re.search(r"```(?:\w*\n)?(.*?)```", after_header, re.DOTALL)
        if code_block_match:
            return code_block_match.group(1).strip()
        return ""

    def get_configuration_summary(self) -> Dict[str, Any]:
        return {
            'config_path': str(self.config_path),
            'persona_length': len(self._persona),
            'engine_count': len(self._settings),
            'engines': list(self._settings.keys()) if self._settings else []
        }

    def get_diffusiond_url(self) -> Optional[str]:
        if hasattr(self, "diffusiond_settings"):
            return self.diffusiond_settings.get("url")
        return None

# Compare this ConfigParser with the one in conductor.py:
# - If this file exists and is used, it should contain the main ConfigParser logic.
# - If it is less complete or outdated compared to the one in conductor.py, you should update it.

# Key points to compare:
# - The ConfigParser in conductor.py parses SETTINGS.md, extracts persona, model configs, and diffusiond settings.
# - It provides methods: parse_settings, get_conversational_persona, get_configuration_summary, get_diffusiond_url, etc.
# - If conductor/utils/config_parser.py is missing any of these features, it is less complete.

# Recommendation:
# If the ConfigParser in conductor.py is more complete, move its implementation to conductor/utils/config_parser.py,
# and import it in conductor.py. Remove any duplicate or outdated code.

# Example (if you want to use the more complete version everywhere):
# 1. Copy the full ConfigParser class from conductor.py to conductor/utils/config_parser.py.
# 2. In conductor.py, replace the class definition with:
#    from conductor.utils.config_parser import ConfigParser