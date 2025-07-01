import re
from typing import Dict, List


class ConfigParser:
    def __init__(self, settings_path: str = "conductor/SETTINGS.md"):
        self.settings_path = settings_path
        self.engine_configs = {}
        self.conversational_persona = ""

    def parse_settings(self) -> Dict[str, Dict]:
        """Parse SETTINGS.md and return engine configurations"""
        with open(self.settings_path, 'r') as f:
            content = f.read()

        # Extract conversational persona
        persona_match = re.search(r'```\n(.*?)\n```', content, re.DOTALL)
        if persona_match:
            self.conversational_persona = persona_match.group(1)

        # Parse table (simplified - you'd want proper markdown parsing)
        lines = content.split('\n')
        table_started = False

        for line in lines:
            if '| Category/Area |' in line:
                table_started = True
                continue
            if table_started and line.startswith('|') and '---' not in line:
                parts = [p.strip() for p in line.split('|')[1:-1]]
                if len(parts) >= 6:
                    category = parts[0].replace('**', '').lower().replace(' ', '_')
                    self.engine_configs[category] = {
                        'model_name': parts[1],
                        'vram_requirement': parts[2],
                        'technical_model_name': parts[3],
                        'stay_loaded': parts[4].lower() == 'true',
                        'precision': parts[5],
                        'persona': self.conversational_persona if 'conversational' in category or 'reasoning' in category or 'question' in category else None
                    }

        return self.engine_configs