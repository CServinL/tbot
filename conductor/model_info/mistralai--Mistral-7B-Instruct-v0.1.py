# Minimal ModelInfo definition for model_info files to avoid import errors
from dataclasses import dataclass, field

@dataclass
class ModelInfo:
    technical_name: str
    max_context_window: int = 8192
    max_new_tokens: int = 2048
    description: str = ""
    special_flags: dict = field(default_factory=dict)
    stop_patterns: list = field(default_factory=list)

model_info = ModelInfo(
    technical_name="mistralai/Mistral-7B-Instruct-v0.1",
    max_context_window=8192,
    max_new_tokens=2048,
    description="Mistral 7B Instruct with extended context window and higher output limit.",
    special_flags={"supports_flash_attention": True},
    stop_patterns=[
        "[/INST]",  # Mistral-style end of instruction
        "[EOF]",
        "```",
        "\nSources:",
        "\nLimitations:",
        "\n>",
        "\n# ",
        "\n\n"
    ]
)
