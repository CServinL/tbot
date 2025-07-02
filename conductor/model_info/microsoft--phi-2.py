from dataclasses import dataclass, field

@dataclass
class ModelInfo:
    technical_name: str
    max_context_window: int = 2048
    max_new_tokens: int = 1024
    description: str = ""
    special_flags: dict = field(default_factory=dict)
    stop_patterns: list = field(default_factory=list)

model_info = ModelInfo(
    technical_name="microsoft/phi-2",
    max_context_window=2048,
    max_new_tokens=1024,
    description="Phi-2, code and instruction, concise completions.",
    special_flags={},
    stop_patterns=[
        "\n\n",  # Phi models often end completions with double newlines
        "[EOF]",
        "```",
        "\nSources:",
        "\nLimitations:",
        "\n>",
        "\n# "
    ]
)
