from dataclasses import dataclass, field

@dataclass
class ModelInfo:
    technical_name: str
    max_context_window: int = 8192
    max_new_tokens: int = 1024
    description: str = ""
    special_flags: dict = field(default_factory=dict)
    stop_patterns: list = field(default_factory=list)

model_info = ModelInfo(
    technical_name="google/gemma-2b-it",
    max_context_window=8192,
    max_new_tokens=1024,
    description="Gemma 2B IT, creative writing and general tasks.",
    special_flags={},
    stop_patterns=[
        "<eos>",  # End of sequence for Gemma
        "[EOF]",
        "```",
        "\nSources:",
        "\nLimitations:",
        "\n>",
        "\n# ",
        "\n\n"
    ]
)
