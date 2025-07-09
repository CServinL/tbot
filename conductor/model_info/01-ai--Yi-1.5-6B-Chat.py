from dataclasses import dataclass, field

@dataclass
class ModelInfo:
    technical_name: str
    max_context_window: int = 32768
    max_new_tokens: int = 2048
    description: str = ""
    special_flags: dict = field(default_factory=dict)
    stop_patterns: list = field(default_factory=list)

model_info = ModelInfo(
    technical_name="01-ai/Yi-1.5-6B-Chat",
    max_context_window=32768,
    max_new_tokens=2048,
    description="Yi 1.5 6B Chat, long context, for long documents.",
    special_flags={},
    stop_patterns=[
        "<|im_end|>",  # Yi chat end marker
        "[EOF]",
        "```",
        "\nSources:",
        "\nLimitations:",
        "\n>",
        "\n# ",
        "\n\n"
    ]
)
