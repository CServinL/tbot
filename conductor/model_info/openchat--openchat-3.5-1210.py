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
    technical_name="openchat/openchat-3.5-1210",
    max_context_window=8192,
    max_new_tokens=2048,
    description="OpenChat 3.5 7B, conversational, OpenChat format.",
    special_flags={},
    stop_patterns=[
        "<|end_of_turn|>",  # OpenChat's end of turn
        "[EOF]",
        "```",
        "\nSources:",
        "\nLimitations:",
        "\n>",
        "\n# ",
        "\n\n"
    ]
)
