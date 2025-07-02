from dataclasses import dataclass, field

@dataclass
class ModelInfo:
    technical_name: str
    max_context_window: int = 1024
    max_new_tokens: int = 512
    description: str = ""
    special_flags: dict = field(default_factory=dict)
    stop_patterns: list = field(default_factory=list)

model_info = ModelInfo(
    technical_name="facebook/nllb-200-distilled-1.3B",
    max_context_window=1024,
    max_new_tokens=512,
    description="NLLB-200 1.3B, translation, concise output.",
    special_flags={},
    stop_patterns=[
        "</s>",  # NLLB end of sentence
        "[EOF]",
        "\n\n"
    ]
)
