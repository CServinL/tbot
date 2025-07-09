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
    technical_name="Qwen/Qwen1.5-1.8B-Chat",
    max_context_window=8192,
    max_new_tokens=1024,
    description="Qwen 1.5 1.8B Chat, good for math and QA.",
    special_flags={},
    stop_patterns=[
        "<|im_end|>",           # Official Qwen chat end-of-message token
        "<|endoftext|>",        # Standard end-of-text token
        "[EOF]",                # Explicit user-requested end marker
        "\n_____",              # Qwen sometimes outputs a line of underscores before switching language/content
        "\n\n",                 # Double newline often marks end of answer
        "解析：",                # Qwen sometimes starts Chinese explanations with this
        "故答案为：",            # Qwen sometimes starts Chinese answers with this
        "答案为：",              # Another common Chinese answer marker
        "参考答案：",            # "Reference answer:" in Chinese
        "Explanation:",         # English explanation marker
        "Answer:",              # English answer marker
        "Sources:",             # Source listing
        "Limitations:",         # Limitation section
        "```",                  # Code block
        "\n>",                  # Prompt-like marker
        "\n# "                  # Markdown heading
    ]
)
