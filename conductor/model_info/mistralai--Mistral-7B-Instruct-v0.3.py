from conductor.conductor import ModelInfo

model_info = ModelInfo(
    technical_name="mistralai/Mistral-7B-Instruct-v0.3",
    max_context_window=32768,
    max_new_tokens=4096,
    description="Mistral 7B Instruct v0.3 with 32k context window and high output limit.",
    special_flags={"supports_flash_attention": True, "rope_scaling": "dynamic"}
)
