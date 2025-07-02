from conductor.conductor import ModelInfo

model_info = ModelInfo(
    technical_name="mistralai/Mistral-7B-Instruct-v0.1",
    max_context_window=32768,
    max_new_tokens=4096,
    description="Mistral 7B Instruct v0.1",
    special_flags={"supports_flash_attention": True}
)
