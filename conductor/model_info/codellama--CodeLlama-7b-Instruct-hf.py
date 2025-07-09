from conductor.conductor import ModelInfo

model_info = ModelInfo(
    technical_name="codellama/CodeLlama-7b-Instruct-hf",
    max_context_window=16384,
    max_new_tokens=2048,
    description="CodeLlama 7B Instruct, optimized for code generation and completion.",
    special_flags={"code_model": True}
)
