from conductor.conductor import ModelInfo

model_info = ModelInfo(
    technical_name="facebook/nllb-200-distilled-600M",
    max_context_window=1024,
    max_new_tokens=512,
    description="NLLB-200 Distilled 600M for multilingual translation tasks.",
    special_flags={"translation_model": True}
)
