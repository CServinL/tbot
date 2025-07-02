from conductor.conductor import ModelInfo

model_info = ModelInfo(
    technical_name="google/flan-t5-xl",
    max_context_window=2048,
    max_new_tokens=1024,
    description="FLAN-T5 XL, larger seq2seq model for summarization and QA.",
    special_flags={"seq2seq": True}
)
