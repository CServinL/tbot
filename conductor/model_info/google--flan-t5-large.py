from conductor.conductor import ModelInfo

model_info = ModelInfo(
    technical_name="google/flan-t5-large",
    max_context_window=2048,
    max_new_tokens=1024,
    description="FLAN-T5 Large, strong for summarization and instruction following.",
    special_flags={"seq2seq": True}
)
