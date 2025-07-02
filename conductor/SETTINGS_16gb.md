# LLM Server Settings

## Conversational Persona

**Note**: This persona applies only to conversational, general reasoning, and question answering tasks. Other specialized tasks (code generation, translation, etc.) should use task-specific prompting.

```
Never present generated, inferred, speculated, or deduced content as fact. If you cannot verify a claim, say so explicitly:
"I cannot verify this information."
"I do not have access to that information."
If you violate this guideline, correct as follows:
Correction: I made an unverified claim. It was incorrect and should have been labeled.
Do not modify or replace my input text unless I expressly request it.
"My knowledge base does not contain that data."
Label unverified content at the beginning of the sentence: [Inference] [Speculation] [Unverified]
Request clarifications when information is missing; do not guess or fill in gaps.
If any part of the response is unverified, mark the entire response as such.
Do not paraphrase or reinterpret my input unless I ask you to.
When using terms that imply absolute certainty (e.g., prevents, guarantees, will never occur, solves, eliminates, ensures that), label the statement unless you cite a reliable source.
For claims about LLM model behavior (including yourself), add: [Inference] or [Unverified] and clarify that it's based on observed patterns.
If you violate this guideline, correct as follows:
Correction: I made an unverified claim. It was incorrect and should have been labeled.
Do not modify or replace my input text unless I expressly request it.
Do not express compliments lightly
Always try to find counter arguments to postures and to the user prompt.
Express when something is a cultural vias and when it is a political vias.
```

**[Unverified]** This table reflects patterns observed as of January 2025. Model performance rankings and resource requirements can vary significantly based on specific tasks, evaluation methods, and hardware configurations.

| Category/Area | Best Open Source Model | VRAM/RAM Requirements | Technical Model Name | Stay Loaded | Precision |
|---------------|------------------------|----------------------|---------------------|-------------|-----------|
| **General Reasoning** | Llama 3.1 8B | ~8GB (FP16) | `meta-llama/Llama-3.1-8B-Instruct` | true | FP16 |
| **Code Generation** | Llama 3.1 8B | ~8GB (FP16) | `meta-llama/Llama-3.1-8B-Instruct` | true | FP16 |
| **Code Completion** | CodeLlama 7B | ~4GB (4-bit) | `codellama/CodeLlama-7b-hf` | true | 4-bit |
| **Mathematical Reasoning** | Llama 3.1 8B | ~8GB (FP16) | `meta-llama/Llama-3.1-8B-Instruct` | true | FP16 |
| **Creative Writing** | Llama 3.1 8B | ~8GB (FP16) | `meta-llama/Llama-3.1-8B-Instruct` | true | FP16 |
| **Conversational Chat** | Llama 3.1 8B Instruct | ~8GB (FP16) | `meta-llama/Llama-3.1-8B-Instruct` | true | FP16 |
| **Instruction Following** | Llama 3.1 8B Instruct | ~8GB (FP16) | `meta-llama/Llama-3.1-8B-Instruct` | true | FP16 |
| **Translation** | NLLB-200 | ~7GB (FP16) | `facebook/nllb-200-3.3B` | false | FP16 |
| **Summarization** | Llama 3.1 8B | ~8GB (FP16) | `meta-llama/Llama-3.1-8B-Instruct` | true | FP16 |
| **Question Answering** | Llama 3.1 8B | ~8GB (FP16) | `meta-llama/Llama-3.1-8B-Instruct` | true | FP16 |
| **Scientific/Research** | Llama 3.1 70B | ~35GB (4-bit) | `meta-llama/Llama-3.1-70B-Instruct` | false | 4-bit |
| **Legal Document Analysis** | Llama 3.1 70B | ~35GB (4-bit) | `meta-llama/Llama-3.1-70B-Instruct` | false | 4-bit |
| **Code Review/Debugging** | Llama 3.1 8B | ~8GB (FP16) | `meta-llama/Llama-3.1-8B-Instruct` | true | FP16 |
| **Long Context Tasks** | Llama 3.1 8B | ~12GB+ (varies with context) | `meta-llama/Llama-3.1-8B-Instruct` | true | FP16 |
| **Image Generation** | Diffusiond (Stable Diffusion) | External | `diffusiond` | false | N/A |

## Model Consolidation Strategy:
- **Primary Model**: `meta-llama/Llama-3.1-8B-Instruct` (FP16, ~8GB) handles most tasks and stays loaded
- **Speed-Critical**: `codellama/CodeLlama-7b-hf` (4-bit, ~4GB) for fast code completion only
- **Complex Tasks**: `meta-llama/Llama-3.1-70B-Instruct` (4-bit, ~35GB) loaded on-demand for research/legal
- **Translation Only**: `facebook/nllb-200-3.3B` (FP16, ~7GB) loaded on-demand

## Precision Logic:
- **FP16**: Balanced quality/resource usage for normal tasks requiring good accuracy
- **4-bit**: Used only when speed is critical (code completion) or when larger models are needed but RAM is limited (70B models)

## Total Memory Footprint (Stay Loaded):
- Llama 3.1 8B (FP16): ~8GB
- CodeLlama 7B (4-bit): ~4GB
- **Total persistent load**: ~12GB

## Important Notes:
- [Unverified] Resource requirements are estimates and can vary based on batch size, sequence length, and specific implementation
- Performance "best" claims are subjective and task-dependent
- Quantization trades some quality for dramatically reduced resource requirements
- Many specialized models are general models with domain-specific fine-tuning rather than fundamentally different architectures

## Diffusiond Settings
url: http://127.0.0.1:8000
