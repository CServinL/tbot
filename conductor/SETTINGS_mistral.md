# LLM Server Settings - Open Models (No Authentication Required)

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

**[Unverified]** This table uses open-source models that don't require authentication. All models listed are freely accessible.

| Category/Area | Best Open Source Model | VRAM/RAM Requirements | Technical Model Name | Stay Loaded | Precision |
|---------------|------------------------|----------------------|---------------------|-------------|-----------|
| **General Reasoning** | Mistral 7B v0.2 | ~7GB (FP16) | `mistralai/Mistral-7B-Instruct-v0.3` | true | FP16 |
| **Code Generation** | CodeLlama 7B Instruct | ~7GB (FP16) | `codellama/CodeLlama-7b-Instruct-hf` | false | FP16 |
| **Code Completion** | CodeLlama 7B | ~4GB (4-bit) | `codellama/CodeLlama-7b-hf` | false | 4-bit |
| **Mathematical Reasoning** | WizardMath 7B | ~7GB (FP16) | `WizardLM/WizardMath-7B-V1.1` | false | FP16 |
| **Creative Writing** | Mistral 7B v0.2 | ~7GB (FP16) | `mistralai/Mistral-7B-Instruct-v0.3` | true | FP16 |
| **Conversational Chat** | Mistral 7B v0.2 | ~7GB (FP16) | `mistralai/Mistral-7B-Instruct-v0.3` | true | FP16 |
| **Instruction Following** | Mistral 7B v0.2 | ~7GB (FP16) | `mistralai/Mistral-7B-Instruct-v0.3` | true | FP16 |
| **Translation** | NLLB 3.3B | ~3GB (FP16) | `facebook/nllb-200-3.3B` | false | FP16 |
| **Summarization** | Mistral 7B v0.2 | ~7GB (FP16) | `mistralai/Mistral-7B-Instruct-v0.3` | true | FP16 |
| **Question Answering** | Mistral 7B v0.2 | ~7GB (FP16) | `mistralai/Mistral-7B-Instruct-v0.3` | true | FP16 |
| **Scientific/Research** | WizardLM 7B | ~7GB (FP16) | `WizardLM/WizardLM-7B-V1.0` | false | FP16 |
| **Legal Document Analysis** | Mistral 7B v0.2 | ~7GB (FP16) | `mistralai/Mistral-7B-Instruct-v0.3` | false | FP16 |
| **Code Review/Debugging** | CodeLlama 7B Instruct | ~7GB (FP16) | `codellama/CodeLlama-7b-Instruct-hf` | false | FP16 |
| **Long Context Tasks** | Mistral 7B v0.2 | ~12GB+ (varies with context) | `mistralai/Mistral-7B-Instruct-v0.3` | true | FP16 |
| **Image Generation** | Diffusiond (Stable Diffusion) | External | `diffusiond` | false | N/A |

## Model Consolidation Strategy:
- **Primary Model**: `mistralai/Mistral-7B-Instruct-v0.3` (FP16, ~7GB) handles most general tasks and stays loaded
- **Code Tasks**: `codellama/CodeLlama-7b-Instruct-hf` (FP16, ~7GB) for code generation/review
- **Speed-Critical Code**: `codellama/CodeLlama-7b-hf` (4-bit, ~4GB) for fast code completion only
- **Math Specialized**: `WizardLM/WizardMath-7B-V1.1` (FP16, ~7GB) loaded on-demand for complex math
- **Translation Only**: `facebook/nllb-200-3.3B` (FP16, ~3GB) loaded on-demand

## Precision Logic:
- **FP16**: Balanced quality/resource usage for normal tasks requiring good accuracy
- **4-bit**: Used only when speed is critical (code completion) to save memory

## Total Memory Footprint (Stay Loaded):
- Mistral 7B (FP16): ~7GB
- **Total persistent load**: ~7GB (much lighter than Llama setup)

## Important Notes:
- [Unverified] Resource requirements are estimates and can vary based on batch size, sequence length, and specific implementation
- These models are **freely accessible** and don't require HuggingFace authentication
- **Mistral 7B v0.2** is an excellent general-purpose model with strong instruction following
- **CodeLlama models** are Meta's open-source code models (no gating)
- **WizardMath/WizardLM** are fine-tuned variants optimized for specific tasks
- Performance is very good for a 7B model, though not quite at Llama 3.1 8B levels

## Authentication-Free Benefits:
- ✅ **No HuggingFace login required**
- ✅ **No access requests needed**
- ✅ **Works offline after download**
- ✅ **Smaller memory footprint** (7GB vs 16GB)
- ✅ **Faster download and setup**
- ✅ **Good performance** for most tasks

## Upgrading to Llama Models Later:
Once you complete HuggingFace authentication:
1. Request access to Meta Llama models
2. Run `huggingface-cli login`
3. Replace model names with `meta-llama/Llama-3.1-8B-Instruct`
4. Enjoy the improved performance of larger models

## Diffusiond Settings
url: http://127.0.0.1:8000
