"""Model loading and generation helpers (4-bit quantized via bitsandbytes; needs a CUDA GPU)."""

from __future__ import annotations

from typing import Tuple

QWEN = "Qwen/Qwen2.5-3B-Instruct"
SMOLLM = "HuggingFaceTB/SmolLM3-3B"


def _quant_config():
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )


def load_qwen(model_name: str = QWEN):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=_quant_config(), device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return model, tokenizer


def load_smollm(model_name: str = SMOLLM, yarn_factor: float = 2.0):
    """SmolLM3 with YaRN rope scaling to extend its window from 64k to 128k."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config.rope_scaling = {"factor": yarn_factor, "original_max_position_embeddings": 65536, "type": "yarn"}
    config.max_position_embeddings = int(65536 * yarn_factor)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, config=config, quantization_config=_quant_config(), device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return model, tokenizer


def truncate_tokens(tokenizer, text: str, max_tokens: int) -> str:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 50, temperature: float = 0.3) -> Tuple[str, int]:
    """Generate a completion and return (answer, prompt_token_count).

    Only the newly generated tokens are decoded, so the prompt is never echoed back.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    n_prompt = inputs["input_ids"].shape[1]
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    answer = tokenizer.decode(out[0][n_prompt:], skip_special_tokens=True).strip()
    return answer, n_prompt


def ask_smollm(model, tokenizer, context: str, query: str, max_context_tokens: int) -> Tuple[str, int]:
    context = truncate_tokens(tokenizer, context, max_context_tokens)
    messages = [
        {"role": "system", "content": "/no_think"},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    return generate(model, tokenizer, prompt, max_new_tokens=50, temperature=0.6)
