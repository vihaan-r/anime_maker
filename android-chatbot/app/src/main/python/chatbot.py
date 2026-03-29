"""Local on-device chatbot runtime for SmolLM2-135M-Instruct.

Model files expected in:
app/src/main/assets/models/HuggingFaceTB/SmolLM2-135M-Instruct/
"""

from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

_MODEL = None
_TOKENIZER = None


def _model_dir() -> Path:
    base = Path(__file__).resolve().parent.parent / "assets" / "models" / "HuggingFaceTB" / "SmolLM2-135M-Instruct"
    return base


def initialize_model() -> str:
    global _MODEL, _TOKENIZER
    if _MODEL is not None and _TOKENIZER is not None:
        return "already_initialized"

    model_dir = _model_dir()
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    _TOKENIZER = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    _MODEL = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=torch.float32,
    )
    _MODEL.eval()
    return "ok"


def generate_response(prompt: str) -> str:
    global _MODEL, _TOKENIZER
    if _MODEL is None or _TOKENIZER is None:
        initialize_model()

    system_prompt = "You are Whispering Shadows AI, a helpful offline assistant."
    full_prompt = f"<|system|>{system_prompt}</s><|user|>{prompt}</s><|assistant|>"

    inputs = _TOKENIZER(full_prompt, return_tensors="pt")
    with torch.no_grad():
        out = _MODEL.generate(
            **inputs,
            max_new_tokens=96,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    text = _TOKENIZER.decode(out[0], skip_special_tokens=True)

    if "assistant" in text:
        text = text.split("assistant")[-1].strip(": \n")

    return text[-1200:]
