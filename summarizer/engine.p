"""
Model loading and summarization logic, kept separate from the Streamlit UI
so it can be tested or reused (e.g. from a CLI or an API) independently.
"""

from functools import lru_cache

import streamlit as st
import torch
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

from .config import MODEL_DEFAULTS, T5_TASK_PREFIX, GenerationConfig


def get_device() -> torch.device:
    """Use GPU automatically if available, otherwise fall back to CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    """
    Load and cache a (tokenizer, model) pair.

    st.cache_resource means the multi-hundred-MB model is downloaded and
    loaded into memory only once per session, instead of on every single
    click of "Summarize" — this was the single biggest performance issue
    in the original script.
    """
    checkpoint = MODEL_DEFAULTS[model_name]["checkpoint"]
    device = get_device()

    if model_name == "BART":
        tokenizer = BartTokenizer.from_pretrained(checkpoint)
        model = BartForConditionalGeneration.from_pretrained(checkpoint)
    else:
        tokenizer = T5Tokenizer.from_pretrained(checkpoint)
        model = T5ForConditionalGeneration.from_pretrained(checkpoint)

    model.to(device)
    model.eval()
    return tokenizer, model


def _clean_text(text: str) -> str:
    """Collapse whitespace/newlines so the tokenizer sees clean input."""
    return " ".join(text.replace("\n", " ").split())


def summarize(model_name: str, text: str, gen_config: GenerationConfig) -> str:
    """
    Run summarization for the given model and return the generated text.

    Raises ValueError on empty input so the UI can show a friendly message
    instead of crashing on a blank generate() call.
    """
    text = _clean_text(text)
    if not text:
        raise ValueError("Input text is empty after cleaning — please paste some content.")

    tokenizer, model = load_model(model_name)
    device = get_device()

    if model_name == "T5":
        text = T5_TASK_PREFIX + text

    with torch.no_grad():
        input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True).to(device)

        summary_ids = model.generate(
            input_ids,
            num_beams=gen_config.num_beams,
            no_repeat_ngram_size=gen_config.no_repeat_ngram_size,
            length_penalty=gen_config.length_penalty,
            min_length=gen_config.min_length,
            max_length=gen_config.max_length,
            early_stopping=gen_config.early_stopping,
        )

    return tokenizer.decode(
        summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
