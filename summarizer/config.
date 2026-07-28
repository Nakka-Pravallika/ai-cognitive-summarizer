"""
Configuration and default generation parameters for each supported model.

Keeping these in one place makes it easy to tweak defaults or add a new
model later without touching the UI or the inference code.
"""

from dataclasses import dataclass


@dataclass
class GenerationConfig:
    """Parameters passed to model.generate() during summarization."""
    num_beams: int
    no_repeat_ngram_size: int
    length_penalty: float
    min_length: int
    max_length: int
    early_stopping: bool


# Sensible starting points for each model, tuned for short/medium articles.
MODEL_DEFAULTS = {
    "BART": {
        "checkpoint": "facebook/bart-base",
        "config": GenerationConfig(
            num_beams=4,
            no_repeat_ngram_size=3,
            length_penalty=1.0,
            min_length=12,
            max_length=128,
            early_stopping=True,
        ),
    },
    "T5": {
        "checkpoint": "t5-base",
        "config": GenerationConfig(
            num_beams=4,
            no_repeat_ngram_size=3,
            length_penalty=2.0,
            min_length=30,
            max_length=200,
            early_stopping=True,
        ),
    },
}

# T5 requires a task-prefix token sequence ("summarize: ") prepended to input.
T5_TASK_PREFIX = "summarize: "

MAX_INPUT_CHARS = 20_000  # basic guardrail so a huge paste doesn't hang the UI
