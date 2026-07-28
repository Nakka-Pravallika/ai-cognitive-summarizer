# 🧠 AI Cognitive Summarizer

A web app that summarizes long text into a concise summary using pretrained
**BART** and **T5** transformer models from Hugging Face, with an interactive
Streamlit interface for tuning generation parameters.

> Originally built as a final-year B.Tech project; refactored into a clean,
> modular structure with caching, error handling, and an upgraded UI.

## Features

- Choose between **BART** (`facebook/bart-base`) and **T5** (`t5-base`) models
- Tune generation parameters live: number of beams, no-repeat n-gram size,
  length penalty, min/max summary length, early stopping
- Model is loaded once and cached (`st.cache_resource`), instead of being
  reloaded on every click
- Automatic GPU detection (falls back to CPU if none available)
- Input validation and friendly error messages instead of raw stack traces
- Character-count / compression-ratio feedback on the generated summary

## Project structure

```
ai_cognitive_summarizer/
├── app.py                   # Streamlit UI entry point
├── summarizer/
│   ├── __init__.py          # Public package API
│   ├── config.py            # Model checkpoints & default generation params
│   └── engine.py            # Model loading (cached) + summarization logic
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/Nakka-Pravallika/ai-cognitive-summarizer.git
cd ai-cognitive-summarizer
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

The app opens in your browser. Paste any text, choose a model in the sidebar,
adjust parameters if you like, and click **Summarize**.

## How it works

1. Text is cleaned (whitespace/newlines normalized) and tokenized.
2. For T5, a `"summarize: "` task-prefix is prepended, as required by the model.
3. The selected pretrained model generates a summary using beam search, with
   the parameters set in the sidebar.
4. The generated token IDs are decoded back into readable text and displayed.

**Note:** This project uses pretrained BART/T5 checkpoints as-is — it does not
fine-tune or train a model from scratch. The work here is the summarization
pipeline, parameter-tuning interface, and app structure around those models.

## Tech stack

Python · PyTorch · Hugging Face Transformers (BART, T5) · Streamlit
