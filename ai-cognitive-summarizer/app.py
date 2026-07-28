"""
AI Cognitive Summarizer
-----------------------
A Streamlit app that summarizes text using pretrained BART or T5
transformer models (Hugging Face), with tunable generation parameters.

Run with:
    streamlit run app.py
"""

import streamlit as st

from summarizer import MODEL_DEFAULTS, GenerationConfig, get_device, summarize

st.set_page_config(
    page_title="AI Cognitive Summarizer",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 AI Cognitive Summarizer")
st.caption("Summarize long text into a concise summary using pretrained BART or T5 transformer models.")

# ---------------------------------------------------------------------------
# Sidebar: model choice + generation parameters
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")

    model_name = st.selectbox("Model", list(MODEL_DEFAULTS.keys()))
    defaults = MODEL_DEFAULTS[model_name]["config"]

    st.caption(f"Running on: **{get_device().type.upper()}**")

    st.subheader("Generation parameters")
    num_beams = st.number_input("Number of beams", min_value=1, max_value=10, value=defaults.num_beams)
    no_repeat_ngram_size = st.number_input(
        "No-repeat n-gram size", min_value=0, max_value=6, value=defaults.no_repeat_ngram_size
    )
    length_penalty = st.number_input(
        "Length penalty", min_value=0.1, max_value=5.0, value=float(defaults.length_penalty), step=0.1
    )
    min_length = st.number_input("Min summary length", min_value=1, max_value=500, value=defaults.min_length)
    max_length = st.number_input("Max summary length", min_value=1, max_value=1000, value=defaults.max_length)
    early_stopping = st.checkbox("Early stopping", value=defaults.early_stopping)

    st.divider()
    st.caption("Tip: higher length penalty favors longer summaries; more beams can improve quality but is slower.")

# ---------------------------------------------------------------------------
# Main area: input + output
# ---------------------------------------------------------------------------
col_input, col_output = st.columns(2)

with col_input:
    st.subheader("Input text")
    text = st.text_area("Paste the text you want summarized", height=350, placeholder="Paste an article, report, or any long text here...")
    st.caption(f"{len(text)} characters")
    generate_clicked = st.button("Summarize", type="primary", use_container_width=True)

with col_output:
    st.subheader("Summary")
    output_placeholder = st.empty()

    if generate_clicked:
        if not text.strip():
            st.warning("Please paste some text before summarizing.")
        else:
            gen_config = GenerationConfig(
                num_beams=int(num_beams),
                no_repeat_ngram_size=int(no_repeat_ngram_size),
                length_penalty=float(length_penalty),
                min_length=int(min_length),
                max_length=int(max_length),
                early_stopping=bool(early_stopping),
            )
            with st.spinner(f"Loading {model_name} and generating summary..."):
                try:
                    summary = summarize(model_name, text, gen_config)
                    output_placeholder.success(summary)
                    st.caption(f"{len(summary)} characters "
                               f"({round(100 * len(summary) / max(len(text), 1), 1)}% of original length)")
                except Exception as exc:  # surface a clean message instead of a stack trace
                    st.error(f"Summarization failed: {exc}")
    else:
        output_placeholder.info("Your summary will appear here.")
