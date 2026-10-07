"""Streamlit interface for the existing trained BiGRU classifier."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import plotly.express as px
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "biGRU.keras"
PREPROCESSOR_PATH = APP_DIR / "biGRU.pkl"

EXAMPLE_TEXTS = [
    "I can't believe how happy I am right now, this is amazing!",
    "I feel so alone and hopeless today.",
    "I am furious that they cancelled the trip at the last minute.",
    "I feel terrified when walking down dark alleyways alone.",
    "I was shocked and completely surprised by the unexpected gift!",
]

st.set_page_config(
    page_title="BiGRU Emotion Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    :root { --ink: #17233b; --muted: #64748b; --accent: #6558e8; }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: linear-gradient(180deg, #f7f8ff 0%, #f8fafc 42%, #ffffff 100%); }
    .block-container { max-width: 1180px; padding-top: 2.3rem; padding-bottom: 4rem; }
    h1, h2, h3 { font-family: 'Manrope', sans-serif; color: var(--ink); }
    .hero { padding: 1.35rem 0 1.2rem; }
    .eyebrow { color: var(--accent); text-transform: uppercase; letter-spacing: .13em;
               font-size: .76rem; font-weight: 700; }
    .subtitle { color: var(--muted); font-size: 1.05rem; margin-top: -.3rem; }
    .result-card { background: linear-gradient(135deg, #6558e8 0%, #5548ce 100%);
                   border-radius: 22px; padding: 1.7rem 1.9rem; color: white;
                   box-shadow: 0 18px 45px rgba(83, 72, 206, .19); }
    .result-caption { color: rgba(255,255,255,.76); font-size: .85rem;
                      text-transform: uppercase; letter-spacing: .1em; font-weight: 700; }
    .result-label { font: 800 2rem 'Manrope', sans-serif; margin: .35rem 0 .7rem; }
    .result-confidence { font-size: 1.2rem; font-weight: 700; }
    .soft-card { background: #fff; border: 1px solid #e8eaf2; border-radius: 18px;
                 padding: 1.25rem 1.4rem; box-shadow: 0 8px 24px rgba(25, 35, 60, .045); }
    .small-muted { color: #64748b; font-size: .9rem; }
    div.stButton > button { border-radius: 11px; font-weight: 600; }
    div.stButton > button[kind="primary"] { background: #6558e8; border-color: #6558e8; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the trained BiGRU model…")
def load_artifacts() -> tuple[Any, Any, str | None]:
    """Load the trusted local model and its saved tokenizer once per app process."""
    if not MODEL_PATH.is_file():
        return None, None, f"Model file not found: `{MODEL_PATH.name}`."
    if not PREPROCESSOR_PATH.is_file():
        return None, None, f"Preprocessing file not found: `{PREPROCESSOR_PATH.name}`."

    try:
        with PREPROCESSOR_PATH.open("rb") as artifact_file:
            tokenizer = pickle.load(artifact_file)
    except Exception as exc:
        return None, None, (
            "The preprocessing file could not be loaded. Ensure it is the trusted "
            f"`biGRU.pkl` created with a compatible Keras version. ({type(exc).__name__})"
        )

    try:
        from tensorflow.keras.models import load_model

        model = load_model(MODEL_PATH)
    except Exception as exc:
        return None, None, (
            "The trained model could not be loaded. Check the TensorFlow/Keras "
            f"runtime compatibility. ({type(exc).__name__})"
        )

    return model, tokenizer, None


def single_shape(shape: Any, description: str) -> tuple[Any, ...]:
    """Return one model input/output shape and reject multi-input/output models."""
    if isinstance(shape, list):
        if len(shape) != 1:
            raise ValueError(f"This app expects one model {description}.")
        shape = shape[0]
    if shape is None:
        raise ValueError(f"The model's {description} shape is unavailable.")
    return tuple(shape)


def model_details(model: Any) -> tuple[int | None, int | None, tuple[Any, ...]]:
    input_shape = single_shape(model.input_shape, "input")
    output_shape = single_shape(model.output_shape, "output")
    if len(input_shape) != 2:
        raise ValueError("The model must accept 2D token sequences: (batch, sequence length).")
    if len(output_shape) != 2 or output_shape[-1] is None:
        raise ValueError("The model must return a 2D class-probability array.")
    sequence_length = int(input_shape[-1]) if input_shape[-1] is not None else None
    class_count = int(output_shape[-1])
    return sequence_length, class_count, input_shape


def run_prediction(
    model: Any, tokenizer: Any, text: str, sequence_length: int | None, input_shape: tuple[Any, ...]
) -> tuple[np.ndarray, list[int], list[int]]:
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=sequence_length)
    if padded.ndim != 2 or padded.shape[0] != 1:
        raise ValueError("The tokenizer did not produce a single valid sequence.")

    fixed_batch = input_shape[0]
    batch_size = int(fixed_batch) if fixed_batch is not None else 1
    if batch_size < 1:
        raise ValueError("The model declares an invalid batch size.")
    model_input = np.repeat(padded, batch_size, axis=0)
    raw_probabilities = np.asarray(model.predict(model_input, verbose=0))
    if raw_probabilities.ndim != 2 or raw_probabilities.shape[0] != batch_size:
        raise ValueError("The model returned an unexpected prediction shape.")
    probabilities = raw_probabilities[0].astype(float)
    if probabilities.size == 0 or not np.all(np.isfinite(probabilities)):
        raise ValueError("The model returned empty or invalid prediction probabilities.")

    token_ids = [int(token) for token in sequences[0]]
    padded_ids = [int(token) for token in padded[0].tolist()]
    return probabilities, token_ids, padded_ids


def class_name(index: int) -> str:
    # The saved pickle contains a tokenizer only, so names are not available.
    return f"Class {index}"


def show_sidebar(model: Any, tokenizer: Any) -> None:
    with st.sidebar:
        st.markdown("## Model Information")
        st.markdown(
            """
            **Model:** BiGRU  
            **Framework:** TensorFlow / Keras  
            **Task:** Emotion classification  
            **Input:** Text  
            **Output:** Class probabilities
            """
        )

        try:
            sequence_length, class_count, input_shape = model_details(model)
            st.markdown(f"**Output classes:** {class_count}")
            st.markdown(f"**Input shape:** `{input_shape}`")
            if sequence_length is not None:
                st.markdown(f"**Sequence length:** {sequence_length}")
            st.markdown("**Available output labels:**")
            st.caption(", ".join(class_name(index) for index in range(class_count)))
        except ValueError as exc:
            st.error(f"Model information unavailable: {exc}")

        st.divider()
        st.markdown("### Preprocessing")
        st.caption(f"Tokenizer type: `{type(tokenizer).__name__}`")
        st.caption("Label mapping: not present in the saved pickle.")


st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Emotion intelligence · powered by your trained model</div>
      <h1>🧠 BiGRU Emotion Analyzer</h1>
      <div class="subtitle">AI-powered emotion classification using a trained Bidirectional GRU neural network</div>
    </div>
    """,
    unsafe_allow_html=True,
)

model, tokenizer, load_error = load_artifacts()
if load_error:
    st.error(load_error)
    st.info("Place `biGRU.keras` and `biGRU.pkl` next to `app.py`, then restart the app.")
    st.stop()

show_sidebar(model, tokenizer)

st.markdown("### Try an example")
st.caption("Select an example to place it in the text box, then press Analyze Emotion.")
example_columns = st.columns(5)
for index, text in enumerate(EXAMPLE_TEXTS):
    short_label = f"Example {index + 1}"
    if example_columns[index].button(short_label, key=f"example_{index}", use_container_width=True):
        st.session_state["user_text"] = text
        st.session_state.pop("prediction_result", None)

with st.form("emotion_analyzer", clear_on_submit=False):
    user_text = st.text_area(
        "Enter your text",
        key="user_text",
        height=180,
        placeholder="I can't believe how happy I am right now, this is amazing!",
        help="Your text is passed to the trained model as plain text.",
    )
    analyze = st.form_submit_button(
        "✨  Analyze Emotion",
        type="primary",
        use_container_width=True,
    )

if analyze:
    if not user_text.strip():
        st.warning("Enter some text before analyzing.")
    else:
        try:
            sequence_length, class_count, input_shape = model_details(model)
            probabilities, token_ids, padded_ids = run_prediction(
                model, tokenizer, user_text, sequence_length, input_shape
            )
            if probabilities.size != class_count:
                raise ValueError(
                    "The number of predicted probabilities does not match the model's class count."
                )
            predicted_index = int(np.argmax(probabilities))
            confidence = float(np.max(probabilities))
            st.session_state["prediction_result"] = {
                "probabilities": probabilities.tolist(),
                "predicted_index": predicted_index,
                "confidence": confidence,
                "token_ids": token_ids,
                "padded_ids": padded_ids,
                "text": user_text,
            }
        except Exception as exc:
            st.error(
                "The text could not be analyzed. The input or model output may be "
                f"incompatible. ({type(exc).__name__})"
            )

result = st.session_state.get("prediction_result")
if result:
    st.markdown("")
    left, right = st.columns([0.9, 1.1], gap="large")
    with left:
        st.markdown(
            f"""
            <div class="result-card">
              <div class="result-caption">Predicted emotion · output index {result['predicted_index']}</div>
              <div class="result-label">{class_name(result['predicted_index']).upper()}</div>
              <div class="result-caption">Confidence</div>
              <div class="result-confidence">{result['confidence']:.1%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.warning(
            "Emotion names are not saved in `biGRU.pkl`; this result is shown by "
            "output index until the training label order is provided."
        )

    with right:
        st.markdown("### Class probabilities")
        chart_data = sorted(
            [
                {"Output class": class_name(index), "Probability": float(probability)}
                for index, probability in enumerate(result["probabilities"])
            ],
            key=lambda row: row["Probability"],
            reverse=True,
        )
        chart = px.bar(
            chart_data,
            x="Probability",
            y="Output class",
            orientation="h",
            color="Probability",
            color_continuous_scale=["#c8c4ff", "#6558e8"],
            range_x=[0, 1],
        )
        chart.update_traces(
            texttemplate="%{x:.1%}",
            textposition="outside",
            cliponaxis=False,
            marker_line_width=0,
        )
        chart.update_layout(
            height=330,
            margin=dict(l=4, r=34, t=8, b=8),
            xaxis=dict(title=None, tickformat=".0%", showgrid=True, gridcolor="#edf0f6"),
            yaxis=dict(title=None, categoryorder="total ascending"),
            coloraxis_showscale=False,
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(color="#334155"),
        )
        st.plotly_chart(chart, use_container_width=True, config={"displayModeBar": False})

    with st.expander("🔢 View Tokenization"):
        st.markdown("**Original text**")
        st.code(result["text"], language=None)
        st.markdown("**Tokenized sequence (token IDs)**")
        st.code(str(result["token_ids"]), language="text")
        st.markdown("**Padded sequence**")
        st.code(str(result["padded_ids"]), language="text")
        st.caption(
            "Tokenization and padding are generated from the saved tokenizer and "
            "the model's input sequence length."
        )

with st.expander("🔍 How does the model work?"):
    st.markdown(
        """
        **Text** → **Tokenizer** → **Integer Sequences** → **Padding** →  
        **Embedding** → **Bidirectional GRU** → **Dense Layer** → **Softmax** → **Prediction**

        - **Text:** You enter a sentence in ordinary language.
        - **Tokenizer:** The saved tokenizer turns known words into their trained integer IDs.
        - **Integer sequences:** Each word is represented as a number the neural network can use.
        - **Padding:** The sequence is padded or trimmed to the length expected by the model.
        - **Embedding:** Each word ID is represented as a learned numeric vector.
        - **Bidirectional GRU:** The recurrent layers process word context in both directions.
        - **Dense layer:** The final layer scores each output class.
        - **Softmax:** Scores are converted into class probabilities; the largest probability is selected.

        **Label limitation:** `biGRU.pkl` contains the tokenizer, but no emotion-label mapping.
        Until the original training label order is supplied, output indices are shown as
        `Class 0`, `Class 1`, and so on. These are not guessed emotion names.
        """
    )
