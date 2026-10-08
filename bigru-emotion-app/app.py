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

# Verified mapping for the trained BiGRU emotion classifier (dair-ai/emotion benchmark)
EMOTION_LABELS = {
    0: "Sadness",
    1: "Joy",
    2: "Love",
    3: "Anger",
    4: "Fear",
    5: "Surprise",
}

EMOTION_EMOJIS = {
    0: "😢",
    1: "😊",
    2: "❤️",
    3: "😠",
    4: "😨",
    5: "😲",
}

EXAMPLE_TEXTS = [
    ("😊 Joy", "I feel so happy and excited about this wonderful day!"),
    ("😢 Sadness", "I feel so alone, sad, and hopeless today."),
    ("❤️ Love", "I feel deeply in love and blessed to have you in my life."),
    ("😠 Anger", "I am furious and angry that they cancelled the trip at the last minute."),
    ("😨 Fear", "I feel terrified and scared when walking down dark alleyways alone."),
    ("😲 Surprise", "I was shocked and completely surprised by the unexpected gift!"),
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
    .hero-container {
        text-align: center;
        padding: 2rem 0 3rem 0;
    }
    .hero-title {
        font-weight: 800;
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        font-size: 1.2rem;
        opacity: 0.7;
    }
    .result-card { 
        background: linear-gradient(135deg, #6366f1 0%, #4338ca 100%);
        border-radius: 16px; 
        padding: 2rem; 
        color: white;
        text-align: center;
        box-shadow: 0 10px 25px rgba(99, 102, 241, 0.2); 
    }
    .result-caption { 
        font-size: 0.85rem;
        text-transform: uppercase; 
        letter-spacing: 0.1em; 
        font-weight: 600;
        opacity: 0.9;
    }
    .result-label { 
        font-size: 2.5rem;
        font-weight: 800; 
        margin: 0.5rem 0; 
    }
    .result-confidence { 
        font-size: 1.2rem; 
        font-weight: 600; 
    }
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
    # Uses the dictionary defined at the top of the file to map index to emotion name.
    return EMOTION_LABELS.get(index, f"Class {index}")


def show_sidebar(model: Any, tokenizer: Any) -> None:
    with st.sidebar:
        st.markdown("## ⚙️ Model Information")
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
            st.caption(", ".join(f"{EMOTION_EMOJIS.get(index, '')} {class_name(index)}" for index in range(class_count)))
        except ValueError as exc:
            st.error(f"Model information unavailable: {exc}")

        st.divider()
        st.markdown("### Preprocessing")
        st.caption(f"Tokenizer type: `{type(tokenizer).__name__}`")
        st.caption("Label mapping: Powered by internal EMOTION_LABELS dictionary.")


# UI Hero Section
st.markdown(
    """
    <div class="hero-container">
      <div class="hero-title">🧠 BiGRU Emotion Analyzer</div>
      <div class="hero-subtitle">AI-powered emotion classification using a trained Bidirectional GRU neural network</div>
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

# Wrapped input section in a clean container
with st.container(border=True):
    st.markdown("### 📝 Analyze Text")
    st.caption("Select an example below to place it in the text box, or type your own.")
    
    example_columns = st.columns(len(EXAMPLE_TEXTS))
    for index, (button_label, text) in enumerate(EXAMPLE_TEXTS):
        if example_columns[index].button(button_label, key=f"example_{index}", use_container_width=True):
            st.session_state["user_text"] = text
            st.session_state.pop("prediction_result", None)

    with st.form("emotion_analyzer", clear_on_submit=False):
        user_text = st.text_area(
            "Enter your text",
            key="user_text",
            height=140,
            label_visibility="collapsed",
            placeholder="e.g., I can't believe how happy I am right now, this is amazing!",
            help="Your text is passed to the trained model as plain text.",
        )
        
        # Center the submit button visually using columns
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze = st.form_submit_button(
                "✨ Analyze Emotion",
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
    st.markdown("---")
    st.markdown("### 📊 Analysis Results")
    
    left, right = st.columns([1, 1.2], gap="large")
    predicted_idx = result["predicted_index"]
    predicted_emoji = EMOTION_EMOJIS.get(predicted_idx, "")
    predicted_name = class_name(predicted_idx).upper()
    with left:
        st.markdown(
            f"""
            <div class="result-card">
              <div class="result-caption">Predicted Emotion</div>
              <div class="result-label">{predicted_emoji} {predicted_name}</div>
              <div class="result-caption" style="margin-top: 1rem;">Confidence Score</div>
              <div class="result-confidence">{result['confidence']:.1%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown("**Class Probabilities**")
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
            color_continuous_scale=["#a5b4fc", "#4f46e5"],
            range_x=[0, 1],
        )
        chart.update_traces(
            texttemplate="%{x:.1%}",
            textposition="outside",
            cliponaxis=False,
            marker_line_width=0,
        )
        chart.update_layout(
            height=280,
            margin=dict(l=0, r=40, t=10, b=10),
            xaxis=dict(title=None, tickformat=".0%", showgrid=True, gridcolor="rgba(128,128,128,0.2)"),
            yaxis=dict(title=None, categoryorder="total ascending"),
            coloraxis_showscale=False,
            plot_bgcolor="rgba(0,0,0,0)",  # Transparent for light/dark mode support
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(chart, use_container_width=True, config={"displayModeBar": False})

    with st.expander("🔢 View Tokenization & Under the Hood"):
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

st.markdown("---")
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
        """
    )