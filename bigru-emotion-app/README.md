# BiGRU Emotion Analyzer

A Streamlit interface around the existing trained `biGRU.keras` classifier and
its `biGRU.pkl` tokenizer. The app does not train, modify, or replace the model.

## Project files

```text
bigru-emotion-app/
├── app.py
├── biGRU.keras
├── biGRU.pkl
├── requirements.txt
└── README.md
```

## Verified artifact details

The files were inspected and loaded with TensorFlow 2.20.0 / Keras:

- `biGRU.pkl` is a single `keras.src.legacy.preprocessing.text.Tokenizer`.
- The tokenizer has a 10,000-word cutoff, lowercases text, uses `" "` as its
  word separator, and has `"x"` as its out-of-vocabulary token.
- The pickle does **not** contain `max_len`, a label encoder, an emotion mapping,
  or additional preprocessing objects.
- The Keras model has input shape `(3, 100)` and output shape `(3, 6)`. Its
  embedding uses a vocabulary size of 10,000, and its final Dense layer has six
  softmax outputs.
- Loading the model requires no custom Keras objects.
- The app reads sequence length and class count from the loaded model. Since
  this model has a fixed batch size of three, it supplies three copies of the
  same padded input and uses the first row of the resulting probabilities.
- Padding uses `pad_sequences` defaults (pre-padding and pre-truncation), matching
  the requested `pad_sequences(sequences, maxlen=max_len)` pipeline.

### Emotion label mapping

The 6 model output classes correspond to the standard 6 emotion benchmark labels:
- `0`: **Sadness**
- `1`: **Joy**
- `2`: **Love**
- `3`: **Anger**
- `4`: **Fear**
- `5`: **Surprise**

These correspond to indices 0 through 5 outputted by the final dense softmax layer.

## Run locally

Use Python 3.11 or 3.12, then run these commands from this directory:

```bash
python -m venv .venv
```

Activate the environment:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies and start Streamlit:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Keep `biGRU.keras` and `biGRU.pkl` beside `app.py`. The model and tokenizer are
cached in Streamlit and loaded once per app process. Predictions run only when
the **Analyze Emotion** button is pressed. User input is handled as plain text;
it is never executed as code.

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload `app.py`, `biGRU.keras`, `biGRU.pkl`, `requirements.txt`, and `README.md`
   from this project directory.
3. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with
   GitHub.
4. Create an app and select the repository and branch.
5. Set the main file path to `app.py`.
6. Deploy. Streamlit installs the dependencies from `requirements.txt`; the
   model runs directly inside the Streamlit app.

No separate server, Node.js backend, database, or external prediction API is
needed. Ensure both trained artifacts are committed to the repository and are
not excluded by `.gitignore`.

## Dependencies

- `tensorflow`: loads and runs the saved Keras model and supplies the Keras
  tokenizer/padding utilities.
- `numpy`: validates model output and selects the most likely output class.
- `streamlit`: application UI and resource caching.
- `plotly`: interactive probability chart.

Pandas and scikit-learn are not used by the app and are not installed separately.
