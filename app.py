"""Streamlit app: predict a penguin's species from its measurements.

Run locally:  streamlit run app.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "models" / "penguin_model.joblib"

st.set_page_config(
    page_title="Palmer Penguins Classifier",
    page_icon="🐧",
    layout="centered"
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.title("🐧 Palmer Penguins — Species Predictor")
st.write(
    "Enter a penguin's physical measurements to predict its species. "
    "Model: **Logistic Regression** with feature engineering (`bill_ratio`) and robust pre-processing "
    "trained on the Palmer Penguins dataset (≈ **99.3 %** cross-validated macro F1 score)."
)

if not MODEL_PATH.exists():
    st.error("Model file not found. Run `python scripts/train_model.py` first.")
    st.stop()

model = load_model()

# Sidebar input controls
st.sidebar.header("📏 Measurements")
st.sidebar.markdown("Adjust the morphological values below:")

bill_length = st.sidebar.slider("Bill length (mm)", 30.0, 62.0, 44.0, 0.1)
bill_depth = st.sidebar.slider("Bill depth (mm)", 12.0, 23.0, 17.0, 0.1)
flipper = st.sidebar.slider("Flipper length (mm)", 165, 235, 200)
mass = st.sidebar.slider("Body mass (g)", 2500, 6500, 4000, 50)
sex = st.sidebar.selectbox("Sex (optional)", ["Unknown", "Male", "Female"])

st.sidebar.markdown("---")
st.sidebar.markdown(
    "[📂 GitHub Repository](https://github.com/farah-assia/palmer-penguins-eda)  \n"
    "Created by **Assia Farah**"
)

# Feature engineering row matching training pipeline
bill_ratio = bill_length / bill_depth

row = pd.DataFrame([{
    "bill_length_mm": bill_length,
    "bill_depth_mm": bill_depth,
    "flipper_length_mm": float(flipper),
    "body_mass_g": float(mass),
    "bill_ratio": bill_ratio,
    "sex": np.nan if sex == "Unknown" else sex.upper(),
}])

# Inference
proba = model.predict_proba(row)[0]
probs = pd.Series(proba, index=model.classes_).sort_values(ascending=False)
predicted_species = probs.index[0]
confidence = probs.iloc[0] * 100

# Species styling & descriptions
species_info = {
    "Adelie": {
        "color": "🔴",
        "description": "Short and deep bill, smaller body mass, ubiquitous across Biscoe, Dream, and Torgersen islands."
    },
    "Chinstrap": {
        "color": "🔵",
        "description": "Longer culmen, distinctive facial marking, intermediate body mass, nesting on Dream island."
    },
    "Gentoo": {
        "color": "🟢",
        "description": "Noticeably larger body mass (≈ +37%), long flippers (> 205 mm), slender beak, exclusively on Biscoe island."
    }
}

info = species_info.get(predicted_species, {"color": "🐧", "description": ""})

# Display prediction
st.markdown("---")
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"Prediction: {info['color']} **{predicted_species}**")
    st.write(info["description"])

with col2:
    st.metric(label="Model Confidence", value=f"{confidence:.1f} %")

# Diagnostic warning
if probs.iloc[0] < 0.8:
    st.warning("⚠️ The model is not very confident: these measurements fall near morphological boundary regions between species.")

# Probability breakdown
st.markdown("### 📊 Probability Distribution")
st.bar_chart(probs)

with st.expander("🔍 View Computed Input Vector"):
    display_row = row.copy()
    display_row["sex"] = sex
    st.dataframe(display_row.round(2))
    st.caption(f"Engineered feature `bill_ratio`: {bill_ratio:.3f}")

st.caption(
    "**Limitations & Scientific Framing:** Trained on 342 penguins from the Palmer Station LTER dataset "
    "(Gorman et al., 2014, PLoS ONE). Measurements far outside the natural biological range may yield unreliable results. "
    "Educational machine learning project, not an operational ecological survey tool."
)
