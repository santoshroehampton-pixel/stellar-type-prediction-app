import joblib
import numpy as np
import streamlit as st


st.set_page_config(
    page_title="Stellar Spectral Type Prediction",
    page_icon="⭐",
    layout="centered",
)


@st.cache_resource
def load_artifacts():
    model = joblib.load("stellar_type_prediction_model.pkl")
    encoder = joblib.load("label_encoder.pkl")
    return model, encoder


st.title("Stellar Spectral Type Prediction")
st.write(
    "Enter the astrophysical measurements below to predict the star's spectral type."
)

try:
    model, encoder = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model files are missing. Add stellar_type_prediction_model.pkl and "
        "label_encoder.pkl to the same folder as app.py."
    )
    st.stop()


col1, col2 = st.columns(2)

with col1:
    st_teff = st.number_input("Effective temperature (K)", min_value=0.0, value=5778.0)
    st_rad = st.number_input("Stellar radius", min_value=0.0, value=1.0)
    st_mass = st.number_input("Stellar mass", min_value=0.0, value=1.0)
    st_met = st.number_input("Metallicity", value=0.0)
    st_logg = st.number_input("Surface gravity (log g)", value=4.44)

with col2:
    sy_dist = st.number_input("System distance (parsecs)", min_value=0.0, value=10.0)
    sy_vmag = st.number_input("Visual magnitude", value=5.0)
    sy_kmag = st.number_input("K-band magnitude", value=4.0)
    sy_gaiamag = st.number_input("Gaia magnitude", value=5.0)


if st.button("Predict stellar type", type="primary", use_container_width=True):
    input_data = np.array(
        [[
            st_teff,
            st_rad,
            st_mass,
            st_met,
            st_logg,
            sy_dist,
            sy_vmag,
            sy_kmag,
            sy_gaiamag,
        ]]
    )
    prediction = model.predict(input_data)
    predicted_class = encoder.inverse_transform(prediction)[0]
    st.success(f"Predicted spectral type: {predicted_class}")

