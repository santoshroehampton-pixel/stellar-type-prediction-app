import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Stellar Dashboard & Spectral Type Prediction",
    page_icon="🌌",
    layout="wide",
)

st.markdown("""
<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}
.block-container {
    padding-top:0.5rem;
    padding-bottom:0rem;
    padding-left:1rem;
    padding-right:1rem;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv("cleaned_host_star_dataset.csv")

@st.cache_resource
def load_artifacts():
    model = joblib.load("stellar_type_prediction_model.pkl")
    encoder = joblib.load("label_encoder.pkl")
    return model, encoder

df = load_data()
model, encoder = load_artifacts()

st.title("🌌 Stellar & Exoplanet Analysis Dashboard")

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Stellar Dashboard",
    page_icon="🌌",
    layout="wide"
)


# Removing Streamlit padding
st.markdown(
    """
    <style>

    #MainMenu {visibility:hidden;}
    footer {visibility:hidden;}
    header {visibility:hidden;}

    .block-container {
        padding-top:0.5rem;
        padding-bottom:0rem;
        padding-left:1rem;
        padding-right:1rem;
    }

    div[data-testid="stVerticalBlock"] {
        gap:0.3rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)



# ==================================================
# LOAD DATA
# ==================================================

df = pd.read_csv(r"cleaned_host_star_dataset.csv")


# ==================================================
# SIDEBAR FILTER
# ==================================================

st.sidebar.title("🔎 Filters")


spectral_filter = st.sidebar.multiselect(
    "Spectral Type",
    options=df["st_spectype"].dropna().unique(),
    default=df["st_spectype"].dropna().unique()
)


df = df[
    df["st_spectype"].isin(spectral_filter)
]



# ==================================================
# TITLE
# ==================================================

st.title("🌌 Stellar & Exoplanet Analysis Dashboard")



# ==================================================
# KPI CARDS
# ==================================================

k1,k2,k3,k4 = st.columns(4)


with k1:
    st.metric(
        "⭐ Total Stars",
        f"{len(df):,}"
    )


with k2:
    st.metric(
        "🌈 Spectral Types",
        df["st_spectype"].nunique()
    )


with k3:
    st.metric(
        "🌡 Avg Temperature",
        f"{df['st_teff'].mean():,.0f} K"
    )


with k4:
    st.metric(
        "⚖ Avg Mass",
        f"{df['st_mass'].mean():.2f}"
    )



# ==================================================
# CHART 1
# Spectral Type Distribution
# ==================================================

spectral_counts = (
    df["st_spectype"]
    .value_counts()
    .reset_index()
)

spectral_counts.columns = [
    "Spectral Type",
    "Count"
]


fig1 = px.bar(
    spectral_counts,
    x="Spectral Type",
    y="Count",
    color="Spectral Type",
    text="Count",
    height=280,
    title="⭐ Stellar Spectral Types"
)


fig1.update_layout(
    margin=dict(l=10,r=10,t=40,b=10)
)



# ==================================================
# CHART 2
# Discovery Methods
# ==================================================

method_counts = (
    df["discoverymethod"]
    .value_counts()
    .reset_index()
)

method_counts.columns=[
    "Method",
    "Count"
]


fig2 = px.bar(
    method_counts,
    x="Method",
    y="Count",
    text="Count",
    color="Method",
    height=280,
    title="🔭 Discovery Methods"
)


fig2.update_layout(
    margin=dict(l=10,r=10,t=40,b=10),
    xaxis_tickangle=-35
)



# ==================================================
# CHART 3
# Discovery Trend
# ==================================================

year_counts = (
    df["disc_year"]
    .value_counts()
    .sort_index()
    .reset_index()
)

year_counts.columns=[
    "Year",
    "Discoveries"
]


fig3 = px.line(
    year_counts,
    x="Year",
    y="Discoveries",
    markers=True,
    height=280,
    title="📈 Discovery Trend"
)


fig3.update_layout(
    margin=dict(l=10,r=10,t=40,b=10)
)



# ==================================================
# CHART 4
# Temperature vs Mass
# ==================================================

fig4 = px.scatter(
    df,
    x="st_teff",
    y="st_mass",
    color="st_spectype",
    hover_data=df.columns,
    height=280,
    title="🌡 Temperature vs Mass"
)


fig4.update_layout(
    margin=dict(l=10,r=10,t=40,b=10)
)



# ==================================================
# CHART 5
# Mass vs Radius
# ==================================================

fig5 = px.scatter(
    df,
    x="st_mass",
    y="st_rad",
    color="st_spectype",
    hover_data=df.columns,
    height=280,
    title="⭐ Mass vs Radius"
)


fig5.update_layout(
    margin=dict(l=10,r=10,t=40,b=10)
)



# ==================================================
# CHART 6
# Correlation Heatmap
# ==================================================

corr = df[
    [
        "st_teff",
        "st_rad",
        "st_mass",
        "st_met",
        "st_logg",
        "sy_dist"
    ]
].corr()


fig6 = go.Figure(
    data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale="RdBu",
        text=corr.round(2).values,
        texttemplate="%{text}"
    )
)


fig6.update_layout(
    title="🔥 Correlation Heatmap",
    height=280,
    margin=dict(l=10,r=10,t=40,b=10)
)



# ==================================================
# DASHBOARD GRID
# ==================================================

row1 = st.columns(3)

with row1[0]:
    st.plotly_chart(
        fig1,
        use_container_width=True
    )


with row1[1]:
    st.plotly_chart(
        fig2,
        use_container_width=True
    )


with row1[2]:
    st.plotly_chart(
        fig3,
        use_container_width=True
    )



row2 = st.columns(3)

with row2[0]:
    st.plotly_chart(
        fig4,
        use_container_width=True
    )


with row2[1]:
    st.plotly_chart(
        fig5,
        use_container_width=True
    )


with row2[2]:
    st.plotly_chart(
        fig6,
        use_container_width=True
    )


st.divider()

st.header("⭐ Stellar Spectral Type Prediction")
st.write(
    "Enter the astrophysical measurements below to predict the star's spectral type."
)

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
    input_data = np.array([[
        st_teff,
        st_rad,
        st_mass,
        st_met,
        st_logg,
        sy_dist,
        sy_vmag,
        sy_kmag,
        sy_gaiamag,
    ]])

    prediction = model.predict(input_data)
    predicted_class = encoder.inverse_transform(prediction)[0]
    st.success(f"Predicted spectral type: {predicted_class}")
