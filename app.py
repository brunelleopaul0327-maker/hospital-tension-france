import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error

# ── Configuration ───────────────────────────────────────────────
st.set_page_config(
    page_title="Validation Modèle Tension Hospitalière",
    page_icon="🏥",
    layout="wide"
)

# ── Titre ──────────────────────────────────────────────────────
st.title("🏥 Validation du Modèle de Tension Hospitalière")
st.markdown(
    "Comparaison des **prédictions à J+7** avec les **valeurs réelles sur 2023**, par région."
)

# ── Chargement des données ─────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_parquet("data/processed/hopitaux_clean.parquet")

df = load_data()

# ── Fonction de feature engineering ────────────────────────────
def create_features(df):
    df = df.copy()
    df = df.sort_values(["reg", "jour"]).reset_index(drop=True)
    
    df["lag_1"] = df.groupby("reg")["tx_prev_SC"].shift(1)
    df["lag_7"] = df.groupby("reg")["tx_prev_SC"].shift(7)
    df["lag_14"] = df.groupby("reg")["tx_prev_SC"].shift(14)
    
    df["rolling_7"] = df.groupby("reg")["tx_prev_SC"].rolling(7).mean().reset_index(level=0, drop=True)
    df["rolling_14"] = df.groupby("reg")["tx_prev_SC"].rolling(14).mean().reset_index(level=0, drop=True)
    
    df["rolling_std_7"] = df.groupby("reg")["tx_prev_SC"].rolling(7).std().reset_index(level=0, drop=True)
    
    df["tendance_7j"] = df.groupby("reg")["tx_prev_SC"].diff(7)
    
    return df

df = create_features(df)

# ── Features utilisées ─────────────────────────────────────────
features = [
    "lag_1", "lag_7", "lag_14",
    "rolling_7", "rolling_14",
    "rolling_std_7",
    "mois", "semaine",
    "tx_prev_hosp",
    "tendance_7j"
]

# ── Chargement modèle par région ───────────────────────────────
@st.cache_resource
def load_model(region_code):
    return joblib.load(f"data/processed/models_regions/model_region_{region_code}.joblib")

# ── Noms des régions ───────────────────────────────────────────
regions = {
    1: "Guadeloupe", 2: "Martinique", 3: "Guyane", 4: "La Réunion",
    6: "Mayotte", 11: "Île-de-France", 24: "Centre-Val de Loire",
    27: "Bourgogne-Franche-Comté", 28: "Normandie", 32: "Hauts-de-France",
    44: "Grand Est", 52: "Pays de la Loire", 53: "Bretagne",
    75: "Nouvelle-Aquitaine", 76: "Occitanie", 84: "Auvergne-Rhône-Alpes",
    93: "PACA", 94: "Corse"
}

# ── Sidebar ────────────────────────────────────────────────────
st.sidebar.header("Paramètres")
region_nom = st.sidebar.selectbox("Choisir une région", list(regions.values()))
region_code = [k for k, v in regions.items() if v == region_nom][0]

# ── Charger modèle ─────────────────────────────────────────────
model = load_model(region_code)

# ── Filtrer données région ─────────────────────────────────────
df_region = df[df["reg"] == region_code].copy()
df_region = df_region.sort_values("jour").reset_index(drop=True)

# ── Historique ─────────────────────────────────────────────────
st.subheader(f"📈 Historique — {region_nom}")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df_region["jour"], df_region["tx_prev_SC"], color="steelblue")
ax.set_xlabel("Date")
ax.set_ylabel("Taux soins critiques")
ax.set_title("Historique du taux d'occupation")
st.pyplot(fig)

# ── Prédictions 2023 ───────────────────────────────────────────
st.subheader(f"📊 Validation du modèle (2023) — {region_nom}")

# target
df_region["cible"] = df_region["tx_prev_SC"].shift(-7)

# données 2023
df_2023 = df_region[df_region["annee"] == 2023].copy()

# nettoyage
df_2023 = df_2023.dropna(subset=features + ["cible"])

# prédictions
X = df_2023[features]
df_2023["prediction"] = model.predict(X)

# recalage temporel
df_2023["date_cible"] = df_2023["jour"] + pd.Timedelta(days=7)

# ── Graphique réel vs prédiction ───────────────────────────────
fig2, ax2 = plt.subplots(figsize=(12, 4))

ax2.plot(
    df_2023["date_cible"],
    df_2023["cible"],
    label="Réel",
    color="black"
)

ax2.plot(
    df_2023["date_cible"],
    df_2023["prediction"],
    label="Prédiction",
    color="red"
)

ax2.set_title("Prédictions vs Réel (J+7)")
ax2.set_xlabel("Date")
ax2.set_ylabel("Taux soins critiques")
ax2.legend()

st.pyplot(fig2)

# ── Métriques ─────────────────────────────────────────────────
mae = mean_absolute_error(df_2023["cible"], df_2023["prediction"])
mean_value = df_2023["cible"].mean()

if mean_value != 0:
    mae_relatif = mae / mean_value
else:
    mae_relatif = np.nan

col1, col2 = st.columns(2)
col1.metric("MAE 2023", f"{mae:.3f}")
col2.metric("MAE relatif", f"{mae_relatif:.2%}" if not np.isnan(mae_relatif) else "N/A")

# ── Erreur dans le temps ───────────────────────────────────────
st.subheader("📉 Erreur du modèle")

df_2023["erreur"] = df_2023["prediction"] - df_2023["cible"]

st.line_chart(df_2023.set_index("date_cible")["erreur"])