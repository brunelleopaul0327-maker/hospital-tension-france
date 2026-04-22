import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

# ── Configuration de la page ───────────────────────────────────────────────
st.set_page_config(
    page_title="Tension Hospitalière France",
    page_icon="🏥",
    layout="wide"
)

# ── Titre ──────────────────────────────────────────────────────────────────
st.title("🏥 Analyse et Prédiction de la Tension Hospitalière en France")
st.markdown("Prédiction du taux d'occupation des soins critiques à **J+7** par région")

# ── Chargement des données ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_parquet("data/processed/hopitaux_clean.parquet")

@st.cache_resource
def load_model():
    model = joblib.load("data/processed/model_final.joblib")
    features = joblib.load("data/processed/features_final.joblib")
    return model, features

df = load_data()
model, features = load_model()

# ── Noms des régions ───────────────────────────────────────────────────────
regions = {
    1: "Guadeloupe", 2: "Martinique", 3: "Guyane", 4: "La Réunion",
    6: "Mayotte", 11: "Île-de-France", 24: "Centre-Val de Loire",
    27: "Bourgogne-Franche-Comté", 28: "Normandie", 32: "Hauts-de-France",
    44: "Grand Est", 52: "Pays de la Loire", 53: "Bretagne",
    75: "Nouvelle-Aquitaine", 76: "Occitanie", 84: "Auvergne-Rhône-Alpes",
    93: "PACA", 94: "Corse"
}

# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.header("Paramètres")
region_nom = st.sidebar.selectbox("Choisir une région", list(regions.values()))
region_code = [k for k, v in regions.items() if v == region_nom][0]

# ── Filtrer les données pour la région ────────────────────────────────────
df_region = df[df["reg"] == region_code].sort_values("jour").copy()

# ── Graphique historique ───────────────────────────────────────────────────
st.subheader(f"📈 Historique — {region_nom}")
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df_region["jour"], df_region["tx_prev_SC"], color="steelblue")
ax.set_xlabel("Date")
ax.set_ylabel("Taux soins critiques")
ax.set_title(f"Taux d'occupation soins critiques — {region_nom}")
st.pyplot(fig)


# ── Prédiction J+7 ─────────────────────────────────────────────────────────
st.subheader(f"🔮 Prédiction J+7 — {region_nom}")

# Prendre les dernières valeurs connues pour faire la prédiction
derniere_ligne = df_region.iloc[-1]

# Calculer les features nécessaires
lag_1 = derniere_ligne["tx_prev_SC"]
tx_prev_hosp = derniere_ligne["tx_prev_hosp"]
tendance_7j = df_region["tx_prev_SC"].iloc[-1] - df_region["tx_prev_SC"].iloc[-7]
tendance_14j = df_region["tx_prev_SC"].iloc[-1] - df_region["tx_prev_SC"].iloc[-14]

# Créer le dataframe de prédiction
X_pred = pd.DataFrame([[lag_1, tx_prev_hosp, tendance_7j, tendance_14j, region_code]], 
                       columns=features)

prediction = model.predict(X_pred)[0]

# ── Niveau d'alerte ────────────────────────────────────────────────────────
if prediction < 2:
    niveau = "🟢 Faible"
    couleur = "green"
elif prediction < 4:
    niveau = "🟡 Modéré"
    couleur = "orange"
else:
    niveau = "🔴 Élevé"
    couleur = "red"

# ── Affichage ──────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
col1.metric("Taux actuel", f"{lag_1:.2f}")
col2.metric("Prédiction J+7", f"{prediction:.2f}", f"{prediction - lag_1:+.2f}")
col3.metric("Niveau d'alerte", niveau)






# ── Graphique prédiction vs historique récent ──────────────────────────────
st.subheader("📊 Historique récent et prédiction")

# 60 derniers jours
df_recent = df_region.tail(60).copy()
derniere_date = df_recent["jour"].iloc[-1]
date_prediction = derniere_date + pd.Timedelta(days=7)

fig2, ax2 = plt.subplots(figsize=(12, 4))
ax2.plot(df_recent["jour"], df_recent["tx_prev_SC"], color="steelblue", label="Historique")
ax2.scatter(date_prediction, prediction, color="red", zorder=5, s=100, label=f"Prédiction J+7 : {prediction:.2f}")
ax2.axvline(derniere_date, color="gray", linestyle="--", alpha=0.5, label="Aujourd'hui")
ax2.set_xlabel("Date")
ax2.set_ylabel("Taux soins critiques")
ax2.legend()
plt.tight_layout()
st.pyplot(fig2)