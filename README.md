# 🏥 Analyse et Prédiction de la Tension Hospitalière en France

🔗 **[Demo live →](https://hospital-tension-france-rtetjsgxyw6c4ry3xtfqde.streamlit.app/)**

## Objectif
Prédire le taux d'occupation des soins critiques à **J+7** pour chaque région française, à partir de données ouvertes (data.gouv.fr).

## Résultats
| Modèle | MAE | RMSE |
|---|---|---|
| Random Forest (final) | 0.195 | 0.262 |

## Stack technique
| Catégorie | Outils |
|---|---|
| Data | pandas, pyarrow |
| ML | scikit-learn, xgboost, lightgbm |
| Dashboard | Streamlit |
| Versioning | Git, GitHub |

## Lancer le projet en local
```bash
git clone https://github.com/brunelleopaul0327-maker/hospital-tension-france.git
cd hospital-tension-france
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

## Source des données
- [data.gouv.fr — Données hospitalières COVID-19](https://www.data.gouv.fr/fr/datasets/donnees-hospitalieres-relatives-a-lepidemie-de-covid-19/)