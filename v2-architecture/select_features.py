import pandas as pd
import numpy as np
import glob
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIG ---
DATA_DIR = 'data_processed'
SAMPLE_SIZE = 100000 # On prend un échantillon pour aller vite

print("--- ANALYSE D'IMPORTANCE DES FEATURES (SOTA) ---")

# 1. Chargement des données en vrac
files = glob.glob(os.path.join(DATA_DIR, "*_final.csv"))
data_list = []

for f in files[:10]: # On prend 10 fichiers représentatifs
    df = pd.read_csv(f, index_col=0, parse_dates=True)
    data_list.append(df)

full_df = pd.concat(data_list)
full_df.replace([np.inf, -np.inf], np.nan, inplace=True)
full_df.dropna(inplace=True)

# On équilibre les classes pour ne pas biaiser l'analyse
g = full_df.groupby('Target')
full_df = g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True))

print(f"Données analysées : {len(full_df)} lignes")

# 2. Random Forest (Le Juge)
features = [c for c in full_df.columns if c not in ['Target']]
X = full_df[features]
y = full_df['Target']

print("Entraînement du Random Forest pour scanner les features...")
rf = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
rf.fit(X, y)

# 3. Résultats
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

print("\n--- CLASSEMENT DES FEATURES ---")
top_features = []
for f in range(X.shape[1]):
    print(f"{f+1}. {features[indices[f]]:<20} ({importances[indices[f]]:.4f})")
    if f < 15: # On garde le TOP 15
        top_features.append(features[indices[f]])

# Sauvegarde de la liste pour le modèle
joblib.dump(top_features, 'selected_features.pkl')
print(f"\n✅ TOP 15 Features sauvegardées dans 'selected_features.pkl'")

# Visualisation
plt.figure(figsize=(12, 8))
sns.barplot(x=importances[indices], y=[features[i] for i in indices])
plt.title("Importance des Features (Random Forest)")
plt.savefig("feature_importance.png")
print("Graphique 'feature_importance.png' généré.")