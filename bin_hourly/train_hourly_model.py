import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
import os
import glob
from tqdm import tqdm

print(f"--- Entraînement du Modèle Horaire Généraliste ---")
print(f"GPU Disponibles: {tf.config.list_physical_devices('GPU')}")

# --- 1. CHARGEMENT ET COMBINAISON DES DONNÉES "SUPER-ENRICHIES" ---

INPUT_DIR = 'donnees_horaire_enrichies'
all_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))

all_X_scaled_list = []
all_y_list = []
features = None # On le définira dynamiquement

print(f"Chargement et traitement de {len(all_files)} fichiers de données horaires...")

for file in tqdm(all_files, desc="Fichiers traités"):
    df = pd.read_csv(file, index_col='datetime', parse_dates=True)
    df.dropna(inplace=True)
    
    # Définition automatique des features : on prend tout sauf la cible et 'adj close' (redondant)
    if features is None:
        features = df.columns.drop(['Target', 'adj close'])
        print(f"Détection automatique de {len(features)} features: {list(features)}")
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(df[features])
    
    all_X_scaled_list.append(X_scaled)
    all_y_list.append(df['Target'])

combined_X_scaled = np.concatenate(all_X_scaled_list, axis=0)
combined_y = pd.concat(all_y_list)
print(f"Données finales combinées. Dimensions X: {combined_X_scaled.shape}")

# --- 2. CRÉATION DES SÉQUENCES HORAIRES ---

look_back = 70  # ~10 jours de bourse en heures
X_sequences, y_sequences = [], []

# Utilisation de tqdm pour la barre de progression sur cette longue opération
for i in tqdm(range(len(combined_X_scaled) - look_back), desc="Création des séquences"):
    X_sequences.append(combined_X_scaled[i:i+look_back])
    y_sequences.append(combined_y.iloc[i + look_back - 1])

X_sequences = np.array(X_sequences)
y_sequences = np.array(y_sequences)

print(f"Séquences créées. Dimensions X: {X_sequences.shape}")

# --- 3. SÉPARATION DES DONNÉES ---
X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_sequences, test_size=0.2, shuffle=True, random_state=42)
print(f"Données prêtes. Entraînement sur {len(X_train)} séquences.")

# --- 4. CONSTRUCTION DU MODÈLE HORAIRE ---
model = Sequential([
    Input(shape=(look_back, len(features))),
    # Couches LSTM plus grandes pour gérer plus de features
    LSTM(units=80, return_sequences=True), 
    Dropout(0.2),
    LSTM(units=80, return_sequences=False),
    Dropout(0.2),
    Dense(units=40, activation='relu'), # Couche dense un peu plus grande aussi
    Dense(units=1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# --- 5. ENTRAÎNEMENT ---
print("\n--- Début de l'entraînement du modèle horaire ---")
history = model.fit(
    X_train, y_train, 
    epochs=20, # On peut réduire un peu les epochs car le dataset est bien plus grand
    batch_size=128, # Batch size plus grand pour un entraînement plus rapide
    validation_split=0.1
)

# --- 6. ÉVALUATION ET SAUVEGARDE ---
print("\n--- Évaluation du modèle horaire sur les données de test ---")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Perte (Loss) sur le set de test : {loss:.4f}")
print(f"Précision (Accuracy) sur le set de test : {accuracy*100:.2f}%")

MODEL_SAVE_PATH = "hourly_general_model_v1.keras"
model.save(MODEL_SAVE_PATH)
print(f"\nModèle Horaire V1 sauvegardé dans : {MODEL_SAVE_PATH}")