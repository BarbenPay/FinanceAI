import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import os

# --- 1. CONFIGURATION ---
TICKER_TO_FINETUNE = 'TSLA'
GENERAL_MODEL_PATH = 'ultimate_hourly_model.keras' # Notre nouveau champion !
DATA_DIR = 'donnees_horaire_ULTRA_enrichies'
FINETUNED_MODEL_SAVE_PATH = f'ultimate_finetuned_model_{TICKER_TO_FINETUNE}.keras'

print(f"--- Affinage ULTIME du modèle horaire pour : {TICKER_TO_FINETUNE} ---")
print(f"Chargement du modèle généraliste depuis : {GENERAL_MODEL_PATH}")

# --- 2. CHARGEMENT DU MODÈLE ET DES DONNÉES SPÉCIFIQUES ---
try:
    model = load_model(GENERAL_MODEL_PATH)
except Exception as e:
    print(f"ERREUR : Le modèle '{GENERAL_MODEL_PATH}' est introuvable. Avez-vous lancé le tuner ultime ?")
    exit()

# Charger les données ultra-enrichies pour notre action cible
data_path = os.path.join(DATA_DIR, f"{TICKER_TO_FINETUNE}_hourly_data.csv") # Attention au nom de fichier
if not os.path.exists(data_path):
    print(f"ERREUR : Le fichier de données '{data_path}' est introuvable.")
    exit()

df = pd.read_csv(data_path, index_col='datetime', parse_dates=True)
df.dropna(inplace=True)

# Définition automatique des features pour être 100% compatible avec le modèle chargé
features = df.columns.drop(['Target', 'adj close'])
print(f"Préparation des données avec {len(features)} features.")
X = df[features]
y = df['Target']

# --- 3. PRÉPARATION DES SÉQUENCES (pour une seule action) ---
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)

look_back = 70 # Doit être le même que celui utilisé pour entraîner le modèle généraliste
X_sequences, y_sequences = [], []
for i in range(len(X_scaled) - look_back):
    X_sequences.append(X_scaled[i:i+look_back])
    y_sequences.append(y.iloc[i + look_back - 1])
X_sequences = np.array(X_sequences)
y_sequences = np.array(y_sequences)

# Division SANS mélange pour préserver la chronologie de l'action
X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_sequences, test_size=0.2, shuffle=False, random_state=42)
print(f"Données de {TICKER_TO_FINETUNE} prêtes. Entraînement sur {len(X_train)} séquences.")

# --- 4. RE-COMPILATION ET AFFINAGE ---
# On utilise un faible taux d'apprentissage pour ne pas "casser" les connaissances du modèle généraliste
low_learning_rate_optimizer = Adam(learning_rate=0.0001) 
model.compile(optimizer=low_learning_rate_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

print("\n--- Début de l'affinage final (fine-tuning) ---")
# On entraîne en surveillant la performance sur le set de test à chaque étape
history = model.fit(
    X_train, y_train,
    epochs=20, # 20 epochs devraient nous montrer si une amélioration est possible
    batch_size=32,
    validation_data=(X_test, y_test)
)

# --- 5. ÉVALUATION FINALE ET VERDICT ---
print("\n--- Évaluation du champion affiné sur les données de test ---")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Perte finale (Loss) : {loss:.4f}")
print(f"Précision finale (Accuracy) : {accuracy*100:.2f}%")

model.save(FINETUNED_MODEL_SAVE_PATH)
print(f"\nModèle champion affiné pour {TICKER_TO_FINETUNE} sauvegardé dans : {FINETUNED_MODEL_SAVE_PATH}")

print("\n--- COMPARAISON ---")
print(f"Meilleur score du spécialiste JOURNALIER : ~55.65%")
print(f"Score final de ce spécialiste HORAIRE : {accuracy*100:.2f}%")
if accuracy*100 > 55.65:
    print("\nVICTOIRE ! Le modèle horaire spécialisé est le nouveau champion absolu !")
else:
    print("\nLe modèle journalier spécialisé reste le plus performant pour cette tâche.")