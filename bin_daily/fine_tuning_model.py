import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model # Nouvelle importation !
from tensorflow.keras.optimizers import Adam # Pour spécifier le learning rate
import os

# --- 0. CONFIGURATION ---
TICKER_TO_FINETUNE = 'TSLA'
GENERAL_MODEL_PATH = 'financial_model_generalist_v1.keras'
FINETUNED_MODEL_SAVE_PATH = f'finetuned_model_{TICKER_TO_FINETUNE}.keras'


print(f"--- Affinage du modèle pour l'action : {TICKER_TO_FINETUNE} ---")
print(f"TensorFlow Version: {tf.__version__}")
print(f"Chargement du modèle généraliste depuis : {GENERAL_MODEL_PATH}")


# --- 1. CHARGER LE MODÈLE GÉNÉRALISTE ---
try:
    model = load_model(GENERAL_MODEL_PATH)
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")
    print("Assurez-vous que le fichier 'financial_model_generalist_v1.keras' est dans le même dossier.")
    exit()

model.summary()


# --- 2. PRÉPARER LES DONNÉES SPÉCIFIQUES À L'ACTION CIBLE (comme dans le 1er script) ---

FILE_PATH = os.path.join('donnees_finales', f'{TICKER_TO_FINETUNE}_final_data.csv')
df = pd.read_csv(FILE_PATH, index_col='Date', parse_dates=True)

features = ['Close', 'High', 'Low', 'Open', 'Volume', 'SMA_20', 'SMA_50', 'RSI_14', 'google_trend_score']
X = df[features]
y = df['Target']

# Important : On utilise un nouveau scaler juste pour cette action
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)

# Création des séquences
look_back = 60
X_sequences, y_sequences = [], []

for i in range(len(X_scaled) - look_back):
    X_sequences.append(X_scaled[i:i+look_back])
    y_sequences.append(y.iloc[i + look_back - 1])

X_sequences = np.array(X_sequences)
y_sequences = np.array(y_sequences)

# Division SANS mélange, car on affûte sur l'histoire de cette action
X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_sequences, test_size=0.2, shuffle=False)

print(f"Données de {TICKER_TO_FINETUNE} prêtes. Entraînement sur {len(X_train)} échantillons, test sur {len(X_test)}.")


# --- 3. RE-COMPILER LE MODÈLE AVEC UN FAIBLE LEARNING RATE ---
# C'est l'étape la plus importante de l'affinage !
# Le learning rate par défaut d'Adam est 0.001. On prend 10x moins.
low_learning_rate_optimizer = Adam(learning_rate=0.0001)

model.compile(optimizer=low_learning_rate_optimizer, loss='binary_crossentropy', metrics=['accuracy'])


# --- 4. AFFINAGE (FINE-TUNING) ---
print("\n--- Début de l'affinage (Fine-tuning) ---")

history = model.fit(
    X_train, y_train,
    epochs=10, # On peut faire quelques epochs de plus car le learning rate est bas
    batch_size=32,
    validation_data=(X_test, y_test) # On observe la performance sur le set de test à chaque epoch
)


# --- 5. ÉVALUATION FINALE ET SAUVEGARDE ---
print("\n--- Évaluation du modèle affiné sur les données de test ---")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Perte finale (Loss) : {loss:.4f}")
print(f"Précision finale (Accuracy) : {accuracy*100:.2f}%")

model.save(FINETUNED_MODEL_SAVE_PATH)
print(f"\nModèle affiné pour {TICKER_TO_FINETUNE} sauvegardé dans : {FINETUNED_MODEL_SAVE_PATH}")