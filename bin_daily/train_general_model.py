import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
import os
import glob # Pour trouver tous les fichiers facilement

print(f"TensorFlow Version: {tf.__version__}")
print(f"GPU Disponibles: {tf.config.list_physical_devices('GPU')}")

# --- 1. CHARGEMENT ET COMBINAISON DES DONNÉES DE TOUTES LES ACTIONS ---

# Chemin vers les données finales
PATH_TO_DATA = 'donnees_finales'
all_files = glob.glob(os.path.join(PATH_TO_DATA, "*.csv"))

all_X_scaled_list = []
all_y_list = []

# Colonnes qui seront utilisées comme features
features = ['Close', 'High', 'Low', 'Open', 'Volume', 'SMA_20', 'SMA_50', 'RSI_14', 'google_trend_score']

print(f"Chargement et traitement de {len(all_files)} fichiers d'actions...")

for file in all_files:
    df = pd.read_csv(file, index_col='Date', parse_dates=True)
    
    # Étape cruciale : On met à l'échelle chaque action INDIVIDUELLEMENT
    # Car un prix de 500$ pour NVDA n'a pas la même signification qu'un prix de 10$ pour AMC
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(df[features])
    
    all_X_scaled_list.append(X_scaled)
    all_y_list.append(df['Target'])

# Combiner toutes les données en de grands tableaux numpy et pandas
combined_X_scaled = np.concatenate(all_X_scaled_list, axis=0)
combined_y = pd.concat(all_y_list)

print(f"Toutes les données sont combinées. Dimensions X: {combined_X_scaled.shape}, Dimensions y: {combined_y.shape}")

# --- 2. CRÉATION DES SÉQUENCES ---

look_back = 60
X_sequences, y_sequences = [], []

for i in range(len(combined_X_scaled) - look_back):
    X_sequences.append(combined_X_scaled[i:i+look_back])
    y_sequences.append(combined_y.iloc[i + look_back - 1])

X_sequences = np.array(X_sequences)
y_sequences = np.array(y_sequences)

print(f"Séquences créées. Dimensions X: {X_sequences.shape}")

# --- 3. SÉPARATION EN DONNÉES D'ENTRAÎNEMENT ET DE TEST ---

# Changement majeur : Maintenant, nous MÉLANGEONS les données (shuffle=True)
# Chaque séquence de 60 jours est une "mini histoire" indépendante. Les mélanger
# force le modèle à apprendre des schémas généraux.
X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_sequences, test_size=0.2, shuffle=True, random_state=42)

print(f"Données prêtes. Entraînement sur {len(X_train)} échantillons, test sur {len(X_test)} échantillons.")


# --- 4. CONSTRUCTION DU MODÈLE (la même architecture qu'avant) ---

model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=25, activation='relu'),
    Dense(units=1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# --- 5. ENTRAÎNEMENT DU MODÈLE GÉNÉRALISTE ---

print("\n--- Début de l'entraînement du modèle généraliste ---")
history = model.fit(
    X_train, y_train, 
    epochs=25,
    batch_size=64, # On peut augmenter le batch size car on a plus de données
    validation_split=0.1
)

# --- 6. ÉVALUATION ET SAUVEGARDE ---
print("\n--- Évaluation du modèle généraliste sur les données de test ---")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Perte (Loss) sur le set de test : {loss:.4f}")
print(f"Précision (Accuracy) sur le set de test : {accuracy*100:.2f}%")

# Sauvegarder le cerveau de notre IA !
MODEL_SAVE_PATH = "financial_model_generalist_v1.keras"
model.save(MODEL_SAVE_PATH)
print(f"\nModèle entraîné et sauvegardé avec succès dans : {MODEL_SAVE_PATH}")