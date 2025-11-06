import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
import os

# --- 1. CHARGEMENT ET PRÉPARATION DES DONNÉES ---

# Choisir l'action pour le développement
TICKER = 'TSLA'
FILE_PATH = os.path.join('donnees_finales', f'{TICKER}_final_data.csv')

# Charger les données
df = pd.read_csv(FILE_PATH, index_col='Date', parse_dates=True)

# Séparer les "features" (données d'entrée, X) et la "target" (ce qu'on veut prédire, y)
features = ['Close', 'High', 'Low', 'Open', 'Volume', 'SMA_20', 'SMA_50', 'RSI_14', 'google_trend_score']
X = df[features]
y = df['Target']

# Mise à l'échelle des features : Très important pour les réseaux de neurones !
# Le scaler transforme toutes les valeurs pour qu'elles soient entre 0 et 1.
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)

# Création des séquences pour le LSTM
# Le LSTM a besoin de regarder une "fenêtre" de jours passés pour prédire le jour suivant.
look_back = 60  # On regarde les 60 derniers jours pour prédire le 61ème.
X_sequences, y_sequences = [], []

for i in range(len(X_scaled) - look_back):
    X_sequences.append(X_scaled[i:i+look_back])
    y_sequences.append(y.iloc[i + look_back - 1]) # La cible correspond au dernier jour de la séquence

X_sequences = np.array(X_sequences)
y_sequences = np.array(y_sequences)

# Diviser en sets d'entraînement et de test
# Important : pour les séries temporelles, on ne mélange pas les données ! (shuffle=False)
X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_sequences, test_size=0.2, shuffle=False)

print(f"Dimensions des données d'entraînement (X) : {X_train.shape}") # Ex: (800, 60, 9) -> 800 échantillons, de 60 jours, avec 9 features
print(f"Dimensions des données de test (X) : {X_test.shape}")


# --- 2. CONSTRUCTION DU MODÈLE LSTM ---

model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])), # Couche d'entrée avec la bonne forme
    
    # Première couche LSTM avec Dropout pour éviter le sur-apprentissage
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    
    # Deuxième couche LSTM
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    
    # Couche "classique" de neurones (Dense)
    Dense(units=25, activation='relu'),
    
    # Couche de sortie
    # 1 neurone car la sortie est binaire (0 ou 1)
    # Activation 'sigmoid' pour donner une probabilité entre 0 et 1
    Dense(units=1, activation='sigmoid')
])

# Compilation du modèle : on définit l'optimiseur, la fonction de perte et les métriques
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Afficher un résumé de l'architecture du modèle
model.summary()


# --- 3. ENTRAÎNEMENT ET ÉVALUATION ---

print("\n--- Début de l'entraînement du modèle ---")

# Entraînement !
history = model.fit(
    X_train, 
    y_train, 
    epochs=25,          # Nombre de fois que le modèle va voir toutes les données
    batch_size=32,      # Taille des lots de données traités à chaque fois
    validation_split=0.1 # Utiliser 10% des données d'entraînement pour la validation
)

print("\n--- Fin de l'entraînement ---")

# Évaluation du modèle sur les données de test (qu'il n'a jamais vues)
print("\n--- Évaluation du modèle sur les données de test ---")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Perte (Loss) sur le set de test : {loss:.4f}")
print(f"Précision (Accuracy) sur le set de test : {accuracy*100:.2f}%")