import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
import os
import glob

print(f"--- Entraînement du Modèle V2 (avec Sentiment des News) ---")
print(f"TensorFlow Version: {tf.__version__}")

# --- 1. CHARGEMENT DES DONNÉES DE SENTIMENT ---
NEWS_SENTIMENT_PATH = 'daily_news_sentiment.csv'
if not os.path.exists(NEWS_SENTIMENT_PATH):
    print(f"ERREUR : Fichier de sentiment '{NEWS_SENTIMENT_PATH}' introuvable.")
    exit()

print("Chargement des données de sentiment...")
df_sentiment = pd.read_csv(NEWS_SENTIMENT_PATH, parse_dates=['Date'])


# --- 2. CHARGEMENT, FUSION ET PRÉPARATION DES DONNÉES D'ACTIONS ---
PATH_TO_DATA = 'donnees_finales' # Le dossier avec SMA, RSI, Google Trends
all_files = glob.glob(os.path.join(PATH_TO_DATA, "*.csv"))

# Changement : on ajoute la nouvelle feature à notre liste
features = ['Close', 'High', 'Low', 'Open', 'Volume', 'SMA_20', 'SMA_50', 'RSI_14', 'google_trend_score', 'NewsSentiment']

all_dataframes = []

print(f"Chargement et fusion pour {len(all_files)} fichiers d'actions...")
for file in all_files:
    # Charger les données boursières
    df_stock = pd.read_csv(file, index_col='Date', parse_dates=True)
    ticker = os.path.basename(file).replace('_final_data.csv', '')
    
    # Isoler les données de sentiment pour ce ticker
    df_sentiment_ticker = df_sentiment[df_sentiment['Ticker'] == ticker].set_index('Date')[['NewsSentiment']]
    
    # Fusionner les deux DataFrames. On utilise une 'left' join pour garder tous les jours de bourse.
    df_merged = df_stock.merge(df_sentiment_ticker, left_index=True, right_index=True, how='left')
    
    # ===== GESTION DES DONNÉES DE SENTIMENT MANQUANTES =====
    # 1. Forward Fill : Propager la dernière valeur connue
    df_merged['NewsSentiment'].fillna(method='ffill', inplace=True)
    # 2. Back Fill (pour le début du dataset) : Remplir les premières valeurs avec la première news connue
    df_merged['NewsSentiment'].fillna(method='bfill', inplace=True)
    # 3. Fillna(0) : S'il n'y a eu AUCUNE news pour cette action, on met tout à 0 (neutre)
    df_merged['NewsSentiment'].fillna(0, inplace=True)
    
    # Assurer que toutes les features sont présentes
    df_merged = df_merged[features + ['Target']]
    all_dataframes.append(df_merged)


# --- 3. PRÉPARATION FINALE ET SÉQUENÇAGE (similaire au script précédent) ---
all_X_scaled_list = []
all_y_list = []

for df in all_dataframes:
    scaler = MinMaxScaler(feature_range=(0, 1))
    # On scale les 10 features
    X_scaled = scaler.fit_transform(df[features])
    
    all_X_scaled_list.append(X_scaled)
    all_y_list.append(df['Target'])

combined_X_scaled = np.concatenate(all_X_scaled_list, axis=0)
combined_y = pd.concat(all_y_list)
print(f"Données finales combinées et mises à l'échelle. Dimensions X: {combined_X_scaled.shape}")

look_back = 60
X_sequences, y_sequences = [], []

for i in range(len(combined_X_scaled) - look_back):
    X_sequences.append(combined_X_scaled[i:i+look_back])
    y_sequences.append(combined_y.iloc[i + look_back - 1])

X_sequences = np.array(X_sequences)
y_sequences = np.array(y_sequences)

print(f"Séquences créées. Dimensions X: {X_sequences.shape}") # La forme doit maintenant être (..., 60, 10)

# Séparation avec mélange
X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_sequences, test_size=0.2, shuffle=True, random_state=42)

# --- 4. CONSTRUCTION ET ENTRAÎNEMENT DU MODÈLE V2 ---
# L'architecture reste la même, seule la forme de l'input change.
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])), # shape[2] sera maintenant 10 !
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=25, activation='relu'),
    Dense(units=1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary() # Vérifiez que l'Input Shape est bien (None, 60, 10)

print("\n--- Début de l'entraînement du modèle V2 ---")
history = model.fit(
    X_train, y_train, 
    epochs=25,
    batch_size=64,
    validation_split=0.1
)

# --- 5. ÉVALUATION ET SAUVEGARDE ---
print("\n--- Évaluation du modèle V2 sur les données de test ---")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Perte (Loss) sur le set de test : {loss:.4f}")
print(f"Précision (Accuracy) sur le set de test : {accuracy*100:.2f}%")

MODEL_SAVE_PATH = "financial_model_generalist_v2.keras"
model.save(MODEL_SAVE_PATH)
print(f"\nModèle V2 sauvegardé dans : {MODEL_SAVE_PATH}")