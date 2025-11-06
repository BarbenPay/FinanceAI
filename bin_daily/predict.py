import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from pytrends.request import TrendReq
import finnhub
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import os
import time # On importe la bibliothèque time pour la pause

# --- 0. CONFIGURATION & INITIALISATION ---

TICKER_TO_PREDICT = 'TSLA'
FINNHUB_API_KEY = "d43vudpr01qge0cvr7d0d43vudpr01qge0cvr7dg"
FINETUNED_MODEL_PATH = f'finetuned_model_{TICKER_TO_PREDICT}.keras'

# La liste de features que notre modèle connaît (SANS news_sentiment)
FEATURES = ['Close', 'High', 'Low', 'Open', 'Volume', 'SMA_20', 'SMA_50', 'RSI_14', 'google_trend_score']

print(f"--- Lancement de la Prédiction pour {TICKER_TO_PREDICT} ---")

if not os.path.exists(FINETUNED_MODEL_PATH):
    print(f"ERREUR : Modèle '{FINETUNED_MODEL_PATH}' non trouvé.")
    exit()

# --- ÉTAPE 1 : COLLECTE DES DONNÉES ---

print("1/4 : Récupération des données de marché...")
end_date = datetime.now()
start_date = end_date - timedelta(days=180)

df = yf.download(TICKER_TO_PREDICT, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), auto_adjust=False, group_by='ticker')

# ===== LA CORRECTION DÉFINITIVE =====
# On force le renommage des colonnes pour qu'elles soient simples et propres.
# La bibliothèque pandas-ta cherchera 'close' en minuscules.
df.columns = ['Open', 'High', 'Low', 'Close', 'Adj close', 'Volume']
# ====================================

if len(df) < 60:
    print(f"ERREUR : Pas assez de données de marché ({len(df)} jours trouvés).")
    exit()

print("2/4 : Calcul des indicateurs techniques...")
df.ta.sma(length=20, append=True)
df.ta.sma(length=50, append=True)
df.ta.rsi(length=14, append=True)

print("3/4 : Récupération de l'intérêt Google Trends...")
try:
    # On ajoute une pause pour être respectueux de l'API
    time.sleep(2)
    pytrends = TrendReq(hl='en-US', tz=360)
    timeframe = f'{df.index.min().strftime("%Y-%m-%d")} {df.index.max().strftime("%Y-%m-%d")}'
    pytrends.build_payload(kw_list=[TICKER_TO_PREDICT], timeframe=timeframe, geo='US')
    trend_df = pytrends.interest_over_time()

    if not trend_df.empty:
        trend_df = trend_df.drop(columns='isPartial')
        daily_trend = trend_df.reindex(df.index, method='ffill').fillna(0)
        df['google_trend_score'] = daily_trend[TICKER_TO_PREDICT]
    else:
        df['google_trend_score'] = 0
except Exception as e:
    print(f"AVERTISSEMENT: Impossible de récupérer les données Google Trends ({e}). Score mis à 0.")
    df['google_trend_score'] = 0

print("4/4 : (Non utilisé par le modèle V1) Analyse du sentiment des news...")
# Ici on mettrait la logique FinnHub pour une V2 du modèle

# --- ÉTAPE 2 : PRÉPARATION FINALE DES DONNÉES ---
print("\nPréparation des données pour le modèle...")
df.dropna(inplace=True)
last_60_days = df.tail(60)

if len(last_60_days) < 60:
    print(f"ERREUR : Pas assez de données après nettoyage ({len(last_60_days)} jours).")
    exit()

scaler = MinMaxScaler(feature_range=(0, 1))

# On ne scale que les colonnes que le modèle a appris à utiliser !
scaled_data = scaler.fit_transform(last_60_days[FEATURES])
X_predict = np.array([scaled_data])

print(f"Données d'entrée prêtes. Forme : {X_predict.shape}")

# --- ÉTAPE 3 : CHARGEMENT DU MODÈLE ET PRÉDICTION ---
print(f"Chargement du cerveau IA depuis : {FINETUNED_MODEL_PATH}...")
model = load_model(FINETUNED_MODEL_PATH)
print("Lancement de la prédiction...")
prediction_proba = model.predict(X_predict)[0][0]

# --- ÉTAPE 4 : INTERPRÉTATION ---
threshold = 0.5
prediction_class = 1 if prediction_proba >= threshold else 0
print("\n--- VERDICT FINAL ---")
print(f"Action: {TICKER_TO_PREDICT}")
print(f"Probabilité de hausse calculée : {prediction_proba * 100:.2f}%")
if prediction_class == 1:
    print("Prédiction : HAUSSE PROBABLE pour la prochaine journée de trading.")
else:
    print("Prédiction : BAISSE ou STAGNATION PROBABLE pour la prochaine journée de trading.")
print("---------------------\n")
print("AVERTISSEMENT : Modèle expérimental. Pas un conseil en investissement.")