import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- 1. CONFIGURATION DU BACKTEST HORAIRE ---
TICKER_TO_BACKTEST = 'TSLA'
MODEL_PATH = 'ultimate_hourly_model.keras' 
DATA_DIR = 'donnees_horaire_ULTRA_enrichies'

INITIAL_CASH = 10000.0
# On assouplit légèrement les seuils pour encourager le trading horaire
BUY_CONFIDENCE_THRESHOLD = 0.52
SELL_CONFIDENCE_THRESHOLD = 0.48

print(f"--- Lancement du Backtest Horaire pour {TICKER_TO_BACKTEST} ---")

# --- 2. CHARGEMENT DES DONNÉES "ULTRA" ET DU MODÈLE "ULTIME" ---
if not os.path.exists(MODEL_PATH):
    print(f"ERREUR : Le modèle '{MODEL_PATH}' n'a pas été trouvé.")
    exit()

model = load_model(MODEL_PATH)

data_path = os.path.join(DATA_DIR, f"{TICKER_TO_BACKTEST}_hourly_data.csv")
if not os.path.exists(data_path):
    print(f"ERREUR : Le fichier de données '{data_path}' est introuvable.")
    exit()

df = pd.read_csv(data_path, index_col='datetime', parse_dates=True)
df.dropna(inplace=True)

# Définition dynamique des features pour être 100% compatible
features_to_drop = ['Target', 'adj close']
features = [col for col in df.columns if col not in features_to_drop]

scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(df[features])
y = df['Target']

look_back = 70
X_sequences, y_sequences = [], []
for i in range(len(X_scaled) - look_back):
    X_sequences.append(X_scaled[i:i+look_back])
    y_sequences.append(y.iloc[i + look_back - 1])
X_sequences = np.array(X_sequences)

# Division SANS mélange pour simuler le passage du temps
train_size = int(len(X_sequences) * 0.8)
X_test = X_sequences[train_size:]
df_test = df.iloc[train_size + look_back:]

# --- 3. LE SIMULATEUR DE TRADING HORAIRE ---
print(f"Simulation du trading sur {len(X_test)} heures de bourse...")

cash = INITIAL_CASH
shares = 0
portfolio_history = []
trades_history = []

# tqdm pour voir la simulation progresser
for i in tqdm(range(len(X_test)), desc="Simulation horaire"):
    current_portfolio_value = cash + shares * df_test['close'].iloc[i]
    portfolio_history.append(current_portfolio_value)
    
    current_sequence = np.expand_dims(X_test[i], axis=0)
    prediction_proba = model.predict(current_sequence, verbose=0)[0][0] # verbose=0 pour ne pas spammer la console
    
    if shares == 0 and prediction_proba > BUY_CONFIDENCE_THRESHOLD:
        shares_to_buy = cash / df_test['close'].iloc[i]
        shares = shares_to_buy
        cash = 0
        trades_history.append(f"ACHAT @ {df_test.index[i]}: {shares:.2f} actions à {df_test['close'].iloc[i]:.2f}$ (Confiance: {prediction_proba:.2%})")

    elif shares > 0 and prediction_proba < SELL_CONFIDENCE_THRESHOLD:
        cash_from_sale = shares * df_test['close'].iloc[i]
        cash = cash_from_sale
        shares = 0
        trades_history.append(f"VENTE @ {df_test.index[i]}: {cash_from_sale:.2f}$ (Confiance: {1-prediction_proba:.2%})")
        
# --- 4. RÉSULTATS ---
final_portfolio_value = portfolio_history[-1]
total_return_pct = ((final_portfolio_value - INITIAL_CASH) / INITIAL_CASH) * 100

buy_and_hold_start_price = df_test['close'].iloc[0]
buy_and_hold_end_price = df_test['close'].iloc[-1]
buy_and_hold_return_pct = ((buy_and_hold_end_price - buy_and_hold_start_price) / buy_and_hold_start_price) * 100

print("\n" + "="*50)
print("--- RÉSULTATS DU BACKTEST HORAIRE ---")
print("="*50)
print(f"Période de test : de {df_test.index[0]} à {df_test.index[-1]}")
print(f"Capital Initial : {INITIAL_CASH:,.2f}$")
print(f"Valeur Finale du Portefeuille : {final_portfolio_value:,.2f}$")
print("-"*(len("Valeur Finale du Portefeuille : ,.f$")+5))
print(f"Performance de l'IA : {total_return_pct:+.2f}%")
print(f"Performance 'Buy and Hold' : {buy_and_hold_return_pct:+.2f}%")
print("="*50)
print(f"Nombre de trades effectués : {len(trades_history)}")
if len(trades_history) > 0:
    print("Détail des trades :")
    for trade in trades_history:
        print(f" - {trade}")
        
# --- 5. VISUALISATION ---
plt.figure(figsize=(15, 8))
plt.plot(df_test.index, portfolio_history, label="Performance de l'IA Horaire")
buy_and_hold_value = (INITIAL_CASH / df_test['close'].iloc[0]) * df_test['close']
plt.plot(df_test.index, buy_and_hold_value, label=f"Stratégie 'Buy and Hold'", linestyle='--', alpha=0.7)
plt.title(f"Backtest du Modèle Horaire Ultime vs. Buy and Hold pour {TICKER_TO_BACKTEST}")
plt.xlabel("Date")
plt.ylabel("Valeur du Portefeuille ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('backtest_result_hourly.png')
print("\nUn graphique 'backtest_result_hourly.png' a été sauvegardé.")