import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import os
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION DU BACKTEST ---

TICKER_TO_BACKTEST = 'TSLA'
# On utilise notre meilleur modèle pour TESLA : le modèle V1 qui a été affiné !
MODEL_PATH = f'finetuned_model_{TICKER_TO_BACKTEST}.keras' 

# Paramètres de la simulation
INITIAL_CASH = 10000.0
BUY_CONFIDENCE_THRESHOLD = 0.55  # On achète seulement si le modèle est sûr à >55%
SELL_CONFIDENCE_THRESHOLD = 0.45 # On vend seulement si le modèle prédit une baisse (<50%) avec conviction

print(f"--- Lancement du Backtest pour {TICKER_TO_BACKTEST} ---")

# --- 2. CHARGEMENT DES DONNÉES ET DU MODÈLE ---

if not os.path.exists(MODEL_PATH):
    print(f"ERREUR : Le modèle '{MODEL_PATH}' n'a pas été trouvé.")
    exit()

# Charger le cerveau de l'IA
model = load_model(MODEL_PATH)

# Charger les données brutes
FILE_PATH = os.path.join('donnees_finales', f'{TICKER_TO_BACKTEST}_final_data.csv')
df = pd.read_csv(FILE_PATH, index_col='Date', parse_dates=True)
features = ['Close', 'High', 'Low', 'Open', 'Volume', 'SMA_20', 'SMA_50', 'RSI_14', 'google_trend_score']

# Préparation (scaling et séquençage), comme pour l'entraînement
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(df[features])
y = df['Target']
look_back = 60
X_sequences, y_sequences = [], []
for i in range(len(X_scaled) - look_back):
    X_sequences.append(X_scaled[i:i+look_back])
    y_sequences.append(y.iloc[i + look_back - 1])
X_sequences = np.array(X_sequences)

# Division SANS mélange pour garder la chronologie
# Nous allons simuler le trading sur la partie 'test' du dataset
train_size = int(len(X_sequences) * 0.8)
X_train, X_test = X_sequences[:train_size], X_sequences[train_size:]
df_test = df.iloc[train_size + look_back:] # DataFrame original correspondant au set de test

# --- 3. LE SIMULATEUR DE TRADING ---
print(f"Simulation du trading sur {len(X_test)} jours...")

cash = INITIAL_CASH
shares = 0
portfolio_history = []
trades_history = [] # Pour garder une trace des trades

for i in range(len(X_test)):
    current_portfolio_value = cash + shares * df_test['Close'].iloc[i]
    portfolio_history.append(current_portfolio_value)
    
    # Préparer la séquence actuelle pour la prédiction
    current_sequence = np.expand_dims(X_test[i], axis=0)
    
    # Faire la prédiction pour le LENDEMAIN
    prediction_proba = model.predict(current_sequence)[0][0]
    
    # --- Application de la stratégie ---
    # Si on n'a pas d'action et que le signal est fort à la hausse
    if shares == 0 and prediction_proba > BUY_CONFIDENCE_THRESHOLD:
        # Achat !
        shares_to_buy = cash / df_test['Close'].iloc[i]
        shares = shares_to_buy
        cash = 0
        trades_history.append(f"ACHAT @ {df_test.index[i].date()}: {shares:.2f} actions à {df_test['Close'].iloc[i]:.2f}$")

    # Si on a une action et que le signal est fort à la baisse
    elif shares > 0 and prediction_proba < SELL_CONFIDENCE_THRESHOLD:
        # Vente !
        cash_from_sale = shares * df_test['Close'].iloc[i]
        cash = cash_from_sale
        shares = 0
        trades_history.append(f"VENTE @ {df_test.index[i].date()}: {cash_from_sale:.2f}$")
        
# --- 4. CALCUL DES RÉSULTATS ---
final_portfolio_value = portfolio_history[-1]
total_return_pct = ((final_portfolio_value - INITIAL_CASH) / INITIAL_CASH) * 100

# Calcul de la performance "Buy and Hold"
buy_and_hold_start_price = df_test['Close'].iloc[0]
buy_and_hold_end_price = df_test['Close'].iloc[-1]
buy_and_hold_return_pct = ((buy_and_hold_end_price - buy_and_hold_start_price) / buy_and_hold_start_price) * 100

print("\n--- RÉSULTATS DU BACKTEST ---")
print(f"Période de test : de {df_test.index[0].date()} à {df_test.index[-1].date()}")
print(f"Capital Initial : {INITIAL_CASH:,.2f}$")
print(f"Valeur Finale du Portefeuille : {final_portfolio_value:,.2f}$")
print("---------------------------------")
print(f"Performance de l'IA : {total_return_pct:+.2f}%")
print(f"Performance 'Buy and Hold' : {buy_and_hold_return_pct:+.2f}%")
print("---------------------------------")
print(f"Nombre de trades effectués : {len(trades_history)}")
if len(trades_history) > 0:
    print("Détail des trades :")
    for trade in trades_history:
        print(f" - {trade}")
        
# --- 5. VISUALISATION (Optionnel mais très utile) ---
plt.figure(figsize=(14, 7))
plt.plot(df_test.index, portfolio_history, label="Performance de l'IA")
# Tracer la courbe Buy & Hold
buy_and_hold_value = (INITIAL_CASH / df_test['Close'].iloc[0]) * df_test['Close']
plt.plot(df_test.index, buy_and_hold_value, label=f"Stratégie 'Buy and Hold'", linestyle='--')

plt.title(f"Backtest du Modèle IA vs. Buy and Hold pour {TICKER_TO_BACKTEST}")
plt.xlabel("Date")
plt.ylabel("Valeur du Portefeuille ($)")
plt.legend()
plt.grid(True)
plt.savefig('backtest_result.png')
print("\nUn graphique 'backtest_result.png' a été sauvegardé.")