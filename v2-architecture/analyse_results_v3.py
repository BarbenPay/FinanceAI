import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os
import glob
from tqdm import tqdm

# --- CONFIGURATION ---
THRESHOLD = 0.50  # Le seuil qui a donné 192 trades (le plus statistiquement fiable)
MODEL_PATH = 'transformer_model_v3.keras'
SCALER_PATH = 'scaler_transformer.pkl'
FEATURES_PATH = 'selected_features.pkl'
DATA_DIR = 'data_processed'
SPLIT_DATE = '2024-06-01'
SEQ_LEN = 60
INITIAL_CAPITAL = 10000
FIXED_BET_SIZE = 2000 # Mise fixe de 2000$ par trade pour le test "Réaliste"
FEES = 0.001 

print(f"--- ANALYSE FINALE V3 (Seuil {THRESHOLD}) ---")

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
features = joblib.load(FEATURES_PATH)

test_dfs = []
files = glob.glob(f"{DATA_DIR}/*_final.csv")
for f in tqdm(files, desc="Chargement"):
    df = pd.read_csv(f, index_col=0, parse_dates=True).replace([np.inf, -np.inf], np.nan).dropna()
    df = df[df.index >= SPLIT_DATE]
    if len(df) > SEQ_LEN: 
        df['Ticker'] = os.path.basename(f).split('_')[0]
        test_dfs.append(df)

# Simulation
trade_log = []
equity_compounding = [INITIAL_CAPITAL]
equity_fixed = [INITIAL_CAPITAL]

capital_comp = INITIAL_CAPITAL
capital_fixed = INITIAL_CAPITAL

dates = []

print("Simulation trade par trade...")

# On doit trier tous les trades par date pour avoir une vraie courbe temporelle
# Astuce : on pré-calcule tout
all_signals = []

for df in test_dfs:
    try:
        X = scaler.transform(df[features])
    except: continue
        
    X_seq = []
    for i in range(SEQ_LEN, len(X)):
        X_seq.append(X[i-SEQ_LEN:i])
    X_seq = np.array(X_seq)
    
    if len(X_seq) == 0: continue
    
    preds = model.predict(X_seq, verbose=0, batch_size=2048)
    real_prices = df['close'].iloc[SEQ_LEN:].values
    idx_dates = df.index[SEQ_LEN:]
    
    position = 0
    entry_price = 0
    entry_date = None
    
    for j, probas in enumerate(preds):
        price = real_prices[j]
        date = idx_dates[j]
        
        # Logique Achat
        if position == 0 and probas[2] > THRESHOLD:
            position = 1
            entry_price = price
            entry_date = date
            
        # Logique Vente
        elif position == 1:
            if probas[0] > THRESHOLD or probas[1] > 0.8:
                position = 0
                exit_price = price
                raw_ret = (exit_price - entry_price) / entry_price
                
                all_signals.append({
                    'Date': date,
                    'Ticker': df['Ticker'].iloc[0],
                    'Return': raw_ret
                })

# Tri chronologique (Essentiel pour voir l'évolution réelle)
all_signals.sort(key=lambda x: x['Date'])

print(f"\nTraitement de {len(all_signals)} trades chronologiques...")

for trade in all_signals:
    ret = trade['Return']
    
    # 1. Mode Compounding (Risqué)
    # On mise tout le capital courant
    amount_comp = capital_comp
    gain_comp = amount_comp * ret
    cost_comp = amount_comp * FEES * 2
    capital_comp += (gain_comp - cost_comp)
    equity_compounding.append(capital_comp)
    
    # 2. Mode Fixe (Prudent)
    # On mise toujours 2000$ (ou le reste du capital si < 2000)
    amount_fixed = min(FIXED_BET_SIZE, capital_fixed)
    gain_fixed = amount_fixed * ret
    cost_fixed = amount_fixed * FEES * 2
    capital_fixed += (gain_fixed - cost_fixed)
    equity_fixed.append(capital_fixed)
    
    trade_log.append(trade)
    dates.append(trade['Date'])

# Résultats
total_return_comp = ((capital_comp - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
total_return_fixed = ((capital_fixed - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100

print("\n" + "="*40)
print(f"RÉSULTAT FINAL (192 Trades)")
print(f"Mode Compounding (Théorique) : {total_return_comp:+.2f}% -> ${capital_comp:,.0f}")
print(f"Mode Fixed Bet (Réaliste)    : {total_return_fixed:+.2f}% -> ${capital_fixed:,.0f}")
print("="*40)

# Graphique
plt.figure(figsize=(12, 6))

# Courbe 1 : Réaliste (Échelle de gauche)
ax1 = plt.gca()
ax1.plot(range(len(equity_fixed)), equity_fixed, color='blue', label='Capital (Mise Fixe 2000$)')
ax1.set_ylabel('Capital (Mode Prudent)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Courbe 2 : Compounding (Échelle de droite car exponentielle)
ax2 = ax1.twinx()
ax2.plot(range(len(equity_compounding)), equity_compounding, color='orange', linestyle='--', label='Capital (Compounding)')
ax2.set_ylabel('Capital (Mode Théorique)', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

plt.title(f"Backtest V3 Chronologique : Transformer SOTA (Seuil {THRESHOLD})")
plt.grid(True, alpha=0.3)
plt.savefig('equity_curve_v3.png')
print("\nGraphique de performance généré : 'equity_curve_v3.png'")