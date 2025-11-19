import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import os
import glob
from tqdm import tqdm

# --- CONFIG V3 ---
MODEL_PATH = 'transformer_model_v3.keras'
SCALER_PATH = 'scaler_transformer.pkl'
FEATURES_PATH = 'selected_features.pkl'
DATA_DIR = 'data_processed'
SPLIT_DATE = '2024-06-01'
SEQ_LEN = 60
INITIAL_CAPITAL = 10000
FEES = 0.001 

print("--- BACKTEST FINAL V3 (TRANSFORMER SOTA) ---")

# 1. Chargement
if not os.path.exists(MODEL_PATH):
    print("ERREUR: Modèle manquant.")
    exit()

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
features = joblib.load(FEATURES_PATH)
print(f"Features ({len(features)}): {features}")

# 2. Préparation Data
test_dfs = []
files = glob.glob(f"{DATA_DIR}/*_final.csv")

for f in tqdm(files, desc="Chargement"):
    df = pd.read_csv(f, index_col=0, parse_dates=True).replace([np.inf, -np.inf], np.nan).dropna()
    df = df[df.index >= SPLIT_DATE]
    if len(df) > SEQ_LEN: test_dfs.append(df)

# 3. Moteur de Simulation
def run_simulation(threshold):
    capital = INITIAL_CAPITAL
    trades = 0
    wins = 0
    
    for df in test_dfs:
        # Préparation spécifique avec les features SOTA
        try:
            X = scaler.transform(df[features])
        except Exception as e:
            print(f"Erreur feature: {e}")
            continue
            
        X_seq = []
        for i in range(SEQ_LEN, len(X)):
            X_seq.append(X[i-SEQ_LEN:i])
        X_seq = np.array(X_seq)
        
        if len(X_seq) == 0: continue

        # Prédiction Rapide
        preds = model.predict(X_seq, verbose=0, batch_size=2048)
        
        # Trading Loop
        position = 0 
        entry_price = 0
        real_prices = df['close'].iloc[SEQ_LEN:].values
        
        for j, probas in enumerate(preds):
            price = real_prices[j]
            
            # BUY SIGNAL (Classe 2)
            if position == 0 and probas[2] > threshold:
                position = 1
                entry_price = price
                capital -= capital * FEES
                trades += 1
            
            # SELL SIGNAL (Classe 0) ou TAKE PROFIT sur signal WAIT fort
            elif position == 1:
                # On vend si le modèle dit Vendre (0) OU s'il est très confiant sur Wait (1)
                # (Si le modèle est sûr qu'il ne se passe rien, on sort pour ne pas payer de frais de swap/risque)
                if probas[0] > threshold or probas[1] > 0.8: 
                    position = 0
                    ret = (price - entry_price) / entry_price
                    capital += capital * ret
                    capital -= capital * FEES
                    if ret > 0: wins += 1

    roi = ((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    wr = (wins / trades * 100) if trades > 0 else 0
    print(f"Seuil {threshold:.2f} -> ROI: {roi:+.2f}% | Trades: {trades} | WR: {wr:.1f}%")

# 4. Scan
for t in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
    run_simulation(t)