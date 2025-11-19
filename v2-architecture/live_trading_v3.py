import os
import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import joblib
import json
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from tqdm import tqdm

# --- CONFIGURATION DU BOT V3 ---
# Tickers (Ton univers + les favoris)
TICKERS = [
    "NVDA", "AMD", "TSLA", "COIN", "SHOP", "PLTR", "SNOW", "NET", "U",
    "RIVN", "LCID", "PLUG", "ENPH", "MRNA", "CRSP", "TDOC", "AMC", "GME", "SPCE",
    "MARA", "MSTR"
]

# Fichiers système
MODEL_PATH = 'transformer_model_v3.keras'
SCALER_PATH = 'scaler_transformer.pkl'
FEATURES_PATH = 'selected_features.pkl'
PORTFOLIO_FILE = 'portfolio_v3_live.json'

# Paramètres Trading
INITIAL_CAPITAL = 10000.0
MAX_POSITIONS = 5             # Diversification (1/5 du capital par ligne)
CONFIDENCE_THRESHOLD = 0.50   # Seuil validé par le Backtest V3
START_DATE_SIMULATION = "2025-11-01 00:00:00" # Date de départ forçée

# Paramètres IA
SEQ_LEN = 60

print("--- BOT DE TRADING V3 (SOTA TRANSFORMER) ---")

# 1. Chargement des composants IA
if not os.path.exists(MODEL_PATH):
    print(f"ERREUR: {MODEL_PATH} manquant.")
    exit()

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
selected_features = joblib.load(FEATURES_PATH)
print(f"✅ IA chargée. Features actives : {len(selected_features)}")

# 2. Gestion du Portefeuille (JSON)
if os.path.exists(PORTFOLIO_FILE):
    with open(PORTFOLIO_FILE, 'r') as f:
        portfolio = json.load(f)
    print(f"✅ Portefeuille existant chargé. Cash: {portfolio['cash']:.2f}$")
else:
    print(f"⚠️ Initialisation nouveau portefeuille (Départ: {START_DATE_SIMULATION})")
    portfolio = {
        'last_update': START_DATE_SIMULATION, # Point de départ
        'cash': INITIAL_CAPITAL,
        'positions': {}, # {ticker: {'shares': x, 'buy_price': y}}
        'history': []
    }

# Détermination de la période de rattrapage
last_update = pd.to_datetime(portfolio['last_update'], utc=True)
now = pd.to_datetime(datetime.now(), utc=True)

if last_update >= now:
    print("Le portefeuille est déjà à jour.")
    exit()

print(f"🔄 Rattrapage du trading : {last_update} -> {now}")

# --- FONCTIONS DE DONNÉES ---

def get_macro_data(start, end):
    """Récupère la macro sur la période large"""
    tickers = {"^VIX": "macro_VIX", "^TNX": "macro_US10Y"}
    macro_df = pd.DataFrame()
    
    # Marge de sécurité de 60 jours avant pour les indicateurs
    fetch_start = start - timedelta(days=60)
    
    for sym, name in tickers.items():
        try:
            df = yf.download(sym, start=fetch_start, end=end, interval="1h", progress=False, auto_adjust=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = [c[0].lower() for c in df.columns]
            else: df.columns = [c.lower() for c in df.columns]
            
            if df.index.tz is None: df.index = df.index.tz_localize('UTC')
            else: df.index = df.index.tz_convert('UTC')

            df[f'{name}_level'] = df['close']
            df[f'{name}_ret'] = df['close'].pct_change()
            
            cols = [c for c in df.columns if 'macro_' in c]
            
            if macro_df.empty: macro_df = df[cols]
            else: macro_df = macro_df.join(df[cols], how='outer')
        except: pass
    return macro_df.ffill().bfill()

def get_ticker_data(ticker, start, end, macro_df):
    """Prépare les features pour un ticker"""
    try:
        fetch_start = start - timedelta(days=60)
        df = yf.download(ticker, start=fetch_start, end=end, interval="1h", progress=False, auto_adjust=False)
        
        if df.empty: return None

        if isinstance(df.columns, pd.MultiIndex): df.columns = [c[0].lower() for c in df.columns]
        else: df.columns = [c.lower() for c in df.columns]
        
        if df.index.tz is None: df.index = df.index.tz_localize('UTC')
        else: df.index = df.index.tz_convert('UTC')

        # Fusion Macro
        df = df.join(macro_df, how='left').ffill()

        # Feature Engineering SOTA
        df['ret_1h'] = df['close'].pct_change()
        df['ret_vol'] = df['volume'].pct_change()
        
        try:
            df.ta.bbands(length=20, std=2.0, append=True)
            df.ta.macd(fast=12, slow=26, signal=9, append=True)
            df.ta.atr(length=14, append=True)
        except: pass

        # Sélection et Nettoyage
        # On ne garde que la période "Utile" (après la dernière update)
        # Mais on a besoin des 60h d'avant pour le tenseur
        mask_useful = df.index > (start - timedelta(hours=SEQ_LEN + 5))
        df = df[mask_useful].copy()
        
        return df
    except: return None

# --- 1. TÉLÉCHARGEMENT GLOBAL ---
print("1. Téléchargement des données...")
macro_data = get_macro_data(last_update, now)
data_pool = {}

for ticker in tqdm(TICKERS, desc="Tickers"):
    df = get_ticker_data(ticker, last_update, now, macro_data)
    if df is not None and len(df) > SEQ_LEN:
        data_pool[ticker] = df

# --- 2. BOUCLE TEMPORELLE (RATTRAPAGE HEURE PAR HEURE) ---
# On doit trouver l'union de tous les index temporels pour avancer step-by-step
all_indices = sorted(list(set().union(*[d.index for d in data_pool.values()])))
simulation_indices = [t for t in all_indices if t > last_update]

print(f"2. Simulation sur {len(simulation_indices)} heures de bourse...")

for current_time in tqdm(simulation_indices, desc="Live Trading"):
    
    # A. Valorisation Portefeuille (Mark to Market)
    total_equity = portfolio['cash']
    for t, pos in portfolio['positions'].items():
        if t in data_pool and current_time in data_pool[t].index:
            price = data_pool[t].loc[current_time]['close']
            total_equity += pos['shares'] * price
        else:
            # Si pas de cotation à cette heure précise, on garde la valeur d'achat (approximatif)
            total_equity += pos['shares'] * pos['buy_price']

    # B. Analyse IA (Batch)
    predictions = {} # ticker: [sell_prob, wait_prob, buy_prob]
    
    for ticker, df in data_pool.items():
        if current_time not in df.index: continue
        
        # Construction du Tenseur (Les 60h précédentes)
        loc_idx = df.index.get_loc(current_time)
        if loc_idx < SEQ_LEN: continue
        
        try:
            # Extraction des features exactes
            X_slice = df.iloc[loc_idx-SEQ_LEN : loc_idx][selected_features]
            X_scaled = scaler.transform(X_slice)
            
            # Prédiction directe (On pourrait batcher pour aller plus vite, mais ici c'est lisible)
            seq = np.expand_dims(X_scaled, axis=0)
            pred = model.predict(seq, verbose=0)[0]
            predictions[ticker] = pred
            
        except Exception: continue

    # C. Décisions de Trading
    # 1. Ventes (Sell Signals ou Stop-Loss implicite via modèle)
    for ticker in list(portfolio['positions'].keys()):
        if ticker in predictions:
            probs = predictions[ticker]
            
            # Signal VENTE (Classe 0) OU Signal WAIT très fort (Sortie d'ennui)
            if probs[0] > CONFIDENCE_THRESHOLD or probs[1] > 0.85:
                price = data_pool[ticker].loc[current_time]['close']
                shares = portfolio['positions'][ticker]['shares']
                proceeds = shares * price
                
                portfolio['cash'] += proceeds
                del portfolio['positions'][ticker]
                
                type_vente = "SIGNAL" if probs[0] > CONFIDENCE_THRESHOLD else "WAIT_EXIT"
                msg = f"[{current_time}] 🔻 VENTE {ticker} ({type_vente}): {shares:.2f} @ {price:.2f}$ -> Cash: {portfolio['cash']:.2f}$"
                print(f"\n{msg}")
                portfolio['history'].append(msg)

    # 2. Achats (Buy Signals)
    # On calcule combien on peut investir par ligne
    # On veut max 5 positions. Donc capital_par_position = Total_Equity / 5
    target_pos_size = total_equity / MAX_POSITIONS
    available_slots = MAX_POSITIONS - len(portfolio['positions'])
    
    if available_slots > 0 and portfolio['cash'] > 100.0: # Il faut un minimum de cash
        # On trie les opportunités par confiance
        candidates = []
        for t, probs in predictions.items():
            if t not in portfolio['positions'] and probs[2] > CONFIDENCE_THRESHOLD:
                candidates.append((t, probs[2])) # (Ticker, Confiance)
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        for ticker, conf in candidates[:available_slots]:
            # On investit le max possible (Target Size ou le Cash restant)
            invest_amount = min(target_pos_size, portfolio['cash'])
            
            if invest_amount > 50.0: # Minimum trade
                price = data_pool[ticker].loc[current_time]['close']
                shares = invest_amount / price
                
                portfolio['positions'][ticker] = {
                    'shares': shares,
                    'buy_price': price,
                    'date': str(current_time)
                }
                portfolio['cash'] -= invest_amount
                
                msg = f"[{current_time}] 🚀 ACHAT {ticker} (Conf: {conf:.2f}): {shares:.2f} @ {price:.2f}$"
                print(f"\n{msg}")
                portfolio['history'].append(msg)

# --- 3. SAUVEGARDE ET RAPPORT ---
portfolio['last_update'] = str(now)
with open(PORTFOLIO_FILE, 'w') as f:
    json.dump(portfolio, f, indent=4)

print("\n" + "="*40)
print("ÉTAT DU PORTEFEUILLE LIVE V3")
print("="*40)

total_val = portfolio['cash']
print(f"LIQUIDITÉS : {portfolio['cash']:,.2f}$")

if portfolio['positions']:
    print("\nPOSITIONS OUVERTES :")
    for t, p in portfolio['positions'].items():
        # Prix actuel (le tout dernier dispo)
        try:
            curr_price = data_pool[t].iloc[-1]['close']
            val = p['shares'] * curr_price
            pnl = ((curr_price - p['buy_price']) / p['buy_price']) * 100
            total_val += val
            print(f" - {t:<6} : {p['shares']:.2f} actions | PnL: {pnl:+.2f}% | Val: {val:,.2f}$")
        except:
            print(f" - {t:<6} : Données indisponibles pour valorisation.")
            total_val += p['shares'] * p['buy_price'] # Fallback

print("-" * 40)
roi = ((total_val - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
print(f"VALEUR TOTALE : {total_val:,.2f}$")
print(f"PERFORMANCE   : {roi:+.2f}%")
print("="*40)