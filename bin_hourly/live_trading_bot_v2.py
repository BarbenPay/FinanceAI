import os
from curl_cffi import requests
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras import mixed_precision
import yfinance as yf
import pandas_ta as ta
from datetime import datetime
import json

# --- AJOUTÉ : CRÉATION DU DOSSIER DE DÉBOGAGE ---
DEBUG_CSV_PATH = 'debug_data'
os.makedirs(DEBUG_CSV_PATH, exist_ok=True)
# ----------------------------------------------

# Session pour yfinance (déjà corrigé)
session = requests.Session(impersonate="chrome110")

# --- 1. ACTIVATION ACCÉLÉRATION MATÉRIELLE ---
print("Activation de la politique de précision mixte...")
try:
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    print(f"Politique de calcul activée : {mixed_precision.global_policy().name}")
except Exception:
    print("Impossible d'activer la précision mixte.")

# --- 2. CONFIGURATION DU BOT ---
UNIVERSE_OF_STOCKS = [
    "NVDA", "AMD", "TSLA", "COIN", "SHOP", "PLTR", "SNOW", "NET", "U",
    "RIVN", "LCID", "PLUG", "ENPH", "MRNA", "CRSP", "TDOC", "AMC", "GME", "SPCE"
]
MODEL_PATH = 'ultimate_hourly_model.keras'
PORTFOLIO_FILE = 'portfolio_prudent.json'
INITIAL_CASH = 10000.0
BUY_CONFIDENCE_THRESHOLD = 0.52
SELL_CONFIDENCE_THRESHOLD = 0.48
MAX_CAPITAL_PER_POSITION = 0.25
MAX_OPEN_POSITIONS = 4

print(f"\n--- BOT DE TRADING PRUDENT ---")
if os.path.exists(PORTFOLIO_FILE):
    with open(PORTFOLIO_FILE, 'r') as f:
        portfolio = json.load(f)
    portfolio['last_update'] = pd.to_datetime(portfolio['last_update'], utc=True)
else:
    now = datetime.now()
    portfolio = {'start_date': str(now),'last_update': str(now),'cash': INITIAL_CASH,'positions': {},'history': []}
    with open(PORTFOLIO_FILE, 'w') as f: json.dump(portfolio, f, indent=4)
    portfolio['last_update'] = pd.to_datetime(portfolio['last_update'], utc=True)

start_of_catchup = portfolio['last_update']
end_of_catchup = pd.to_datetime(datetime.now(), utc=True)
if (end_of_catchup - start_of_catchup).total_seconds() < 3600:
    print("\nDernière mise à jour trop récente. Aucune action nécessaire.")
    exit()

print(f"\nPériode à simuler : de {start_of_catchup} à {end_of_catchup}")
fetch_start_date = start_of_catchup - pd.Timedelta(days=20) 
all_market_data = yf.download(tickers=UNIVERSE_OF_STOCKS, start=fetch_start_date, end=end_of_catchup, interval='1h', auto_adjust=False, group_by='ticker', progress=False,session=session)
if all_market_data.empty:
    print("Aucune nouvelle donnée de marché.")
    exit()

processed_data = {}
model = load_model(MODEL_PATH)
look_back = model.input_shape[1]
features = None

for ticker in UNIVERSE_OF_STOCKS:
    # --- MODIFIÉ : Correction de la condition de boucle ---
    if ticker not in all_market_data.columns.get_level_values(0):
        print(f"Aucune donnée brute téléchargée pour {ticker}, passage au suivant.")
        continue
    # ----------------------------------------------------
    
    df = all_market_data[ticker].copy()
    #df.columns = df.columns.droplevel(1) # Aplatir le MultiIndex
    df.columns = [col.lower() for col in df.columns]

    # --- AJOUTÉ : Sauvegarde des données BRUTES pour débogage ---
    raw_filepath = os.path.join(DEBUG_CSV_PATH, f"{ticker}_raw.csv")
    df.to_csv(raw_filepath)
    # -----------------------------------------------------------
    
    # Calcul de tous les indicateurs...
    df.ta.sma(length=8, append=True); df.ta.sma(length=40, append=True)
    df.ta.rsi(length=14, append=True); df.ta.macd(append=True)
    df.ta.bbands(length=40, append=True); df.ta.obv(append=True)
    df.ta.atr(length=14, append=True); df.ta.stochrsi(length=14, append=True)
    df.ta.cci(length=20, append=True); df.ta.adx(length=14, append=True)
    df.ta.aroon(length=14, append=True); df.ta.willr(length=14, append=True)
    
    df.dropna(inplace=True)

    # --- AJOUTÉ : Sauvegarde des données TRAITÉES pour débogage ---
    if not df.empty:
        processed_filepath = os.path.join(DEBUG_CSV_PATH, f"{ticker}_processed.csv")
        df.to_csv(processed_filepath)
    # -------------------------------------------------------------
    
    if df.empty:
        print(f"Le DataFrame pour {ticker} est vide après l'ajout des indicateurs et dropna().")
        continue
    
    if features is None:
        features = [col for col in df.columns if col not in ['adj close']]
        
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    processed_data[ticker] = df

# --- MODIFIÉ : Correction du calcul de l'index commun ---
if not processed_data:
    common_index = []
else:
    list_of_sets = [set(df.index) for df in processed_data.values()]
    # On prend le premier set et on trouve l'intersection avec tous les autres
    intersection_result = list_of_sets[0].intersection(*list_of_sets[1:])
    common_index = sorted(list(intersection_result))
# ------------------------------------------------------

catchup_index = [ts for ts in common_index if ts >= start_of_catchup and ts <= end_of_catchup]

# --- BOUCLE DE SIMULATION ---
print(f"Début de la simulation heure par heure sur {len(catchup_index)} nouvelles périodes...")
for timestamp in tqdm(catchup_index, desc="Simulation Prudente"):
    total_portfolio_value = portfolio['cash']
    for ticker, data in portfolio['positions'].items():
        if timestamp in all_market_data[ticker].index:
            current_price = all_market_data[ticker].loc[timestamp]['Close']
            total_portfolio_value += data['shares'] * current_price

    sequences_to_predict, tickers_in_batch = [], []
    for ticker, df in processed_data.items():
        try:
            end_loc = df.index.get_loc(timestamp)
            start_loc = end_loc - look_back
            if start_loc < 0: continue
            sequence = df.iloc[start_loc:end_loc][features].values
            sequences_to_predict.append(sequence)
            tickers_in_batch.append(ticker)
        except (KeyError, IndexError): continue
            
    if not sequences_to_predict: continue
    all_probas = model.predict(np.array(sequences_to_predict), verbose=0)
    predictions = {ticker: proba[0] for ticker, proba in zip(tickers_in_batch, all_probas)}

    positions_to_sell = [ticker for ticker in portfolio['positions'] if predictions.get(ticker, 0.5) < SELL_CONFIDENCE_THRESHOLD]
    for ticker in positions_to_sell:
        price = all_market_data[ticker].loc[timestamp]['Close']
        cash_from_sale = portfolio['positions'][ticker]['shares'] * price
        portfolio['cash'] += cash_from_sale
        trade_log = f"{timestamp} VENTE de {ticker}: {portfolio['positions'][ticker]['shares']:.2f} actions à {price:.2f}$ = {cash_from_sale:,.2f}$"
        print(f"\n{trade_log}"); portfolio['history'].append(trade_log)
        del portfolio['positions'][ticker]

    available_slots = MAX_OPEN_POSITIONS - len(portfolio['positions'])
    if available_slots > 0 and portfolio['cash'] > 0:
        buy_signals = {ticker: proba for ticker, proba in predictions.items() if proba > BUY_CONFIDENCE_THRESHOLD and ticker not in portfolio['positions']}
        if buy_signals:
            sorted_signals = sorted(buy_signals.items(), key=lambda item: item[1], reverse=True)
            capital_per_new_position = (total_portfolio_value * MAX_CAPITAL_PER_POSITION)
            for i in range(min(len(sorted_signals), available_slots)):
                ticker, proba = sorted_signals[i]
                investment_amount = min(capital_per_new_position, portfolio['cash'])
                if investment_amount <= 0: break
                price = all_market_data[ticker].loc[timestamp]['Close']
                shares_to_buy = investment_amount / price
                portfolio['cash'] -= investment_amount
                portfolio['positions'][ticker] = {'shares': shares_to_buy, 'buy_price': price}
                trade_log = f"{timestamp} ACHAT de {ticker}: {shares_to_buy:.2f} actions à {price:.2f}$ (Capital Alloué: {investment_amount:,.2f}$)"
                print(f"\n{trade_log}"); portfolio['history'].append(trade_log)

# --- Sauvegarde du portefeuille ---
portfolio['last_update'] = str(end_of_catchup)
with open(PORTFOLIO_FILE, 'w') as f:
    json.dump(portfolio, f, indent=4)
print("\n--- Simulation terminée. Portefeuille mis à jour et sauvegardé. ---")

# Rapport final
final_value = portfolio['cash']
print(f"\nCash disponible : {portfolio['cash']:,.2f}$")
if portfolio['positions']:
    print("Positions Actuelles :")
    for ticker, data in portfolio['positions'].items():
        if not all_market_data[ticker].empty:
            last_price = all_market_data[ticker].iloc[-1]['Close']
            position_value = data['shares'] * last_price
            final_value += position_value
            print(f" - {ticker}: {data['shares']:.2f} actions, valeur: {position_value:,.2f}$")
print("-" * 30)
print(f"Valeur Totale du Portefeuille : {final_value:,.2f}$")
print(f"Performance depuis le début : {( (final_value - INITIAL_CASH) / INITIAL_CASH ):+.2%}")