import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras import mixed_precision

# ===== PARTIE 1 : ACTIVATION DE L'ACCÉLÉRATION MATÉRIELLE (TENSOR CORES) =====
print("Activation de la politique de précision mixte pour une inférence accélérée...")
# Sur les GPU modernes (RTX 20xx, 30xx, 40xx), 'mixed_float16' est la politique qui active les Tensor Cores.
try:
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    print(f"Politique de calcul activée : {mixed_precision.global_policy().name}")
except Exception as e:
    print(f"Impossible d'activer la précision mixte (peut-être pas sur un GPU compatible) : {e}")
# =============================================================================


# --- 2. CONFIGURATION DU BACKTEST ---
# L'univers des actions que notre bot a le droit de trader
UNIVERSE_OF_STOCKS = ['AMC', 'ENPH','SNOW'] 
# On utilise notre meilleur modèle généraliste, car il doit analyser plusieurs actions
MODEL_PATH = 'ultimate_hourly_model.keras'
# Le dossier contenant les données horaires les plus riches que nous ayons créées
DATA_DIR = 'donnees_horaire_ULTRA_enrichies'

# Paramètres de la simulation
INITIAL_CASH = 10000.0
BUY_CONFIDENCE_THRESHOLD = 0.52  # Seuil pour déclencher un achat
SELL_CONFIDENCE_THRESHOLD = 0.48 # Seuil pour déclencher une vente

print(f"\n--- Lancement du Backtest Multi-Actifs sur l'univers : {UNIVERSE_OF_STOCKS} ---")


# --- 3. CHARGEMENT ET PRÉPARATION DE L'ENSEMBLE DES DONNÉES ---
if not os.path.exists(MODEL_PATH):
    print(f"ERREUR : Le modèle '{MODEL_PATH}' est introuvable.")
    exit()

# Le modèle est chargé en respectant la politique de précision mixte, ce qui accélérera les .predict()
model = load_model(MODEL_PATH)
look_back = model.input_shape[1] # On récupère dynamiquement la mémoire du modèle (70)

data_dict = {}
features = None
print("Chargement et préparation des données pour chaque action de l'univers...")

for ticker in UNIVERSE_OF_STOCKS:
    data_path = os.path.join(DATA_DIR, f"{ticker}_hourly_data.csv") # Attention, le nom est _hourly_data.csv
    if not os.path.exists(data_path):
        print(f"AVERTISSEMENT: Fichier pour {ticker} introuvable, il sera ignoré.")
        continue
    
    df = pd.read_csv(data_path, index_col='datetime', parse_dates=True)
    df.dropna(inplace=True)
    
    if features is None:
        # On définit la liste de features à partir du premier fichier chargé pour garantir la cohérence
        features_to_drop = ['Target', 'adj close']
        features = [col for col in df.columns if col not in features_to_drop]
    
    # Préparation des séquences X pour chaque action
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(df[features])
    X_sequences = np.array([X_scaled[i:i+look_back] for i in range(len(X_scaled) - look_back)])
    
    data_dict[ticker] = {
        'X_sequences': X_sequences,
        'dataframe': df.iloc[look_back:] # DataFrame original, aligné avec les séquences
    }

# Trouver l'index commun : la simulation ne peut tourner que sur les heures où TOUTES les actions ont des données
common_index = sorted(list(set.intersection(*(set(data['dataframe'].index) for data in data_dict.values()))))


# --- 4. LE SIMULATEUR DE PORTFEUILLE DYNAMIQUE ---
print(f"\nSimulation sur un index de temps commun de {len(common_index)} heures...")

cash = INITIAL_CASH
position = {'ticker': None, 'shares': 0} # Le bot ne peut détenir qu'une action à la fois
portfolio_history = []
trades_history = []

for timestamp in tqdm(common_index, desc="Simulation Multi-Actifs Accélérée"):
    # Calculer la valeur actuelle du portefeuille
    current_portfolio_value = cash
    if position['ticker']:
        current_price = data_dict[position['ticker']]['dataframe'].loc[timestamp]['close']
        current_portfolio_value = position['shares'] * current_price
    portfolio_history.append(current_portfolio_value)
    
    # 1. SCANNER L'UNIVERS D'ACTIONS
    predictions = {}
    for ticker in data_dict:
        try:
            # Trouver l'index de la séquence pour cette heure et cette action
            idx = data_dict[ticker]['dataframe'].index.get_loc(timestamp)
            current_sequence = np.expand_dims(data_dict[ticker]['X_sequences'][idx], axis=0)
            predictions[ticker] = model.predict(current_sequence, verbose=0)[0][0]
        except KeyError: # Au cas où il y aurait un désalignement de données
            continue

    # 2. LOGIQUE DE DÉCISION
    # D'abord, on gère la position actuelle : faut-il vendre ?
    if position['ticker'] and predictions.get(position['ticker'], 0.5) < SELL_CONFIDENCE_THRESHOLD:
        price = data_dict[position['ticker']]['dataframe'].loc[timestamp]['close']
        cash = position['shares'] * price
        trades_history.append(f"VENTE de {position['ticker']} @ {timestamp.date()}: {cash:,.2f}$ (Confiance Baisse: {1 - predictions[position['ticker']]:.2%})")
        position = {'ticker': None, 'shares': 0}

    # Ensuite, et seulement si on est en cash, on cherche une nouvelle opportunité
    if not position['ticker']:
        best_signal = {'ticker': None, 'strength': 0, 'proba': 0.5}
        for ticker, proba in predictions.items():
            strength = abs(proba - 0.5) # Force du signal = distance par rapport à 50%
            if strength > best_signal['strength']:
                best_signal = {'ticker': ticker, 'strength': strength, 'proba': proba}

        # On agit seulement si le meilleur signal est un signal d'ACHAT et qu'il est assez fort
        if best_signal['proba'] > BUY_CONFIDENCE_THRESHOLD:
            price = data_dict[best_signal['ticker']]['dataframe'].loc[timestamp]['close']
            shares_to_buy = cash / price
            position = {'ticker': best_signal['ticker'], 'shares': shares_to_buy}
            trades_history.append(f"ACHAT de {position['ticker']} @ {timestamp.date()}: {shares_to_buy:.2f} actions à {price:.2f}$ (Confiance Hausse: {best_signal['proba']:.2%})")
            cash = 0

# --- 5. RÉSULTATS FINAUX ---
final_portfolio_value = portfolio_history[-1]
total_return_pct = ((final_portfolio_value - INITIAL_CASH) / INITIAL_CASH) * 100

print("\n" + "="*50)
print("--- RÉSULTATS DU BACKTEST MULTI-ACTIFS ---")
print("="*50)
print(f"Période de test : de {common_index[0].date()} à {common_index[-1].date()}")
print(f"Capital Initial : {INITIAL_CASH:,.2f}$")
print(f"Valeur Finale du Portefeuille : {final_portfolio_value:,.2f}$")
print("-"*(len("Valeur Finale du Portefeuille : ,.f$")+5))
print(f"Performance de l'IA : {total_return_pct:+.2f}%")
print("="*50)
print(f"Nombre de trades effectués : {len(trades_history)}")
if trades_history:
    print("Détail des trades :")
    for trade in trades_history:
        print(f" - {trade}")
else:
    print("Le bot n'a effectué aucun trade.")

# --- 6. VISUALISATION ---
plt.figure(figsize=(15, 8))
plt.plot(common_index, portfolio_history, label="Performance du Portefeuille de l'IA")
plt.title(f"Backtest du Portefeuille Dynamique sur {len(UNIVERSE_OF_STOCKS)} Actions")
plt.xlabel("Date")
plt.ylabel("Valeur du Portefeuille ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('backtest_result_multi_asset.png')
print("\nUn graphique 'backtest_result_multi_asset.png' a été sauvegardé.")