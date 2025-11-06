# Créez ce script, exécutez-le une fois, puis passez au tuner.
# C'est une version finale de votre add_hourly_features.py
import pandas as pd
import pandas_ta as ta
import os
from tqdm import tqdm

INPUT_DIR = 'donnees_bourse_horaire'
OUTPUT_DIR = 'donnees_horaire_ULTRA_enrichies' # Le dernier dossier !

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

files_to_process = [f for f in os.listdir(INPUT_DIR) if f.endswith('.csv')]
for filename in tqdm(files_to_process, desc="Super-Enrichissement Ultime"):
    try:
        df = pd.read_csv(os.path.join(INPUT_DIR, filename))
        date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col], utc=True)
        df.set_index(date_col, inplace=True)
        df.index.name = 'datetime'
        
        # TOUS nos indicateurs
        df.ta.sma(length=8, append=True)
        df.ta.sma(length=40, append=True)
        df.ta.rsi(length=14, append=True)
        df.ta.macd(append=True)
        df.ta.bbands(length=40, append=True)
        df.ta.obv(append=True)
        df.ta.atr(length=14, append=True)
        df.ta.stochrsi(length=14, append=True)
        df.ta.cci(length=20, append=True)
        # Indicateurs ultimes
        df.ta.adx(length=14, append=True) # Force de tendance
        df.ta.aroon(length=14, append=True) # "Fraîcheur" de tendance
        df.ta.willr(length=14, append=True) # Momentum rapide
        
        df['future_close'] = df['close'].shift(-1)
        df['Target'] = (df['future_close'] > df['close']).astype(int)
        df.dropna(inplace=True)
        df.drop('future_close', axis=1, inplace=True)
        df.to_csv(os.path.join(OUTPUT_DIR, filename))
    except Exception as e:
        print(f"\nErreur sur {filename}: {e}")