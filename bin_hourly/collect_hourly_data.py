import yfinance as yf
import pandas as pd
import os
from tqdm import tqdm

# --- Configuration ---
TICKERS_LIST = [
    "NVDA", "AMD", "TSLA", "COIN", "SHOP", "SQ", "PLTR", "SNOW", "NET", "U",
    "RIVN", "LCID", "PLUG", "ENPH", "MRNA", "CRSP", "TDOC", "AMC", "GME", "SPCE"
]
OUTPUT_DIR = 'donnees_bourse_horaire'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("--- Démarrage de la collecte des données horaires (max 730 jours) ---")

# --- Boucle sur tous les tickers ---
for ticker in tqdm(TICKERS_LIST, desc="Actions traitées"):
    try:
        data = yf.download(
            tickers=ticker,
            period="730d",
            interval="1h",
            auto_adjust=False,
            progress=False
        )
        
        if data.empty:
            print(f"\n   -> Aucune donnée horaire trouvée pour {ticker}.")
            continue
            
        # ===== LA CORRECTION FINALE =====
        # 1. Vérifier si les colonnes sont un MultiIndex (des tuples)
        if isinstance(data.columns, pd.MultiIndex):
            # 2. Si oui, on "aplatit" les noms en ne gardant que le premier élément de chaque tuple
            data.columns = [col[0] for col in data.columns]
        
        # 3. Maintenant qu'on est sûr que ce sont des chaînes de caractères, on met tout en minuscules.
        data.columns = [col.lower() for col in data.columns]
        # ================================
        
        # Sauvegarde
        file_path = os.path.join(OUTPUT_DIR, f"{ticker}_hourly_data.csv")
        data.to_csv(file_path)
        
    except Exception as e:
        print(f"\n   -> ERREUR INATTENDUE pour {ticker}: {e}")

print(f"\n--- Collecte terminée. Données sauvegardées dans le dossier '{OUTPUT_DIR}'. ---")